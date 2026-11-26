//! Trains the mirrored evaluation network:
//! (768 x input buckets -> hidden)x2 -> 8 output buckets, with the board
//! mirrored so the viewer's king is on files a-d.
//!
//! usage: bucketed --data <shuffled.bin> [--hidden N] [--layout single|four]
//!                 [--id NAME] [--superbatches N] [--batches N] [--wdl X]
//!                 [--lr X] [--threads N]
//!
//! The hidden size, the bucket layout, `NUM_OUTPUT_BUCKETS`, `QA`, `QB` and
//! `SCALE` must match `src/nnue.rs` in the engine (with `MIRRORED = true`).

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const NUM_OUTPUT_BUCKETS: usize = 8;
/// King bucket per king square on files a-d (rank 1 first); e-h are mirrored.
/// `single`: mirroring only, one set of input weights.
const LAYOUT_SINGLE: [usize; 32] = [0; 32];
/// `four`: separate weights for a king in the corner, on the rest of the back
/// rank, on the second rank, and anywhere further up the board.
#[rustfmt::skip]
const LAYOUT_FOUR: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

struct Args {
    data: Vec<String>,
    hidden: usize,
    layout: [usize; 32],
    id: String,
    superbatches: usize,
    /// Batches of 16384 positions per superbatch.
    batches: usize,
    /// Weight of the game result in the training target (the rest is the search score).
    wdl: f32,
    lr: f32,
    threads: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        data: Vec::new(),
        hidden: 512,
        layout: LAYOUT_FOUR,
        id: "net".to_string(),
        superbatches: 40,
        batches: 6104,
        wdl: 0.5,
        lr: 0.001,
        threads: 4,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        let mut value = || iter.next().unwrap_or_else(|| panic!("missing value for {flag}"));
        match flag.as_str() {
            "--data" => args.data.push(value()),
            "--hidden" => args.hidden = value().parse().expect("number"),
            "--layout" => {
                args.layout = match value().as_str() {
                    "single" => LAYOUT_SINGLE,
                    "four" => LAYOUT_FOUR,
                    other => panic!("unknown layout {other}"),
                }
            }
            "--id" => args.id = value(),
            "--superbatches" => args.superbatches = value().parse().expect("number"),
            "--batches" => args.batches = value().parse().expect("number"),
            "--wdl" => args.wdl = value().parse().expect("number"),
            "--lr" => args.lr = value().parse().expect("number"),
            "--threads" => args.threads = value().parse().expect("number"),
            other => panic!("unknown option {other}"),
        }
    }
    assert!(!args.data.is_empty(), "at least one --data file is required");
    args
}

fn main() {
    let args = parse_args();

    let hidden = args.hidden;
    let layout = args.layout;
    let input_buckets = get_num_buckets(&layout);

    // With several king buckets a "factoriser" is trained alongside: a set of
    // weights shared by all buckets, folded into each bucket when saving.
    let l0w = if input_buckets > 1 {
        SavedFormat::id("l0w").transform(move |store, weights| {
            let factoriser = store.get("l0f").values.f32().repeat(input_buckets);
            weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
        })
    } else {
        SavedFormat::id("l0w")
    };

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(layout))
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            l0w.round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            // Transposed so each output bucket's weights are contiguous.
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let mut l0 = builder.new_affine("l0", 768 * input_buckets, hidden);
            if input_buckets > 1 {
                let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
                l0.weights = l0.weights + l0f.repeat(input_buckets);
            }

            let l1 = builder.new_affine("l1", 2 * hidden, NUM_OUTPUT_BUCKETS);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden)).select(output_buckets)
        });

    if input_buckets > 1 {
        // The saved first-layer weight is the sum of two trained weights, so
        // each is clipped more tightly to keep the sum inside the quantised range.
        let stricter_clipping = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
        trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
        trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);
    }

    let schedule = TrainingSchedule {
        net_id: args.id.clone(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: args.batches,
            start_superbatch: 1,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: args.wdl },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: args.lr,
            final_lr: args.lr * 0.01,
            final_superbatch: args.superbatches,
        },
        save_rate: args.superbatches,
    };

    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let files: Vec<&str> = args.data.iter().map(String::as_str).collect();
    let data_loader = loader::DirectSequentialDataLoader::new(&files);

    trainer.run(&schedule, &settings, &data_loader);
}
