//! Trains the plain evaluation network: (768 -> hidden)x2 -> 1.
//!
//! usage: simple --data <shuffled.bin> [--hidden N] [--id NAME] [--superbatches N]
//!               [--batches N] [--wdl X] [--lr X] [--threads N]
//!
//! The data file holds 32-byte records as written by `tabx datagen`
//! (bullet's `ChessBoard` format) and must be shuffled first.
//!
//! The hidden size, `QA`, `QB` and `SCALE` must match `src/nnue.rs` in the engine.

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

struct Args {
    data: Vec<String>,
    hidden: usize,
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
        hidden: 256,
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

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        // Quantised layout read by the engine: feature weights and biases
        // scaled by QA, output weights by QB, output bias by QA * QB.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, hidden);
            let l1 = builder.new_affine("l1", 2 * hidden, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

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
