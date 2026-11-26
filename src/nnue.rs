//! NNUE evaluation: a `(768 x INPUT_BUCKETS -> HIDDEN) x 2 -> OUTPUT_BUCKETS` network.
//!
//! Inputs are one bit per (piece colour, piece type, square), seen once from
//! each side's point of view. Optionally the board is mirrored so that the
//! viewer's own king is always on files a-d, and a separate set of input
//! weights is used depending on where that king stands ("king buckets").
//!
//! The first layer's output for both points of view (the "accumulator") is
//! updated incrementally as moves are made: a move only changes a few inputs,
//! so only a few weight columns are added or subtracted. Only when a king
//! changes bucket or crosses the mirror line is that side's half recomputed.
//!
//! The output layer applies a squared clipped ReLU to both accumulators (side
//! to move first) and takes a dot product with one of several weight sets,
//! chosen by the number of pieces on the board ("output buckets").
//!
//! Weights are trained with the `trainer/` crate and stored quantised: first
//! layer times `QA`, output weights times `QB`. The constants below must
//! match the trainer.

use crate::position::Position;
use crate::types::*;

// HIDDEN, BUCKET_LAYOUT and NET_NAME, written by build.rs for the network it
// compiles in (`nets/gen4.bin` unless the environment names another).
include!(concat!(env!("OUT_DIR"), "/net.rs"));
const OUTPUT_BUCKETS: usize = 8;
/// Mirror the board horizontally when the viewer's king is on files e-h.
const MIRRORED: bool = true;
const INPUT_BUCKETS: usize = max_bucket(&BUCKET_LAYOUT) + 1;

const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

/// Largest evaluation the network may return; keeps evaluations clear of mate scores.
const MAX_EVAL: i32 = 30_000;

const fn max_bucket(layout: &[usize; 32]) -> usize {
    let mut max = 0;
    let mut i = 0;
    while i < 32 {
        if layout[i] > max {
            max = layout[i];
        }
        i += 1;
    }
    max
}

/// `BUCKET_LAYOUT` expanded to all 64 squares.
const KING_BUCKETS: [usize; 64] = {
    let mut table = [0; 64];
    let mut sq = 0;
    while sq < 64 {
        let file = sq % 8;
        let mirrored_file = if file > 3 { 7 - file } else { file };
        table[sq] = BUCKET_LAYOUT[(sq / 8) * 4 + mirrored_file];
        sq += 1;
    }
    table
};

#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Column([i16; HIDDEN]);

/// Layout of the network file written by the trainer.
#[repr(C)]
struct Network {
    /// One column per input feature, king bucket by king bucket.
    feature_weights: [Column; 768 * INPUT_BUCKETS],
    feature_bias: Column,
    /// Per output bucket: weights for the side to move's accumulator, then the opponent's.
    output_weights: [[Column; 2]; OUTPUT_BUCKETS],
    output_bias: [i16; OUTPUT_BUCKETS],
}

// SAFETY: `Network` consists only of `i16` values and padding, the file has
// exactly its size (checked at compile time by `transmute`), and any bit
// pattern is a valid `i16`.
static NET: Network = unsafe {
    std::mem::transmute(*include_bytes!(concat!(env!("OUT_DIR"), "/net.bin")))
};

/// How one side views the board: which input weights to use and whether
/// squares are mirrored. Depends only on that side's king square.
#[derive(Clone, Copy, PartialEq, Eq)]
struct View {
    /// Offset of the king bucket's block of feature weights.
    bucket_offset: usize,
    /// XOR mask applied to squares: 56 flips ranks (Black), 7 mirrors files.
    square_flip: usize,
}

impl View {
    #[inline(always)]
    fn new(perspective: Color, king_sq: Square) -> View {
        let mut flip = if perspective == Color::White { 0 } else { 56 };
        let relative_king = king_sq.idx() ^ flip;
        if MIRRORED && relative_king % 8 > 3 {
            flip ^= 7;
        }
        View { bucket_offset: 768 * KING_BUCKETS[relative_king], square_flip: flip }
    }

    /// Weight column of the input feature for `piece` on `sq`: own pieces
    /// first, then the opponent's.
    #[inline(always)]
    fn weights(self, perspective: Color, piece: Piece, sq: Square) -> &'static [i16; HIDDEN] {
        let theirs = (piece.color() != perspective) as usize;
        let index =
            self.bucket_offset + theirs * 384 + piece.piece_type().idx() * 64 + (sq.idx() ^ self.square_flip);
        &NET.feature_weights[index].0
    }
}

/// The pieces a move puts on the board and takes off it: everything needed
/// to update an accumulator later, without the position the move was made in.
#[derive(Clone, Copy)]
pub struct Delta {
    adds: [(Piece, Square); 2],
    subs: [(Piece, Square); 2],
    add_count: u8,
    sub_count: u8,
    /// King squares after the move, White's then Black's.
    kings: [Square; 2],
    /// Whether each side's half has to be rebuilt because its king changed
    /// bucket or crossed the mirror line.
    refresh: [bool; 2],
}

impl Delta {
    /// A null move changes nothing.
    pub const NULL: Delta = Delta {
        adds: [(Piece::NONE, Square::NONE); 2],
        subs: [(Piece::NONE, Square::NONE); 2],
        add_count: 0,
        sub_count: 0,
        kings: [Square::NONE; 2],
        refresh: [false; 2],
    };

    /// Piece changes of `mv`, played in `pos`.
    #[inline]
    pub fn from_move(pos: &Position, mv: Move) -> Delta {
        let us = pos.stm();
        let (from, to) = (mv.from(), mv.to());
        let piece = pos.piece_on(from);
        let mut delta = Delta {
            adds: [(piece, to); 2],
            subs: [(piece, from); 2],
            add_count: 1,
            sub_count: 1,
            kings: [pos.king_sq(Color::White), pos.king_sq(Color::Black)],
            refresh: [false; 2],
        };
        if mv.is_castle() {
            let kingside = to.file() > from.file();
            let rook = Piece::new(us, PieceType::Rook);
            delta.adds[0] = (piece, Square::from_fr(if kingside { 6 } else { 2 }, from.rank()));
            delta.adds[1] = (rook, Square::from_fr(if kingside { 5 } else { 3 }, from.rank()));
            delta.subs[1] = (rook, to);
            (delta.add_count, delta.sub_count) = (2, 2);
        } else {
            if mv.is_promo() {
                delta.adds[0] = (Piece::new(us, mv.promo_type()), to);
            }
            if mv.is_capture() {
                let cap_sq = if mv.is_en_passant() { Square(to.0 ^ 8) } else { to };
                delta.subs[1] = (pos.piece_on(cap_sq), cap_sq);
                delta.sub_count = 2;
            }
        }
        if piece.piece_type() == PieceType::King {
            let king_to = delta.adds[0].1;
            delta.refresh[us.idx()] = View::new(us, from) != View::new(us, king_to);
            delta.kings[us.idx()] = king_to;
        }
        delta
    }

    /// True if applying this delta needs the position after the move.
    #[inline(always)]
    pub fn needs_position(&self) -> bool {
        self.refresh[0] | self.refresh[1]
    }
}

/// First-layer outputs from White's and Black's point of view.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Accumulator {
    vals: [[i16; HIDDEN]; 2],
}

impl Accumulator {
    pub fn new(pos: &Position) -> Accumulator {
        let mut acc = Accumulator { vals: [[0; HIDDEN]; 2] };
        acc.refresh(pos);
        acc
    }

    /// Recomputes both halves from scratch.
    pub fn refresh(&mut self, pos: &Position) {
        self.refresh_side(pos, Color::White);
        self.refresh_side(pos, Color::Black);
    }

    /// Recomputes one side's half from scratch.
    fn refresh_side(&mut self, pos: &Position, perspective: Color) {
        let view = View::new(perspective, pos.king_sq(perspective));
        let vals = &mut self.vals[perspective.idx()];
        *vals = NET.feature_bias.0;
        for sq in pos.occupied() {
            let weights = view.weights(perspective, pos.piece_on(sq), sq);
            for i in 0..HIDDEN {
                vals[i] = vals[i].wrapping_add(weights[i]);
            }
        }
    }

    /// Sets `self` to `parent` with the move `mv` applied; `pos` is the position before the move.
    #[cfg(test)]
    pub fn update(&mut self, parent: &Accumulator, pos: &Position, mv: Move) {
        let mut next = *pos;
        next.make_move(mv);
        self.apply(parent, &Delta::from_move(pos, mv), &next);
    }

    /// Sets `self` to `parent` with the piece changes in `delta` applied.
    /// `after` is the position after the move; it is only read when
    /// `delta.needs_position()`.
    pub fn apply(&mut self, parent: &Accumulator, delta: &Delta, after: &Position) {
        let (adds, subs) = (&delta.adds, &delta.subs);
        for perspective in [Color::White, Color::Black] {
            if delta.refresh[perspective.idx()] {
                self.refresh_side(after, perspective);
                continue;
            }
            let input = &parent.vals[perspective.idx()];
            let output = &mut self.vals[perspective.idx()];
            if delta.add_count == 0 {
                *output = *input;
                continue;
            }
            let view = View::new(perspective, delta.kings[perspective.idx()]);
            let weights = |(piece, sq): (Piece, Square)| view.weights(perspective, piece, sq);
            match (delta.add_count, delta.sub_count) {
                (1, 1) => {
                    let (a0, s0) = (weights(adds[0]), weights(subs[0]));
                    for i in 0..HIDDEN {
                        output[i] = input[i].wrapping_add(a0[i]).wrapping_sub(s0[i]);
                    }
                }
                (1, _) => {
                    let (a0, s0, s1) = (weights(adds[0]), weights(subs[0]), weights(subs[1]));
                    for i in 0..HIDDEN {
                        output[i] = input[i].wrapping_add(a0[i]).wrapping_sub(s0[i]).wrapping_sub(s1[i]);
                    }
                }
                _ => {
                    let (a0, a1) = (weights(adds[0]), weights(adds[1]));
                    let (s0, s1) = (weights(subs[0]), weights(subs[1]));
                    for i in 0..HIDDEN {
                        output[i] = input[i]
                            .wrapping_add(a0[i])
                            .wrapping_add(a1[i])
                            .wrapping_sub(s0[i])
                            .wrapping_sub(s1[i]);
                    }
                }
            }
        }
    }

    /// Evaluation of `pos` in centipawns from the side to move's point of view.
    /// The accumulator must belong to `pos`.
    #[inline]
    pub fn evaluate(&self, pos: &Position) -> i32 {
        let stm = pos.stm();
        let bucket = if OUTPUT_BUCKETS > 1 {
            (pos.occupied().count() as usize - 2) / 32usize.div_ceil(OUTPUT_BUCKETS)
        } else {
            0
        };
        let weights = &NET.output_weights[bucket];
        let sum = screlu_dot(&self.vals[stm.idx()], &weights[0].0)
            + screlu_dot(&self.vals[(!stm).idx()], &weights[1].0);
        // The activation is squared, so the sum carries a factor of QA * QA * QB.
        let output = (sum / QA as i64 + NET.output_bias[bucket] as i64) * SCALE as i64 / (QA * QB) as i64;
        (output as i32).clamp(-MAX_EVAL, MAX_EVAL)
    }
}

/// Sum over `i` of `clamp(acc[i], 0, QA)^2 * weights[i]`.
#[cfg(target_feature = "avx2")]
#[inline(always)]
fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i64 {
    use std::arch::x86_64::*;
    const _: () = assert!(HIDDEN.is_multiple_of(16));
    // SAFETY: AVX2 is enabled at compile time (checked by the `cfg`), and all
    // loads stay inside the two arrays because HIDDEN is a multiple of 16.
    unsafe {
        let zero = _mm256_setzero_si256();
        let qa = _mm256_set1_epi16(QA as i16);
        let mut sum = _mm256_setzero_si256();
        for i in (0..HIDDEN).step_by(16) {
            let v = _mm256_loadu_si256(acc.as_ptr().add(i).cast());
            let w = _mm256_loadu_si256(weights.as_ptr().add(i).cast());
            let v = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
            // v * w fits in 16 bits because |w| <= 127 (the trainer clips
            // output weights), so (v * w) * v can use the widening multiply-add.
            let product = _mm256_madd_epi16(_mm256_mullo_epi16(v, w), v);
            sum = _mm256_add_epi32(sum, product);
        }
        let mut lanes = [0i32; 8];
        _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum);
        lanes.iter().map(|&x| x as i64).sum()
    }
}

#[cfg(not(target_feature = "avx2"))]
#[inline(always)]
fn screlu_dot(acc: &[i16; HIDDEN], weights: &[i16; HIDDEN]) -> i64 {
    let mut sum = 0i64;
    for i in 0..HIDDEN {
        let v = (acc[i] as i32).clamp(0, QA);
        sum += (v * v * weights[i] as i32) as i64;
    }
    sum
}

/// Evaluates a position from scratch (no incremental state needed).
pub fn evaluate(pos: &Position) -> i32 {
    Accumulator::new(pos).evaluate(pos)
}
