//! Self-play training data generation for the neural network evaluation.
//!
//! Every thread plays games against itself from randomised openings with a
//! small fixed node budget per move, and records quiet positions together with
//! the search score and the final game result.
//!
//! An opening is either a number of random moves from the start position
//! (the default: 8 or 9) or a position from a file of openings (`--book`),
//! followed by at most a few random moves. Random moves alone give positions
//! that look nothing like the ones that come up in real games: pieces and
//! kings wander, nobody castles. A network trained only on such games has
//! seen little normal development, and it showed in its play.
//!
//! Output format: a flat file of 32-byte records (little endian), always from
//! the point of view of the side to move ("us"). If Black is to move the
//! board is mirrored vertically so that "us" always plays up the board.
//!
//! | bytes | field                                                            |
//! |-------|------------------------------------------------------------------|
//! | 0..8  | occupancy bitboard (after mirroring)                             |
//! | 8..24 | one nibble per occupied square in ascending square order:        |
//! |       | bit 3 = piece belongs to the opponent, bits 0-2 = piece type     |
//! | 24..26| search score in centipawns (i16), from our point of view         |
//! | 26    | game result from our point of view: 0 loss, 1 draw, 2 win        |
//! | 27    | our king square                                                  |
//! | 28    | opponent king square, as seen from the opponent's side (^ 56)    |
//! | 29..32| unused                                                           |

use crate::movegen::{MoveList, generate_all};
use crate::position::Position;
use crate::search::{self, Limits, MATE_BOUND, Report, Shared, ThreadData};
use crate::types::*;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const RECORD_SIZE: usize = 32;

pub struct Config {
    pub out: String,
    pub threads: usize,
    /// Stop once this many positions have been written.
    pub positions: u64,
    /// Node budget per move (the search finishes its current iteration).
    pub nodes: u64,
    /// Start games from double Fischer random positions instead of the standard one.
    pub dfrc: bool,
    /// Opening positions (FEN or EPD lines), one list per `--book` file. A
    /// game then starts from a random line of a random list.
    pub books: Vec<Vec<String>>,
    /// Random moves played before the engine takes over: a number from the
    /// first value to the second, drawn for every game.
    pub random_plies: (usize, usize),
    pub seed: u64,
    /// Transposition table size per thread in MB. Searches here are tiny, so
    /// a small table that stays in the CPU cache is faster than a large one.
    pub hash_mb: usize,
}

/// Openings whose first search score exceeds this are thrown away as too one-sided.
const MAX_OPENING_SCORE: i32 = 1000;
/// A score beyond this for `WIN_PLIES` plies in a row ends the game as a win.
const WIN_SCORE: i32 = 2500;
const WIN_PLIES: u32 = 4;
/// A score within this for `DRAW_PLIES` plies in a row (after move 40) ends the game as a draw.
const DRAW_SCORE: i32 = 8;
const DRAW_PLIES: u32 = 12;
const DRAW_MIN_MOVE: u16 = 40;
const MAX_GAME_PLIES: usize = 800;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 16) % n as u64) as usize
    }
}

/// Packs a position into the 32-byte record described in the module docs.
/// `result` is from our (the side to move's) point of view.
pub fn pack(pos: &Position, score: i32, result: u8) -> [u8; RECORD_SIZE] {
    let us = pos.stm();
    let flip = if us == Color::Black { 56 } else { 0 };
    let mut squares = [(0u8, 0u8); 32];
    let mut count = 0;
    for sq in pos.occupied() {
        let piece = pos.piece_on(sq);
        let theirs = (piece.color() != us) as u8;
        squares[count] = (sq.0 ^ flip, theirs << 3 | piece.piece_type() as u8);
        count += 1;
    }
    // Mirroring reverses the rank order, so sort back into ascending squares.
    squares[..count].sort_unstable();

    let mut record = [0u8; RECORD_SIZE];
    let mut occ = 0u64;
    for (i, &(sq, nibble)) in squares[..count].iter().enumerate() {
        occ |= 1u64 << sq;
        record[8 + i / 2] |= nibble << (4 * (i & 1));
    }
    record[0..8].copy_from_slice(&occ.to_le_bytes());
    let score = score.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
    record[24..26].copy_from_slice(&score.to_le_bytes());
    record[26] = result;
    record[27] = pos.king_sq(us).0 ^ flip;
    record[28] = pos.king_sq(!us).0 ^ flip ^ 56;
    record
}

/// Plays one game and returns its samples, or `None` if the opening was unusable.
fn play_game(
    cfg: &Config,
    rng: &mut Rng,
    shared: &mut Shared,
    td: &mut ThreadData,
) -> Option<Vec<[u8; RECORD_SIZE]>> {
    let mut pos = if !cfg.books.is_empty() {
        let book = &cfg.books[rng.below(cfg.books.len())];
        Position::from_fen(&book[rng.below(book.len())]).ok()?
    } else if cfg.dfrc {
        Position::frc(rng.below(960), rng.below(960))
    } else {
        Position::startpos()
    };
    let mut keys: Vec<u64> = Vec::with_capacity(256);
    let mut legal = MoveList::new();

    // Random moves. From the start position: 8 or 9 by default, so both sides
    // get to move first equally often.
    let (fewest, most) = cfg.random_plies;
    let random_plies = fewest + rng.below(most - fewest + 1);
    for _ in 0..random_plies {
        legal.clear();
        generate_all(&pos, &mut legal);
        if legal.is_empty() {
            return None;
        }
        keys.push(pos.hash());
        pos.make_move(legal[rng.below(legal.len())]);
    }

    shared.tt.clear();
    td.clear();
    let report = Report { chess960: true, print: false, multipv: 1 };
    // (record, side to move) for every sampled position; results are filled in at the end.
    let mut samples: Vec<([u8; RECORD_SIZE], Color)> = Vec::with_capacity(128);
    let (mut win_plies, mut draw_plies) = (0u32, 0u32);
    let mut first = true;

    // Final result from White's point of view: 0 loss, 1 draw, 2 win.
    let white_result: u8 = loop {
        legal.clear();
        generate_all(&pos, &mut legal);
        let us = pos.stm();
        let loss_for_us = if us == Color::White { 0 } else { 2 };
        if legal.is_empty() {
            break if pos.in_check() { loss_for_us } else { 1 };
        }
        let window = pos.repetition_window().min(keys.len());
        let repeated = keys[keys.len() - window..].iter().any(|&k| k == pos.hash());
        if pos.halfmove() >= 100
            || pos.is_insufficient_material()
            || repeated
            || keys.len() >= MAX_GAME_PLIES
        {
            break 1;
        }

        let mut limits = Limits::new();
        limits.soft_nodes = cfg.nodes;
        limits.nodes = cfg.nodes * 8;
        shared.stop.store(false, Ordering::Relaxed);
        let result = search::run(shared, std::slice::from_mut(td), &pos, &keys, &limits, report);
        let score = result.score;

        if first && score.abs() > MAX_OPENING_SCORE {
            return None;
        }
        first = false;

        // Adjudication keeps games short and the results reliable.
        if score.abs() >= MATE_BOUND {
            break if score > 0 { 2 - loss_for_us } else { loss_for_us };
        }
        win_plies = if score.abs() >= WIN_SCORE { win_plies + 1 } else { 0 };
        if win_plies >= WIN_PLIES {
            break if score > 0 { 2 - loss_for_us } else { loss_for_us };
        }
        draw_plies =
            if pos.fullmove() >= DRAW_MIN_MOVE && score.abs() <= DRAW_SCORE { draw_plies + 1 } else { 0 };
        if draw_plies >= DRAW_PLIES {
            break 1;
        }

        // Only quiet positions make good training targets for a static evaluation.
        let best = result.best_move;
        if !pos.in_check() && !best.is_capture() && !best.is_promo() {
            samples.push((pack(&pos, score, 1), us));
        }

        keys.push(pos.hash());
        pos.make_move(best);
    };

    Some(
        samples
            .into_iter()
            .map(|(mut record, us)| {
                record[26] = if us == Color::White { white_result } else { 2 - white_result };
                record
            })
            .collect(),
    )
}

pub fn run(cfg: Config) {
    let file = OpenOptions::new().create(true).append(true).open(&cfg.out);
    let file = match file {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cannot open {}: {e}", cfg.out);
            return;
        }
    };
    let writer = Mutex::new(BufWriter::with_capacity(1 << 20, file));
    let written = AtomicU64::new(0);
    let games = AtomicU64::new(0);
    let discarded = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let start = Instant::now();
    let openings = if !cfg.books.is_empty() {
        format!("{} book", cfg.books.iter().map(Vec::len).sum::<usize>())
    } else if cfg.dfrc {
        "DFRC".to_string()
    } else {
        "standard".to_string()
    };
    println!(
        "datagen: {} threads, {} nodes/move, target {} positions, {openings} openings with {} to {} random moves, writing to {}",
        cfg.threads, cfg.nodes, cfg.positions, cfg.random_plies.0, cfg.random_plies.1, cfg.out
    );

    std::thread::scope(|scope| {
        for id in 0..cfg.threads {
            let (cfg, writer, written, games, discarded, done) =
                (&cfg, &writer, &written, &games, &discarded, &done);
            std::thread::Builder::new()
                .stack_size(crate::uci::SEARCH_STACK_SIZE)
                .spawn_scoped(scope, move || {
                    let mut rng = Rng((cfg.seed ^ 0x9E37_79B9_7F4A_7C15)
                        .wrapping_mul(id as u64 * 2 + 1)
                        .wrapping_add(0x1234_5678_9ABC_DEF1 * (id as u64 + 1)));
                    for _ in 0..16 {
                        rng.next();
                    }
                    let mut shared = Shared::new(cfg.hash_mb);
                    let mut td = ThreadData::new();
                    while !done.load(Ordering::Relaxed) {
                        let Some(samples) = play_game(cfg, &mut rng, &mut shared, &mut td) else {
                            discarded.fetch_add(1, Ordering::Relaxed);
                            continue;
                        };
                        let mut out = writer.lock().unwrap();
                        if done.load(Ordering::Relaxed) {
                            break;
                        }
                        for record in &samples {
                            out.write_all(record).expect("failed to write training data");
                        }
                        games.fetch_add(1, Ordering::Relaxed);
                        let total = written.fetch_add(samples.len() as u64, Ordering::Relaxed)
                            + samples.len() as u64;
                        if total >= cfg.positions {
                            done.store(true, Ordering::Relaxed);
                        }
                    }
                })
                .expect("failed to spawn datagen thread");
        }

        // Progress report from the calling thread.
        let mut last = Instant::now();
        while !done.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(200));
            if last.elapsed() >= Duration::from_secs(30) {
                last = Instant::now();
                let n = written.load(Ordering::Relaxed);
                let secs = start.elapsed().as_secs_f64();
                println!(
                    "{n} positions, {} games ({} openings discarded), {:.0} positions/s, {:.1} min elapsed",
                    games.load(Ordering::Relaxed),
                    discarded.load(Ordering::Relaxed),
                    n as f64 / secs,
                    secs / 60.0
                );
            }
        }
    });

    writer.lock().unwrap().flush().expect("failed to flush training data");
    let n = written.load(Ordering::Relaxed);
    println!(
        "done: {n} positions from {} games in {:.1} min",
        games.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64() / 60.0
    );
}
