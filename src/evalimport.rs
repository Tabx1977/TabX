//! Turns Lichess's database of position evaluations into training records,
//! or into an index in which a position can be looked up.
//!
//!     zstd -dc lichess_db_eval.jsonl.zst | tabx import-evals --out data/lichess.bin
//!     zstd -dc lichess_db_eval.jsonl.zst | tabx index-evals --out tools/lichess-evals/evals.idx
//!
//! The database (https://database.lichess.org/#evals, licence CC0) has one
//! line of JSON per position: the position, and one or more evaluations of it
//! by Stockfish, each with the depth it reached and its best lines:
//!
//!     {"fen":"7r/1p3k2/... b - -","evals":[{"pvs":[{"cp":69,"line":"f7g7 e6e2 ..."},...],"knodes":4189972,"depth":46}]}
//!
//! Of each position the deepest evaluation is used, and of that the first
//! line. Scores in the database are from White's point of view; a record
//! holds the score of the side to move, like the records of `datagen`. The
//! records carry no game result (the byte says "draw"), so a network has to
//! be trained on them with the weight of the result at zero (`--wdl 0`).
//!
//! Left out, for the same reasons as in `datagen`: positions with a forced
//! mate, positions in which the side to move is in check or whose best move
//! is a capture or a promotion (a static evaluation is trained on quiet
//! positions), and evaluations shallower than `--min-depth`.
//!
//! Stockfish's centipawns are smaller than this engine's: over the positions
//! of 112 games between the two, Stockfish's score was 0.78 times the
//! engine's in the middlegame. The search's margins were tuned for the
//! engine's own scale, so scores are multiplied by `--scale` percent (128)
//! on the way in.
//!
//! The database says more than one score per position. Most evaluations give
//! several lines (Stockfish's best move, its second best and so on, each with
//! its score and the moves it expects to follow), and a line's score is also
//! the score of every position along it, from a search that many moves
//! shallower. With `--lines N --plies M` the records are those positions
//! instead: the ones after the first M moves of each of an evaluation's first
//! N lines, under the same rules as above (the move that follows in the line
//! stands in for the best move). The positions after the second-best and
//! third-best moves are the valuable part: positions in which a side has just
//! played something inferior, with Stockfish's verdict on how inferior.
//! `--known FILE` names a file of records (the plain import) whose positions
//! are left out, and no position is written twice.
//!
//! The records come out in the order of the database, and a trainer wants
//! them shuffled. Shuffling a file needs room for a second copy of it; with
//! billions of records that is more disk than there may be. `--shards N`
//! deals the records out at random among N files instead (and those of
//! `--also FILE` with them, for the records of the plain import), each small
//! enough to be shuffled in memory and joined to the others afterwards
//! (`scripts/join_shards.py`).

use crate::datagen::{RECORD_SIZE, pack};
use crate::movegen::{MoveList, generate_all};
use crate::position::Position;
use crate::types::PieceType::Pawn;
use crate::types::*;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufWriter, Read, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

pub struct Config {
    pub out: String,
    pub threads: usize,
    /// Evaluations that reached a lower depth are not used.
    pub min_depth: u32,
    /// Percent by which scores are multiplied.
    pub scale: i32,
    /// Stop after this many lines (0: read everything).
    pub limit: u64,
    /// Lines of each evaluation whose positions become records (0: the
    /// evaluated position itself is the record, the plain import).
    pub lines: usize,
    /// Moves along each of those lines.
    pub plies: usize,
    /// A file of records whose positions are not to be written again.
    pub known: Option<String>,
    /// Write to this many files, `<out>.000` and so on, a record to a file
    /// picked at random (0: one file, in the order of the database).
    pub shards: usize,
    /// A file of records to be dealt out among the shards as well.
    pub also: Option<String>,
}

/// Scores are clamped to this after scaling.
const MAX_SCORE: i32 = 3200;
/// Lines handed to a worker at a time.
const BATCH: usize = 4096;

/// Why a line did not become a record.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Skipped {
    Malformed,
    Shallow,
    Mate,
    BadPosition,
    InCheck,
    NoisyBestMove,
}

const REASONS: [(Skipped, &str); 6] = [
    (Skipped::Malformed, "could not be read"),
    (Skipped::Shallow, "evaluation too shallow"),
    (Skipped::Mate, "forced mate"),
    (Skipped::BadPosition, "not a usable position"),
    (Skipped::InCheck, "side to move in check"),
    (Skipped::NoisyBestMove, "best move a capture or promotion"),
];

/// Position of `needle` in `hay` at or after `from`.
fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay[from..].windows(needle.len()).position(|window| window == needle).map(|at| at + from)
}

/// The integer that follows the next colon at or after `from`, and the index just past it.
fn number_after_colon(hay: &[u8], from: usize) -> Option<(i64, usize)> {
    let mut at = find(hay, b":", from)? + 1;
    while at < hay.len() && hay[at] == b' ' {
        at += 1;
    }
    let negative = at < hay.len() && hay[at] == b'-';
    if negative {
        at += 1;
    }
    let start = at;
    let mut value: i64 = 0;
    while at < hay.len() && hay[at].is_ascii_digit() {
        value = value.saturating_mul(10).saturating_add((hay[at] - b'0') as i64);
        at += 1;
    }
    if at == start {
        return None;
    }
    Some((if negative { -value } else { value }, at))
}

/// The string value that follows the next colon at or after `from`.
fn string_after_colon(hay: &[u8], from: usize) -> Option<&[u8]> {
    let colon = find(hay, b":", from)?;
    let open = find(hay, b"\"", colon)? + 1;
    let close = find(hay, b"\"", open)?;
    Some(&hay[open..close])
}

/// What one line of the database says: the position, and the deepest
/// evaluation's depth, score (None for a forced mate) and best move.
struct Entry<'a> {
    fen: &'a [u8],
    depth: u32,
    cp: Option<i32>,
    /// Moves to the mate if the score is a forced mate; negative if Black mates.
    mate: Option<i32>,
    best_move: &'a [u8],
    /// Evaluations the line holds, and the most lines any one of them gives.
    evals: u32,
    most_lines: u32,
}

fn read_line(line: &[u8]) -> Option<Entry<'_>> {
    let fen = string_after_colon(line, find(line, b"\"fen\"", 0)?)?;
    let mut best: Option<Entry> = None;
    let (mut evals, mut most_lines) = (0, 0);
    let mut at = find(line, b"\"evals\"", 0)?;
    // Every evaluation is {"pvs":[{first line},...],"knodes":N,"depth":D}.
    while let Some(pvs) = find(line, b"\"pvs\"", at) {
        let first = find(line, b"{", pvs)?;
        let end_of_first = find(line, b"}", first)?;
        let pv = &line[first..end_of_first];
        let (cp, mate) = match find(pv, b"\"cp\"", 0) {
            Some(key) => (Some(number_after_colon(pv, key)?.0.clamp(-100_000, 100_000) as i32), None),
            None => {
                let moves_to_mate = number_after_colon(pv, find(pv, b"\"mate\"", 0)?)?.0;
                (None, Some(moves_to_mate.clamp(-10_000, 10_000) as i32))
            }
        };
        let moves = string_after_colon(pv, find(pv, b"\"line\"", 0)?)?;
        let best_move = moves.split(|&byte| byte == b' ').next()?;
        let depth_key = find(line, b"\"depth\"", end_of_first)?;
        let (depth, after) = number_after_colon(line, depth_key)?;
        let depth = depth.clamp(0, 10_000) as u32;
        evals += 1;
        // One opening brace for each line of this evaluation.
        most_lines = most_lines.max(line[pvs..depth_key].iter().filter(|&&byte| byte == b'{').count() as u32);
        if best.as_ref().is_none_or(|entry| depth > entry.depth) {
            best = Some(Entry { fen, depth, cp, mate, best_move, evals: 0, most_lines: 0 });
        }
        at = after;
    }
    best.map(|entry| Entry { evals, most_lines, ..entry })
}

fn square(text: &[u8]) -> Option<Square> {
    let (file, rank) = (text.first()?.wrapping_sub(b'a'), text.get(1)?.wrapping_sub(b'1'));
    (file < 8 && rank < 8).then(|| Square(rank * 8 + file))
}

/// Converts one line into a record, or says why not.
pub fn convert(line: &[u8], min_depth: u32, scale: i32) -> Result<[u8; RECORD_SIZE], Skipped> {
    let entry = read_line(line).ok_or(Skipped::Malformed)?;
    if entry.depth < min_depth {
        return Err(Skipped::Shallow);
    }
    let cp = entry.cp.ok_or(Skipped::Mate)?;
    // The position code assumes one king a side; look before handing it over.
    let board = entry.fen.split(|&byte| byte == b' ').next().unwrap_or(&[]);
    let count = |piece: u8| board.iter().filter(|&&byte| byte == piece).count();
    if count(b'K') != 1 || count(b'k') != 1 {
        return Err(Skipped::BadPosition);
    }
    let fen = std::str::from_utf8(entry.fen).map_err(|_| Skipped::Malformed)?;
    let pos = Position::from_fen(fen).map_err(|_| Skipped::BadPosition)?;
    let pawns = pos.pieces(Color::White, Pawn) | pos.pieces(Color::Black, Pawn);
    if (pawns & (Bitboard::rank(0) | Bitboard::rank(7))).any() || pos.occupied().count() > 32 {
        return Err(Skipped::BadPosition);
    }
    if pos.in_check() {
        return Err(Skipped::InCheck);
    }
    // The best move, in UCI notation. Castling may be written as the king
    // taking its own rook, which is not a capture here.
    let (from, to) = (
        square(entry.best_move).ok_or(Skipped::Malformed)?,
        square(entry.best_move.get(2..).unwrap_or(&[])).ok_or(Skipped::Malformed)?,
    );
    let us = pos.stm();
    let mover = pos.piece_on(from);
    let target = pos.piece_on(to);
    if mover == Piece::NONE || mover.color() != us {
        return Err(Skipped::BadPosition);
    }
    let captures = target != Piece::NONE && target.color() != us;
    let en_passant = mover.piece_type() == Pawn && target == Piece::NONE && from.0 % 8 != to.0 % 8;
    if captures || en_passant || entry.best_move.len() > 4 {
        return Err(Skipped::NoisyBestMove);
    }
    let from_white = (cp as i64 * scale as i64 / 100).clamp(-MAX_SCORE as i64, MAX_SCORE as i64) as i32;
    let score = if us == Color::White { from_white } else { -from_white };
    Ok(pack(&pos, score, 1))
}

/// One evaluation of a database line: its depth, and each of its lines as the score for
/// White (None for a forced mate) and the moves.
type Evaluation<'a> = (u32, Vec<(Option<i32>, &'a [u8])>);

/// Everything a line of the database says: the position and all its evaluations.
fn read_evaluations(line: &[u8]) -> Option<(&[u8], Vec<Evaluation<'_>>)> {
    let fen = string_after_colon(line, find(line, b"\"fen\"", 0)?)?;
    let mut evaluations = Vec::new();
    let mut at = find(line, b"\"evals\"", 0)?;
    // Every evaluation is {"pvs":[{line},...],"knodes":N,"depth":D}.
    while let Some(pvs) = find(line, b"\"pvs\"", at) {
        let depth_key = find(line, b"\"depth\"", pvs)?;
        let (depth, after) = number_after_colon(line, depth_key)?;
        let mut lines = Vec::new();
        let mut cursor = pvs;
        while let Some(open) = find(line, b"{", cursor) {
            if open >= depth_key {
                break;
            }
            let close = find(line, b"}", open)?;
            let pv = &line[open..close];
            let score = match find(pv, b"\"cp\"", 0) {
                Some(key) => Some(number_after_colon(pv, key)?.0.clamp(-100_000, 100_000) as i32),
                None => None,
            };
            let moves = match find(pv, b"\"line\"", 0) {
                Some(key) => string_after_colon(pv, key)?,
                None => b"",
            };
            lines.push((score, moves));
            cursor = close;
        }
        evaluations.push((depth.clamp(0, 10_000) as u32, lines));
        at = after;
    }
    Some((fen, evaluations))
}

/// The legal move of `pos` that `text` names in UCI notation; castling may be
/// written as the king's own move or as the king taking its rook.
fn move_named(pos: &Position, text: &[u8]) -> Option<Move> {
    let from = square(text)?;
    let to = square(text.get(2..)?)?;
    let promotion = match text.get(4) {
        None => None,
        Some(b'n') => Some(PieceType::Knight),
        Some(b'b') => Some(PieceType::Bishop),
        Some(b'r') => Some(PieceType::Rook),
        Some(b'q') => Some(PieceType::Queen),
        Some(_) => return None,
    };
    let mut list = MoveList::new();
    generate_all(pos, &mut list);
    list.iter().copied().find(|mv| {
        if mv.from() != from {
            return false;
        }
        if mv.is_castle() {
            // Stored as king takes rook; the king lands on the g- or c-file.
            let kingside = mv.to().file() > from.file();
            let lands = Square::from_fr(if kingside { 6 } else { 2 }, from.rank());
            return promotion.is_none() && (to == mv.to() || to == lands);
        }
        mv.to() == to && (if mv.is_promo() { promotion == Some(mv.promo_type()) } else { promotion.is_none() })
    })
}

/// The records of one database line in the mode of `--lines`: the positions along the
/// first `lines` lines of each evaluation, up to `plies` moves in, deepest evaluation first.
pub fn records_along_lines(
    line: &[u8],
    min_depth: u32,
    scale: i32,
    lines: usize,
    plies: usize,
    out: &mut Vec<[u8; RECORD_SIZE]>,
) {
    let Some((fen, mut evaluations)) = read_evaluations(line) else { return };
    let board = fen.split(|&byte| byte == b' ').next().unwrap_or(&[]);
    let count = |piece: u8| board.iter().filter(|&&byte| byte == piece).count();
    if count(b'K') != 1 || count(b'k') != 1 {
        return;
    }
    let Ok(fen) = std::str::from_utf8(fen) else { return };
    let Ok(root) = Position::from_fen(fen) else { return };
    let pawns = root.pieces(Color::White, Pawn) | root.pieces(Color::Black, Pawn);
    if (pawns & (Bitboard::rank(0) | Bitboard::rank(7))).any() || root.occupied().count() > 32 {
        return;
    }
    evaluations.sort_by_key(|(depth, _)| std::cmp::Reverse(*depth));
    for (depth, pvs) in &evaluations {
        for (score, moves) in pvs.iter().take(lines) {
            let Some(cp) = *score else { continue };
            let from_white = (cp as i64 * scale as i64 / 100).clamp(-MAX_SCORE as i64, MAX_SCORE as i64) as i32;
            let mut pos = root;
            let mut texts = moves.split(|&byte| byte == b' ').filter(|text| !text.is_empty());
            let mut next = texts.next();
            for ply in 1..=plies {
                // The search that gave the score had this many plies less for the position.
                if *depth < min_depth + ply as u32 {
                    break;
                }
                let Some(mv) = next.and_then(|text| move_named(&pos, text)) else { break };
                pos.make_move(mv);
                next = texts.next();
                // The move that follows stands in for the position's best move.
                let Some(reply) = next.and_then(|text| move_named(&pos, text)) else { break };
                if pos.in_check() || reply.is_capture() || reply.is_promo() || reply.is_en_passant() {
                    continue;
                }
                let score = if pos.stm() == Color::White { from_white } else { -from_white };
                out.push(pack(&pos, score, 1));
            }
        }
    }
}

/// What tells two records' positions apart: the pieces as the record stores
/// them (seen from the side to move), mixed into 64 bits.
fn position_key(record: &[u8]) -> u64 {
    let part = |at: usize| u64::from_le_bytes(record[at..at + 8].try_into().unwrap());
    let mut key = part(0).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ part(8).rotate_left(23).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ part(16).rotate_left(47).wrapping_mul(0x1656_67B1_9E37_79F9);
    key ^= key >> 32;
    key = key.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    key ^ (key >> 32)
}

/// The keys are mixed already: the set uses them as they are.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed");
    }
    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }
}

type KeySet = HashSet<u64, BuildHasherDefault<KeyHasher>>;

/// The `--lines` mode: writes the positions along the lines of the database on standard
/// input, each once, leaving out those of the records in `cfg.known`.
pub fn run_lines(cfg: Config) {
    let mut seen = KeySet::default();
    if let Some(path) = &cfg.known {
        let mut file = match File::open(path) {
            Ok(file) => std::io::BufReader::with_capacity(1 << 22, file),
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return;
            }
        };
        if let Ok(meta) = std::fs::metadata(path) {
            seen.reserve(meta.len() as usize / RECORD_SIZE);
        }
        let mut record = [0u8; RECORD_SIZE];
        while file.read_exact(&mut record).is_ok() {
            seen.insert(position_key(&record));
        }
        eprintln!("{} positions known from {path}", seen.len());
    }
    let known = seen.len();
    let names: Vec<String> = if cfg.shards == 0 {
        vec![cfg.out.clone()]
    } else {
        (0..cfg.shards).map(|shard| format!("{}.{shard:03}", cfg.out)).collect()
    };
    let mut outs = Vec::new();
    for name in &names {
        match File::create(name) {
            Ok(file) => outs.push(BufWriter::with_capacity(1 << 20, file)),
            Err(e) => {
                eprintln!("cannot write {name}: {e}");
                return;
            }
        }
    }
    // Which file a record goes to: a random one, by a generator of its own.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut pick = move |count: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 16) % count as u64) as usize
    };
    let mut dealt = 0u64;
    if let Some(path) = &cfg.also {
        let mut file = match File::open(path) {
            Ok(file) => std::io::BufReader::with_capacity(1 << 22, file),
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return;
            }
        };
        let mut record = [0u8; RECORD_SIZE];
        while file.read_exact(&mut record).is_ok() {
            let shard = pick(outs.len());
            outs[shard].write_all(&record).expect("failed to write training data");
            dealt += 1;
        }
        eprintln!("{dealt} records of {path} dealt out");
    }
    let (to_workers, from_reader) = mpsc::sync_channel::<Vec<Vec<u8>>>(cfg.threads * 2);
    let from_reader = Arc::new(Mutex::new(from_reader));
    let (to_writer, from_workers) = mpsc::sync_channel::<Vec<[u8; RECORD_SIZE]>>(cfg.threads * 2);

    let (mut found, mut written) = (0u64, 0u64);
    let mut lines_read = 0u64;
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            let (from_reader, to_writer) = (Arc::clone(&from_reader), to_writer.clone());
            let (min_depth, scale, lines, plies) = (cfg.min_depth, cfg.scale, cfg.lines, cfg.plies);
            scope.spawn(move || {
                loop {
                    let batch = match from_reader.lock().unwrap().recv() {
                        Ok(batch) => batch,
                        Err(_) => return,
                    };
                    let mut records = Vec::with_capacity(batch.len() * 4);
                    for line in &batch {
                        records_along_lines(line, min_depth, scale, lines, plies, &mut records);
                    }
                    if to_writer.send(records).is_err() {
                        return;
                    }
                }
            });
        }
        drop(to_writer);
        // This thread reads; a second one keeps the set of positions and writes.
        let seen = &mut seen;
        let outs = &mut outs;
        let writer = scope.spawn(move || {
            let (mut found, mut written) = (0u64, 0u64);
            for records in from_workers {
                for record in &records {
                    found += 1;
                    if seen.insert(position_key(record)) {
                        let shard = if outs.len() == 1 { 0 } else { pick(outs.len()) };
                        outs[shard].write_all(record).expect("failed to write training data");
                        written += 1;
                    }
                }
            }
            for out in outs.iter_mut() {
                out.flush().expect("failed to write training data");
            }
            (found, written)
        });
        lines_read = feed(cfg.limit, to_workers);
        (found, written) = writer.join().expect("writer thread");
    });
    println!(
        "{lines_read} lines read, {found} positions along the lines, {written} of them new and written to {} \
         ({known} positions were known beforehand)",
        cfg.out
    );
    if cfg.shards > 0 {
        println!("    in {} files, {}.000 and so on, with {dealt} records of the --also file", cfg.shards, cfg.out);
    }
}

/// Reads standard input and hands its lines to the workers in batches, until the
/// input ends or `limit` lines are read (0: no limit). Returns the number of lines.
fn feed(limit: u64, to_workers: mpsc::SyncSender<Vec<Vec<u8>>>) -> u64 {
    let stdin = std::io::stdin();
    let mut input = std::io::BufReader::with_capacity(1 << 22, stdin.lock());
    let mut batch = Vec::with_capacity(BATCH);
    let mut lines_read = 0u64;
    loop {
        let mut line = Vec::with_capacity(1024);
        match input.read_until(b'\n', &mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        lines_read += 1;
        batch.push(line);
        if batch.len() == BATCH {
            if to_workers.send(std::mem::replace(&mut batch, Vec::with_capacity(BATCH))).is_err() {
                break;
            }
            if lines_read % (BATCH as u64 * 2048) == 0 {
                eprintln!("{lines_read} lines read");
            }
        }
        if limit != 0 && lines_read >= limit {
            break;
        }
    }
    if !batch.is_empty() {
        let _ = to_workers.send(batch);
    }
    lines_read
}

pub fn run(cfg: Config) {
    let file = match File::create(&cfg.out) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("cannot write {}: {e}", cfg.out);
            return;
        }
    };
    let mut out = BufWriter::with_capacity(1 << 22, file);
    let (to_workers, from_reader) = mpsc::sync_channel::<Vec<Vec<u8>>>(cfg.threads * 2);
    let from_reader = Arc::new(Mutex::new(from_reader));
    let (to_writer, from_workers) = mpsc::sync_channel::<(Vec<u8>, [u64; REASONS.len()])>(cfg.threads * 2);

    let mut written = 0u64;
    let mut skipped = [0u64; REASONS.len()];
    let mut lines_read = 0u64;
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            let (from_reader, to_writer) = (Arc::clone(&from_reader), to_writer.clone());
            let (min_depth, scale) = (cfg.min_depth, cfg.scale);
            scope.spawn(move || {
                loop {
                    let batch = match from_reader.lock().unwrap().recv() {
                        Ok(batch) => batch,
                        Err(_) => return,
                    };
                    let mut records = Vec::with_capacity(batch.len() * RECORD_SIZE);
                    let mut reasons = [0u64; REASONS.len()];
                    for line in &batch {
                        match convert(line, min_depth, scale) {
                            Ok(record) => records.extend_from_slice(&record),
                            Err(why) => reasons[REASONS.iter().position(|(r, _)| *r == why).unwrap()] += 1,
                        }
                    }
                    if to_writer.send((records, reasons)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(to_writer);

        // This thread reads; a second one writes what the workers hand back.
        let writer = scope.spawn(|| {
            let mut written = 0u64;
            let mut skipped = [0u64; REASONS.len()];
            for (records, reasons) in from_workers {
                out.write_all(&records).expect("failed to write training data");
                written += (records.len() / RECORD_SIZE) as u64;
                for (total, count) in skipped.iter_mut().zip(reasons) {
                    *total += count;
                }
            }
            out.flush().expect("failed to write training data");
            (written, skipped)
        });

        lines_read = feed(cfg.limit, to_workers);
        (written, skipped) = writer.join().expect("writer thread");
    });

    println!("{lines_read} lines read, {written} positions written to {}", cfg.out);
    for ((_, text), count) in REASONS.iter().zip(skipped) {
        println!("    left out, {text}: {count}");
    }
}

/// Settings of `index-evals`.
pub struct IndexConfig {
    pub out: String,
    pub threads: usize,
    /// Stop after this many lines (0: read everything).
    pub limit: u64,
}

/// Bytes of one record of the index.
pub const INDEX_RECORD: usize = 16;
/// A forced mate in n moves is stored as this minus n, negative if Black mates.
pub const INDEX_MATE: i32 = 32_000;
/// Centipawn scores are clamped to this.
pub const INDEX_MAX_CP: i32 = 20_000;

/// The key of a position in the index: the FNV-1a hash of the text the
/// database has for it (the first four fields of a FEN).
pub fn index_key(fen: &[u8]) -> u64 {
    fen.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// One line of the database as a record of the index, little-endian:
///
///     bytes 0-7    key of the position (`index_key`)
///     bytes 8-9    score for White of the deepest evaluation: centipawns, or a forced mate (`INDEX_MATE`)
///     bytes 10-11  its best move: from square, to square << 6, promotion << 12 (1 knight, 2 bishop, 3 rook, 4 queen)
///     byte  12     its depth
///     byte  13     evaluations the database has of the position
///     byte  14     the most lines any one of them gives
///     byte  15     0
///
/// Squares count from a1 = 0 along the ranks. The move is as the database
/// writes it: castling may be the king taking its own rook.
pub fn index_record(line: &[u8]) -> Option<[u8; INDEX_RECORD]> {
    let entry = read_line(line)?;
    let score = match (entry.cp, entry.mate) {
        (Some(cp), _) => cp.clamp(-INDEX_MAX_CP, INDEX_MAX_CP),
        (None, Some(mate)) => (INDEX_MATE - mate.abs().min(1000)) * if mate < 0 { -1 } else { 1 },
        (None, None) => return None,
    };
    let from = square(entry.best_move)?;
    let to = square(entry.best_move.get(2..)?)?;
    let promotion: u16 = match entry.best_move.get(4) {
        None => 0,
        Some(b'n') => 1,
        Some(b'b') => 2,
        Some(b'r') => 3,
        Some(b'q') => 4,
        Some(_) => return None,
    };
    let mut record = [0u8; INDEX_RECORD];
    record[..8].copy_from_slice(&index_key(entry.fen).to_le_bytes());
    record[8..10].copy_from_slice(&(score as i16).to_le_bytes());
    record[10..12].copy_from_slice(&(from.0 as u16 | (to.0 as u16) << 6 | promotion << 12).to_le_bytes());
    record[12] = entry.depth.min(255) as u8;
    record[13] = entry.evals.min(255) as u8;
    record[14] = entry.most_lines.min(255) as u8;
    Some(record)
}

fn record_key(record: &[u8; INDEX_RECORD]) -> u64 {
    u64::from_le_bytes(record[..8].try_into().unwrap())
}

/// Writes the index of the database on standard input: one record a position, sorted by key,
/// so that a position is found by bisection without reading the database again.
pub fn index(cfg: IndexConfig) {
    let (to_workers, from_reader) = mpsc::sync_channel::<Vec<Vec<u8>>>(cfg.threads * 2);
    let from_reader = Arc::new(Mutex::new(from_reader));
    let (to_collector, from_workers) = mpsc::sync_channel::<(Vec<[u8; INDEX_RECORD]>, u64)>(cfg.threads * 2);

    let mut records: Vec<[u8; INDEX_RECORD]> = Vec::new();
    let mut unreadable = 0u64;
    let mut lines_read = 0u64;
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads {
            let (from_reader, to_collector) = (Arc::clone(&from_reader), to_collector.clone());
            scope.spawn(move || {
                loop {
                    let batch = match from_reader.lock().unwrap().recv() {
                        Ok(batch) => batch,
                        Err(_) => return,
                    };
                    let mut found = Vec::with_capacity(batch.len());
                    let mut bad = 0u64;
                    for line in &batch {
                        match index_record(line) {
                            Some(record) => found.push(record),
                            None => bad += 1,
                        }
                    }
                    if to_collector.send((found, bad)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(to_collector);
        let collector = scope.spawn(|| {
            let mut all: Vec<[u8; INDEX_RECORD]> = Vec::new();
            let mut bad = 0u64;
            for (found, count) in from_workers {
                all.extend_from_slice(&found);
                bad += count;
            }
            (all, bad)
        });
        lines_read = feed(cfg.limit, to_workers);
        (records, unreadable) = collector.join().expect("collector thread");
    });

    records.sort_unstable_by_key(record_key);
    let shared = records.windows(2).filter(|pair| record_key(&pair[0]) == record_key(&pair[1])).count();
    let written = File::create(&cfg.out).and_then(|file| {
        let mut out = BufWriter::with_capacity(1 << 22, file);
        for record in &records {
            out.write_all(record)?;
        }
        out.flush()
    });
    if let Err(e) = written {
        eprintln!("cannot write {}: {e}", cfg.out);
        return;
    }
    println!("{lines_read} lines read, {} positions indexed in {}", records.len(), cfg.out);
    if unreadable > 0 {
        println!("    left out, could not be read: {unreadable}");
    }
    if shared > 0 {
        println!("    positions that share their key with another: {shared}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record_score(record: &[u8; RECORD_SIZE]) -> i32 {
        i16::from_le_bytes([record[24], record[25]]) as i32
    }

    #[test]
    fn the_deepest_evaluation_and_its_first_line_are_used() {
        // Black to move; the deeper evaluation says +69 for White, so -69 for the side to move.
        let line = br#"{"fen":"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -","evals":[{"pvs":[{"cp":163,"line":"h8d8 d1e1"}],"knodes":10,"depth":20},{"pvs":[{"cp":69,"line":"f7g7 e6e2 h8d8"},{"cp":163,"line":"h8d8 d1e1"}],"knodes":4189972,"depth":46}]}"#;
        let record = convert(line, 16, 100).expect("a record");
        assert_eq!(record_score(&record), -69);
        assert_eq!(record[26], 1);
        // The same position as datagen would store it.
        let pos = Position::from_fen("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -").unwrap();
        assert_eq!(record, pack(&pos, -69, 1));
        // Scaled on the way in.
        assert_eq!(record_score(&convert(line, 16, 128).unwrap()), -88);
        // Too shallow if more depth is asked for than the database has.
        assert_eq!(convert(line, 50, 100), Err(Skipped::Shallow));
    }

    #[test]
    fn mates_checks_and_captures_are_left_out() {
        let mate = br#"{"fen":"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -","evals":[{"pvs":[{"mate":1,"line":"a1a8"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(convert(mate, 16, 100), Err(Skipped::Mate));
        let check = br#"{"fen":"4k3/8/8/8/8/8/4r3/4K3 w - -","evals":[{"pvs":[{"cp":-500,"line":"e1e2"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(convert(check, 16, 100), Err(Skipped::InCheck));
        let capture = br#"{"fen":"4k3/8/8/3p4/4P3/8/8/4K3 w - -","evals":[{"pvs":[{"cp":20,"line":"e4d5 e8d7"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(convert(capture, 16, 100), Err(Skipped::NoisyBestMove));
        let promotion = br#"{"fen":"4k3/P7/8/8/8/8/8/4K3 w - -","evals":[{"pvs":[{"cp":900,"line":"a7a8q"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(convert(promotion, 16, 100), Err(Skipped::NoisyBestMove));
        // Castling written as king takes rook is a quiet move.
        let castling = br#"{"fen":"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -","evals":[{"pvs":[{"cp":10,"line":"e1h1 e8h8"}],"knodes":5,"depth":30}]}"#;
        assert!(convert(castling, 16, 100).is_ok());
    }

    #[test]
    fn index_records_hold_the_deepest_evaluation() {
        // FNV-1a, 64 bits: the published values for "" and "a".
        assert_eq!(index_key(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(index_key(b"a"), 0xaf63_dc4c_8601_ec8c);

        let line = br#"{"fen":"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -","evals":[{"pvs":[{"cp":163,"line":"h8d8 d1e1"}],"knodes":10,"depth":20},{"pvs":[{"cp":69,"line":"f7g7 e6e2 h8d8"},{"cp":163,"line":"h8d8 d1e1"}],"knodes":4189972,"depth":46}]}"#;
        let record = index_record(line).expect("a record");
        assert_eq!(record_key(&record), index_key(b"7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"));
        assert_eq!(i16::from_le_bytes([record[8], record[9]]), 69);
        // f7 is square 53, g7 square 54.
        assert_eq!(u16::from_le_bytes([record[10], record[11]]), 53 | 54 << 6);
        assert_eq!(record[12..], [46, 2, 2, 0]);

        let mate = br#"{"fen":"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -","evals":[{"pvs":[{"mate":1,"line":"a1a8"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(i16::from_le_bytes(index_record(mate).unwrap()[8..10].try_into().unwrap()), 31_999);
        let mated = br#"{"fen":"6k1/8/8/8/8/8/5PPP/r5K1 b - -","evals":[{"pvs":[{"mate":-3,"line":"a1a2"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(i16::from_le_bytes(index_record(mated).unwrap()[8..10].try_into().unwrap()), -31_997);
        let promotion = br#"{"fen":"4k3/P7/8/8/8/8/8/4K3 w - -","evals":[{"pvs":[{"cp":900,"line":"a7a8q"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(u16::from_le_bytes(index_record(promotion).unwrap()[10..12].try_into().unwrap()), 48 | 56 << 6 | 4 << 12);
        assert_eq!(index_record(br#"{"fen":"8/8/8/8/8/8/8/8 w - -","evals":[]}"#), None);
    }

    #[test]
    fn positions_along_the_lines_carry_the_lines_scores() {
        // White to move. Line 1: 1.Nf3 Nf6 2.d4, +30. Line 2: 1.a3 e5 2.Nc3, -10 for White.
        let line = br#"{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -","evals":[{"pvs":[{"cp":30,"line":"g1f3 g8f6 d2d4"},{"cp":-10,"line":"a2a3 e7e5 b1c3"}],"knodes":10,"depth":30}]}"#;
        let mut records = Vec::new();
        records_along_lines(line, 16, 100, 5, 2, &mut records);
        // After 1.Nf3 (Black to move, -30 for it), after 1.Nf3 Nf6 (White, +30),
        // after 1.a3 (Black, +10), after 1.a3 e5 (White, -10).
        let scores: Vec<i32> = records.iter().map(record_score).collect();
        assert_eq!(scores, [-30, 30, 10, -10]);
        let mut pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        pos.make_move(crate::movegen::parse_uci_move(&pos, "g1f3").unwrap());
        assert_eq!(records[0], pack(&pos, -30, 1));
        // One line only, one move in.
        records.clear();
        records_along_lines(line, 16, 100, 1, 1, &mut records);
        assert_eq!(records.iter().map(record_score).collect::<Vec<_>>(), [-30]);
        // A search of depth 30 says nothing of depth 29 or more about the second move's position.
        records.clear();
        records_along_lines(line, 29, 100, 5, 2, &mut records);
        assert_eq!(records.len(), 2);
        // The same position does not get the same key as another one.
        assert_ne!(position_key(&records[0]), position_key(&records[1]));
    }

    #[test]
    fn lines_stop_at_captures_checks_and_their_end() {
        // 1.e4 d5 2.exd5: the position after 1...d5 has a capture as its next move and is
        // left out; the one after 1.e4 is kept. The line ends after 2.exd5, so the position
        // after it has no move to stand in for its best one.
        let line = br#"{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -","evals":[{"pvs":[{"cp":40,"line":"e2e4 d7d5 e4d5"}],"knodes":10,"depth":30}]}"#;
        let mut records = Vec::new();
        records_along_lines(line, 16, 100, 5, 4, &mut records);
        assert_eq!(records.iter().map(record_score).collect::<Vec<_>>(), [-40]);
        // A forced mate is no score, and a move that is not legal ends its line.
        let mate = br#"{"fen":"6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -","evals":[{"pvs":[{"mate":1,"line":"a1a8"},{"cp":500,"line":"a1a7 g8f8 a7a9 f8e8"}],"knodes":5,"depth":30}]}"#;
        records.clear();
        records_along_lines(mate, 16, 100, 5, 4, &mut records);
        assert_eq!(records.len(), 1);
        // Castling written as the king taking its rook.
        let castling = br#"{"fen":"r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -","evals":[{"pvs":[{"cp":10,"line":"e1h1 e8h8 a2a3"},{"cp":5,"line":"e1c1 e8c8 a2a3"}],"knodes":5,"depth":30}]}"#;
        records.clear();
        records_along_lines(castling, 16, 100, 5, 2, &mut records);
        assert_eq!(records.len(), 4);
    }

    #[test]
    fn broken_lines_are_reported_not_fatal() {
        assert_eq!(convert(b"", 16, 100), Err(Skipped::Malformed));
        assert_eq!(convert(br#"{"fen":"8/8/8/8/8/8/8/8 w - -","evals":[]}"#, 16, 100), Err(Skipped::Malformed));
        let no_kings = br#"{"fen":"8/8/8/8/8/8/8/R7 w - -","evals":[{"pvs":[{"cp":500,"line":"a1a2"}],"knodes":5,"depth":30}]}"#;
        assert_eq!(convert(no_kings, 16, 100), Err(Skipped::BadPosition));
    }
}
