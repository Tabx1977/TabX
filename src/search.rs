//! Iterative-deepening alpha-beta search.

use crate::history::{self, History, NO_CONT};
use crate::movegen::{MoveList, generate_all};
use crate::movepick::MovePicker;
use crate::nnue::{Accumulator, Delta};
use crate::params;
use crate::position::{Position, SEE_VALUES};
use crate::tt::{Bound, NO_EVAL, TranspositionTable};
use crate::types::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub const MAX_PLY: usize = 128;
pub const MATE: i32 = 32_000;
pub const INFINITE: i32 = 32_001;
/// Scores with an absolute value above this are "mate in N plies" scores.
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32;

const SE_MAX_DOUBLE: i32 = 6;
const TM_MIN_DEPTH: i32 = 6;
const ASPIRATION_MIN_DEPTH: i32 = 5;

/// How often (in nodes) a thread looks at the clock and the stop flag.
const CHECK_INTERVAL: u64 = 1024;

/// When to stop searching.
#[derive(Clone)]
pub struct Limits {
    pub start: Instant,
    /// Maximum iteration depth.
    pub depth: i32,
    /// Hard node limit for the main thread.
    pub nodes: u64,
    /// Do not start another iteration once this many nodes have been searched.
    pub soft_nodes: u64,
    /// Do not start another iteration once this much time has passed.
    pub soft_time: Option<Duration>,
    /// Abort the search once this much time has passed.
    pub hard_time: Option<Duration>,
    /// Keep the result back until `stop` arrives (UCI `go infinite`).
    pub infinite: bool,
    /// What a draw is worth less than nothing to the side to move at the
    /// root, in centipawns. Positive against an opponent the engine expects
    /// to beat: it then plays on in a level position instead of repeating
    /// moves. Negative makes a draw welcome.
    pub contempt: i32,
}

impl Limits {
    pub fn new() -> Limits {
        Limits {
            start: Instant::now(),
            depth: MAX_PLY as i32 - 1,
            nodes: u64::MAX,
            soft_nodes: u64::MAX,
            soft_time: None,
            hard_time: None,
            infinite: false,
            contempt: 0,
        }
    }
}

/// State shared by all search threads.
pub struct Shared {
    pub stop: AtomicBool,
    /// The search is thinking on the opponent's time (UCI `go ponder`). Its
    /// time limits do not apply and it reports nothing until `ponderhit`
    /// clears the flag or `stop` ends the search.
    pub pondering: AtomicBool,
    /// Set by the search while pondering once it has used the time it would
    /// have taken for the move: `ponderhit` then ends it at once.
    pub stop_on_ponderhit: AtomicBool,
    pub nodes: AtomicU64,
    pub tt: TranspositionTable,
}

impl Shared {
    pub fn new(hash_mb: usize) -> Shared {
        Shared {
            stop: AtomicBool::new(false),
            pondering: AtomicBool::new(false),
            stop_on_ponderhit: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            tt: TranspositionTable::new(hash_mb),
        }
    }
}

/// Mate scores are stored relative to the node they were found in ("mate in
/// N from here") rather than relative to the root.
#[inline(always)]
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score + ply as i32
    } else if score <= -MATE_BOUND {
        score - ply as i32
    } else {
        score
    }
}

#[inline(always)]
fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_BOUND {
        score - ply as i32
    } else if score <= -MATE_BOUND {
        score + ply as i32
    } else {
        score
    }
}

/// Per-thread state that survives from one search to the next.
pub struct ThreadData {
    history: History,
    /// Late move reduction in plies, indexed by `[depth][move number]`: for
    /// quiet moves, and for captures and promotions.
    lmr: Box<[[u8; 64]; 64]>,
    lmr_noisy: Box<[[u8; 64]; 64]>,
    /// Network accumulator for the position at each ply of the search. They
    /// are computed lazily: `deltas[ply]` holds the piece changes that lead
    /// from ply - 1 to ply, and `computed[ply]` says whether they were applied.
    accumulators: Vec<Accumulator>,
    deltas: Vec<Delta>,
    computed: Vec<bool>,
    /// Position at a ply, kept only when its accumulator needs a rebuild
    /// (a king changed bucket), which cannot be done from the delta alone.
    rebuild_positions: Vec<Position>,
}

impl ThreadData {
    pub fn new() -> ThreadData {
        let mut td = ThreadData {
            history: History::new(),
            lmr: Box::new([[0u8; 64]; 64]),
            lmr_noisy: Box::new([[0u8; 64]; 64]),
            accumulators: vec![Accumulator::new(&Position::startpos()); MAX_PLY + 2],
            deltas: vec![Delta::NULL; MAX_PLY + 2],
            computed: vec![false; MAX_PLY + 2],
            rebuild_positions: vec![Position::startpos(); MAX_PLY + 2],
        };
        td.build_lmr_table();
        td
    }

    fn build_lmr_table(&mut self) {
        let table = |base: i32, divisor: i32, table: &mut [[u8; 64]; 64]| {
            for (depth, row) in table.iter_mut().enumerate().skip(1) {
                for (moves, entry) in row.iter_mut().enumerate().skip(1) {
                    let r = base as f64 / 100.0 + (depth as f64).ln() * (moves as f64).ln() / (divisor as f64 / 100.0);
                    *entry = r.max(0.0) as u8;
                }
            }
        };
        table(params::lmr_base(), params::lmr_divisor(), &mut self.lmr);
        table(params::lmr_noisy_base(), params::lmr_noisy_divisor(), &mut self.lmr_noisy);
    }

    /// Forgets everything learned in previous searches (UCI `ucinewgame`).
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Per-ply search state.
#[derive(Clone, Copy)]
struct StackEntry {
    /// Quiet moves that caused a cutoff at this ply in sibling nodes.
    killers: [Move; 2],
    /// Static evaluation of the position at this ply (`NO_EVAL` when in check).
    static_eval: i32,
    /// Move being searched from this ply; `Move::NONE` for a null move.
    mv: Move,
    /// Continuation history row of that move (`NO_CONT` for a null move).
    cont: usize,
    /// Move left out of the search of this node (singular extension test).
    excluded: Move,
    /// Number of double extensions on the path to this node.
    double_extensions: i32,
}

impl StackEntry {
    const EMPTY: StackEntry = StackEntry {
        killers: [Move::NONE; 2],
        static_eval: NO_EVAL,
        mv: Move::NONE,
        cont: NO_CONT,
        excluded: Move::NONE,
        double_extensions: 0,
    };
}

pub struct SearchResult {
    pub best_move: Move,
    /// The reply the search expects, `Move::NONE` if it has none.
    pub ponder_move: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

/// Output settings for a search.
#[derive(Clone, Copy)]
pub struct Report {
    pub chess960: bool,
    /// Print `info` lines and the final `bestmove`.
    pub print: bool,
    /// Number of best lines to search and report (UCI MultiPV).
    pub multipv: usize,
}

struct Search<'a> {
    shared: &'a Shared,
    limits: &'a Limits,
    td: &'a mut ThreadData,
    stack: Vec<StackEntry>,
    report: Report,
    is_main: bool,
    nodes: u64,
    flushed_nodes: u64,
    seldepth: usize,
    stopped: bool,
    root_depth: i32,
    /// Hashes of every position before the current one, game history included.
    keys: Vec<u64>,
    pv: Vec<[Move; MAX_PLY + 1]>,
    pv_len: [usize; MAX_PLY + 1],
    /// Best root move found so far, with its score and line. Updated as soon
    /// as a root move improves on the previous best, even mid-iteration.
    root_best: Move,
    root_score: i32,
    root_pv: Vec<Move>,
    /// Nodes spent below each root move in the current iteration, `[from][to]`.
    root_nodes: Box<[[u64; 64]; 64]>,
    /// Root moves already reported as better lines in this iteration (MultiPV).
    root_excluded: Vec<Move>,
}

/// Searches `root` with all threads in `threads` and returns the main thread's result.
/// `game_keys` are the hashes of the earlier positions of the game, oldest first.
pub fn run(
    shared: &Shared,
    threads: &mut [ThreadData],
    root: &Position,
    game_keys: &[u64],
    limits: &Limits,
    report: Report,
) -> SearchResult {
    // The caller clears `shared.stop` before starting, so that a stop request
    // racing with the start of the search is never lost.
    shared.nodes.store(0, Ordering::Relaxed);
    shared.tt.new_search();
    // In tuning builds the reduction table depends on adjustable parameters.
    #[cfg(feature = "tune")]
    for td in threads.iter_mut() {
        td.build_lmr_table();
    }

    let (main, helpers) = threads.split_first_mut().expect("at least one search thread");
    let result = std::thread::scope(|scope| {
        for td in helpers {
            std::thread::Builder::new()
                .stack_size(crate::uci::SEARCH_STACK_SIZE)
                .spawn_scoped(scope, move || {
                    Search::new(shared, limits, td, game_keys, report, false).iterate(root);
                })
                .expect("failed to spawn search thread");
        }
        let result = Search::new(shared, limits, main, game_keys, report, true).iterate(root);
        // "go infinite" must not report before the GUI says stop, and a
        // search on the opponent's time not before the opponent has moved.
        while (limits.infinite || shared.pondering.load(Ordering::SeqCst)) && !shared.stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
        }
        shared.stop.store(true, Ordering::Relaxed);
        result
    });

    if report.print {
        if result.ponder_move.is_none() {
            println!("bestmove {}", result.best_move.to_uci(report.chess960));
        } else {
            println!(
                "bestmove {} ponder {}",
                result.best_move.to_uci(report.chess960),
                result.ponder_move.to_uci(report.chess960)
            );
        }
    }
    result
}

impl<'a> Search<'a> {
    fn new(
        shared: &'a Shared,
        limits: &'a Limits,
        td: &'a mut ThreadData,
        game_keys: &[u64],
        report: Report,
        is_main: bool,
    ) -> Search<'a> {
        let mut keys = Vec::with_capacity(game_keys.len() + MAX_PLY + 8);
        keys.extend_from_slice(game_keys);
        Search {
            shared,
            limits,
            td,
            stack: vec![StackEntry::EMPTY; MAX_PLY + 2],
            report,
            is_main,
            nodes: 0,
            flushed_nodes: 0,
            seldepth: 0,
            stopped: false,
            root_depth: 0,
            keys,
            pv: vec![[Move::NONE; MAX_PLY + 1]; MAX_PLY + 1],
            pv_len: [0; MAX_PLY + 1],
            root_best: Move::NONE,
            root_score: 0,
            root_pv: Vec::new(),
            root_nodes: Box::new([[0; 64]; 64]),
            root_excluded: Vec::new(),
        }
    }

    fn iterate(&mut self, root: &Position) -> SearchResult {
        let mut legal = MoveList::new();
        generate_all(root, &mut legal);
        let mut result =
            SearchResult { best_move: Move::NONE, ponder_move: Move::NONE, score: 0, depth: 0, nodes: 0 };
        if legal.is_empty() {
            return result;
        }
        result.best_move = legal[0];
        self.td.accumulators[0].refresh(root);
        self.td.computed[0] = true;

        // With MultiPV the root is searched once per requested line, each
        // time excluding the best moves of the lines before it.
        let multipv = self.report.multipv.clamp(1, legal.len());
        let mut lines = vec![(Move::NONE, 0); multipv];
        // How many iterations in a row the best move has stayed the same.
        let mut stability = 0u32;
        'deepening: for depth in 1..=self.limits.depth {
            self.root_depth = depth;
            self.seldepth = 0;
            *self.root_nodes = [[0; 64]; 64];
            let nodes_before = self.nodes;
            let previous_best = result.best_move;
            self.root_excluded.clear();
            #[allow(clippy::needless_range_loop)]
            for index in 0..multipv {
                self.root_best = lines[index].0;
                let score = self.aspiration_search(root, depth, lines[index].1);
                if self.stopped {
                    // An interrupted iteration may still have found a move that
                    // beats the previous best; it was searched to full depth, so use it.
                    if index == 0 && !self.root_best.is_none() && self.root_best != result.best_move {
                        result.best_move = self.root_best;
                        result.score = self.root_score;
                        if self.is_main && self.report.print {
                            self.flush_nodes();
                            self.print_info(depth, self.root_score, 1);
                        }
                    }
                    break 'deepening;
                }
                lines[index] = (self.root_best, score);
                self.root_excluded.push(self.root_best);
                if index == 0 {
                    result.best_move = self.root_best;
                    result.score = score;
                    result.depth = depth;
                }
                if self.is_main && self.report.print {
                    self.flush_nodes();
                    self.print_info(depth, score, index + 1);
                }
            }
            if self.is_main {
                self.flush_nodes();
                if let Some(soft) = self.limits.soft_time {
                    // Spend less time when the choice looks settled: the best
                    // move took most of the effort and has not changed for a
                    // while. Spend more when it is still in doubt.
                    stability = if result.best_move == previous_best { stability + 1 } else { 0 };
                    let best = result.best_move;
                    let best_nodes = self.root_nodes[best.from().idx()][best.to().idx()];
                    let fraction = best_nodes as f64 / (self.nodes - nodes_before).max(1) as f64;
                    let node_scale =
                        (params::tm_node_base() as f64 - params::tm_node_slope() as f64 * fraction) / 100.0;
                    let stability_scale = (params::tm_stability_base() as f64
                        - params::tm_stability_step() as f64 * stability.min(6) as f64)
                        / 100.0;
                    let scale = if depth >= TM_MIN_DEPTH { node_scale * stability_scale } else { 1.0 };
                    let time_is_up = self.limits.start.elapsed().as_secs_f64() >= soft.as_secs_f64() * scale;
                    // On the opponent's time the search carries on, and only
                    // notes that it would answer at once if the expected move
                    // is played. The flag is written before the second look
                    // at `pondering`, and `ponderhit` clears `pondering`
                    // before it reads the flag, so one of the two always
                    // sees the other.
                    self.shared.stop_on_ponderhit.store(time_is_up, Ordering::SeqCst);
                    if time_is_up && !self.shared.pondering.load(Ordering::SeqCst) {
                        break;
                    }
                }
                if self.nodes >= self.limits.soft_nodes {
                    break;
                }
            }
        }
        self.flush_nodes();
        result.nodes = self.shared.nodes.load(Ordering::Relaxed);
        if self.is_main {
            result.ponder_move = self.expected_reply(root, result.best_move);
        }
        result
    }

    /// The reply the search expects to its best move, which the GUI may let
    /// the engine think about during the opponent's turn: the second move of
    /// the best line, or else the hash move of the position after the best
    /// move. `Move::NONE` if neither is a legal move there.
    fn expected_reply(&self, root: &Position, best: Move) -> Move {
        let mut next = *root;
        next.make_move(best);
        let mut replies = MoveList::new();
        generate_all(&next, &mut replies);
        if self.root_pv.first() == Some(&best)
            && let Some(&reply) = self.root_pv.get(1)
            && replies.contains(&reply)
        {
            return reply;
        }
        match self.shared.tt.probe(next.hash()) {
            Some(entry) if replies.contains(&entry.mv) => entry.mv,
            _ => Move::NONE,
        }
    }

    /// Searches the root with a narrow window around the previous iteration's
    /// score, widening it whenever the result falls outside.
    fn aspiration_search(&mut self, root: &Position, depth: i32, previous: i32) -> i32 {
        if depth < ASPIRATION_MIN_DEPTH {
            return self.search(root, -INFINITE, INFINITE, depth, 0, false);
        }
        let mut delta = params::aspiration_delta();
        let mut alpha = (previous - delta).max(-INFINITE);
        let mut beta = (previous + delta).min(INFINITE);
        loop {
            let score = self.search(root, alpha, beta, depth, 0, false);
            if self.stopped {
                return score;
            }
            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (score - delta).max(-INFINITE);
            } else if score >= beta {
                beta = (score + delta).min(INFINITE);
            } else {
                return score;
            }
            delta += delta / 2;
            if delta > 1000 {
                alpha = -INFINITE;
                beta = INFINITE;
            }
        }
    }

    fn flush_nodes(&mut self) {
        self.shared.nodes.fetch_add(self.nodes - self.flushed_nodes, Ordering::Relaxed);
        self.flushed_nodes = self.nodes;
    }

    fn print_info(&self, depth: i32, score: i32, multipv: usize) {
        let elapsed = self.limits.start.elapsed();
        let nodes = self.shared.nodes.load(Ordering::Relaxed);
        let nps = (nodes as f64 / elapsed.as_secs_f64().max(1e-6)) as u64;
        let score_text = if score.abs() >= MATE_BOUND {
            let plies = MATE - score.abs();
            let moves = (plies + 1) / 2;
            format!("mate {}", if score > 0 { moves } else { -moves })
        } else {
            format!("cp {score}")
        };
        let pv: Vec<String> = self.root_pv.iter().map(|m| m.to_uci(self.report.chess960)).collect();
        println!(
            "info depth {depth} seldepth {} multipv {multipv} score {score_text} nodes {nodes} nps {nps} hashfull {} time {} pv {}",
            self.seldepth,
            self.shared.tt.hashfull(),
            elapsed.as_millis(),
            pv.join(" ")
        );
    }

    /// Static evaluation of the position at `ply` from the side to move's point of view.
    #[inline(always)]
    fn evaluate(&mut self, pos: &Position, ply: usize) -> i32 {
        if !self.td.computed[ply] {
            self.compute_accumulators(ply);
        }
        self.td.accumulators[ply].evaluate(pos)
    }

    /// Brings the accumulator at `ply` up to date by applying the pending
    /// piece changes since the nearest computed ancestor (the root always is).
    fn compute_accumulators(&mut self, ply: usize) {
        let mut first = ply;
        while !self.td.computed[first - 1] {
            first -= 1;
        }
        for p in first..=ply {
            let (parents, children) = self.td.accumulators.split_at_mut(p);
            children[0].apply(&parents[p - 1], &self.td.deltas[p], &self.td.rebuild_positions[p]);
            self.td.computed[p] = true;
        }
    }

    /// Records the piece changes of `mv` (played in `pos` at `ply`, giving
    /// `next`) for the child's accumulator, which is only computed if it gets evaluated.
    #[inline(always)]
    fn push_accumulator(&mut self, pos: &Position, next: &Position, ply: usize, mv: Move) {
        let delta = Delta::from_move(pos, mv);
        if delta.needs_position() {
            self.td.rebuild_positions[ply + 1] = *next;
        }
        self.td.deltas[ply + 1] = delta;
        self.td.computed[ply + 1] = false;
    }

    /// Counts a node and periodically checks whether the search has to stop.
    #[inline(always)]
    fn visit_node(&mut self) {
        self.nodes += 1;
        if self.is_main && self.nodes >= self.limits.nodes {
            self.stop_all();
        }
        if self.nodes.is_multiple_of(CHECK_INTERVAL) {
            self.poll();
        }
    }

    #[cold]
    fn stop_all(&mut self) {
        self.stopped = true;
        self.shared.stop.store(true, Ordering::Relaxed);
    }

    fn poll(&mut self) {
        self.flush_nodes();
        // The first iteration always completes so there is a move to play.
        if self.root_depth <= 1 {
            return;
        }
        if self.shared.stop.load(Ordering::Relaxed) {
            self.stopped = true;
        } else if self.is_main
            && let Some(hard) = self.limits.hard_time
            && self.limits.start.elapsed() >= hard
            && !self.shared.pondering.load(Ordering::Relaxed)
        {
            self.stop_all();
        }
    }

    /// Draw by repetition: the position occurred earlier inside the search
    /// tree, or twice before in the game.
    fn is_repetition(&self, pos: &Position, ply: usize) -> bool {
        let len = self.keys.len();
        let window = pos.repetition_window().min(len);
        let mut seen_before = false;
        let mut distance = 4;
        while distance <= window {
            if self.keys[len - distance] == pos.hash() {
                if distance < ply || seen_before {
                    return true;
                }
                seen_before = true;
            }
            distance += 2;
        }
        false
    }

    /// The score of a drawn position `ply` plies from the root, for the side
    /// to move there: the root's side to move thinks `contempt` less of a draw.
    #[inline(always)]
    fn draw_score(&self, ply: usize) -> i32 {
        if ply.is_multiple_of(2) { -self.limits.contempt } else { self.limits.contempt }
    }

    fn is_draw(&self, pos: &Position, ply: usize) -> bool {
        if pos.halfmove() >= 100 {
            // Checkmate on the move that completes the fifty-move count still wins.
            if !pos.in_check() {
                return true;
            }
            let mut list = MoveList::new();
            generate_all(pos, &mut list);
            if !list.is_empty() {
                return true;
            }
        }
        pos.is_insufficient_material() || self.is_repetition(pos, ply)
    }

    fn update_pv(&mut self, ply: usize, mv: Move) {
        let child_len = self.pv_len[ply + 1];
        let (head, tail) = self.pv.split_at_mut(ply + 1);
        head[ply][0] = mv;
        head[ply][1..=child_len].copy_from_slice(&tail[0][..child_len]);
        self.pv_len[ply] = child_len + 1;
    }

    /// Continuation history rows for a node at `ply`: the opponent's last
    /// move and our own move before it.
    #[inline(always)]
    fn conts(&self, ply: usize) -> [usize; 2] {
        [
            if ply >= 1 { self.stack[ply - 1].cont } else { NO_CONT },
            if ply >= 2 { self.stack[ply - 2].cont } else { NO_CONT },
        ]
    }

    /// After a cutoff: rewards the move that caused it and penalises the
    /// moves of the same kind that were tried before it.
    #[allow(clippy::too_many_arguments)]
    fn update_stats(
        &mut self,
        pos: &Position,
        ply: usize,
        best: Move,
        quiets_tried: &[Move],
        noisy_tried: &[Move],
        depth: i32,
    ) {
        let bonus = history::bonus(depth);
        if best.is_quiet() {
            let killers = &mut self.stack[ply].killers;
            if killers[0] != best {
                killers[1] = killers[0];
                killers[0] = best;
            }
            let conts = self.conts(ply);
            self.td.history.update_quiet(pos, best, conts, bonus);
            for &mv in quiets_tried {
                self.td.history.update_quiet(pos, mv, conts, -bonus);
            }
        } else {
            self.td.history.update_capture(pos, best, bonus);
        }
        // Captures that failed to cut off are penalised whatever move did.
        for &mv in noisy_tried {
            self.td.history.update_capture(pos, mv, -bonus);
        }
    }

    /// `cut_node` is true when this node is expected to fail high (a
    /// zero-window search of a move that is not the parent's first).
    fn search(
        &mut self,
        pos: &Position,
        mut alpha: i32,
        beta: i32,
        mut depth: i32,
        ply: usize,
        cut_node: bool,
    ) -> i32 {
        if depth <= 0 {
            return self.qsearch(pos, alpha, beta, ply);
        }
        self.pv_len[ply] = 0;
        self.visit_node();
        if self.stopped {
            return 0;
        }
        let root = ply == 0;
        // In a principal variation node the window is open; everywhere else
        // the search only has to prove a bound (zero-width window).
        let pv_node = beta - alpha > 1;
        if !root {
            if self.is_draw(pos, ply) {
                return self.draw_score(ply);
            }
            if ply >= MAX_PLY {
                return self.evaluate(pos, ply);
            }
        }
        self.seldepth = self.seldepth.max(ply);

        // Transposition table: reuse the result of an earlier search of this
        // position if it was deep enough, otherwise at least try its best move first.
        let in_check = pos.in_check();
        // Set while testing whether the hash move is singular (see below).
        let excluded = self.stack[ply].excluded;
        let mut tt_move = Move::NONE;
        let mut tt_score = 0;
        let mut tt_depth = -1;
        let mut tt_bound = Bound::None;
        // The network's output as stored in the table, and the same after
        // correction history (which is what the search works with).
        let mut raw_eval = NO_EVAL;
        let mut static_eval = NO_EVAL;
        // `eval` is the best available estimate of the position's value: the
        // static evaluation, or a search score from the table if that is more informative.
        let mut eval = NO_EVAL;
        if let Some(entry) = self.shared.tt.probe(pos.hash()) {
            tt_move = entry.mv;
            tt_score = score_from_tt(entry.score, ply);
            tt_depth = entry.depth;
            tt_bound = entry.bound;
            let usable = match entry.bound {
                Bound::Exact => true,
                Bound::Lower => tt_score >= beta,
                Bound::Upper => tt_score <= alpha,
                Bound::None => false,
            };
            if !pv_node && excluded.is_none() && entry.depth >= depth && usable {
                return tt_score;
            }
            if !in_check {
                raw_eval = if entry.eval != NO_EVAL { entry.eval } else { self.evaluate(pos, ply) };
                static_eval = self.td.history.correct_eval(pos, raw_eval);
                eval = match entry.bound {
                    Bound::Exact => tt_score,
                    Bound::Lower if tt_score > static_eval => tt_score,
                    Bound::Upper if tt_score < static_eval => tt_score,
                    _ => static_eval,
                };
            }
        } else if !in_check {
            raw_eval = self.evaluate(pos, ply);
            static_eval = self.td.history.correct_eval(pos, raw_eval);
            eval = static_eval;
        }
        self.stack[ply].static_eval = static_eval;
        // Is our position getting better compared to our previous turn?
        let improving = !in_check
            && ply >= 2
            && self.stack[ply - 2].static_eval != NO_EVAL
            && static_eval > self.stack[ply - 2].static_eval;

        if !pv_node && !in_check && excluded.is_none() {
            // Razoring: far below alpha at low depth, check with a quiescence
            // search whether captures can save the position; if not, give up.
            if depth <= params::razor_max_depth() && static_eval + params::razor_margin() * depth < alpha {
                let score = self.qsearch(pos, alpha - 1, alpha, ply);
                if score < alpha {
                    return score;
                }
            }

            // Reverse futility pruning: the position is so far above beta that
            // a shallow search is very unlikely to bring it back below.
            if depth <= params::rfp_max_depth()
                && eval - params::rfp_margin() * (depth - improving as i32) >= beta
                && eval < MATE_BOUND
            {
                return eval;
            }

            // Null move pruning: if we could pass and still beat beta after a
            // reduced search, the position is good enough to cut off. Not
            // tried with only pawns left, where passing can be the best "move".
            if depth >= 3
                && eval >= beta
                && !self.stack[ply - 1].mv.is_none()
                && pos.has_non_pawn_material(pos.stm())
            {
                let reduction = params::nmp_base()
                    + depth / params::nmp_depth_div()
                    + ((eval - beta) / params::nmp_eval_div()).min(3);
                let mut next = *pos;
                next.make_null();
                self.stack[ply].mv = Move::NONE;
                self.stack[ply].cont = NO_CONT;
                self.td.deltas[ply + 1] = Delta::NULL;
                self.td.computed[ply + 1] = false;
                self.keys.push(pos.hash());
                let score = -self.search(&next, -beta, -beta + 1, depth - reduction, ply + 1, !cut_node);
                self.keys.pop();
                if self.stopped {
                    return 0;
                }
                if score >= beta {
                    // Do not return unproven mate scores from a null search.
                    return if score >= MATE_BOUND { beta } else { score };
                }
            }
        }

        // Internal iterative reduction: without a hash move the ordering here
        // is poor, so search shallower now and come back with a move next time.
        if depth >= params::iir_min_depth() && tt_move.is_none() && (pv_node || cut_node) {
            depth -= 1;
        }

        let first = if root && !self.root_best.is_none() { self.root_best } else { tt_move };
        let killers = self.stack[ply].killers;
        // Killers are only meaningful among sibling nodes.
        self.stack[ply + 1].killers = [Move::NONE; 2];
        let mut picker = MovePicker::new(first, killers);
        let conts = self.conts(ply);

        let original_alpha = alpha;
        let mut best_move = Move::NONE;
        let mut best_score = -INFINITE;
        let mut quiets_tried = [Move::NONE; 64];
        let mut noisy_tried = [Move::NONE; 32];
        let (mut quiet_count, mut noisy_count) = (0, 0);
        let mut moves_seen = 0;
        let mut moves_played = 0;
        while let Some(mv) = picker.next(pos, &self.td.history, conts) {
            if mv == excluded || (root && self.root_excluded.contains(&mv)) {
                continue;
            }
            moves_seen += 1;
            let is_quiet = mv.is_quiet();
            // Moves are only pruned once some move is known not to lose.
            let can_prune = !root && best_score > -MATE_BOUND;

            let gives_check = pos.gives_check(mv);

            if can_prune && is_quiet && !gives_check {
                // Late move pruning: with good move ordering, quiet moves this
                // far down the list almost never matter at low depth.
                if depth <= params::lmp_max_depth()
                    && moves_played >= (params::lmp_base() + depth * depth) / (2 - improving as i32)
                {
                    picker.skip_quiets();
                    continue;
                }
                // Futility pruning: the static evaluation is so far below
                // alpha that a quiet move will not make up the difference.
                if depth <= params::fp_max_depth()
                    && !in_check
                    && static_eval + params::fp_base() + params::fp_margin() * depth <= alpha
                {
                    continue;
                }
                // History pruning: quiet moves that have consistently failed
                // in similar positions are not worth a shallow search.
                if depth <= params::hp_max_depth()
                    && self.td.history.quiet_score(pos, mv, conts) < params::hp_margin() * depth
                {
                    continue;
                }
            }

            // Static exchange pruning: at low depth, skip moves that lose
            // material according to the exchange evaluation.
            if can_prune && depth <= params::see_max_depth() {
                let margin = if is_quiet { params::see_quiet_margin() } else { params::see_noisy_margin() };
                if !pos.see_ge(mv, margin * depth) {
                    continue;
                }
            }

            let mut next = *pos;
            next.make_move(mv);
            // Singular extension: if the hash move is much better than every
            // alternative (a reduced search without it fails well below its
            // score), it is searched deeper. If instead the alternatives are
            // also good enough, the node can be cut or the move reduced.
            let mut extension = 0;
            if !root
                && depth >= params::se_min_depth()
                && mv == tt_move
                && excluded.is_none()
                && tt_depth >= depth - 3
                && tt_bound != Bound::Upper
                && tt_score.abs() < MATE_BOUND
                && ply < 2 * self.root_depth as usize
            {
                let singular_beta = tt_score - depth * params::se_beta_mult() / 16;
                self.stack[ply].excluded = mv;
                let score = self.search(pos, singular_beta - 1, singular_beta, (depth - 1) / 2, ply, cut_node);
                self.stack[ply].excluded = Move::NONE;
                if self.stopped {
                    return 0;
                }
                if score < singular_beta {
                    extension = 1;
                    if !pv_node
                        && score < singular_beta - params::se_double_margin()
                        && self.stack[ply].double_extensions < SE_MAX_DOUBLE
                    {
                        extension = 2;
                    }
                } else if singular_beta >= beta {
                    // Multi-cut: even without the hash move this node fails high.
                    return singular_beta;
                } else if tt_score >= beta || cut_node {
                    extension = -1;
                }
            }
            self.stack[ply + 1].double_extensions =
                self.stack[ply].double_extensions + (extension >= 2) as i32;

            moves_played += 1;
            let nodes_before = self.nodes;
            self.stack[ply].mv = mv;
            self.stack[ply].cont = History::cont_row(pos.piece_on(mv.from()), mv.to());
            self.push_accumulator(pos, &next, ply, mv);
            self.shared.tt.prefetch(next.hash());
            self.keys.push(pos.hash());

            // Principal variation search: the first move gets the full window.
            // Later moves are expected to fail low, so they are first tried
            // with a zero-width window and only re-searched if they surprise.
            let new_depth = depth - 1 + extension;
            let mut score;
            if moves_played == 1 {
                score = -self.search(&next, -beta, -alpha, new_depth, ply + 1, !pv_node && !cut_node);
            } else {
                // Late move reductions: moves ordered late are unlikely to be
                // best, so they are searched to a reduced depth first.
                let mut reduction = 0;
                if depth >= 3 && is_quiet {
                    reduction = self.td.lmr[depth.min(63) as usize][(moves_played as usize).min(63)] as i32;
                    reduction -= pv_node as i32;
                    reduction -= gives_check as i32;
                    reduction -= (mv == killers[0] || mv == killers[1]) as i32;
                    // Moves with a good track record are reduced less, bad ones more.
                    reduction -= (self.td.history.quiet_score(pos, mv, conts) / params::lmr_history_div()).clamp(-2, 2);
                    reduction += cut_node as i32;
                    reduction = reduction.clamp(0, new_depth - 1);
                } else if depth >= 3 {
                    // Captures and promotions this far down the list: the
                    // most promising ones came first, so these are reduced
                    // as well, by less, and by how such captures have done.
                    reduction = self.td.lmr_noisy[depth.min(63) as usize][(moves_played as usize).min(63)] as i32;
                    reduction -= pv_node as i32;
                    reduction -= gives_check as i32;
                    reduction -=
                        (self.td.history.capture_score(pos, mv) / params::lmr_noisy_history_div()).clamp(-2, 2);
                    reduction += cut_node as i32;
                    reduction = reduction.clamp(0, new_depth - 1);
                }
                score = -self.search(&next, -alpha - 1, -alpha, new_depth - reduction, ply + 1, true);
                if reduction > 0 && score > alpha {
                    score = -self.search(&next, -alpha - 1, -alpha, new_depth, ply + 1, !cut_node);
                }
                if score > alpha && score < beta {
                    score = -self.search(&next, -beta, -alpha, new_depth, ply + 1, false);
                }
            }
            self.keys.pop();
            if self.stopped {
                return 0;
            }
            if root {
                self.root_nodes[mv.from().idx()][mv.to().idx()] += self.nodes - nodes_before;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mv;
                    self.update_pv(ply, mv);
                    if root {
                        self.root_best = mv;
                        self.root_score = score;
                        self.root_pv.clear();
                        self.root_pv.extend_from_slice(&self.pv[0][..self.pv_len[0]]);
                    }
                    if alpha >= beta {
                        self.update_stats(
                            pos,
                            ply,
                            mv,
                            &quiets_tried[..quiet_count],
                            &noisy_tried[..noisy_count],
                            depth,
                        );
                        break;
                    }
                }
            }
            if is_quiet {
                if quiet_count < quiets_tried.len() {
                    quiets_tried[quiet_count] = mv;
                    quiet_count += 1;
                }
            } else if noisy_count < noisy_tried.len() {
                noisy_tried[noisy_count] = mv;
                noisy_count += 1;
            }
        }

        if moves_seen == 0 {
            // With a move excluded, "no other move" means the excluded one is forced.
            return if !excluded.is_none() {
                alpha
            } else if in_check {
                -MATE + ply as i32
            } else {
                self.draw_score(ply)
            };
        }

        // A search with a move left out says nothing about the position as a whole.
        if excluded.is_none() && !(root && !self.root_excluded.is_empty()) {
            let bound = if best_score >= beta {
                Bound::Lower
            } else if best_score > original_alpha {
                Bound::Exact
            } else {
                Bound::Upper
            };
            self.shared.tt.store(pos.hash(), best_move, score_to_tt(best_score, ply), raw_eval, depth, bound, pv_node);

            // Correction history: remember how far the search result was from
            // the static evaluation, unless the difference is explained by a
            // capture or the score is only a bound on the wrong side.
            if !in_check
                && !best_move.is_noisy()
                && !(bound == Bound::Lower && best_score <= static_eval)
                && !(bound == Bound::Upper && best_score >= static_eval)
            {
                self.td.history.update_correction(pos, best_score - static_eval, depth);
            }
        }
        best_score
    }

    /// Quiescence search: resolves captures (and check evasions) so that the
    /// static evaluation is only trusted in quiet positions.
    fn qsearch(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.pv_len[ply] = 0;
        self.visit_node();
        if self.stopped {
            return 0;
        }
        if ply >= MAX_PLY {
            return self.evaluate(pos, ply);
        }
        self.seldepth = self.seldepth.max(ply);
        let pv_node = beta - alpha > 1;

        let in_check = pos.in_check();
        let mut tt_move = Move::NONE;
        let mut raw_eval = NO_EVAL;
        if let Some(entry) = self.shared.tt.probe(pos.hash()) {
            tt_move = entry.mv;
            let tt_score = score_from_tt(entry.score, ply);
            let usable = match entry.bound {
                Bound::Exact => true,
                Bound::Lower => tt_score >= beta,
                Bound::Upper => tt_score <= alpha,
                Bound::None => false,
            };
            if !pv_node && usable {
                return tt_score;
            }
            if !in_check && entry.eval != NO_EVAL {
                raw_eval = entry.eval;
            }
        }

        let mut best_move = Move::NONE;
        let mut best_score = -INFINITE;
        // Upper estimate of what a capture can gain on top of the static evaluation.
        let mut futility_base = -INFINITE;
        if !in_check {
            // Standing pat: the side to move may decline all captures.
            if raw_eval == NO_EVAL {
                raw_eval = self.evaluate(pos, ply);
            }
            best_score = self.td.history.correct_eval(pos, raw_eval);
            futility_base = best_score + params::qs_futility_margin();
            if best_score >= beta {
                return best_score;
            }
            alpha = alpha.max(best_score);
        }

        // In check every evasion is searched; otherwise only captures and
        // queen promotions that do not lose material.
        let mut picker =
            if in_check { MovePicker::new(tt_move, [Move::NONE; 2]) } else { MovePicker::new_noisy(tt_move) };
        let mut moves_seen = 0;
        let conts = self.conts(ply);
        while let Some(mv) = picker.next(pos, &self.td.history, conts) {
            moves_seen += 1;
            // Futility pruning: even winning the captured piece for free
            // would not bring the score up to alpha.
            if !in_check && mv.is_capture() && !mv.is_promo() {
                let victim =
                    if mv.is_en_passant() { PieceType::Pawn } else { pos.piece_on(mv.to()).piece_type() };
                let futility = futility_base + SEE_VALUES[victim.idx()];
                if futility <= alpha {
                    best_score = best_score.max(futility);
                    continue;
                }
            }
            let mut next = *pos;
            next.make_move(mv);
            self.stack[ply].mv = mv;
            self.stack[ply].cont = History::cont_row(pos.piece_on(mv.from()), mv.to());
            self.push_accumulator(pos, &next, ply, mv);
            self.shared.tt.prefetch(next.hash());
            self.keys.push(pos.hash());
            let score = -self.qsearch(&next, -beta, -alpha, ply + 1);
            self.keys.pop();
            if self.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    alpha = score;
                    best_move = mv;
                    self.update_pv(ply, mv);
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        if in_check && moves_seen == 0 {
            return -MATE + ply as i32;
        }

        let bound = if best_score >= beta { Bound::Lower } else { Bound::Upper };
        self.shared.tt.store(pos.hash(), best_move, score_to_tt(best_score, ply), raw_eval, 0, bound, pv_node);
        best_score
    }
}
