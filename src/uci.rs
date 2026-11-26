//! UCI protocol front end.
//!
//! The main thread reads commands from stdin. `go` hands the search to a
//! separate thread so that `stop`, `isready` and `quit` are answered at once.

use crate::bench;
use crate::nnue;
use crate::movegen::{self, parse_uci_move};
use crate::position::Position;
use crate::search::{self, Limits, Report, Shared, ThreadData};
use crate::types::*;
use std::io::BufRead;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

pub const ENGINE_NAME: &str = "TabX";
pub const ENGINE_AUTHOR: &str = "the TabX developers";
pub const SEARCH_STACK_SIZE: usize = 16 * 1024 * 1024;
pub const DEFAULT_HASH_MB: usize = 16;
const MAX_HASH_MB: usize = 1 << 20;

pub struct Options {
    pub threads: usize,
    /// Safety margin in milliseconds subtracted from the clock on every move.
    pub move_overhead: u64,
    pub chess960: bool,
    pub multipv: usize,
    /// Whether the GUI lets the engine think on the opponent's time (`go ponder`).
    pub ponder: bool,
    /// Centipawns a draw is worth less than nothing, whoever the opponent is.
    pub contempt: i32,
    /// The engine's own rating on the opponent's scale, 0 if not known.
    pub own_rating: i32,
    /// The opponent's rating, from `UCI_Opponent`.
    pub opponent_rating: Option<i32>,
}

/// Contempt for an opponent of another rating: this many centipawns for every
/// hundred rating points the engine is ahead, up to `MAX_CONTEMPT`, and down
/// to `-MAX_RESPECT` if it is behind.
const CONTEMPT_PER_100: i32 = 15;
const MAX_CONTEMPT: i32 = 40;
const MAX_RESPECT: i32 = 15;

/// What a draw is worth less than nothing against this opponent (see
/// `Limits::contempt`). On Lichess the bot drew 13 of 15 games with one
/// network, among them games against bots 150 to 300 points below it in
/// which it had stood better and then repeated moves: against those a draw
/// loses rating points, and against the bots above it a draw gains them.
pub fn contempt_for(options: &Options) -> i32 {
    let by_rating = match options.opponent_rating {
        Some(theirs) if options.own_rating > 0 && theirs > 0 => {
            ((options.own_rating - theirs) * CONTEMPT_PER_100 / 100).clamp(-MAX_RESPECT, MAX_CONTEMPT)
        }
        _ => 0,
    };
    options.contempt + by_rating
}

pub struct Engine {
    shared: Arc<Shared>,
    /// Per-thread search state; `None` while a search thread has borrowed it.
    threads: Option<Vec<ThreadData>>,
    search: Option<JoinHandle<Vec<ThreadData>>>,
    /// The running search was started with `go infinite`: it only ends on `stop`.
    unbounded: bool,
    pos: Position,
    /// Hashes of all earlier positions of the current game.
    keys: Vec<u64>,
    options: Options,
}

/// Runs the UCI loop until `quit` or end of input.
pub fn run() {
    let mut engine = Engine::new();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if !engine.handle(&line) {
            break;
        }
    }
    engine.shutdown();
}

impl Engine {
    pub fn new() -> Engine {
        Engine {
            shared: Arc::new(Shared::new(DEFAULT_HASH_MB)),
            threads: Some(vec![ThreadData::new()]),
            search: None,
            unbounded: false,
            pos: Position::startpos(),
            keys: Vec::new(),
            options: Options {
                threads: 1,
                move_overhead: 10,
                chess960: false,
                multipv: 1,
                ponder: false,
                contempt: 0,
                own_rating: 0,
                opponent_rating: None,
            },
        }
    }

    /// Blocks until the running search (if any) has finished.
    fn wait(&mut self) {
        if let Some(handle) = self.search.take() {
            self.threads = Some(handle.join().expect("search thread panicked"));
        }
    }

    /// Whether the running search cannot end by itself: `go infinite`, or
    /// `go ponder` before `ponderhit`.
    fn search_is_open_ended(&self) -> bool {
        self.unbounded || self.shared.pondering.load(Ordering::SeqCst)
    }

    /// Makes the engine ready for a command that needs the search finished.
    /// An open-ended search is stopped first; waiting for it would never end.
    fn finish_search(&mut self) {
        if self.search_is_open_ended() {
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        self.wait();
    }

    /// Exclusive access to the shared search state; only valid while no search is running.
    fn shared_mut(&mut self) -> &mut Shared {
        Arc::get_mut(&mut self.shared).expect("no search is running")
    }

    fn shutdown(&mut self) {
        // At end of input let a bounded search finish, so that piping
        // "go depth 10" into the engine still prints a result.
        self.finish_search();
    }

    /// Handles one command line. Returns false when the engine should exit.
    pub fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = tokens.first() else { return true };
        match command {
            "uci" => {
                println!("id name {ENGINE_NAME} {}", env!("CARGO_PKG_VERSION"));
                println!("id author {ENGINE_AUTHOR}");
                println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name MultiPV type spin default 1 min 1 max 256");
                println!("option name Ponder type check default false");
                println!("option name Move Overhead type spin default 10 min 0 max 5000");
                println!("option name UCI_Chess960 type check default false");
                println!("option name Contempt type spin default 0 min -100 max 100");
                println!("option name Own Rating type spin default 0 min 0 max 4000");
                println!("option name UCI_Opponent type string default <empty>");
                #[cfg(feature = "tune")]
                for (name, default, min, max, _) in crate::params::LIST {
                    println!("option name {name} type spin default {default} min {min} max {max}");
                }
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                self.finish_search();
                for td in self.threads.as_mut().unwrap() {
                    td.clear();
                }
                self.shared_mut().tt.clear();
            }
            "setoption" => {
                self.finish_search();
                self.set_option(&tokens[1..]);
            }
            "position" => {
                self.finish_search();
                self.set_position(&tokens[1..]);
            }
            "go" => {
                self.finish_search();
                self.go(&tokens[1..]);
            }
            "stop" => self.shared.stop.store(true, Ordering::Relaxed),
            "ponderhit" => {
                // The opponent played the expected move: the search carries on
                // as an ordinary one. Its time is counted from the start of
                // the pondering, so the move costs less on our own clock.
                self.shared.pondering.store(false, Ordering::SeqCst);
                if self.shared.stop_on_ponderhit.load(Ordering::SeqCst) {
                    self.shared.stop.store(true, Ordering::Relaxed);
                }
            }
            "quit" => {
                self.shared.stop.store(true, Ordering::Relaxed);
                self.unbounded = false;
                return false;
            }
            // Non-standard conveniences.
            "wait" => self.wait(),
            "d" => {
                self.wait();
                println!("{}", self.pos);
            }
            "eval" => {
                self.wait();
                println!("eval: {} cp (side to move)", nnue::evaluate(&self.pos));
            }
            "bench" => {
                self.wait();
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH);
                bench::run(depth);
            }
            // Tuning builds: list the adjustable parameters as "name default min max step".
            #[cfg(feature = "tune")]
            "params" => {
                for (name, default, min, max, step) in crate::params::LIST {
                    println!("{name} {default} {min} {max} {step}");
                }
                println!("paramsok");
            }
            "perft" => {
                self.wait();
                let depth = tokens.get(1).and_then(|d| d.parse().ok()).unwrap_or(1);
                perft_divide(&self.pos, depth, self.options.chess960);
            }
            _ => println!("info string unknown command: {line}"),
        }
        true
    }

    fn set_option(&mut self, tokens: &[&str]) {
        // setoption name <name with spaces> value <value>
        let value_at = tokens.iter().position(|&t| t == "value").unwrap_or(tokens.len());
        let name = tokens.get(1..value_at).unwrap_or(&[]).join(" ").to_ascii_lowercase();
        let value = tokens.get(value_at + 1..).unwrap_or(&[]).join(" ");
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    self.shared_mut().tt.resize(mb.clamp(1, MAX_HASH_MB));
                }
            }
            "threads" => {
                if let Ok(n) = value.parse::<usize>() {
                    let n = n.clamp(1, 256);
                    self.options.threads = n;
                    self.threads = Some((0..n).map(|_| ThreadData::new()).collect());
                }
            }
            "move overhead" => {
                if let Ok(ms) = value.parse::<u64>() {
                    self.options.move_overhead = ms.min(5000);
                }
            }
            "multipv" => {
                if let Ok(n) = value.parse::<usize>() {
                    self.options.multipv = n.clamp(1, 256);
                }
            }
            // Tells the engine that the GUI will use `go ponder`: the moves
            // it guesses right will cost little on its own clock, which
            // `allocate_time` takes into account.
            "ponder" => self.options.ponder = value.eq_ignore_ascii_case("true"),
            "uci_chess960" => self.options.chess960 = value.eq_ignore_ascii_case("true"),
            "contempt" => {
                if let Ok(cp) = value.parse::<i32>() {
                    self.options.contempt = cp.clamp(-100, 100);
                }
            }
            "own rating" => {
                if let Ok(rating) = value.parse::<i32>() {
                    self.options.own_rating = rating.clamp(0, 4000);
                }
            }
            // "<title> <rating> <computer|human> <name>", any of the first two "none".
            "uci_opponent" => {
                self.options.opponent_rating = value.split_whitespace().nth(1).and_then(|text| text.parse().ok());
            }
            #[cfg(feature = "tune")]
            other => {
                if let Ok(v) = value.parse::<i32>() {
                    crate::params::set(other, v);
                }
            }
            #[cfg(not(feature = "tune"))]
            _ => {}
        }
    }

    fn set_position(&mut self, tokens: &[&str]) {
        let moves_at = tokens.iter().position(|&t| t == "moves").unwrap_or(tokens.len());
        let mut pos = match tokens.first() {
            Some(&"startpos") => Position::startpos(),
            Some(&"fen") => match Position::from_fen(&tokens[1..moves_at].join(" ")) {
                Ok(pos) => pos,
                Err(e) => {
                    println!("info string {e}");
                    return;
                }
            },
            _ => {
                println!("info string expected 'startpos' or 'fen'");
                return;
            }
        };
        let mut keys = Vec::new();
        for text in tokens.iter().skip(moves_at + 1) {
            let Some(mv) = parse_uci_move(&pos, text) else {
                println!("info string illegal move: {text}");
                return;
            };
            keys.push(pos.hash());
            pos.make_move(mv);
        }
        self.pos = pos;
        self.keys = keys;
    }

    fn go(&mut self, tokens: &[&str]) {
        let mut limits = Limits::new();
        let mut ponder = false;
        let (mut time, mut inc, mut movestogo, mut movetime) = (None, 0u64, None, None);
        let (our_time, our_inc) =
            if self.pos.stm() == Color::White { ("wtime", "winc") } else { ("btime", "binc") };
        let mut iter = tokens.iter();
        while let Some(&token) = iter.next() {
            let mut number = || iter.next().and_then(|v| v.parse::<i64>().ok()).map(|v| v.max(0) as u64);
            match token {
                "infinite" => limits.infinite = true,
                "ponder" => ponder = true,
                "depth" => limits.depth = number().map_or(limits.depth, |d| (d as i32).clamp(1, limits.depth)),
                "nodes" => limits.nodes = number().unwrap_or(u64::MAX),
                "movetime" => movetime = number(),
                "movestogo" => movestogo = number(),
                t if t == our_time => time = number(),
                t if t == our_inc => inc = number().unwrap_or(0),
                _ => {}
            }
        }
        let overhead = self.options.move_overhead;
        if let Some(ms) = movetime {
            let budget = Duration::from_millis(ms.saturating_sub(overhead).max(1));
            limits.soft_time = Some(budget);
            limits.hard_time = Some(budget);
        } else if let Some(ms) = time {
            // Plies played before this move, from the position's move number.
            let game_ply =
                (self.pos.fullmove() as u64 - 1) * 2 + u64::from(self.pos.stm() == Color::Black);
            let (soft, hard) = allocate_time(ms, inc, movestogo, overhead, game_ply, self.options.ponder);
            // For whoever reads a log of the game afterwards: what the
            // engine meant to spend, next to what it did spend.
            println!("info string time for this move: soft {} ms, hard {} ms", soft.as_millis(), hard.as_millis());
            limits.soft_time = Some(soft);
            limits.hard_time = Some(hard);
        }

        limits.contempt = contempt_for(&self.options);
        if limits.contempt != 0 {
            println!("info string contempt {} centipawns", limits.contempt);
        }

        let mut threads = self.threads.take().expect("search state is available");
        let shared = Arc::clone(&self.shared);
        let pos = self.pos;
        let keys = self.keys.clone();
        let report = Report { chess960: self.options.chess960, print: true, multipv: self.options.multipv };
        self.unbounded = limits.infinite;
        // Reset before spawning so a `stop` or `ponderhit` that arrives right
        // away is not lost.
        self.shared.stop.store(false, Ordering::Relaxed);
        self.shared.stop_on_ponderhit.store(false, Ordering::SeqCst);
        self.shared.pondering.store(ponder, Ordering::SeqCst);
        limits.start = Instant::now();
        let handle = std::thread::Builder::new()
            .stack_size(SEARCH_STACK_SIZE)
            .spawn(move || {
                search::run(&shared, &mut threads, &pos, &keys, &limits, report);
                threads
            })
            .expect("failed to spawn search thread");
        self.search = Some(handle);
    }
}

/// Splits the remaining clock time into a soft limit (do not start another
/// iteration) and a hard limit (abort the search). All values in milliseconds;
/// `game_ply` is the number of plies played before this move, and `ponder`
/// says whether the engine thinks on the opponent's time in this game.
///
/// The search uses about the soft limit on an average move: less when the
/// best move is settled, more when it keeps changing.
fn allocate_time(
    time_left: u64,
    inc: u64,
    movestogo: Option<u64>,
    overhead: u64,
    game_ply: u64,
    ponder: bool,
) -> (Duration, Duration) {
    let available = time_left.saturating_sub(overhead).max(1);
    let (soft, hard) = match movestogo {
        // A fixed number of moves to the next time control: equal shares.
        Some(moves) => {
            let base = available / moves.clamp(1, 50) + inc * 3 / 4;
            (base * 6 / 10, base * 3)
        }
        // The whole game on one clock. The budget is the clock plus the
        // increments of the next moves, and a move gets a fixed share of it,
        // so the reserve is spent in the middlegame rather than carried to
        // the end of the game. The first moves get a smaller share.
        None => {
            let budget = available + inc * crate::params::tm_horizon() as u64;
            let floor = crate::params::tm_ramp_floor().clamp(0, 100) as u64;
            let ramp_plies = crate::params::tm_ramp_plies().max(1) as u64;
            let ramp = floor + (100 - floor) * game_ply.min(ramp_plies) / ramp_plies;
            // An increment refills the clock, so the clock can be spent
            // freely. Without one, every move's share shrinks what is left
            // for all later moves, and a long game ends with no time at all:
            // the share is smaller then. The full share needs an increment
            // of at least a two-hundredth of the clock, which the usual
            // time controls with an increment have from the first move.
            let refill = (inc as f64 * 200.0 / available as f64).min(1.0);
            let bare = crate::params::tm_share_bare() as f64;
            let share = bare + (crate::params::tm_share() as f64 - bare) * refill;
            let soft = (budget as f64 * share * ramp as f64 / 100_000.0) as u64;
            // An increment larger than the clock cannot be spent before it arrives.
            let soft = soft.min(available / 5);
            (soft, soft * crate::params::tm_hard_mult() as u64 / 100)
        }
    };
    // With pondering, a move the opponent answers as expected has been
    // thought about on the opponent's clock, and the time is counted from
    // then: such a move costs little or nothing on our own. Without a
    // correction the clock would be used more slowly than intended, so the
    // soft limit is longer. The hard limit stays: it is there for safety.
    let soft = if ponder { soft * crate::params::tm_ponder_scale().max(0) as u64 / 100 } else { soft };
    let hard = hard.min(available * 6 / 10).max(1);
    let soft = soft.min(hard).max(1);
    (Duration::from_millis(soft), Duration::from_millis(hard))
}

#[cfg(test)]
mod tests {
    use super::allocate_time;

    fn ms(time_left: u64, inc: u64, movestogo: Option<u64>, game_ply: u64) -> (u64, u64) {
        let (soft, hard) = allocate_time(time_left, inc, movestogo, 10, game_ply, false);
        (soft.as_millis() as u64, hard.as_millis() as u64)
    }

    fn ms_pondering(time_left: u64, inc: u64, movestogo: Option<u64>, game_ply: u64) -> (u64, u64) {
        let (soft, hard) = allocate_time(time_left, inc, movestogo, 10, game_ply, true);
        (soft.as_millis() as u64, hard.as_millis() as u64)
    }

    #[test]
    fn time_limits_stay_inside_the_clock() {
        for &time_left in &[0u64, 1, 9, 10, 11, 50, 500, 3_000, 60_000, 180_000, 3_600_000] {
            for &inc in &[0u64, 80, 2_000, 30_000] {
                for &movestogo in &[None, Some(1u64), Some(40)] {
                    for &game_ply in &[0u64, 7, 24, 300] {
                        for limits in [ms, ms_pondering] {
                            let (soft, hard) = limits(time_left, inc, movestogo, game_ply);
                            assert!(soft >= 1 && soft <= hard, "{time_left} {inc} {movestogo:?} {game_ply}");
                            assert!(hard <= (time_left * 6 / 10).max(1), "{time_left} {inc} {movestogo:?} {game_ply}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn early_moves_get_less_time() {
        let (first, _) = ms(180_000, 2_000, None, 0);
        let (later, _) = ms(180_000, 2_000, None, 40);
        assert!(first < later);
        // 3 minutes + 2 seconds: between 2 and 6 seconds on the first move.
        assert!((2_000..=6_000).contains(&first), "{first}");
        // Low on the clock with an increment: no more than a fifth of the clock.
        let (soft, _) = ms(5_000, 10_000, None, 60);
        assert!(soft <= 1_000, "{soft}");
    }

    #[test]
    fn the_share_depends_on_the_increment() {
        // 3 minutes, past the opening. With an increment of a two-hundredth
        // of the clock or more: 4% of the clock and 16 increments.
        let (with_increment, _) = ms(180_000, 2_000, None, 40);
        assert_eq!(with_increment, (179_990 + 16 * 2_000) * 40 / 1000);
        let (smallest, _) = ms(180_000, 900, None, 40);
        assert_eq!(smallest, (179_990 + 16 * 900) * 40 / 1000);
        // Without increment: 3% of the clock.
        let (bare, hard) = ms(180_000, 0, None, 40);
        assert_eq!(bare, 179_990 * 30 / 1000);
        assert_eq!(hard, 4 * bare);
        // In between: in between.
        let (small, _) = ms(180_000, 300, None, 40);
        assert!(bare < small && small < (179_990 + 16 * 300) * 40 / 1000, "{small}");
    }

    #[test]
    fn pondering_lengthens_the_soft_limit_only() {
        for &(time_left, inc, movestogo) in &[(180_000u64, 2_000u64, None), (300_000, 0, None), (60_000, 0, Some(20u64))] {
            let (soft, hard) = ms(time_left, inc, movestogo, 40);
            let (soft_pondering, hard_pondering) = ms_pondering(time_left, inc, movestogo, 40);
            assert_eq!(soft_pondering, soft * 125 / 100, "{time_left} {inc} {movestogo:?}");
            assert_eq!(hard_pondering, hard, "{time_left} {inc} {movestogo:?}");
        }
        // Almost no time left: the hard limit still bounds the soft one.
        let (soft, hard) = ms_pondering(300, 0, None, 40);
        assert!(soft <= hard, "{soft} {hard}");
    }
}

pub fn perft_divide(pos: &Position, depth: u32, chess960: bool) {
    let start = Instant::now();
    let mut list = movegen::MoveList::new();
    movegen::generate_all(pos, &mut list);
    let mut total = 0;
    for &mv in list.iter() {
        let mut next = *pos;
        next.make_move(mv);
        let nodes = if depth > 1 { movegen::perft(&next, depth - 1) } else { 1 };
        println!("{}: {nodes}", mv.to_uci(chess960));
        total += nodes;
    }
    let secs = start.elapsed().as_secs_f64();
    println!("\nnodes {total}  time {secs:.3}s  nps {:.0}", total as f64 / secs.max(1e-9));
}
