//! Thinking on the opponent's time (UCI `go ponder`, `ponderhit`), tested on
//! the real engine over its standard input and output.
//!
//! The time limits here are generous: they tell "answers at once" from
//! "never answers", also on a busy machine, and nothing finer.

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Position};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

/// A middlegame position with plenty of reasonable moves.
const MIDDLEGAME: &str = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9";
/// Clock for a search that should still be running when the test moves on.
const LONG_CLOCK: &str = "wtime 600000 btime 600000 winc 1000 binc 1000";
/// "At once", allowing for a busy machine.
const PROMPT_MS: u64 = 2_000;

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
}

impl Engine {
    fn start() -> Engine {
        let mut child = Command::new(env!("CARGO_BIN_EXE_tabx"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("the engine starts");
        let stdin = child.stdin.take().expect("engine input");
        let stdout = child.stdout.take().expect("engine output");
        let (sender, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = Engine { child, stdin, lines };
        engine.send("uci");
        assert!(engine.wait_for("uciok", 10_000).is_some(), "no answer to uci");
        engine.send("isready");
        assert!(engine.wait_for("readyok", 10_000).is_some(), "no answer to isready");
        engine
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("engine accepts input");
        self.stdin.flush().expect("engine accepts input");
    }

    /// The next line that starts with `prefix`, if one arrives within `ms` milliseconds.
    fn wait_for(&mut self, prefix: &str, ms: u64) -> Option<String> {
        let deadline = Instant::now() + Duration::from_millis(ms);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return Some(line),
                Ok(_) => {}
                Err(_) => return None,
            }
        }
    }

    /// Starts pondering on the middlegame position and checks that the engine keeps quiet.
    fn ponder(&mut self, limits: &str, quiet_ms: u64) {
        self.send(&format!("position fen {MIDDLEGAME}"));
        self.send(&format!("go ponder {limits}"));
        assert!(self.wait_for("bestmove", quiet_ms).is_none(), "the engine moved while pondering");
    }

    fn exits_within(&mut self, ms: u64) -> bool {
        let deadline = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Checks a `bestmove X [ponder Y]` line against the position: X must be
/// legal, and Y legal after X. Returns whether a ponder move was given.
fn check_bestmove(fen: &str, mode: CastlingMode, line: &str) -> bool {
    let mut pos: Chess = fen.parse::<Fen>().expect("valid FEN").into_position(mode).expect("legal position");
    let words: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(words[0], "bestmove", "{line}");
    let best = words[1].parse::<UciMove>().expect("a move").to_move(&pos).unwrap_or_else(|_| panic!("illegal: {line}"));
    pos.play_unchecked(best);
    if words.len() == 2 {
        return false;
    }
    assert_eq!((words.len(), words[2]), (4, "ponder"), "{line}");
    words[3].parse::<UciMove>().expect("a move").to_move(&pos).unwrap_or_else(|_| panic!("illegal reply: {line}"));
    true
}

#[test]
fn a_search_names_the_reply_it_expects() {
    let mut engine = Engine::start();
    engine.send(&format!("position fen {MIDDLEGAME}"));
    engine.send("go movetime 300");
    let line = engine.wait_for("bestmove", 10_000).expect("an answer");
    assert!(check_bestmove(MIDDLEGAME, CastlingMode::Standard, &line), "no ponder move: {line}");
}

#[test]
fn the_expected_reply_is_legal_in_chess960() {
    let fen = "bnrqkbnr/pppppppp/8/8/8/8/PPPPPPPP/BNRQKBNR w HChc - 0 1";
    let mut engine = Engine::start();
    engine.send("setoption name UCI_Chess960 value true");
    engine.send(&format!("position fen {fen}"));
    engine.send("go movetime 300");
    let line = engine.wait_for("bestmove", 10_000).expect("an answer");
    check_bestmove(fen, CastlingMode::Chess960, &line);
}

#[test]
fn pondering_goes_on_until_ponderhit() {
    let mut engine = Engine::start();
    // What a GUI that ponders sends first; the engine then allows itself
    // more time for a move.
    engine.send("setoption name Ponder value true");
    // Far more time than the test waits: after ponderhit the search has to
    // carry on by itself and stop when its time for the move is up.
    engine.ponder("wtime 4000 btime 4000 winc 40 binc 40", 100);
    engine.send("ponderhit");
    let line = engine.wait_for("bestmove", 10_000).expect("an answer after ponderhit");
    check_bestmove(MIDDLEGAME, CastlingMode::Standard, &line);
}

#[test]
fn ponderhit_is_answered_at_once_when_the_time_for_the_move_is_up() {
    let mut engine = Engine::start();
    // One second on the clock: the time for the move is long gone when the
    // opponent "moves" after one and a half.
    engine.ponder("wtime 1000 btime 1000 winc 10 binc 10", 1_500);
    engine.send("ponderhit");
    let line = engine.wait_for("bestmove", PROMPT_MS).expect("an answer at once");
    check_bestmove(MIDDLEGAME, CastlingMode::Standard, &line);
}

/// The soft and hard time limits the engine reports for a move, in milliseconds.
fn time_limits(ponder_option: bool, go: &str) -> (u64, u64) {
    let mut engine = Engine::start();
    engine.send(&format!("setoption name Ponder value {ponder_option}"));
    engine.send(&format!("position fen {MIDDLEGAME}"));
    engine.send(go);
    let line = engine.wait_for("info string time for this move", 10_000).expect("the time limits");
    engine.send("stop");
    let numbers: Vec<u64> = line.split_whitespace().filter_map(|word| word.parse().ok()).collect();
    assert_eq!(numbers.len(), 2, "{line}");
    (numbers[0], numbers[1])
}

#[test]
fn the_ponder_option_lengthens_the_time_for_a_move() {
    // The option says that the GUI will let the engine ponder; it counts for
    // every search of the game, whether that search starts as pondering or not.
    for go in ["go wtime 60000 btime 60000 winc 600 binc 600", "go ponder wtime 180000 btime 180000 winc 2000 binc 2000"] {
        let (soft, hard) = time_limits(false, go);
        let (soft_pondering, hard_pondering) = time_limits(true, go);
        assert_eq!(soft_pondering, soft * 125 / 100, "{go}");
        assert_eq!(hard_pondering, hard, "{go}");
    }
}

#[test]
fn stop_ends_pondering() {
    let mut engine = Engine::start();
    engine.ponder(LONG_CLOCK, 300);
    engine.send("stop");
    let line = engine.wait_for("bestmove", PROMPT_MS).expect("an answer at once");
    check_bestmove(MIDDLEGAME, CastlingMode::Standard, &line);
}

#[test]
fn a_finished_search_waits_for_ponderhit() {
    let mut engine = Engine::start();
    engine.ponder("depth 4", 500);
    engine.send("ponderhit");
    let line = engine.wait_for("bestmove", PROMPT_MS).expect("an answer at once");
    check_bestmove(MIDDLEGAME, CastlingMode::Standard, &line);
}

#[test]
fn a_new_position_ends_pondering() {
    // Not allowed by the protocol (the GUI has to send stop first), but the
    // engine must not hang if it happens.
    let after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    let mut engine = Engine::start();
    engine.ponder(LONG_CLOCK, 300);
    engine.send("position startpos moves e2e4");
    engine.send("go movetime 200");
    let first = engine.wait_for("bestmove", PROMPT_MS).expect("the pondering search ends");
    check_bestmove(MIDDLEGAME, CastlingMode::Standard, &first);
    let second = engine.wait_for("bestmove", 10_000).expect("the new search answers");
    check_bestmove(after_e4, CastlingMode::Standard, &second);
}

#[test]
fn quit_ends_pondering() {
    let mut engine = Engine::start();
    engine.ponder(LONG_CLOCK, 300);
    engine.send("quit");
    assert!(engine.exits_within(PROMPT_MS), "the engine is still running");
}
