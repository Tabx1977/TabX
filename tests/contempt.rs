//! What a draw is worth to the engine, by its options and by its opponent's
//! rating, tested on the real engine over its standard input and output.
//!
//! With bare kings every line is a draw, so the score the search reports is
//! exactly the value it gives a draw.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

const BARE_KINGS: &str = "8/8/8/8/8/8/8/K6k w - - 0 1";

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
        let listing = engine.ask("uci", "uciok");
        for option in ["Contempt", "Own Rating", "UCI_Opponent"] {
            assert!(
                listing.iter().any(|line| line.starts_with(&format!("option name {option} type"))),
                "the engine does not offer the option {option}"
            );
        }
        engine
    }

    fn send(&mut self, line: &str) {
        writeln!(self.stdin, "{line}").expect("engine accepts input");
        self.stdin.flush().expect("engine accepts input");
    }

    /// Sends a command and returns every line up to the one that starts with `until`.
    fn ask(&mut self, command: &str, until: &str) -> Vec<String> {
        self.send(command);
        let mut seen = Vec::new();
        loop {
            let line = self.lines.recv_timeout(Duration::from_secs(30)).expect("the engine answers");
            let last = line.starts_with(until);
            seen.push(line);
            if last {
                return seen;
            }
        }
    }

    /// The score of a short search of the bare kings, for the side to move, and
    /// the contempt the engine says it is using (0 if it says nothing).
    fn draw_value(&mut self, settings: &[&str]) -> (i32, i32) {
        for setting in settings {
            self.send(setting);
        }
        self.send(&format!("position fen {BARE_KINGS}"));
        let lines = self.ask("go depth 4", "bestmove");
        let number_after = |line: &String, word: &str| -> Option<i32> {
            let mut words = line.split_whitespace().skip_while(|&w| w != word);
            words.nth(1).and_then(|text| text.parse().ok())
        };
        let score = lines
            .iter()
            .rev()
            .filter(|line| line.starts_with("info depth"))
            .find_map(|line| number_after(line, "cp"))
            .expect("the search reports a score");
        let announced = lines
            .iter()
            .filter(|line| line.starts_with("info string contempt"))
            .find_map(|line| number_after(line, "contempt"))
            .unwrap_or(0);
        (score, announced)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.child.wait();
    }
}

#[test]
fn a_draw_is_worth_nothing_unless_told_otherwise() {
    let mut engine = Engine::start();
    assert_eq!(engine.draw_value(&[]), (0, 0));
}

#[test]
fn contempt_makes_a_draw_a_small_loss_for_the_engine() {
    let mut engine = Engine::start();
    assert_eq!(engine.draw_value(&["setoption name Contempt value 30"]), (-30, 30));
    assert_eq!(engine.draw_value(&["setoption name Contempt value -20"]), (20, -20));
    assert_eq!(engine.draw_value(&["setoption name Contempt value 0"]), (0, 0));
}

#[test]
fn the_opponents_rating_sets_the_contempt() {
    let mut engine = Engine::start();
    // Without its own rating the engine cannot compare.
    assert_eq!(engine.draw_value(&["setoption name UCI_Opponent value BOT 2600 computer weakling"]), (0, 0));
    // 200 points ahead: 15 centipawns for every hundred.
    assert_eq!(engine.draw_value(&["setoption name Own Rating value 2800"]), (-30, 30));
    // Far ahead: no more than 40.
    assert_eq!(engine.draw_value(&["setoption name UCI_Opponent value none 1500 human somebody"]), (-40, 40));
    // Behind: a draw is welcome, by 15 at most.
    assert_eq!(engine.draw_value(&["setoption name UCI_Opponent value GM 3000 computer strong one"]), (15, -15));
    assert_eq!(engine.draw_value(&["setoption name UCI_Opponent value BOT 2850 computer close"]), (7, -7));
    // An opponent without a rating: nothing to go by.
    assert_eq!(engine.draw_value(&["setoption name UCI_Opponent value none none computer unrated"]), (0, 0));
    // A fixed contempt adds to it.
    assert_eq!(
        engine.draw_value(&["setoption name UCI_Opponent value BOT 2700 computer x", "setoption name Contempt value 10"]),
        (-25, 25)
    );
}
