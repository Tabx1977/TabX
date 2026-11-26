mod attacks;
mod bench;
mod datagen;
mod evalimport;
mod history;
mod movegen;
mod movepick;
mod nnue;
mod params;
mod position;
mod search;
mod tt;
#[cfg(test)]
mod tests;
mod types;
mod uci;
mod zobrist;

use position::Position;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => uci::run(),
        Some("bench") => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH);
            bench::run(depth);
        }
        Some("perft") => {
            let depth: u32 = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(6);
            let fen = if args.len() > 2 { args[2..].join(" ") } else { position::START_FEN.into() };
            match Position::from_fen(&fen) {
                Ok(pos) => uci::perft_divide(&pos, depth, false),
                Err(e) => eprintln!("{e}"),
            }
        }
        Some("datagen") => {
            let mut cfg = datagen::Config {
                out: "data/selfplay.bin".into(),
                threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
                positions: 10_000_000,
                nodes: 5000,
                dfrc: false,
                books: Vec::new(),
                random_plies: (8, 9),
                hash_mb: 1,
                seed: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(1, |d| d.as_nanos() as u64),
            };
            let mut random_plies = None;
            let mut iter = args[1..].iter();
            while let Some(arg) = iter.next() {
                let mut value = || iter.next().cloned().unwrap_or_default();
                match arg.as_str() {
                    "--book" => {
                        let path = value();
                        match std::fs::read_to_string(&path) {
                            Ok(text) => {
                                let lines: Vec<String> = text
                                    .lines()
                                    .map(str::trim)
                                    .filter(|line| !line.is_empty())
                                    .map(String::from)
                                    .collect();
                                if lines.is_empty() {
                                    eprintln!("no openings in {path}");
                                    return;
                                }
                                cfg.books.push(lines);
                            }
                            Err(e) => {
                                eprintln!("cannot read {path}: {e}");
                                return;
                            }
                        }
                    }
                    "--random-plies" => {
                        // "N" or "MIN-MAX".
                        let text = value();
                        let (low, high) = text.split_once('-').unwrap_or((&text, &text));
                        match (low.parse::<usize>(), high.parse::<usize>()) {
                            (Ok(low), Ok(high)) if low <= high => random_plies = Some((low, high)),
                            _ => {
                                eprintln!("--random-plies takes a number or a range such as 0-2");
                                return;
                            }
                        }
                    }
                    "--out" => cfg.out = value(),
                    "--threads" => cfg.threads = value().parse().unwrap_or(cfg.threads).max(1),
                    "--positions" => cfg.positions = value().parse().unwrap_or(cfg.positions),
                    "--nodes" => cfg.nodes = value().parse().unwrap_or(cfg.nodes).max(1),
                    "--seed" => cfg.seed = value().parse().unwrap_or(cfg.seed),
                    "--hash" => cfg.hash_mb = value().parse().unwrap_or(cfg.hash_mb).max(1),
                    "--dfrc" => cfg.dfrc = true,
                    other => {
                        eprintln!("unknown datagen option '{other}'");
                        return;
                    }
                }
            }
            // After a book position a few random moves are enough to make the games differ.
            cfg.random_plies = random_plies.unwrap_or(if cfg.books.is_empty() { (8, 9) } else { (0, 2) });
            datagen::run(cfg);
        }
        Some("import-evals") => {
            let mut cfg = evalimport::Config {
                out: "data/lichess.bin".into(),
                threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
                min_depth: 16,
                scale: 128,
                limit: 0,
                lines: 0,
                plies: 2,
                known: None,
                shards: 0,
                also: None,
            };
            let mut iter = args[1..].iter();
            while let Some(arg) = iter.next() {
                let mut value = || iter.next().cloned().unwrap_or_default();
                match arg.as_str() {
                    "--out" => cfg.out = value(),
                    "--threads" => cfg.threads = value().parse().unwrap_or(cfg.threads).max(1),
                    "--min-depth" => cfg.min_depth = value().parse().unwrap_or(cfg.min_depth),
                    "--scale" => cfg.scale = value().parse().unwrap_or(cfg.scale),
                    "--limit" => cfg.limit = value().parse().unwrap_or(0),
                    "--lines" => cfg.lines = value().parse().unwrap_or(0),
                    "--plies" => cfg.plies = value().parse().unwrap_or(cfg.plies).max(1),
                    "--known" => cfg.known = Some(value()),
                    "--shards" => cfg.shards = value().parse().unwrap_or(0),
                    "--also" => cfg.also = Some(value()),
                    other => {
                        eprintln!("unknown import-evals option '{other}'");
                        return;
                    }
                }
            }
            if cfg.lines > 0 {
                evalimport::run_lines(cfg);
            } else {
                evalimport::run(cfg);
            }
        }
        Some("index-evals") => {
            let mut cfg = evalimport::IndexConfig {
                out: "tools/lichess-evals/evals.idx".into(),
                threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
                limit: 0,
            };
            let mut iter = args[1..].iter();
            while let Some(arg) = iter.next() {
                let mut value = || iter.next().cloned().unwrap_or_default();
                match arg.as_str() {
                    "--out" => cfg.out = value(),
                    "--threads" => cfg.threads = value().parse().unwrap_or(cfg.threads).max(1),
                    "--limit" => cfg.limit = value().parse().unwrap_or(0),
                    other => {
                        eprintln!("unknown index-evals option '{other}'");
                        return;
                    }
                }
            }
            evalimport::index(cfg);
        }
        Some(other) => {
            eprintln!("unknown command '{other}'");
            eprintln!("usage: tabx                        UCI mode");
            eprintln!("       tabx bench [depth]");
            eprintln!("       tabx perft <depth> [fen]");
            eprintln!("       tabx datagen [--out FILE] [--threads N] [--positions N] [--nodes N] [--hash MB] [--seed N]");
            eprintln!("                      [--book FILE]... [--random-plies N|MIN-MAX] [--dfrc]");
            eprintln!("       tabx import-evals [--out FILE] [--threads N] [--min-depth N] [--scale PERCENT] [--limit LINES]");
            eprintln!("                      [--lines N [--plies M] [--known FILE] [--shards N] [--also FILE]]");
            eprintln!("       tabx index-evals [--out FILE] [--threads N] [--limit LINES]");
            eprintln!("                      (both read the Lichess evaluation database, uncompressed, from standard input)");
        }
    }
}
