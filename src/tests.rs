//! Correctness tests for the board representation and move generator.
//!
//! Besides the classic perft positions, the generator is checked against an
//! independent implementation (the `shakmaty` crate, a test-only dependency)
//! on thousands of random positions, including Chess960 castling setups.

use crate::attacks::*;
use crate::movegen::*;
use crate::position::*;
use crate::types::*;
use shakmaty::{CastlingMode, Chess, fen::Fen};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn legal_moves(pos: &Position) -> MoveList {
    let mut list = MoveList::new();
    generate_all(pos, &mut list);
    list
}

/// Plays random moves from `pos`, calling `visit` on every position reached.
fn random_playout(mut pos: Position, rng: &mut Rng, plies: usize, visit: &mut dyn FnMut(&Position)) {
    for _ in 0..plies {
        visit(&pos);
        let list = legal_moves(&pos);
        if list.is_empty() || pos.halfmove() >= 100 {
            break;
        }
        pos.make_move(list[rng.below(list.len())]);
    }
}

fn reference_perft(fen: &str, depth: u32) -> Option<u64> {
    let parsed: Fen = fen.parse().ok()?;
    // Random test setups can contain checks no real game could produce; the
    // reference library flags those as errors, which we deliberately ignore.
    let pos: Chess = parsed
        .into_position(CastlingMode::Chess960)
        .or_else(|e| e.ignore_impossible_check())
        .or_else(|e| e.ignore_too_much_material())
        .ok()?;
    Some(shakmaty::perft(&pos, depth))
}

#[test]
fn perft_known_positions() {
    let cases: [(&str, &[u64]); 6] = [
        (START_FEN, &[20, 400, 8902, 197_281, 4_865_609]),
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2039, 97_862, 4_085_603],
        ),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[14, 191, 2812, 43_238, 674_624, 11_030_083]),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9467, 422_333, 15_833_292],
        ),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[44, 1486, 62_379, 2_103_487]),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2079, 89_890, 3_894_594],
        ),
    ];
    for (fen, counts) in cases {
        let pos = Position::from_fen(fen).unwrap();
        for (i, &expected) in counts.iter().enumerate() {
            assert_eq!(perft(&pos, i as u32 + 1), expected, "perft {} of {fen}", i + 1);
        }
        // The published counts above are also confirmed by the reference library.
        assert_eq!(reference_perft(fen, 3), Some(counts[2]), "reference perft 3 of {fen}");
    }
}

#[test]
fn fen_round_trip() {
    for fen in [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3",
    ] {
        assert_eq!(Position::from_fen(fen).unwrap().fen(false), fen);
    }
    // An en passant square that cannot be captured on is dropped.
    let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1").unwrap();
    assert_eq!(pos.ep(), Square::NONE);
    assert!(Position::from_fen("8/8/8/8/8/8/8/K7 w - - 0 1").is_err());
    assert!(Position::from_fen("k7/8/8/8/8/8/8/KR6 w - - 0 1").is_ok());
    assert!(Position::from_fen("k7/8/8/8/8/8/8/KR6 b - - 0 1").is_ok());
    assert!(Position::from_fen("kR6/8/8/8/8/8/8/K7 w - - 0 1").is_err());
}

#[test]
fn chess960_start_positions() {
    assert_eq!(Position::frc(518, 518).fen(false), START_FEN);
    assert_eq!(Position::frc(518, 518).hash(), Position::startpos().hash());
    for n in 0..960 {
        let rank = frc_backrank(n);
        let files = |pt| (0..8).filter(|&f| rank[f] == pt).collect::<Vec<_>>();
        let (bishops, rooks, king) = (files(PieceType::Bishop), files(PieceType::Rook), files(PieceType::King));
        assert_eq!(bishops.len(), 2);
        assert_ne!(bishops[0] % 2, bishops[1] % 2, "bishops on opposite colours");
        assert!(rooks[0] < king[0] && king[0] < rooks[1], "king between the rooks");
        let pos = Position::frc(n, n);
        assert_eq!(pos.castling(), 0b1111);
        assert_eq!(perft(&pos, 2), reference_perft(&pos.fen(true), 2).unwrap());
    }
}

#[test]
fn slider_attacks_match_ray_walk() {
    fn slow(sq: Square, occ: Bitboard, dirs: [(i32, i32); 4]) -> Bitboard {
        let mut attacks = Bitboard::EMPTY;
        for (dr, df) in dirs {
            let (mut r, mut f) = (sq.rank() as i32 + dr, sq.file() as i32 + df);
            while (0..8).contains(&r) && (0..8).contains(&f) {
                let s = Square::from_fr(f as u8, r as u8);
                attacks |= s.bb();
                if occ.contains(s) {
                    break;
                }
                r += dr;
                f += df;
            }
        }
        attacks
    }
    let mut rng = Rng(0xA5A5_1234_DEAD_BEEF);
    for i in 0..64 {
        let sq = Square(i);
        for round in 0..2000 {
            // Mix dense and sparse occupancies.
            let occ = match round % 3 {
                0 => Bitboard(rng.next()),
                1 => Bitboard(rng.next() & rng.next()),
                _ => Bitboard(rng.next() & rng.next() & rng.next()),
            };
            assert_eq!(rook_attacks(sq, occ), slow(sq, occ, [(1, 0), (-1, 0), (0, 1), (0, -1)]));
            assert_eq!(bishop_attacks(sq, occ), slow(sq, occ, [(1, 1), (1, -1), (-1, 1), (-1, -1)]));
        }
    }
}

/// Checks everything that must hold in any reachable position.
fn check_invariants(pos: &Position) {
    let all = legal_moves(pos);

    // Incremental hashes match a from-scratch computation.
    let incremental = (
        pos.hash(),
        pos.pawn_hash(),
        [pos.non_pawn_hash(Color::White), pos.non_pawn_hash(Color::Black)],
    );
    assert_eq!(incremental, pos.compute_hash(), "hash of {}", pos.fen(true));

    // FEN round trip preserves the position.
    let reparsed = Position::from_fen(&pos.fen(true)).unwrap();
    assert_eq!(reparsed.hash(), pos.hash(), "FEN round trip of {}", pos.fen(true));
    assert_eq!(reparsed.fen(true), pos.fen(true));

    // The staged generators partition the full move list.
    let (mut noisy, mut quiet) = (MoveList::new(), MoveList::new());
    generate_noisy(pos, &mut noisy);
    generate_quiet(pos, &mut quiet);
    assert_eq!(noisy.len() + quiet.len(), all.len());
    assert!(noisy.iter().all(|m| m.is_noisy() && all.contains(m)));
    assert!(quiet.iter().all(|m| m.is_quiet() && all.contains(m)));

    // `is_legal` agrees with the generator for every possible 16-bit move.
    for raw in 0..=u16::MAX {
        let mv = Move(raw);
        assert_eq!(pos.is_legal(mv), all.contains(&mv), "is_legal({mv:?}) in {}", pos.fen(true));
    }

    // `gives_check` predicts what making the move shows.
    for &mv in all.iter() {
        let mut next = *pos;
        next.make_move(mv);
        assert_eq!(pos.gives_check(mv), next.in_check(), "gives_check({}) in {}", mv.to_uci(true), pos.fen(true));
    }

    // UCI move text parses back to the same move in both notations.
    for &mv in all.iter() {
        assert_eq!(parse_uci_move(pos, &mv.to_uci(true)), Some(mv));
        if !mv.is_castle() {
            assert_eq!(parse_uci_move(pos, &mv.to_uci(false)), Some(mv));
        }
    }
}

#[test]
fn invariants_on_random_games() {
    let mut rng = Rng(0x1357_9BDF_0246_8ACE);
    for game in 0..60 {
        let start = if game % 3 == 0 {
            Position::startpos()
        } else {
            Position::frc(rng.below(960), rng.below(960))
        };
        random_playout(start, &mut rng, 80, &mut check_invariants);
    }
}

#[test]
fn movegen_matches_reference_on_random_games() {
    let mut rng = Rng(0x0F1E_2D3C_4B5A_6978);
    let mut positions = 0;
    for game in 0..400 {
        let start = if game % 4 == 0 {
            Position::startpos()
        } else {
            Position::frc(rng.below(960), rng.below(960))
        };
        random_playout(start, &mut rng, 120, &mut |pos| {
            let fen = pos.fen(true);
            let expected = reference_perft(&fen, 2).unwrap_or_else(|| panic!("reference rejects {fen}"));
            assert_eq!(perft(pos, 2), expected, "perft 2 of {fen}");
            positions += 1;
        });
    }
    assert!(positions > 10_000);
}

/// Random positions built to stress castling: kings and rooks on their back
/// ranks (Chess960 style) with assorted pieces scattered around them.
#[test]
fn castling_matches_reference_on_random_setups() {
    let mut rng = Rng(0xC0FF_EE00_BAAD_F00D);
    let mut tested = 0;
    let mut with_castling = 0;
    while tested < 30_000 {
        let mut board = [['.'; 8]; 8]; // board[rank][file]
        let mut castling = String::new();
        for (rank, upper) in [(0usize, true), (7usize, false)] {
            // King somewhere on b..g with a rook on each side.
            let king = 1 + rng.below(6);
            let left = rng.below(king);
            let right = king + 1 + rng.below(7 - king);
            let case = |c: char| if upper { c.to_ascii_uppercase() } else { c };
            board[rank][king] = case('k');
            for (file, present) in [(right, rng.below(8) > 0), (left, rng.below(8) > 0)] {
                if present {
                    board[rank][file] = case('r');
                    castling.push(case((b'a' + file as u8) as char));
                }
            }
        }
        for _ in 0..rng.below(10) {
            let (rank, file) = (rng.below(8), rng.below(8));
            let piece = ['q', 'r', 'b', 'n', 'p', 'Q', 'R', 'B', 'N', 'P'][rng.below(10)];
            let is_pawn = piece.eq_ignore_ascii_case(&'p');
            if board[rank][file] == '.' && !(is_pawn && (rank == 0 || rank == 7)) {
                board[rank][file] = piece;
            }
        }
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                if board[rank][file] == '.' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    fen.push(board[rank][file]);
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }
        let stm = if rng.below(2) == 0 { 'w' } else { 'b' };
        let castling = if castling.is_empty() { "-".to_string() } else { castling };
        let fen = format!("{fen} {stm} {castling} - 0 1");

        // Skip setups where the side not to move is in check; both
        // implementations must agree on which positions are valid.
        let ours = Position::from_fen(&fen);
        let reference = reference_perft(&fen, 2);
        assert_eq!(ours.is_ok(), reference.is_some(), "validity of {fen}");
        let Ok(pos) = ours else { continue };
        assert_eq!(perft(&pos, 2), reference.unwrap(), "perft 2 of {fen}");
        assert_eq!(perft(&pos, 1), reference_perft(&fen, 1).unwrap(), "perft 1 of {fen}");
        for &mv in legal_moves(&pos).iter() {
            let mut next = pos;
            next.make_move(mv);
            assert_eq!(pos.gives_check(mv), next.in_check(), "gives_check({}) in {fen}", mv.to_uci(true));
        }
        if legal_moves(&pos).iter().any(|m| m.is_castle()) {
            with_castling += 1;
        }
        tested += 1;
    }
    assert!(with_castling > 3_000, "only {with_castling} positions allowed castling");
}

#[test]
fn bench_positions_are_valid() {
    for fen in crate::bench::POSITIONS {
        let pos = Position::from_fen(fen).unwrap_or_else(|e| panic!("{e}"));
        assert!(reference_perft(&pos.fen(true), 1).is_some(), "reference rejects {fen}");
        assert_eq!(perft(&pos, 2), reference_perft(&pos.fen(true), 2).unwrap(), "perft 2 of {fen}");
    }
}

/// Exact static exchange value computed with the classic "swap list"
/// algorithm, as an independent reference for `Position::see_ge`.
fn see_value(pos: &Position, mv: Move) -> i32 {
    use PieceType::*;
    let (from, to) = (mv.from(), mv.to());
    let value = |pt: PieceType| SEE_VALUES[pt.idx()];
    let mut gain = [0i32; 40];
    let mut depth = 0;
    let mut occ = (pos.occupied() ^ from.bb()) | to.bb();
    let mut on_square = pos.piece_on(from).piece_type();
    if mv.is_en_passant() {
        gain[0] = value(Pawn);
        occ ^= Square(to.0 ^ 8).bb();
    } else if mv.is_capture() {
        gain[0] = value(pos.piece_on(to).piece_type());
    }
    if mv.is_promo() {
        on_square = mv.promo_type();
        gain[0] += value(on_square) - value(Pawn);
    }
    let diagonal = pos.by_type(Bishop) | pos.by_type(Queen);
    let straight = pos.by_type(Rook) | pos.by_type(Queen);
    let mut attackers = pos.attackers_to(to, occ) & occ;
    let mut side = !pos.stm();
    loop {
        let ours = attackers & pos.by_color(side);
        let Some(pt) = PieceType::ALL.into_iter().find(|&pt| (ours & pos.by_type(pt)).any()) else { break };
        if pt == King && (attackers & pos.by_color(!side)).any() {
            break;
        }
        depth += 1;
        gain[depth] = value(on_square) - gain[depth - 1];
        on_square = pt;
        occ ^= (ours & pos.by_type(pt)).lsb().bb();
        attackers |= (bishop_attacks(to, occ) & diagonal) | (rook_attacks(to, occ) & straight);
        attackers &= occ;
        side = !side;
    }
    while depth > 0 {
        gain[depth - 1] = -(-gain[depth - 1]).max(gain[depth]);
        depth -= 1;
    }
    gain[0]
}

#[test]
fn static_exchange_evaluation() {
    // Hand-checked cases: (position, move, exact exchange value).
    let cases = [
        ("1k1r4/1pp4p/p7/4p3/8/P5P1/1PP4P/2K1R3 w - - 0 1", "e1e5", 100),
        ("1k1r3q/1ppn3p/p4b2/4p3/8/P2N2P1/1PP1R1BP/2K1Q3 w - - 0 1", "d3e5", -200),
        ("k7/8/8/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 100),
        ("k7/8/2p5/3p4/4P3/8/8/K7 w - - 0 1", "e4d5", 0),
        ("k7/8/8/3q4/8/8/3R4/K7 w - - 0 1", "d2d5", 900),
        ("k2r4/8/8/3p4/8/8/3Q4/K7 w - - 0 1", "d2d5", -800),
        ("k7/8/8/3p4/8/8/8/K2R4 w - - 0 1", "d1d4", 0),
        ("k7/8/2p5/8/3P4/8/8/K7 b - - 0 1", "c6c5", -100),
    ];
    for (fen, text, expected) in cases {
        let pos = Position::from_fen(fen).unwrap();
        let mv = parse_uci_move(&pos, text).unwrap();
        assert_eq!(see_value(&pos, mv), expected, "reference SEE of {text} in {fen}");
        assert!(pos.see_ge(mv, expected), "see_ge({text}, {expected}) in {fen}");
        assert!(!pos.see_ge(mv, expected + 1), "see_ge({text}, {}) in {fen}", expected + 1);
    }

    // The threshold test must agree with the exact value for every move of
    // many random positions and every threshold.
    let mut rng = Rng(0x5EE5_EE5E_E5EE_5EE5);
    let mut checked = 0u64;
    for _ in 0..300 {
        let start = Position::frc(rng.below(960), rng.below(960));
        random_playout(start, &mut rng, 100, &mut |pos| {
            for &mv in legal_moves(pos).iter() {
                let exact = see_value(pos, mv);
                for threshold in [-900, -500, -300, -200, -100, -1, 0, 1, 100, 200, 300, 500, 900] {
                    assert_eq!(
                        pos.see_ge(mv, threshold),
                        exact >= threshold,
                        "see_ge({}, {threshold}) with exact value {exact} in {}",
                        mv.to_uci(true),
                        pos.fen(true)
                    );
                    checked += 1;
                }
            }
        });
    }
    assert!(checked > 1_000_000);
}

#[test]
fn accumulator_updates_match_refresh() {
    use crate::nnue::Accumulator;
    let mut rng = Rng(0xACC0_ACC0_1357_2468);
    let mut checked = 0;
    for _ in 0..200 {
        let mut pos = Position::frc(rng.below(960), rng.below(960));
        let mut acc = Accumulator::new(&pos);
        for _ in 0..150 {
            let list = legal_moves(&pos);
            if list.is_empty() || pos.halfmove() >= 100 {
                break;
            }
            let mv = list[rng.below(list.len())];
            let mut child = acc;
            child.update(&acc, &pos, mv);
            pos.make_move(mv);
            assert!(child == Accumulator::new(&pos), "after {} to reach {}", mv.to_uci(true), pos.fen(true));
            assert_eq!(child.evaluate(&pos), crate::nnue::evaluate(&pos));
            acc = child;
            checked += 1;
        }
    }
    assert!(checked > 10_000);
}

#[test]
fn move_picker_yields_every_legal_move_once() {
    use crate::history::History;
    use crate::movepick::MovePicker;
    let history = History::new();
    let mut rng = Rng(0x9199_0000_FFFF_1234);
    // Separate generator for use inside the playout callback.
    let mut pick = Rng(0x0BAD_5EED_600D_F00D);
    let mut positions = 0;
    for _ in 0..150 {
        let start = Position::frc(rng.below(960), rng.below(960));
        random_playout(start, &mut rng, 120, &mut |pos| {
            let legal = legal_moves(pos);
            // Hash move and killers may be legal moves, stale moves, or garbage.
            let random_move = |rng: &mut Rng| match rng.below(4) {
                0 => Move::NONE,
                1 => Move(rng.next() as u16),
                _ if legal.is_empty() => Move::NONE,
                _ => legal[rng.below(legal.len())],
            };
            let tt_move = random_move(&mut pick);
            let killers = [random_move(&mut pick), random_move(&mut pick)];

            let mut picker = MovePicker::new(tt_move, killers);
            let mut seen: Vec<Move> = Vec::new();
            while let Some(mv) = picker.next(pos, &history, [crate::history::NO_CONT; 2]) {
                assert!(!seen.contains(&mv), "{} yielded twice in {}", mv.to_uci(true), pos.fen(true));
                assert!(legal.contains(&mv), "illegal {} in {}", mv.to_uci(true), pos.fen(true));
                seen.push(mv);
            }
            assert_eq!(seen.len(), legal.len(), "missing moves in {}", pos.fen(true));
            if legal.contains(&tt_move) {
                assert_eq!(seen[0], tt_move);
            }

            // Quiescence mode: exactly the noisy moves that do not lose material.
            let mut picker = MovePicker::new_noisy(tt_move);
            let mut noisy: Vec<Move> = Vec::new();
            while let Some(mv) = picker.next(pos, &history, [crate::history::NO_CONT; 2]) {
                assert!(!noisy.contains(&mv) && legal.contains(&mv) && mv.is_noisy());
                noisy.push(mv);
            }
            for &mv in legal.iter() {
                if mv.is_noisy() && pos.see_ge(mv, 0) {
                    assert!(noisy.contains(&mv), "{} missing in {}", mv.to_uci(true), pos.fen(true));
                }
            }
            // Anything else it yields can only be the hash move.
            assert!(noisy.iter().all(|&mv| pos.see_ge(mv, 0) || mv == tt_move));

            // Skipping quiets part-way still yields every noisy move exactly once.
            let mut picker = MovePicker::new(tt_move, killers);
            let mut partial: Vec<Move> = Vec::new();
            let cutoff = pick.below(legal.len() + 1);
            while let Some(mv) = picker.next(pos, &history, [crate::history::NO_CONT; 2]) {
                assert!(!partial.contains(&mv));
                partial.push(mv);
                if partial.len() == cutoff {
                    picker.skip_quiets();
                }
            }
            for &mv in legal.iter() {
                if mv.is_noisy() {
                    assert!(partial.contains(&mv), "{} lost after skip_quiets in {}", mv.to_uci(true), pos.fen(true));
                }
            }
            positions += 1;
        });
    }
    assert!(positions > 5_000);
}

/// Rough timings of the hot paths. Run with:
/// `cargo test --release hot_path_timings -- --ignored --nocapture`
#[test]
#[ignore]
fn hot_path_timings() {
    use crate::nnue::{Accumulator, Delta};
    use std::hint::black_box;
    use std::time::Instant;

    let pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    let moves = legal_moves(&pos);
    let acc = Accumulator::new(&pos);
    let n = 2_000_000u32;

    let start = Instant::now();
    let mut sum = 0i64;
    for _ in 0..n {
        sum += black_box(&acc).evaluate(black_box(&pos)) as i64;
    }
    println!("evaluate:        {:6.1} ns", start.elapsed().as_nanos() as f64 / n as f64);

    let start = Instant::now();
    let mut child = acc;
    for i in 0..n {
        let mv = moves[i as usize % moves.len()];
        let delta = Delta::from_move(black_box(&pos), mv);
        child.apply(black_box(&acc), &delta, &pos);
    }
    println!("delta + apply:   {:6.1} ns", start.elapsed().as_nanos() as f64 / n as f64);

    let start = Instant::now();
    for _ in 0..n / 20 {
        let mut fresh = acc;
        fresh.refresh(black_box(&pos));
        sum += black_box(&fresh).evaluate(&pos) as i64;
    }
    println!("refresh + eval:  {:6.1} ns", start.elapsed().as_nanos() as f64 / (n / 20) as f64);

    let start = Instant::now();
    for i in 0..n {
        let mut next = *black_box(&pos);
        next.make_move(moves[i as usize % moves.len()]);
        sum += next.hash() as i64 & 1;
    }
    println!("copy + make:     {:6.1} ns", start.elapsed().as_nanos() as f64 / n as f64);

    let start = Instant::now();
    for _ in 0..n {
        let list = legal_moves(black_box(&pos));
        sum += list.len() as i64;
    }
    println!("generate all:    {:6.1} ns", start.elapsed().as_nanos() as f64 / n as f64);

    let start = Instant::now();
    for i in 0..n {
        sum += black_box(&pos).see_ge(moves[i as usize % moves.len()], 0) as i64;
    }
    println!("see_ge:          {:6.1} ns", start.elapsed().as_nanos() as f64 / n as f64);
    black_box(sum);
}
