//! Board state: piece placement, game state, hashing and move making.
//!
//! `Position` is `Copy` and small. The search uses copy-make: copy the
//! position, apply the move to the copy, and drop the copy to "unmake".

use crate::attacks::*;
use crate::types::PieceType::*;
use crate::types::*;
use crate::zobrist::ZOBRIST;
use std::fmt;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// Piece values used by static exchange evaluation (the king is never captured).
pub const SEE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];

/// Index of a castling right: white king side, white queen side, black king side, black queen side.
#[inline(always)]
pub const fn castle_index(color: Color, kingside: bool) -> usize {
    color.idx() * 2 + if kingside { 0 } else { 1 }
}

#[inline(always)]
pub const fn castle_right(color: Color, kingside: bool) -> u8 {
    1 << castle_index(color, kingside)
}

#[derive(Clone, Copy)]
pub struct Position {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    mailbox: [Piece; 64],
    hash: u64,
    pawn_hash: u64,
    /// Hash of each side's pieces other than pawns (kings included).
    non_pawn_hash: [u64; 2],
    /// Enemy pieces giving check to the side to move.
    checkers: Bitboard,
    /// Pieces of the side to move that are pinned to their own king.
    pinned: Bitboard,
    /// Squares attacked by the side not to move (sliders see through the mover's king).
    threats: Bitboard,
    fullmove: u16,
    plies_from_null: u16,
    stm: Color,
    /// Bitmask of castling rights, see `castle_right`.
    castling: u8,
    /// Starting square of the rook for each castling right (Chess960-aware).
    castle_rooks: [Square; 4],
    ep: Square,
    halfmove: u8,
}

impl Position {
    fn empty() -> Position {
        Position {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            mailbox: [Piece::NONE; 64],
            hash: 0,
            pawn_hash: 0,
            non_pawn_hash: [0; 2],
            checkers: Bitboard::EMPTY,
            pinned: Bitboard::EMPTY,
            threats: Bitboard::EMPTY,
            fullmove: 1,
            plies_from_null: 0,
            stm: Color::White,
            castling: 0,
            castle_rooks: [Square::NONE; 4],
            ep: Square::NONE,
            halfmove: 0,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("start position is valid")
    }

    /// Parses a FEN string. Castling rights may be given as `KQkq`, or as rook
    /// files (`HAha`, Shredder-FEN / X-FEN) for Chess960 positions.
    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let mut parts = fen.split_whitespace();
        let board = parts.next().ok_or("empty FEN")?;
        let stm = parts.next().unwrap_or("w");
        let castling = parts.next().unwrap_or("-");
        let ep = parts.next().unwrap_or("-");
        let halfmove = parts.next().unwrap_or("0");
        let fullmove = parts.next().unwrap_or("1");

        let mut pos = Position::empty();

        let (mut file, mut rank) = (0u8, 7u8);
        for ch in board.chars() {
            match ch {
                '/' => {
                    if file != 8 || rank == 0 {
                        return Err(format!("malformed board in FEN: {fen}"));
                    }
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => file += ch as u8 - b'0',
                _ => {
                    let piece = Piece::from_char(ch)
                        .ok_or_else(|| format!("unknown piece '{ch}' in FEN: {fen}"))?;
                    if file > 7 {
                        return Err(format!("malformed board in FEN: {fen}"));
                    }
                    pos.put_piece(piece, Square::from_fr(file, rank));
                    file += 1;
                }
            }
            if file > 8 {
                return Err(format!("malformed board in FEN: {fen}"));
            }
        }
        if rank != 0 || file != 8 {
            return Err(format!("malformed board in FEN: {fen}"));
        }
        for color in [Color::White, Color::Black] {
            if pos.pieces(color, King).count() != 1 {
                return Err(format!("each side needs exactly one king: {fen}"));
            }
        }
        if (pos.pieces[Pawn.idx()] & (Bitboard::rank(0) | Bitboard::rank(7))).any() {
            return Err(format!("pawn on the first or last rank: {fen}"));
        }

        pos.stm = match stm {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(format!("bad side to move in FEN: {fen}")),
        };
        if pos.stm == Color::Black {
            pos.hash ^= ZOBRIST.side;
        }

        for ch in castling.chars() {
            if ch == '-' {
                continue;
            }
            let color = if ch.is_ascii_uppercase() { Color::White } else { Color::Black };
            let back_rank = if color == Color::White { 0 } else { 7 };
            let ksq = pos.king_sq(color);
            let rooks = pos.pieces(color, Rook) & Bitboard::rank(back_rank);
            let rook_sq = match ch.to_ascii_lowercase() {
                // Outermost rook on the given side of the king.
                'k' => rooks.filter(|r| r.file() > ksq.file()).last(),
                'q' => rooks.into_iter().find(|r| r.file() < ksq.file()),
                f @ 'a'..='h' => {
                    let sq = Square::from_fr(f as u8 - b'a', back_rank);
                    rooks.contains(sq).then_some(sq)
                }
                _ => return Err(format!("bad castling field in FEN: {fen}")),
            };
            let rook_sq = match rook_sq {
                Some(sq) if ksq.rank() == back_rank && sq != ksq => sq,
                _ => return Err(format!("castling right without king/rook in place: {fen}")),
            };
            let kingside = rook_sq.file() > ksq.file();
            pos.castling |= castle_right(color, kingside);
            pos.castle_rooks[castle_index(color, kingside)] = rook_sq;
        }
        pos.hash ^= ZOBRIST.castling[pos.castling as usize];

        if ep != "-" {
            let sq = Square::parse(ep).ok_or_else(|| format!("bad en passant square: {fen}"))?;
            let them = !pos.stm;
            let expected_rank = if pos.stm == Color::White { 5 } else { 2 };
            let pawn_sq = Square(sq.0 ^ 8);
            // Only keep the square if an en passant capture is actually possible,
            // so that equal positions always hash equally.
            if sq.rank() == expected_rank
                && pos.piece_on(pawn_sq) == Piece::new(them, Pawn)
                && pos.piece_on(sq).is_none()
                && (pawn_attacks(them, sq) & pos.pieces(pos.stm, Pawn)).any()
            {
                pos.ep = sq;
                pos.hash ^= ZOBRIST.ep[sq.file() as usize];
            }
        }

        pos.halfmove = halfmove.parse::<u32>().unwrap_or(0).min(255) as u8;
        pos.fullmove = fullmove.parse::<u32>().unwrap_or(1).clamp(1, u16::MAX as u32) as u16;
        pos.plies_from_null = 1024;
        pos.update_check_info();

        let them = !pos.stm;
        if (pos.attackers_to(pos.king_sq(them), pos.occupied()) & pos.colors[pos.stm.idx()]).any()
        {
            return Err(format!("the side not to move is in check: {fen}"));
        }
        Ok(pos)
    }

    /// Chess960 start position from Scharnagl numbers (0..960 each); 518 is the
    /// standard setup. Different numbers for the two sides give "double" FRC.
    pub fn frc(white: usize, black: usize) -> Position {
        let w = frc_backrank(white % 960);
        let b = frc_backrank(black % 960);
        let mut fen = String::new();
        let mut castling = [String::new(), String::new()];
        for (i, rank) in [b, w].iter().enumerate() {
            let white_side = i == 1;
            let king_file = rank.iter().position(|&p| p == King).unwrap();
            let mut rook_files: Vec<usize> = (0..8).filter(|&f| rank[f] == Rook).collect();
            // King-side right first, to match the usual "KQkq" ordering.
            rook_files.sort_by_key(|&f| std::cmp::Reverse(f > king_file));
            for &pt in rank {
                let c = pt.to_char();
                fen.push(if white_side { c.to_ascii_uppercase() } else { c });
            }
            for f in rook_files {
                let c = (b'a' + f as u8) as char;
                castling[i].push(if white_side { c.to_ascii_uppercase() } else { c });
            }
            if !white_side {
                fen.push_str("/pppppppp/8/8/8/8/PPPPPPPP/");
            }
        }
        fen.push_str(&format!(" w {}{} - 0 1", castling[1], castling[0]));
        Position::from_fen(&fen).expect("generated Chess960 position is valid")
    }

    /// FEN of the position. With `chess960` castling rights are written as rook files.
    pub fn fen(&self, chess960: bool) -> String {
        let mut s = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let piece = self.piece_on(Square::from_fr(file, rank));
                if piece.is_none() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push_str(&empty.to_string());
                        empty = 0;
                    }
                    s.push(piece.to_char());
                }
            }
            if empty > 0 {
                s.push_str(&empty.to_string());
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (color, kingside, std_char) in [
                (Color::White, true, 'K'),
                (Color::White, false, 'Q'),
                (Color::Black, true, 'k'),
                (Color::Black, false, 'q'),
            ] {
                if self.castling & castle_right(color, kingside) == 0 {
                    continue;
                }
                if chess960 {
                    let file = self.castle_rooks[castle_index(color, kingside)].file();
                    let c = (b'a' + file) as char;
                    s.push(if color == Color::White { c.to_ascii_uppercase() } else { c });
                } else {
                    s.push(std_char);
                }
            }
        }
        s.push_str(&format!(" {} {} {}", self.ep, self.halfmove, self.fullmove));
        s
    }

    // ------------------------------------------------------------ accessors

    #[inline(always)]
    pub fn stm(&self) -> Color {
        self.stm
    }

    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.mailbox[sq.idx()]
    }

    /// All pieces of one type, both colours.
    #[cfg(test)]
    pub fn by_type(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()]
    }

    /// All pieces of one colour.
    #[inline(always)]
    pub fn by_color(&self, color: Color) -> Bitboard {
        self.colors[color.idx()]
    }

    #[inline(always)]
    pub fn pieces(&self, color: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()] & self.colors[color.idx()]
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }

    #[inline(always)]
    pub fn king_sq(&self, color: Color) -> Square {
        self.pieces(color, King).lsb()
    }

    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        self.checkers
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.checkers.any()
    }

    #[inline(always)]
    pub fn pinned(&self) -> Bitboard {
        self.pinned
    }

    #[inline(always)]
    pub fn threats(&self) -> Bitboard {
        self.threats
    }

    #[inline(always)]
    pub fn ep(&self) -> Square {
        self.ep
    }

    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[inline(always)]
    pub fn pawn_hash(&self) -> u64 {
        self.pawn_hash
    }

    #[inline(always)]
    pub fn non_pawn_hash(&self, color: Color) -> u64 {
        self.non_pawn_hash[color.idx()]
    }

    #[inline(always)]
    pub fn halfmove(&self) -> u8 {
        self.halfmove
    }

    #[inline(always)]
    pub fn fullmove(&self) -> u16 {
        self.fullmove
    }

    /// Number of plies over which a repetition of this position is possible.
    #[inline(always)]
    pub fn repetition_window(&self) -> usize {
        (self.halfmove as usize).min(self.plies_from_null as usize)
    }

    #[cfg(test)]
    pub fn castling(&self) -> u8 {
        self.castling
    }

    #[inline(always)]
    pub fn castle_rook(&self, color: Color, kingside: bool) -> Square {
        self.castle_rooks[castle_index(color, kingside)]
    }


    /// True if `color` has anything besides pawns and the king.
    #[inline(always)]
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        (self.colors[color.idx()] & !(self.pieces[Pawn.idx()] | self.pieces[King.idx()])).any()
    }

    /// Neither side can possibly deliver mate: bare kings, or a single minor piece.
    pub fn is_insufficient_material(&self) -> bool {
        (self.pieces[Pawn.idx()] | self.pieces[Rook.idx()] | self.pieces[Queen.idx()]).is_empty()
            && (self.pieces[Knight.idx()] | self.pieces[Bishop.idx()]).count() <= 1
    }

    /// All pieces (of both colours) attacking `sq` given the occupancy `occ`.
    #[inline]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let bishops = self.pieces[Bishop.idx()] | self.pieces[Queen.idx()];
        let rooks = self.pieces[Rook.idx()] | self.pieces[Queen.idx()];
        (pawn_attacks(Color::White, sq) & self.pieces(Color::Black, Pawn))
            | (pawn_attacks(Color::Black, sq) & self.pieces(Color::White, Pawn))
            | (knight_attacks(sq) & self.pieces[Knight.idx()])
            | (bishop_attacks(sq, occ) & bishops)
            | (rook_attacks(sq, occ) & rooks)
            | (king_attacks(sq) & self.pieces[King.idx()])
    }

    // ------------------------------------------------------ piece placement

    #[inline(always)]
    fn put_piece(&mut self, piece: Piece, sq: Square) {
        debug_assert!(self.mailbox[sq.idx()].is_none());
        let pt = piece.piece_type();
        self.pieces[pt.idx()] |= sq.bb();
        self.colors[piece.color().idx()] |= sq.bb();
        self.mailbox[sq.idx()] = piece;
        let key = ZOBRIST.pieces[piece.idx()][sq.idx()];
        self.hash ^= key;
        if pt == Pawn {
            self.pawn_hash ^= key;
        } else {
            self.non_pawn_hash[piece.color().idx()] ^= key;
        }
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let piece = self.mailbox[sq.idx()];
        debug_assert!(!piece.is_none());
        let pt = piece.piece_type();
        self.pieces[pt.idx()] ^= sq.bb();
        self.colors[piece.color().idx()] ^= sq.bb();
        self.mailbox[sq.idx()] = Piece::NONE;
        let key = ZOBRIST.pieces[piece.idx()][sq.idx()];
        self.hash ^= key;
        if pt == Pawn {
            self.pawn_hash ^= key;
        } else {
            self.non_pawn_hash[piece.color().idx()] ^= key;
        }
        piece
    }

    /// Castling rights that are lost when a piece moves from or to `sq`
    /// (i.e. `sq` is the home square of a rook that may still castle).
    #[inline(always)]
    fn rook_rights(&self, sq: Square) -> u8 {
        let mut mask = 0;
        for (i, &rook_sq) in self.castle_rooks.iter().enumerate() {
            if rook_sq == sq {
                mask |= 1 << i;
            }
        }
        mask
    }

    // ------------------------------------------------------------ make move

    /// Plays `mv`, which must be legal in this position.
    pub fn make_move(&mut self, mv: Move) {
        let us = self.stm;
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let piece = self.mailbox[from.idx()];
        debug_assert!(!piece.is_none() && piece.color() == us, "bad move {mv:?} in {}", self.fen(true));
        let pt = piece.piece_type();

        if self.ep != Square::NONE {
            self.hash ^= ZOBRIST.ep[self.ep.file() as usize];
            self.ep = Square::NONE;
        }
        let old_castling = self.castling;
        self.halfmove = self.halfmove.saturating_add(1);
        self.plies_from_null = self.plies_from_null.saturating_add(1);

        if flag == Move::CASTLE {
            let kingside = to.file() > from.file();
            let rank = from.rank();
            let king_to = Square::from_fr(if kingside { 6 } else { 2 }, rank);
            let rook_to = Square::from_fr(if kingside { 5 } else { 3 }, rank);
            // Lift both pieces before putting them down: in Chess960 their
            // start and destination squares can overlap.
            let rook = self.remove_piece(to);
            self.remove_piece(from);
            self.put_piece(piece, king_to);
            self.put_piece(rook, rook_to);
            self.castling &= !(castle_right(us, true) | castle_right(us, false));
        } else {
            if mv.is_capture() {
                let cap_sq = if flag == Move::EN_PASSANT { Square(to.0 ^ 8) } else { to };
                self.remove_piece(cap_sq);
                self.halfmove = 0;
                self.castling &= !self.rook_rights(to);
            }
            self.remove_piece(from);
            if mv.is_promo() {
                self.put_piece(Piece::new(us, mv.promo_type()), to);
            } else {
                self.put_piece(piece, to);
            }
            match pt {
                Pawn => {
                    self.halfmove = 0;
                    if flag == Move::DOUBLE_PUSH {
                        let ep = Square((from.0 + to.0) / 2);
                        if (pawn_attacks(us, ep) & self.pieces(!us, Pawn)).any() {
                            self.ep = ep;
                            self.hash ^= ZOBRIST.ep[ep.file() as usize];
                        }
                    }
                }
                King => self.castling &= !(castle_right(us, true) | castle_right(us, false)),
                Rook => self.castling &= !self.rook_rights(from),
                _ => {}
            }
        }

        if self.castling != old_castling {
            self.hash ^=
                ZOBRIST.castling[old_castling as usize] ^ ZOBRIST.castling[self.castling as usize];
        }
        self.stm = !us;
        self.hash ^= ZOBRIST.side;
        if us == Color::Black {
            self.fullmove = self.fullmove.saturating_add(1);
        }
        self.update_check_info();
    }

    /// Passes the turn. Must not be called when in check.
    pub fn make_null(&mut self) {
        debug_assert!(self.checkers.is_empty());
        if self.ep != Square::NONE {
            self.hash ^= ZOBRIST.ep[self.ep.file() as usize];
            self.ep = Square::NONE;
        }
        self.halfmove = self.halfmove.saturating_add(1);
        self.plies_from_null = 0;
        self.stm = !self.stm;
        self.hash ^= ZOBRIST.side;
        self.update_check_info();
    }

    /// Recomputes checkers, pinned pieces and enemy threats for the side to move.
    fn update_check_info(&mut self) {
        let us = self.stm;
        let them = !us;
        let ours = self.colors[us.idx()];
        let theirs = self.colors[them.idx()];
        let occ = ours | theirs;
        let ksq = self.king_sq(us);
        let their_bq = (self.pieces[Bishop.idx()] | self.pieces[Queen.idx()]) & theirs;
        let their_rq = (self.pieces[Rook.idx()] | self.pieces[Queen.idx()]) & theirs;

        self.checkers = (pawn_attacks(us, ksq) & self.pieces[Pawn.idx()] & theirs)
            | (knight_attacks(ksq) & self.pieces[Knight.idx()] & theirs)
            | (bishop_attacks(ksq, occ) & their_bq)
            | (rook_attacks(ksq, occ) & their_rq);

        let mut pinned = Bitboard::EMPTY;
        let snipers = (bishop_attacks(ksq, Bitboard::EMPTY) & their_bq)
            | (rook_attacks(ksq, Bitboard::EMPTY) & their_rq);
        for sniper in snipers {
            let blockers = between(ksq, sniper) & occ;
            if !blockers.more_than_one() {
                pinned |= blockers & ours;
            }
        }
        self.pinned = pinned;

        // Sliders look through our king so it cannot step back along a checking ray.
        let occ_no_king = occ ^ ksq.bb();
        let mut threats = pawn_attacks_bb(them, self.pieces[Pawn.idx()] & theirs);
        for sq in self.pieces[Knight.idx()] & theirs {
            threats |= knight_attacks(sq);
        }
        for sq in their_bq {
            threats |= bishop_attacks(sq, occ_no_king);
        }
        for sq in their_rq {
            threats |= rook_attacks(sq, occ_no_king);
        }
        threats |= king_attacks(self.king_sq(them));
        self.threats = threats;
    }

    // ------------------------------------------------------------- legality

    /// Whether the side to move may castle on the given side right now.
    pub fn castle_is_legal(&self, kingside: bool) -> bool {
        let us = self.stm;
        if self.castling & castle_right(us, kingside) == 0 || self.checkers.any() {
            return false;
        }
        let ksq = self.king_sq(us);
        let rook_sq = self.castle_rooks[castle_index(us, kingside)];
        let rank = ksq.rank();
        let king_to = Square::from_fr(if kingside { 6 } else { 2 }, rank);
        let rook_to = Square::from_fr(if kingside { 5 } else { 3 }, rank);
        // Everything the king and rook travel over must be empty, ignoring the two of them.
        let occ = self.occupied() ^ ksq.bb() ^ rook_sq.bb();
        let king_path = between(ksq, king_to) | king_to.bb();
        let rook_path = between(rook_sq, rook_to) | rook_to.bb();
        if ((king_path | rook_path) & occ).any() || (king_path & self.threats).any() {
            return false;
        }
        // Chess960: the castling rook may have been shielding the king's
        // destination from a rook or queen on the back rank.
        let their_rq =
            (self.pieces[Rook.idx()] | self.pieces[Queen.idx()]) & self.colors[(!us).idx()];
        (rook_attacks(king_to, occ) & Bitboard::rank(rank) & their_rq).is_empty()
    }

    /// Whether the pawn on `from` may capture en passant (the ep square must be set).
    pub fn ep_is_legal(&self, from: Square) -> bool {
        let us = self.stm;
        let ksq = self.king_sq(us);
        let cap_sq = Square(self.ep.0 ^ 8);
        // A check by a knight or by another pawn cannot be answered en passant.
        let leapers = self.pieces[Knight.idx()] | self.pieces[Pawn.idx()];
        if (self.checkers & leapers & !cap_sq.bb()).any() {
            return false;
        }
        // After the capture no slider may attack our king. This covers pins of
        // either pawn (including both pawns leaving the same rank) and
        // unanswered slider checks.
        let occ = self.occupied() ^ from.bb() ^ cap_sq.bb() ^ self.ep.bb();
        let theirs = self.colors[(!us).idx()];
        let their_bq = (self.pieces[Bishop.idx()] | self.pieces[Queen.idx()]) & theirs;
        let their_rq = (self.pieces[Rook.idx()] | self.pieces[Queen.idx()]) & theirs;
        (bishop_attacks(ksq, occ) & their_bq).is_empty()
            && (rook_attacks(ksq, occ) & their_rq).is_empty()
    }

    /// Full legality test for an arbitrary 16-bit move, e.g. one read from the
    /// transposition table. Agrees exactly with the move generator.
    pub fn is_legal(&self, mv: Move) -> bool {
        let from = mv.from();
        let to = mv.to();
        let flag = mv.flag();
        let us = self.stm;
        let piece = self.mailbox[from.idx()];
        if piece.is_none() || piece.color() != us {
            return false;
        }
        let pt = piece.piece_type();
        let ours = self.colors[us.idx()];
        let theirs = self.colors[(!us).idx()];
        let occ = ours | theirs;
        let ksq = self.king_sq(us);

        if flag == Move::CASTLE {
            if pt != King {
                return false;
            }
            for kingside in [true, false] {
                if self.castling & castle_right(us, kingside) != 0
                    && self.castle_rooks[castle_index(us, kingside)] == to
                {
                    return self.castle_is_legal(kingside);
                }
            }
            return false;
        }
        if matches!(flag, 3 | 6 | 7) || ours.contains(to) {
            return false;
        }
        let captures_piece = theirs.contains(to);

        if pt == King {
            return (flag == Move::QUIET || flag == Move::CAPTURE)
                && (flag == Move::CAPTURE) == captures_piece
                && king_attacks(from).contains(to)
                && !self.threats.contains(to);
        }
        if self.checkers.more_than_one() {
            return false;
        }

        if pt == Pawn {
            let up: i32 = if us == Color::White { 8 } else { -8 };
            let last_rank = if us == Color::White { 7 } else { 0 };
            if mv.is_promo() != (to.rank() == last_rank) {
                return false;
            }
            if flag == Move::EN_PASSANT {
                return self.ep != Square::NONE
                    && to == self.ep
                    && pawn_attacks(us, from).contains(to)
                    && self.ep_is_legal(from);
            }
            if mv.is_capture() {
                if !captures_piece || !pawn_attacks(us, from).contains(to) {
                    return false;
                }
            } else if flag == Move::DOUBLE_PUSH {
                let start_rank = if us == Color::White { 1 } else { 6 };
                if from.rank() != start_rank
                    || to != from.offset(2 * up)
                    || occ.contains(from.offset(up))
                    || occ.contains(to)
                {
                    return false;
                }
            } else if to != from.offset(up) || occ.contains(to) {
                return false;
            }
        } else {
            if (flag != Move::QUIET && flag != Move::CAPTURE)
                || (flag == Move::CAPTURE) != captures_piece
                || !piece_attacks(pt, from, occ).contains(to)
            {
                return false;
            }
        }

        // The move must answer a check, and a pinned piece must stay on its pin line.
        if self.checkers.any() {
            let checker = self.checkers.lsb();
            if !(between(ksq, checker) | self.checkers).contains(to) {
                return false;
            }
        }
        !self.pinned.contains(from) || line(ksq, from).contains(to)
    }

    /// Whether the legal move `mv` gives check, without making it.
    pub fn gives_check(&self, mv: Move) -> bool {
        let us = self.stm;
        let (from, to) = (mv.from(), mv.to());
        let their_king = self.king_sq(!us);
        let occ = self.occupied();

        if mv.is_castle() {
            // Only the rook can give check; the king is off its old square.
            let kingside = to.file() > from.file();
            let king_to = Square::from_fr(if kingside { 6 } else { 2 }, from.rank());
            let rook_to = Square::from_fr(if kingside { 5 } else { 3 }, from.rank());
            let after = (occ ^ from.bb() ^ to.bb()) | king_to.bb() | rook_to.bb();
            return rook_attacks(rook_to, after).contains(their_king);
        }

        // Direct check by the moved (or promoted) piece from its new square.
        let after = (occ ^ from.bb()) | to.bb();
        let pt = if mv.is_promo() { mv.promo_type() } else { self.piece_on(from).piece_type() };
        let direct = match pt {
            Pawn => pawn_attacks(us, to).contains(their_king),
            King => false,
            _ => piece_attacks(pt, to, after).contains(their_king),
        };
        if direct {
            return true;
        }

        // Discovered check: a slider of ours is uncovered by the piece leaving
        // its square (or, en passant, by the captured pawn disappearing).
        let ours = self.colors[us.idx()];
        let bishops = (self.pieces[Bishop.idx()] | self.pieces[Queen.idx()]) & ours & !from.bb();
        let rooks = (self.pieces[Rook.idx()] | self.pieces[Queen.idx()]) & ours & !from.bb();
        if mv.is_en_passant() {
            let after = after ^ Square(to.0 ^ 8).bb();
            return (bishop_attacks(their_king, after) & bishops).any()
                || (rook_attacks(their_king, after) & rooks).any();
        }
        // Only possible if the piece leaves a line through the enemy king.
        let through = line(from, their_king);
        if through.is_empty() || through.contains(to) {
            return false;
        }
        (bishop_attacks(their_king, after) & bishops & through).any()
            || (rook_attacks(their_king, after) & rooks & through).any()
    }

    /// Static exchange evaluation: does the sequence of captures on the
    /// destination square that `mv` starts win at least `threshold`
    /// centipawns for the side to move, assuming both sides always recapture
    /// with their least valuable piece and may stop when that is better?
    pub fn see_ge(&self, mv: Move, threshold: i32) -> bool {
        if mv.is_castle() {
            return threshold <= 0;
        }
        let from = mv.from();
        let to = mv.to();

        // Material won by the move itself.
        let mut balance = -threshold;
        if mv.is_en_passant() {
            balance += SEE_VALUES[Pawn.idx()];
        } else if mv.is_capture() {
            balance += SEE_VALUES[self.piece_on(to).piece_type().idx()];
        }
        let mut victim = self.piece_on(from).piece_type();
        if mv.is_promo() {
            victim = mv.promo_type();
            balance += SEE_VALUES[victim.idx()] - SEE_VALUES[Pawn.idx()];
        }
        // Not enough even if the piece is never recaptured.
        if balance < 0 {
            return false;
        }
        // Enough even if the piece is lost for nothing.
        balance -= SEE_VALUES[victim.idx()];
        if balance >= 0 {
            return true;
        }

        let mut occ = (self.occupied() ^ from.bb()) | to.bb();
        if mv.is_en_passant() {
            occ ^= Square(to.0 ^ 8).bb();
        }
        let diagonal = self.pieces[Bishop.idx()] | self.pieces[Queen.idx()];
        let straight = self.pieces[Rook.idx()] | self.pieces[Queen.idx()];
        let mut attackers = self.attackers_to(to, occ) & occ;
        let mut side = !self.stm;

        loop {
            let ours = attackers & self.colors[side.idx()];
            if ours.is_empty() {
                break;
            }
            // Recapture with the least valuable piece.
            let mut pt = Pawn;
            for candidate in PieceType::ALL {
                pt = candidate;
                if (ours & self.pieces[candidate.idx()]).any() {
                    break;
                }
            }
            occ ^= (ours & self.pieces[pt.idx()]).lsb().bb();
            // Removing the piece may uncover a slider behind it.
            if matches!(pt, Pawn | Bishop | Queen) {
                attackers |= bishop_attacks(to, occ) & diagonal;
            }
            if matches!(pt, Rook | Queen) {
                attackers |= rook_attacks(to, occ) & straight;
            }
            attackers &= occ;

            side = !side;
            balance = -balance - 1 - SEE_VALUES[pt.idx()];
            if balance >= 0 {
                // A king cannot capture a defended piece: that capture never happened.
                if pt == King && (attackers & self.colors[side.idx()]).any() {
                    side = !side;
                }
                break;
            }
        }
        // The side left to move when the exchange stops has lost it.
        side != self.stm
    }

    /// Recomputes the hash from scratch; used to verify incremental updates.
    #[cfg(test)]
    pub fn compute_hash(&self) -> (u64, u64, [u64; 2]) {
        let (mut hash, mut pawn_hash, mut non_pawn_hash) = (0u64, 0u64, [0u64; 2]);
        for sq in self.occupied() {
            let piece = self.piece_on(sq);
            hash ^= ZOBRIST.pieces[piece.idx()][sq.idx()];
            if piece.piece_type() == Pawn {
                pawn_hash ^= ZOBRIST.pieces[piece.idx()][sq.idx()];
            } else {
                non_pawn_hash[piece.color().idx()] ^= ZOBRIST.pieces[piece.idx()][sq.idx()];
            }
        }
        if self.stm == Color::Black {
            hash ^= ZOBRIST.side;
        }
        hash ^= ZOBRIST.castling[self.castling as usize];
        if self.ep != Square::NONE {
            hash ^= ZOBRIST.ep[self.ep.file() as usize];
        }
        (hash, pawn_hash, non_pawn_hash)
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            write!(f, " {} ", rank + 1)?;
            for file in 0..8 {
                write!(f, " {}", self.piece_on(Square::from_fr(file, rank)).to_char())?;
            }
            writeln!(f)?;
        }
        writeln!(f, "\n    a b c d e f g h\n")?;
        writeln!(f, "FEN: {}", self.fen(false))?;
        write!(f, "Key: {:016x}", self.hash)
    }
}

/// Back-rank arrangement of Chess960 start position `n` (Scharnagl numbering).
pub fn frc_backrank(n: usize) -> [PieceType; 8] {
    fn place(rank: &mut [Option<PieceType>; 8], nth_empty: usize, pt: PieceType) {
        let file = (0..8).filter(|&f| rank[f].is_none()).nth(nth_empty).unwrap();
        rank[file] = Some(pt);
    }
    const KNIGHTS: [(usize, usize); 10] =
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)];

    let mut rank = [None; 8];
    let mut n = n;
    rank[1 + 2 * (n % 4)] = Some(Bishop); // light-squared bishop
    n /= 4;
    rank[2 * (n % 4)] = Some(Bishop); // dark-squared bishop
    n /= 4;
    place(&mut rank, n % 6, Queen);
    n /= 6;
    let (a, b) = KNIGHTS[n];
    place(&mut rank, b, Knight); // higher slot first so the lower index stays valid
    place(&mut rank, a, Knight);
    place(&mut rank, 0, Rook);
    place(&mut rank, 0, King);
    place(&mut rank, 0, Rook);
    rank.map(|p| p.unwrap())
}
