//! Core chess types: colours, piece types, pieces, squares, bitboards and moves.

use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

// ------------------------------------------------------------------ Color

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }
}

impl Not for Color {
    type Output = Color;
    #[inline(always)]
    fn not(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

// -------------------------------------------------------------- PieceType

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    pub const fn to_char(self) -> char {
        ['p', 'n', 'b', 'r', 'q', 'k'][self as usize]
    }
}

// ------------------------------------------------------------------ Piece

/// A coloured piece, encoded as `piece_type * 2 + color` (0..12). 12 = none.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Piece(u8);

impl Piece {
    pub const NONE: Piece = Piece(12);

    #[inline(always)]
    pub const fn new(color: Color, pt: PieceType) -> Piece {
        Piece(((pt as u8) << 1) | color as u8)
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 >= 12
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 & 1 == 0 { Color::White } else { Color::Black }
    }

    /// Panics for `Piece::NONE`.
    #[inline(always)]
    pub const fn piece_type(self) -> PieceType {
        PieceType::ALL[(self.0 >> 1) as usize]
    }

    pub fn from_char(c: char) -> Option<Piece> {
        let pt = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        let color = if c.is_ascii_uppercase() { Color::White } else { Color::Black };
        Some(Piece::new(color, pt))
    }

    pub fn to_char(self) -> char {
        if self.is_none() {
            return '.';
        }
        let c = self.piece_type().to_char();
        if self.color() == Color::White { c.to_ascii_uppercase() } else { c }
    }
}

// ----------------------------------------------------------------- Square

/// A board square, a1 = 0, b1 = 1, ..., h8 = 63. 64 = none.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Square(pub u8);

impl Square {
    pub const NONE: Square = Square(64);

    #[inline(always)]
    pub const fn from_fr(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline(always)]
    pub const fn bb(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }


    /// The square `delta` steps away (no bounds checking beyond a debug assert).
    #[inline(always)]
    pub const fn offset(self, delta: i32) -> Square {
        let v = self.0 as i32 + delta;
        debug_assert!(v >= 0 && v < 64);
        Square(v as u8)
    }

    pub fn parse(s: &str) -> Option<Square> {
        let b = s.as_bytes();
        if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || !(b'1'..=b'8').contains(&b[1]) {
            return None;
        }
        Some(Square::from_fr(b[0] - b'a', b[1] - b'1'))
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 64 {
            return write!(f, "-");
        }
        write!(f, "{}{}", (b'a' + self.file()) as char, (b'1' + self.rank()) as char)
    }
}

// --------------------------------------------------------------- Bitboard

/// A set of squares; bit `i` is square `i`. Iterating yields squares from a1 upwards.
#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    #[inline(always)]
    pub const fn rank(rank: u8) -> Bitboard {
        Bitboard(0xFFu64 << (8 * rank))
    }

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    /// Lowest square in the set. The set must not be empty.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }
}

impl Iterator for Bitboard {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            let sq = Square(self.0.trailing_zeros() as u8);
            self.0 &= self.0 - 1;
            Some(sq)
        }
    }
}

macro_rules! bitboard_ops {
    ($($trait:ident $fn:ident $assign_trait:ident $assign_fn:ident $op:tt),*) => {$(
        impl $trait for Bitboard {
            type Output = Bitboard;
            #[inline(always)]
            fn $fn(self, rhs: Bitboard) -> Bitboard {
                Bitboard(self.0 $op rhs.0)
            }
        }
        impl $assign_trait for Bitboard {
            #[inline(always)]
            fn $assign_fn(&mut self, rhs: Bitboard) {
                self.0 = self.0 $op rhs.0;
            }
        }
    )*};
}
bitboard_ops!(BitAnd bitand BitAndAssign bitand_assign &, BitOr bitor BitOrAssign bitor_assign |, BitXor bitxor BitXorAssign bitxor_assign ^);

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let c = if self.contains(Square::from_fr(file, rank)) { 'X' } else { '.' };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ------------------------------------------------------------------- Move

/// A move packed into 16 bits: from (6) | to (6) << 6 | flag (4) << 12.
///
/// Castling is encoded as "king takes own rook" (`to` is the rook's square),
/// which also covers Chess960. `Move::NONE` (all zero) doubles as the null move.
#[derive(Copy, Clone, PartialEq, Eq, Default, Debug)]
pub struct Move(pub u16);

impl Move {
    pub const NONE: Move = Move(0);

    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const CASTLE: u16 = 2;
    pub const CAPTURE: u16 = 4;
    pub const EN_PASSANT: u16 = 5;
    /// Promotion flags are `PROMO + (piece - Knight)`; add `CAPTURE` for promotion captures.
    pub const PROMO: u16 = 8;
    pub const PROMO_QUEEN: u16 = 11;

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move(from.0 as u16 | (to.0 as u16) << 6 | flag << 12)
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square(((self.0 >> 6) & 63) as u8)
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.0 & (Move::CAPTURE << 12) != 0
    }

    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        self.0 & (Move::PROMO << 12) != 0
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        self.flag() == Move::CASTLE
    }

    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        self.flag() == Move::EN_PASSANT
    }

    /// Piece type promoted to. Only meaningful when `is_promo()`.
    #[inline(always)]
    pub const fn promo_type(self) -> PieceType {
        PieceType::ALL[1 + ((self.0 >> 12) & 3) as usize]
    }

    /// Captures and queen promotions; everything the quiescence search looks at.
    #[inline(always)]
    pub const fn is_noisy(self) -> bool {
        self.is_capture() || self.flag() == Move::PROMO_QUEEN
    }

    #[inline(always)]
    pub const fn is_quiet(self) -> bool {
        !self.is_noisy()
    }

    /// Formats the move in UCI notation. In standard chess castling is written
    /// as the king's two-square move, in Chess960 as king-takes-rook.
    pub fn to_uci(self, chess960: bool) -> String {
        if self.is_none() {
            return "0000".to_string();
        }
        let from = self.from();
        let mut to = self.to();
        if self.is_castle() && !chess960 {
            let file = if to.file() > from.file() { 6 } else { 2 };
            to = Square::from_fr(file, from.rank());
        }
        let mut s = format!("{from}{to}");
        if self.is_promo() {
            s.push(self.promo_type().to_char());
        }
        s
    }
}
