//! Move ordering statistics gathered during the search.

use crate::params;
use crate::position::Position;
use crate::types::*;

pub const HISTORY_MAX: i32 = 16_384;

/// Marks "no previous move" for continuation history (root, or after a null move).
pub const NO_CONT: usize = usize::MAX;

const CONT_ROW: usize = 12 * 64;

/// Number of pawn structures the pawn history tells apart (by pawn hash).
const PAWN_HISTORY_SIZE: usize = 512;

/// Correction history: entries per table, fixed-point scale, and the largest
/// correction (in centipawns) a single table may apply.
const CORRECTION_SIZE: usize = 16_384;
const CORRECTION_GRAIN: i32 = 256;
const CORRECTION_LIMIT: i32 = 64;

/// Tables that record which moves tended to be good in earlier parts of the
/// search. They persist between searches of the same game.
pub struct History {
    /// How often a quiet move caused a cutoff, indexed by `[side to move,
    /// whether the square it leaves is attacked, whether the square it goes
    /// to is attacked][from][to]`: moving a piece out of an attack, or into
    /// one, is a different move from the same one made in peace.
    main: Box<[[[i16; 64]; 64]; 8]>,
    /// The same for captures, indexed by `[moving piece][to][captured piece type]`.
    capture: Box<[[[i16; 6]; 64]; 12]>,
    /// How well a move worked as a follow-up to an earlier move, indexed by
    /// `[earlier piece][earlier to][piece][to]` (stored flat).
    continuation: Vec<i16>,
    /// How well a quiet move worked in positions with the same pawn
    /// structure, indexed by `[pawn hash][piece][to]` (stored flat): where a
    /// piece belongs depends on the pawns more than on anything else.
    pawn: Vec<i16>,
    /// Running average of (search score - static evaluation) for positions
    /// sharing a pawn structure, indexed by `[side to move][pawn hash]`.
    pawn_correction: Box<[[i16; CORRECTION_SIZE]; 2]>,
    /// The same for positions sharing one side's non-pawn pieces, indexed by
    /// `[side to move][piece colour][non-pawn hash]`.
    non_pawn_correction: Box<[[[i16; CORRECTION_SIZE]; 2]; 2]>,
}

impl History {
    pub fn new() -> History {
        History {
            main: Box::new([[[0; 64]; 64]; 8]),
            capture: Box::new([[[0; 6]; 64]; 12]),
            continuation: vec![0; CONT_ROW * CONT_ROW],
            pawn: vec![0; PAWN_HISTORY_SIZE * CONT_ROW],
            pawn_correction: Box::new([[0; CORRECTION_SIZE]; 2]),
            non_pawn_correction: Box::new([[[0; CORRECTION_SIZE]; 2]; 2]),
        }
    }

    pub fn clear(&mut self) {
        *self.main = [[[0; 64]; 64]; 8];
        *self.capture = [[[0; 6]; 64]; 12];
        self.continuation.fill(0);
        self.pawn.fill(0);
        *self.pawn_correction = [[0; CORRECTION_SIZE]; 2];
        *self.non_pawn_correction = [[[0; CORRECTION_SIZE]; 2]; 2];
    }

    /// Adjusts a raw static evaluation by what earlier searches of similar
    /// positions found the evaluation to be off by.
    #[inline(always)]
    pub fn correct_eval(&self, pos: &Position, raw_eval: i32) -> i32 {
        let stm = pos.stm().idx();
        let correction = self.pawn_correction[stm][pos.pawn_hash() as usize % CORRECTION_SIZE] as i32
            + self.non_pawn_correction[stm][0][pos.non_pawn_hash(Color::White) as usize % CORRECTION_SIZE] as i32
            + self.non_pawn_correction[stm][1][pos.non_pawn_hash(Color::Black) as usize % CORRECTION_SIZE] as i32;
        raw_eval + correction / CORRECTION_GRAIN
    }

    /// Records that a search of `pos` to `depth` scored `diff` centipawns
    /// above its static evaluation.
    pub fn update_correction(&mut self, pos: &Position, diff: i32, depth: i32) {
        let stm = pos.stm().idx();
        let weight = (depth + 1).min(16);
        let target = diff.clamp(-CORRECTION_LIMIT * 4, CORRECTION_LIMIT * 4) * CORRECTION_GRAIN;
        let limit = CORRECTION_LIMIT * CORRECTION_GRAIN;
        let blend = |entry: &mut i16| {
            let value = (*entry as i32 * (256 - weight) + target * weight) / 256;
            *entry = value.clamp(-limit, limit) as i16;
        };
        blend(&mut self.pawn_correction[stm][pos.pawn_hash() as usize % CORRECTION_SIZE]);
        blend(&mut self.non_pawn_correction[stm][0][pos.non_pawn_hash(Color::White) as usize % CORRECTION_SIZE]);
        blend(&mut self.non_pawn_correction[stm][1][pos.non_pawn_hash(Color::Black) as usize % CORRECTION_SIZE]);
    }

    /// Which of the main table's eight parts a quiet move of the side to move belongs to.
    #[inline(always)]
    fn main_part(pos: &Position, mv: Move) -> usize {
        let threats = pos.threats();
        pos.stm().idx() * 4 + threats.contains(mv.from()) as usize * 2 + threats.contains(mv.to()) as usize
    }

    #[inline(always)]
    fn pawn_entry(pos: &Position, piece: Piece, to: Square) -> usize {
        (pos.pawn_hash() as usize % PAWN_HISTORY_SIZE) * CONT_ROW + piece.idx() * 64 + to.idx()
    }

    /// Row of the continuation table for a move that put `piece` on `to`.
    #[inline(always)]
    pub fn cont_row(piece: Piece, to: Square) -> usize {
        (piece.idx() * 64 + to.idx()) * CONT_ROW
    }

    #[inline(always)]
    fn cont(&self, row: usize, piece: Piece, to: Square) -> i32 {
        if row == NO_CONT { 0 } else { self.continuation[row + piece.idx() * 64 + to.idx()] as i32 }
    }

    /// Ordering score of a quiet move. `conts` are the continuation rows of
    /// the opponent's last move and of our own move before that.
    #[inline(always)]
    pub fn quiet_score(&self, pos: &Position, mv: Move, conts: [usize; 2]) -> i32 {
        let piece = pos.piece_on(mv.from());
        self.main[History::main_part(pos, mv)][mv.from().idx()][mv.to().idx()] as i32
            + self.cont(conts[0], piece, mv.to())
            + self.cont(conts[1], piece, mv.to())
    }

    /// Ordering score of a quiet move: its record, and how the move has done
    /// with this pawn structure. The second part is kept out of
    /// `quiet_score`, so the thresholds the search compares that with keep
    /// their meaning.
    #[inline(always)]
    pub fn order_score(&self, pos: &Position, mv: Move, conts: [usize; 2]) -> i32 {
        self.quiet_score(pos, mv, conts)
            + self.pawn[History::pawn_entry(pos, pos.piece_on(mv.from()), mv.to())] as i32
    }

    pub fn update_quiet(&mut self, pos: &Position, mv: Move, conts: [usize; 2], bonus: i32) {
        let piece = pos.piece_on(mv.from());
        update(&mut self.pawn[History::pawn_entry(pos, piece, mv.to())], bonus);
        update(&mut self.main[History::main_part(pos, mv)][mv.from().idx()][mv.to().idx()], bonus);
        for row in conts {
            if row != NO_CONT {
                update(&mut self.continuation[row + piece.idx() * 64 + mv.to().idx()], bonus);
            }
        }
    }

    #[inline(always)]
    fn capture_entry(pos: &Position, mv: Move) -> (usize, usize, usize) {
        let piece = pos.piece_on(mv.from());
        let victim = if mv.is_capture() && !mv.is_en_passant() {
            pos.piece_on(mv.to()).piece_type()
        } else {
            // En passant, and queen promotions without a capture.
            PieceType::Pawn
        };
        (piece.idx(), mv.to().idx(), victim.idx())
    }

    #[inline(always)]
    pub fn capture_score(&self, pos: &Position, mv: Move) -> i32 {
        let (piece, to, victim) = History::capture_entry(pos, mv);
        self.capture[piece][to][victim] as i32
    }

    pub fn update_capture(&mut self, pos: &Position, mv: Move, bonus: i32) {
        let (piece, to, victim) = History::capture_entry(pos, mv);
        update(&mut self.capture[piece][to][victim], bonus);
    }
}

/// Moves a history counter towards `bonus`, saturating at +-HISTORY_MAX
/// ("history gravity": the closer to the limit, the smaller the step).
#[inline(always)]
fn update(entry: &mut i16, bonus: i32) {
    let bonus = bonus.clamp(-HISTORY_MAX, HISTORY_MAX);
    *entry += (bonus - *entry as i32 * bonus.abs() / HISTORY_MAX) as i16;
}

/// Size of the history update for a cutoff at the given depth.
#[inline(always)]
pub fn bonus(depth: i32) -> i32 {
    (params::hist_bonus_mult() * depth - params::hist_bonus_offset()).clamp(0, params::hist_bonus_max())
}
