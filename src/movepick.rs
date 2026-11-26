//! Staged move ordering.
//!
//! Moves are produced lazily in the order most likely to cause an early
//! cutoff: the hash move, winning captures, killer moves, quiet moves by
//! history score, and finally losing captures. Later stages (and their move
//! generation and scoring cost) are skipped entirely when an earlier move
//! already refutes the position.

use crate::history::History;
use crate::movegen::{MAX_MOVES, MoveList, generate_noisy, generate_quiet};
use crate::position::Position;
use crate::types::*;

const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20_000];

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Debug)]
enum Stage {
    TtMove,
    GenerateNoisy,
    GoodNoisy,
    FirstKiller,
    SecondKiller,
    GenerateQuiet,
    Quiet,
    BadNoisy,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    tt_move: Move,
    killers: [Move; 2],
    /// Only captures and queen promotions that do not lose material (quiescence search).
    noisy_only: bool,
    /// The search asked for the remaining quiet moves to be skipped.
    no_quiets: bool,
    moves: MoveList,
    scores: [i32; MAX_MOVES],
    /// Next move to consider in the current stage.
    index: usize,
    /// Losing captures are parked in `moves[..bad_count]` until the last stage.
    bad_count: usize,
}

impl MovePicker {
    /// Picker for the main search: every legal move, best guesses first.
    pub fn new(tt_move: Move, killers: [Move; 2]) -> MovePicker {
        MovePicker {
            stage: Stage::TtMove,
            tt_move,
            killers,
            noisy_only: false,
            no_quiets: false,
            moves: MoveList::new(),
            scores: [0; MAX_MOVES],
            index: 0,
            bad_count: 0,
        }
    }

    /// Picker for the quiescence search when not in check: only captures and
    /// queen promotions that do not lose material.
    pub fn new_noisy(tt_move: Move) -> MovePicker {
        let mut picker = MovePicker::new(tt_move, [Move::NONE; 2]);
        picker.noisy_only = true;
        picker
    }

    /// Stops producing quiet moves (the search decided they are not worth trying).
    pub fn skip_quiets(&mut self) {
        self.no_quiets = true;
        if matches!(
            self.stage,
            Stage::FirstKiller | Stage::SecondKiller | Stage::GenerateQuiet | Stage::Quiet
        ) {
            self.stage = Stage::BadNoisy;
            self.index = 0;
        }
    }

    /// Moves the highest-scored move in `moves[index..]` to `index` and returns it.
    #[inline(always)]
    fn select_best(&mut self) -> Move {
        let mut best = self.index;
        for i in self.index + 1..self.moves.len() {
            if self.scores[i] > self.scores[best] {
                best = i;
            }
        }
        self.moves.swap(self.index, best);
        self.scores.swap(self.index, best);
        self.moves[self.index]
    }

    /// Returns the next move to try. `conts` are the continuation history
    /// rows for the previous two plies (see `History::quiet_score`).
    pub fn next(&mut self, pos: &Position, history: &History, conts: [usize; 2]) -> Option<Move> {
        loop {
            match self.stage {
                Stage::TtMove => {
                    self.stage = Stage::GenerateNoisy;
                    if !self.tt_move.is_none()
                        && (!self.noisy_only || self.tt_move.is_noisy())
                        && pos.is_legal(self.tt_move)
                    {
                        return Some(self.tt_move);
                    }
                    // Not usable here: make sure it is not filtered out below.
                    self.tt_move = Move::NONE;
                }
                Stage::GenerateNoisy => {
                    generate_noisy(pos, &mut self.moves);
                    for i in 0..self.moves.len() {
                        let mv = self.moves[i];
                        // Most valuable victim first, then by how well this
                        // kind of capture has worked before.
                        let mut score = history.capture_score(pos, mv);
                        if mv.is_capture() {
                            let victim = if mv.is_en_passant() {
                                PieceType::Pawn
                            } else {
                                pos.piece_on(mv.to()).piece_type()
                            };
                            score += PIECE_VALUES[victim.idx()] * 16;
                        }
                        if mv.is_promo() {
                            score += PIECE_VALUES[mv.promo_type().idx()] * 16;
                        }
                        self.scores[i] = score;
                    }
                    self.stage = Stage::GoodNoisy;
                }
                Stage::GoodNoisy => {
                    if self.index >= self.moves.len() {
                        self.stage = if self.noisy_only {
                            Stage::Done
                        } else if self.no_quiets {
                            self.index = 0;
                            Stage::BadNoisy
                        } else {
                            Stage::FirstKiller
                        };
                        continue;
                    }
                    let mv = self.select_best();
                    self.index += 1;
                    if mv == self.tt_move {
                        continue;
                    }
                    if !pos.see_ge(mv, 0) {
                        // Loses material: try it after everything else. The
                        // slots before `index` are already consumed, so park it there.
                        self.moves[self.bad_count] = mv;
                        self.bad_count += 1;
                        continue;
                    }
                    return Some(mv);
                }
                Stage::FirstKiller | Stage::SecondKiller => {
                    let second = self.stage == Stage::SecondKiller;
                    let killer = self.killers[second as usize];
                    self.stage = if second { Stage::GenerateQuiet } else { Stage::SecondKiller };
                    if !killer.is_none()
                        && killer != self.tt_move
                        && !(second && killer == self.killers[0])
                        && killer.is_quiet()
                        && pos.is_legal(killer)
                    {
                        return Some(killer);
                    }
                }
                Stage::GenerateQuiet => {
                    // Quiet moves are appended after the (consumed) noisy moves.
                    let start = self.moves.len();
                    generate_quiet(pos, &mut self.moves);
                    for i in start..self.moves.len() {
                        self.scores[i] = history.order_score(pos, self.moves[i], conts);
                    }
                    self.index = start;
                    self.stage = Stage::Quiet;
                }
                Stage::Quiet => {
                    if self.index >= self.moves.len() {
                        self.stage = Stage::BadNoisy;
                        self.index = 0;
                        continue;
                    }
                    let mv = self.select_best();
                    self.index += 1;
                    if mv == self.tt_move || mv == self.killers[0] || mv == self.killers[1] {
                        continue;
                    }
                    return Some(mv);
                }
                Stage::BadNoisy => {
                    if self.index >= self.bad_count {
                        self.stage = Stage::Done;
                        continue;
                    }
                    let mv = self.moves[self.index];
                    self.index += 1;
                    return Some(mv);
                }
                Stage::Done => return None,
            }
        }
    }
}
