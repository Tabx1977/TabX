//! Zobrist hashing keys, generated at compile time from a fixed seed.

pub struct Zobrist {
    /// Indexed by `[piece][square]`.
    pub pieces: [[u64; 64]; 12],
    /// XORed in when Black is to move.
    pub side: u64,
    /// Indexed by the 4-bit castling rights mask.
    pub castling: [u64; 16],
    /// Indexed by the file of the en passant square.
    pub ep: [u64; 8],
}

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Zobrist {
    const fn new() -> Zobrist {
        let mut state = 0x5EED_C0DE_1234_5678u64;
        let mut z = Zobrist { pieces: [[0; 64]; 12], side: 0, castling: [0; 16], ep: [0; 8] };
        let mut p = 0;
        while p < 12 {
            let mut s = 0;
            while s < 64 {
                z.pieces[p][s] = splitmix64(&mut state);
                s += 1;
            }
            p += 1;
        }
        z.side = splitmix64(&mut state);
        // Castling keys are composed from one key per right so that the key of
        // a rights mask is the XOR of its individual rights; index 0 stays 0.
        let mut rights = [0u64; 4];
        let mut i = 0;
        while i < 4 {
            rights[i] = splitmix64(&mut state);
            i += 1;
        }
        let mut mask = 0;
        while mask < 16 {
            let mut bit = 0;
            while bit < 4 {
                if mask & (1 << bit) != 0 {
                    z.castling[mask] ^= rights[bit];
                }
                bit += 1;
            }
            mask += 1;
        }
        let mut f = 0;
        while f < 8 {
            z.ep[f] = splitmix64(&mut state);
            f += 1;
        }
        z
    }
}

pub static ZOBRIST: Zobrist = Zobrist::new();
