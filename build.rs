//! Build script: precomputes attack lookup tables so the engine can embed them
//! as plain `static` data (no runtime initialisation, no lazy-init checks).
//!
//! Outputs into `$OUT_DIR`:
//!   * `magics.rs`   - `ROOK_MAGICS` / `BISHOP_MAGICS` constants and the table length
//!   * `sliders.bin` - shared rook/bishop attack table ("fancy" magic bitboards)
//!   * `leapers.bin` - pawn (2x64), knight (64) and king (64) attacks
//!   * `lines.bin`   - BETWEEN (64x64) followed by LINE (64x64)
//!   * `net.bin`     - the evaluation network compiled into the engine
//!   * `net.rs`      - its shape: hidden layer width and king buckets
//!
//! Squares are numbered a1 = 0, b1 = 1, ..., h8 = 63.
//!
//! The network is `nets/gen4.bin` unless the environment says otherwise:
//!
//!     TABX_NET=nets/other.bin TABX_NET_HIDDEN=768 TABX_NET_LAYOUT=four cargo build --release
//!
//! `TABX_NET` is the file (relative to this directory, or absolute),
//! `TABX_NET_HIDDEN` the width of its hidden layer and `TABX_NET_LAYOUT` its
//! king buckets, "single" (one set of input weights) or "four" (see
//! `LAYOUT_FOUR`). They have to describe the file: if they do not, its size
//! does not fit and `src/nnue.rs` does not compile.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const KNIGHT_STEPS: [(i32, i32); 8] =
    [(2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)];
const KING_STEPS: [(i32, i32); 8] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// The network compiled in when the environment names no other (see above).
const DEFAULT_NET: &str = "nets/gen4.bin";
const DEFAULT_HIDDEN: usize = 256;
const DEFAULT_LAYOUT: &str = "single";
/// King bucket of each king square on files a-d, rank 1 first, for the
/// "four" layout: the corner, the rest of the back rank, the second rank, and
/// anywhere further up the board. Files e-h are mirrored.
#[rustfmt::skip]
const LAYOUT_FOUR: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_H: u64 = FILE_A << 7;
const RANK_1: u64 = 0xFF;
const RANK_8: u64 = 0xFF << 56;

fn on_board(rank: i32, file: i32) -> bool {
    (0..8).contains(&rank) && (0..8).contains(&file)
}

/// Attacks of a sliding piece on `sq` with blockers `occ`. The first blocker
/// in each direction is included in the attack set.
fn ray_attacks(sq: usize, occ: u64, dirs: &[(i32, i32); 4]) -> u64 {
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    let mut attacks = 0u64;
    for &(dr, df) in dirs {
        let (mut r, mut f) = (r0 + dr, f0 + df);
        while on_board(r, f) {
            let bit = 1u64 << (r * 8 + f);
            attacks |= bit;
            if occ & bit != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

fn step_attacks(sq: usize, steps: &[(i32, i32)]) -> u64 {
    let (r0, f0) = ((sq / 8) as i32, (sq % 8) as i32);
    let mut attacks = 0u64;
    for &(dr, df) in steps {
        let (r, f) = (r0 + dr, f0 + df);
        if on_board(r, f) {
            attacks |= 1u64 << (r * 8 + f);
        }
    }
    attacks
}

/// xorshift64* generator; deterministic so builds are reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Random number with few bits set; good magic candidates are sparse.
    fn sparse(&mut self) -> u64 {
        self.next() & self.next() & self.next()
    }
}

struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: u32,
}

/// Finds a magic multiplier for every square and appends the corresponding
/// attack sets to `table`.
fn build_magics(dirs: &[(i32, i32); 4], rng: &mut Rng, table: &mut Vec<u64>) -> Vec<Magic> {
    let mut out = Vec::with_capacity(64);
    for sq in 0..64usize {
        // Board edges never change the attack set, unless the piece is on that edge.
        let rank_bb = 0xFFu64 << (8 * (sq / 8));
        let file_bb = FILE_A << (sq % 8);
        let edges = ((RANK_1 | RANK_8) & !rank_bb) | ((FILE_A | FILE_H) & !file_bb);
        let mask = ray_attacks(sq, 0, dirs) & !edges;
        let bits = mask.count_ones();
        let size = 1usize << bits;
        let shift = 64 - bits;

        // Enumerate every blocker subset of the mask (carry-rippler trick).
        let mut occs = Vec::with_capacity(size);
        let mut refs = Vec::with_capacity(size);
        let mut subset = 0u64;
        loop {
            occs.push(subset);
            refs.push(ray_attacks(sq, subset, dirs));
            subset = subset.wrapping_sub(mask) & mask;
            if subset == 0 {
                break;
            }
        }

        let offset = table.len();
        table.resize(offset + size, 0);
        let mut epoch = vec![0u32; size];
        let mut attempt = 0u32;
        let magic = loop {
            let candidate = rng.sparse();
            if (candidate.wrapping_mul(mask) >> 56).count_ones() < 6 {
                continue;
            }
            attempt += 1;
            let slots = &mut table[offset..offset + size];
            let mut ok = true;
            for (&occ, &att) in occs.iter().zip(&refs) {
                let idx = (occ.wrapping_mul(candidate) >> shift) as usize;
                if epoch[idx] < attempt {
                    epoch[idx] = attempt;
                    slots[idx] = att;
                } else if slots[idx] != att {
                    ok = false;
                    break;
                }
            }
            if ok {
                break candidate;
            }
        };
        out.push(Magic { mask, magic, shift, offset: offset as u32 });
    }
    out
}

fn to_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out = Path::new(&out_dir);

    // --- sliders -----------------------------------------------------------
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut table = Vec::new();
    let rook = build_magics(&ROOK_DIRS, &mut rng, &mut table);
    let bishop = build_magics(&BISHOP_DIRS, &mut rng, &mut table);

    let mut src = String::new();
    writeln!(src, "pub const SLIDER_TABLE_LEN: usize = {};", table.len()).unwrap();
    for (name, magics) in [("ROOK_MAGICS", &rook), ("BISHOP_MAGICS", &bishop)] {
        writeln!(src, "pub static {name}: [Magic; 64] = [").unwrap();
        for m in magics {
            writeln!(
                src,
                "    Magic {{ mask: {:#018x}, magic: {:#018x}, shift: {}, offset: {} }},",
                m.mask, m.magic, m.shift, m.offset
            )
            .unwrap();
        }
        writeln!(src, "];").unwrap();
    }
    fs::write(out.join("magics.rs"), src).unwrap();
    fs::write(out.join("sliders.bin"), to_bytes(&table)).unwrap();

    // --- leapers -----------------------------------------------------------
    let mut leapers = Vec::with_capacity(256);
    for sq in 0..64 {
        leapers.push(step_attacks(sq, &[(1, -1), (1, 1)])); // white pawn
    }
    for sq in 0..64 {
        leapers.push(step_attacks(sq, &[(-1, -1), (-1, 1)])); // black pawn
    }
    for sq in 0..64 {
        leapers.push(step_attacks(sq, &KNIGHT_STEPS));
    }
    for sq in 0..64 {
        leapers.push(step_attacks(sq, &KING_STEPS));
    }
    fs::write(out.join("leapers.bin"), to_bytes(&leapers)).unwrap();

    // --- between / line ----------------------------------------------------
    let mut between = vec![0u64; 64 * 64];
    let mut line = vec![0u64; 64 * 64];
    for a in 0..64usize {
        for b in 0..64usize {
            if a == b {
                continue;
            }
            for dirs in [&ROOK_DIRS, &BISHOP_DIRS] {
                if ray_attacks(a, 0, dirs) & (1u64 << b) != 0 {
                    between[a * 64 + b] =
                        ray_attacks(a, 1u64 << b, dirs) & ray_attacks(b, 1u64 << a, dirs);
                    line[a * 64 + b] = (ray_attacks(a, 0, dirs) & ray_attacks(b, 0, dirs))
                        | (1u64 << a)
                        | (1u64 << b);
                }
            }
        }
    }
    between.extend_from_slice(&line);
    fs::write(out.join("lines.bin"), to_bytes(&between)).unwrap();

    // --- network -------------------------------------------------------------
    for name in ["TABX_NET", "TABX_NET_HIDDEN", "TABX_NET_LAYOUT"] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let net = env::var("TABX_NET").unwrap_or_else(|_| DEFAULT_NET.to_string());
    // An absolute path stays as it is; a relative one is taken from here.
    let net_path = Path::new(&manifest).join(&net);
    println!("cargo:rerun-if-changed={}", net_path.display());
    let bytes = fs::read(&net_path).unwrap_or_else(|e| panic!("cannot read the network {}: {e}", net_path.display()));
    let hidden = match env::var("TABX_NET_HIDDEN") {
        Ok(text) => text.parse().unwrap_or_else(|_| panic!("TABX_NET_HIDDEN must be a number, not {text:?}")),
        Err(_) => DEFAULT_HIDDEN,
    };
    let layout = env::var("TABX_NET_LAYOUT").unwrap_or_else(|_| DEFAULT_LAYOUT.to_string());
    let buckets = match layout.as_str() {
        "single" => [0; 32],
        "four" => LAYOUT_FOUR,
        other => panic!("TABX_NET_LAYOUT must be \"single\" or \"four\", not {other:?}"),
    };
    let mut src = String::new();
    writeln!(src, "/// Width of the network's hidden layer.").unwrap();
    writeln!(src, "pub const HIDDEN: usize = {hidden};").unwrap();
    writeln!(src, "/// King bucket for each king square on files a-d, from the viewer's point of").unwrap();
    writeln!(src, "/// view, rank 1 first. Files e-h use the mirrored square.").unwrap();
    writeln!(src, "const BUCKET_LAYOUT: [usize; 32] = {buckets:?};").unwrap();
    let stem = net_path.file_stem().and_then(|s| s.to_str()).unwrap_or("network");
    writeln!(src, "/// Name of the network file, for `uci` and `bench` output.").unwrap();
    writeln!(src, "pub const NET_NAME: &str = {stem:?};").unwrap();
    fs::write(out.join("net.rs"), src).unwrap();
    fs::write(out.join("net.bin"), bytes).unwrap();
}
