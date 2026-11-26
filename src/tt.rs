//! Transposition table shared by all search threads.
//!
//! Entries are 10 bytes, three to a 32-byte cluster. Everything is stored in
//! atomics and accessed with relaxed ordering: on x86 that compiles to plain
//! loads and stores, and it keeps concurrent access free of data races. A
//! reader can still see an entry that another thread overwrote for a different
//! position (the 16-bit key check is not perfect), so the search treats every
//! entry as a hint: moves from the table are validated before use.

use crate::types::Move;
use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU64, Ordering::Relaxed};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    None = 0,
    /// The true score is at most the stored score (search failed low).
    Upper = 1,
    /// The true score is at least the stored score (search failed high).
    Lower = 2,
    Exact = 3,
}

/// Stored instead of a static evaluation when there is none (e.g. in check).
pub const NO_EVAL: i32 = i16::MIN as i32;

/// Search depths are stored with this offset so that 0 can mean "empty slot".
const DEPTH_OFFSET: i32 = 2;
const AGE_BITS: u32 = 5;
const AGE_MASK: u8 = (1 << AGE_BITS) - 1;
const ENTRIES_PER_CLUSTER: usize = 3;

#[derive(Copy, Clone)]
pub struct TtEntry {
    pub mv: Move,
    pub score: i32,
    pub eval: i32,
    pub depth: i32,
    pub bound: Bound,
}

/// Packed entry word: key (16) | move (16) | score (16) | depth (8) | flags (8),
/// where flags = bound (2) | was-on-principal-variation (1, stored but not
/// used by the search yet) | age (5).
#[repr(C, align(32))]
struct Cluster {
    data: [AtomicU64; ENTRIES_PER_CLUSTER],
    evals: [AtomicI16; ENTRIES_PER_CLUSTER],
    _pad: [u8; 2],
}

const _: () = assert!(std::mem::size_of::<Cluster>() == 32);

#[inline(always)]
fn depth8(data: u64) -> u8 {
    (data >> 48) as u8
}

#[inline(always)]
fn flags(data: u64) -> u8 {
    (data >> 56) as u8
}

pub struct TranspositionTable {
    clusters: Box<[Cluster]>,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new(megabytes: usize) -> TranspositionTable {
        TranspositionTable { clusters: allocate(megabytes), age: AtomicU8::new(0) }
    }

    pub fn resize(&mut self, megabytes: usize) {
        // Free the old table first so both never exist at the same time.
        self.clusters = Box::new([]);
        self.clusters = allocate(megabytes);
        self.age.store(0, Relaxed);
    }

    pub fn clear(&mut self) {
        for cluster in self.clusters.iter_mut() {
            for i in 0..ENTRIES_PER_CLUSTER {
                *cluster.data[i].get_mut() = 0;
                *cluster.evals[i].get_mut() = 0;
            }
        }
        self.age.store(0, Relaxed);
    }

    /// Marks the start of a new search; older entries become preferred replacement victims.
    pub fn new_search(&self) {
        self.age.store((self.age.load(Relaxed) + 1) & AGE_MASK, Relaxed);
    }

    #[inline(always)]
    fn cluster(&self, hash: u64) -> &Cluster {
        // Maps the hash uniformly onto the table without a modulo.
        let index = ((hash as u128 * self.clusters.len() as u128) >> 64) as usize;
        // SAFETY: the product of a 64-bit hash and `len`, shifted right by 64, is below `len`.
        unsafe { self.clusters.get_unchecked(index) }
    }

    /// Hints the CPU to load the cluster for `hash` into cache.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetching is a hint and never faults, whatever the address.
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T0>(self.cluster(hash) as *const Cluster as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }

    pub fn probe(&self, hash: u64) -> Option<TtEntry> {
        let cluster = self.cluster(hash);
        let key = hash as u16;
        for i in 0..ENTRIES_PER_CLUSTER {
            let data = cluster.data[i].load(Relaxed);
            if data as u16 == key && depth8(data) != 0 {
                let entry_flags = flags(data);
                // Refresh the age so entries that are still useful are kept.
                let age = self.age.load(Relaxed);
                if entry_flags >> 3 != age {
                    let refreshed = (entry_flags & 7) | (age << 3);
                    cluster.data[i].store((data & !(0xFF << 56)) | (refreshed as u64) << 56, Relaxed);
                }
                return Some(TtEntry {
                    mv: Move((data >> 16) as u16),
                    score: (data >> 32) as u16 as i16 as i32,
                    eval: cluster.evals[i].load(Relaxed) as i32,
                    depth: depth8(data) as i32 - DEPTH_OFFSET,
                    bound: match entry_flags & 3 {
                        1 => Bound::Upper,
                        2 => Bound::Lower,
                        3 => Bound::Exact,
                        _ => Bound::None,
                    },
                });
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store(&self, hash: u64, mv: Move, score: i32, eval: i32, depth: i32, bound: Bound, pv: bool) {
        let cluster = self.cluster(hash);
        let key = hash as u16;
        let age = self.age.load(Relaxed);

        // Use the slot already holding this position, else an empty one, else
        // the least valuable one (shallow and old entries go first).
        let worth = |data: u64| {
            let relative_age = (age.wrapping_sub(flags(data) >> 3)) & AGE_MASK;
            depth8(data) as i32 - 8 * relative_age as i32
        };
        let mut slot = 0;
        let mut old = cluster.data[0].load(Relaxed);
        for (i, entry) in cluster.data.iter().enumerate() {
            let data = entry.load(Relaxed);
            if depth8(data) == 0 || data as u16 == key {
                slot = i;
                old = data;
                break;
            }
            if worth(data) < worth(old) {
                slot = i;
                old = data;
            }
        }

        let same_position = depth8(old) != 0 && old as u16 == key;
        // Keep the old best move if this search did not find one.
        let mv = if mv.is_none() && same_position { Move((old >> 16) as u16) } else { mv };
        let new_depth8 = (depth + DEPTH_OFFSET).clamp(1, 255);

        // Do not let a shallow non-exact result overwrite a much deeper one for the same position.
        let overwrite = bound == Bound::Exact
            || !same_position
            || new_depth8 + 2 * pv as i32 > depth8(old) as i32 - 4
            || flags(old) >> 3 != age;
        if overwrite {
            let new_flags = bound as u8 | (pv as u8) << 2 | age << 3;
            let data = key as u64
                | (mv.0 as u64) << 16
                | (score as i16 as u16 as u64) << 32
                | (new_depth8 as u64) << 48
                | (new_flags as u64) << 56;
            cluster.data[slot].store(data, Relaxed);
            cluster.evals[slot].store(eval as i16, Relaxed);
        } else if (old >> 16) as u16 != mv.0 {
            cluster.data[slot].store((old & !(0xFFFF << 16)) | (mv.0 as u64) << 16, Relaxed);
        }
    }

    /// Approximate fill level in permille, for the UCI `hashfull` info field.
    pub fn hashfull(&self) -> usize {
        let age = self.age.load(Relaxed);
        let sample = self.clusters.len().min(1000);
        let mut used = 0;
        for cluster in &self.clusters[..sample] {
            for slot in &cluster.data {
                let data = slot.load(Relaxed);
                if depth8(data) != 0 && flags(data) >> 3 == age {
                    used += 1;
                }
            }
        }
        used * 1000 / (sample * ENTRIES_PER_CLUSTER).max(1)
    }
}

fn allocate(megabytes: usize) -> Box<[Cluster]> {
    let count = (megabytes.max(1) * 1024 * 1024 / std::mem::size_of::<Cluster>()).max(1);
    let layout = Layout::array::<Cluster>(count).expect("transposition table size overflows");
    // SAFETY: the layout has non-zero size; zeroed memory is a valid `Cluster`
    // (atomics and padding bytes); the pointer comes from the global allocator
    // with exactly the layout `Box<[Cluster]>` will free it with.
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut Cluster;
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, count))
    }
}
