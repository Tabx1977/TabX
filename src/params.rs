//! Search parameters.
//!
//! In a normal build every parameter is a compile-time constant. Built with
//! `--features tune`, each one becomes a UCI option instead, so an automatic
//! tuner (`scripts/spsa.py`) can adjust the values of a running engine.
//!
//! Each line is: name = default, minimum, maximum, tuning step.

macro_rules! params {
    ($($(#[$doc:meta])* $name:ident = $default:expr, $min:expr, $max:expr, $step:expr;)*) => {
        $(
            $(#[$doc])*
            #[cfg(not(feature = "tune"))]
            #[inline(always)]
            pub const fn $name() -> i32 {
                $default
            }

            #[cfg(feature = "tune")]
            pub mod $name {
                pub static VALUE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new($default);
            }

            $(#[$doc])*
            #[cfg(feature = "tune")]
            #[inline(always)]
            pub fn $name() -> i32 {
                $name::VALUE.load(std::sync::atomic::Ordering::Relaxed)
            }
        )*

        /// Name, default, minimum, maximum and tuning step of every parameter.
        #[cfg(feature = "tune")]
        pub const LIST: &[(&str, i32, i32, i32, f64)] =
            &[$((stringify!($name), $default, $min, $max, $step as f64)),*];

        /// Sets a parameter by name; returns false if there is no such parameter.
        #[cfg(feature = "tune")]
        pub fn set(name: &str, value: i32) -> bool {
            $(
                if name == stringify!($name) {
                    $name::VALUE.store(value, std::sync::atomic::Ordering::Relaxed);
                    return true;
                }
            )*
            false
        }
    };
}

params! {
    /// Aspiration window half-width at the start of each iteration.
    aspiration_delta = 16, 5, 60, 3;

    /// Reverse futility pruning: margin per ply of depth.
    rfp_margin = 67, 30, 160, 8;
    rfp_max_depth = 7, 4, 14, 1;

    /// Razoring: margin per ply of depth.
    razor_margin = 256, 100, 500, 25;
    razor_max_depth = 1, 1, 6, 1;

    /// Null move pruning: reduction = base + depth / depth_div + min((eval - beta) / eval_div, 3).
    nmp_base = 4, 1, 6, 1;
    nmp_depth_div = 3, 2, 8, 1;
    nmp_eval_div = 208, 80, 400, 20;

    /// Late move reductions: reduction = base/100 + ln(depth) * ln(moves) / (divisor/100).
    lmr_base = 84, 0, 200, 10;
    lmr_divisor = 198, 120, 400, 15;
    /// History score that changes the reduction by one ply.
    lmr_history_div = 7483, 2000, 20000, 800;
    /// The same formula for captures and promotions, which are reduced less,
    /// and the capture history score that changes their reduction by one ply.
    lmr_noisy_base = -13, -100, 100, 10;
    lmr_noisy_divisor = 259, 150, 500, 15;
    lmr_noisy_history_div = 6034, 2000, 20000, 800;

    /// Late move pruning: quiet moves allowed = (base + depth^2) / (2 - improving).
    lmp_base = 4, 1, 10, 1;
    lmp_max_depth = 8, 4, 14, 1;

    /// Futility pruning: static eval + base + margin * depth <= alpha.
    fp_base = 85, 0, 300, 20;
    fp_margin = 64, 30, 200, 10;
    fp_max_depth = 9, 4, 14, 1;

    /// History pruning: skip quiets scoring below margin * depth.
    hp_margin = -2979, -8000, -500, 300;
    hp_max_depth = 4, 1, 8, 1;

    /// Static exchange pruning margins per ply of depth.
    see_quiet_margin = -58, -150, -10, 8;
    see_noisy_margin = -85, -250, -20, 12;
    see_max_depth = 8, 4, 14, 1;

    /// Singular extensions: singular beta = hash score - depth * beta_mult / 16.
    se_min_depth = 6, 5, 12, 1;
    se_beta_mult = 17, 6, 40, 2;
    se_double_margin = 24, 0, 60, 4;

    /// Internal iterative reductions.
    iir_min_depth = 6, 2, 8, 1;

    /// Quiescence futility margin on top of the captured piece's value.
    qs_futility_margin = 188, 50, 350, 15;

    /// History bonus = clamp(mult * depth - offset, 0, max).
    hist_bonus_mult = 177, 50, 400, 15;
    hist_bonus_offset = 46, -100, 300, 20;
    hist_bonus_max = 1431, 500, 4000, 150;

    /// Time management (all in hundredths): soft limit is scaled by
    /// (node_base - node_slope * best move's node share) *
    /// (stability_base - stability_step * iterations the best move was stable).
    tm_node_base = 190, 120, 280, 8;
    tm_node_slope = 130, 60, 220, 8;
    tm_stability_base = 125, 100, 180, 5;
    tm_stability_step = 7, 0, 15, 1;

    /// Time management, dividing the clock (`allocate_time` in uci.rs): the
    /// budget is the clock plus `horizon` increments, a move gets `share`
    /// thousandths of it, or `share_bare` thousandths when there is no
    /// increment. The first move of a game gets `ramp_floor` percent of
    /// that, rising to all of it after `ramp_plies` plies. The hard limit is
    /// `hard_mult` percent of the soft limit. When the engine thinks on the
    /// opponent's time (the UCI option Ponder), the soft limit is
    /// `ponder_scale` percent of what it would be.
    tm_horizon = 16, 4, 40, 3;
    tm_share = 40, 20, 70, 4;
    tm_share_bare = 30, 15, 50, 3;
    tm_ramp_floor = 50, 20, 100, 8;
    tm_ramp_plies = 24, 4, 60, 4;
    tm_hard_mult = 400, 200, 800, 40;
    tm_ponder_scale = 125, 100, 200, 10;
}
