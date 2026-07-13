//! Forgetting curve — Ebbinghaus-style configurable decay model.
//!
//! This module provides [`ForgettingCurve`], a small, dependency-free abstraction
//! over the three retention models most commonly used to simulate agent memory
//! decay: Ebbinghaus exponential, power-law, and linear. Pick the model that
//! best matches the retention profile you want to emulate (see [`ForgettingModel`]).
//!
//! ## When to use this directly
//!
//! You rarely need to. The curve is wired into [`ShortTermMemory`](crate::ShortTermMemory)
//! by default and is consulted automatically on every recall. Construct a
//! [`ForgettingCurve`] directly only when you want a custom model, a non-default
//! half-life, or need to compute `retention(t)` / `time_until_threshold(threshold)`
//! outside of a memory store — for example, to schedule a periodic
//! [`MemoryConsolidation`](crate::MemoryConsolidation) pass.

use core::f64;

/// Preset forgetting models for [`ForgettingCurve`].
///
/// Each variant matches a well-known retention profile from cognitive
/// psychology; see the variant docs for the formula and typical use case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForgettingModel {
    /// Ebbinghaus exponential decay: `R = e^(-t/S)` where `S` is the memory
    /// stability (a half-life analog). At `t = S · ln(2) ≈ 0.693·S`, retention
    /// is exactly 0.5. Use this for biologically plausible memory decay — it is
    /// the default for [`ShortTermMemory`](crate::ShortTermMemory).
    Ebbinghaus {
        /// Memory stability. Larger = slower decay. Must be `> 0.0` for the
        /// curve to be meaningful; non-positive values make `retention` return
        /// `0.0` everywhere (treated as "no memory retained").
        stability: f64,
    },
    /// Power-law decay: `R = (1 + t)^(-alpha)`. Has a heavy tail — retention
    /// drops fast early and then lingers. Often fits skill/knowledge retention
    /// data better than exponential decay.
    PowerLaw {
        /// Exponent. Larger = faster decay. Must be `> 0.0`; non-positive values
        /// make `retention` return `1.0` everywhere (treated as "perfect memory").
        alpha: f64,
    },
    /// Linear decay: `R = 1 - (t / horizon)`, clamped to `[0, 1]`. Reaches
    /// exactly 0 at `t = horizon`. Use this for hard-deadline expiry semantics
    /// (e.g. "this credential is forgotten at exactly tick 1000").
    Linear {
        /// Tick at which retention reaches 0. Must be `> 0.0`; non-positive
        /// values make `retention` return `0.0` everywhere.
        horizon: f64,
    },
}

/// A configurable forgetting curve that computes retention over time.
///
/// Construct with [`ForgettingCurve::new`], the [`Default`] implementation
/// (Ebbinghaus with a 100-tick half-life), or the
/// [`ForgettingCurve::ebbinghaus_with_half_life`] convenience constructor.
#[derive(Debug, Clone)]
pub struct ForgettingCurve {
    model: ForgettingModel,
}

impl ForgettingCurve {
    /// Create a new forgetting curve with the given [`ForgettingModel`].
    #[must_use]
    pub fn new(model: ForgettingModel) -> Self {
        Self { model }
    }

    /// Convenience constructor for an [`ForgettingModel::Ebbinghaus`] curve
    /// parameterised by its half-life rather than its stability.
    ///
    /// At `t = half_life`, retention is exactly 0.5. Internally this solves
    /// `R = e^(-t/S)` for `S` given `R(t=half_life) = 0.5`, yielding
    /// `S = half_life / ln(2)`.
    #[must_use]
    pub fn ebbinghaus_with_half_life(half_life: f64) -> Self {
        let stability = half_life / f64::consts::LN_2;
        Self::new(ForgettingModel::Ebbinghaus { stability })
    }

    /// Compute retention probability at time `t` (the elapsed time since the
    /// memory was stored).
    ///
    /// Negative `t` is clamped to `0.0` (retention at or before the storage
    /// instant is `1.0`, modulo the degenerate-parameter rules documented on
    /// each [`ForgettingModel`] variant).
    ///
    /// The result is always in `[0.0, 1.0]` for well-formed parameters.
    #[must_use]
    pub fn retention(&self, t: f64) -> f64 {
        let t = if t < 0.0 { 0.0 } else { t };
        match self.model {
            ForgettingModel::Ebbinghaus { stability } => {
                if stability <= 0.0 {
                    return 0.0;
                }
                (-t / stability).exp()
            }
            ForgettingModel::PowerLaw { alpha } => {
                if alpha <= 0.0 {
                    return 1.0;
                }
                (1.0 + t).powf(-alpha)
            }
            ForgettingModel::Linear { horizon } => {
                if horizon <= 0.0 {
                    return 0.0;
                }
                (1.0 - t / horizon).clamp(0.0, 1.0)
            }
        }
    }

    /// Time at which retention drops below `threshold`.
    ///
    /// Returns `Some(0.0)` when retention is already below `threshold` at
    /// `t = 0` (including the degenerate-parameter cases), `Some(INFINITY)` when
    /// `threshold <= 0.0`, and `None` when retention never drops below the
    /// threshold (e.g. a [`ForgettingModel::PowerLaw`] with `alpha <= 0.0`,
    /// where retention is the constant `1.0`).
    #[must_use]
    pub fn time_until_threshold(&self, threshold: f64) -> Option<f64> {
        if threshold <= 0.0 {
            return Some(f64::INFINITY);
        }
        match self.model {
            ForgettingModel::Ebbinghaus { stability } => {
                if stability <= 0.0 {
                    return Some(0.0);
                }
                if threshold >= 1.0 {
                    return Some(0.0);
                }
                Some(-stability * threshold.ln())
            }
            ForgettingModel::PowerLaw { alpha } => {
                if alpha <= 0.0 {
                    return None;
                }
                if threshold >= 1.0 {
                    return Some(0.0);
                }
                Some(threshold.powf(-1.0 / alpha) - 1.0)
            }
            ForgettingModel::Linear { horizon } => {
                if horizon <= 0.0 {
                    // Retention is the constant 0.0 (see `retention`), so any
                    // positive threshold is already below it at t = 0.
                    return Some(0.0);
                }
                if threshold >= 1.0 {
                    return Some(0.0);
                }
                Some(horizon * (1.0 - threshold))
            }
        }
    }

    /// Returns a reference to the underlying [`ForgettingModel`].
    #[must_use]
    pub fn model(&self) -> &ForgettingModel {
        &self.model
    }
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self::ebbinghaus_with_half_life(100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ebbinghaus_half_life_is_exact() {
        // Sanity: at t = half_life, retention must be exactly 0.5.
        let curve = ForgettingCurve::ebbinghaus_with_half_life(100.0);
        assert!((curve.retention(100.0) - 0.5).abs() < 1e-12);
        assert!((curve.retention(0.0) - 1.0).abs() < 1e-12);
        // e^(-200/S) where S = 100/ln(2): = e^(-2 ln 2) = 1/4
        assert!((curve.retention(200.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn time_until_threshold_round_trips_ebbinghaus() {
        let curve = ForgettingCurve::ebbinghaus_with_half_life(50.0);
        let t = curve.time_until_threshold(0.25).unwrap();
        // retention(t) should equal the threshold (within float tolerance).
        assert!((curve.retention(t) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn time_until_threshold_round_trips_powerlaw() {
        let curve = ForgettingCurve::new(ForgettingModel::PowerLaw { alpha: 1.5 });
        let t = curve.time_until_threshold(0.3).unwrap();
        assert!((curve.retention(t) - 0.3).abs() < 1e-9);
    }

    #[test]
    fn linear_round_trips_and_clamps() {
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: 10.0 });
        assert!((curve.retention(0.0) - 1.0).abs() < 1e-12);
        assert!((curve.retention(5.0) - 0.5).abs() < 1e-12);
        assert!((curve.retention(10.0) - 0.0).abs() < 1e-12);
        // Past the horizon → clamped to 0 (complete forgetting).
        assert!((curve.retention(11.0) - 0.0).abs() < 1e-12);
        let t = curve.time_until_threshold(0.5).unwrap();
        assert!((t - 5.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_parameters_are_consistent() {
        // Ebbinghaus with non-positive stability: retention is the constant 0,
        // so any positive threshold is already below it at t = 0.
        let curve = ForgettingCurve::new(ForgettingModel::Ebbinghaus { stability: 0.0 });
        assert_eq!(curve.retention(0.0), 0.0);
        assert_eq!(curve.time_until_threshold(0.5), Some(0.0));

        // PowerLaw with non-positive alpha: retention is the constant 1.0, so
        // it never drops below any positive threshold.
        let curve = ForgettingCurve::new(ForgettingModel::PowerLaw { alpha: 0.0 });
        assert_eq!(curve.retention(0.0), 1.0);
        assert_eq!(curve.time_until_threshold(0.5), None);

        // Linear with non-positive horizon: retention is the constant 0.0
        // (previously the code returned None here — that contradicted
        // `retention`, which already said 0).
        let curve = ForgettingCurve::new(ForgettingModel::Linear { horizon: -1.0 });
        assert_eq!(curve.retention(0.0), 0.0);
        assert_eq!(curve.time_until_threshold(0.5), Some(0.0));
    }

    #[test]
    fn negative_time_is_clamped_to_zero() {
        let curve = ForgettingCurve::default();
        assert!((curve.retention(-5.0) - curve.retention(0.0)).abs() < 1e-12);
    }

    #[test]
    fn zero_threshold_is_infinity() {
        let curve = ForgettingCurve::default();
        assert_eq!(curve.time_until_threshold(0.0), Some(f64::INFINITY));
    }
}
