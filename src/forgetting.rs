//! Forgetting curve — Ebbinghaus-style configurable decay model.

use core::f64;

/// Preset forgetting models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForgettingModel {
    /// Ebbinghaus exponential decay: `R = e^(-t/S)`
    /// where S is memory stability (half-life analog).
    Ebbinghaus { stability: f64 },
    /// Power-law decay: `R = (1 + t)^(-alpha)`
    PowerLaw { alpha: f64 },
    /// Linear decay: `R = 1 - (t / horizon)`, clamped to [0, 1].
    Linear { horizon: f64 },
}

/// A configurable forgetting curve that computes retention over time.
#[derive(Debug, Clone)]
pub struct ForgettingCurve {
    model: ForgettingModel,
}

impl ForgettingCurve {
    /// Create a new forgetting curve with the given model.
    pub fn new(model: ForgettingModel) -> Self {
        Self { model }
    }

    /// Convenience: Ebbinghaus model with a given half-life.
    /// At `t = half_life`, retention is ~0.5.
    pub fn ebbinghaus_with_half_life(half_life: f64) -> Self {
        // R = e^(-t * ln(2) / half_life) => stability = half_life / ln(2)
        let stability = half_life / f64::consts::LN_2;
        Self::new(ForgettingModel::Ebbinghaus { stability })
    }

    /// Compute retention probability at time `t` (non-negative).
    pub fn retention(&self, t: f64) -> f64 {
        let t = if t < 0.0 { 0.0 } else { t };
        match self.model {
            ForgettingModel::Ebbinghaus { stability } => {
                if stability <= 0.0 { return 0.0; }
                (-t / stability).exp()
            }
            ForgettingModel::PowerLaw { alpha } => {
                if alpha <= 0.0 { return 1.0; }
                (1.0 + t).powf(-alpha)
            }
            ForgettingModel::Linear { horizon } => {
                if horizon <= 0.0 { return 0.0; }
                (1.0 - t / horizon).max(0.0).min(1.0)
            }
        }
    }

    /// Time at which retention drops below `threshold` (0 < threshold <= 1).
    /// Returns `None` if retention never drops below threshold.
    pub fn time_until_threshold(&self, threshold: f64) -> Option<f64> {
        if threshold <= 0.0 {
            return Some(f64::INFINITY);
        }
        match self.model {
            ForgettingModel::Ebbinghaus { stability } => {
                if stability <= 0.0 { return Some(0.0); }
                if threshold >= 1.0 { return Some(0.0); }
                Some(-stability * threshold.ln())
            }
            ForgettingModel::PowerLaw { alpha } => {
                if alpha <= 0.0 { return None; }
                if threshold >= 1.0 { return Some(0.0); }
                Some(threshold.powf(-1.0 / alpha) - 1.0)
            }
            ForgettingModel::Linear { horizon } => {
                if threshold >= 1.0 { return Some(0.0); }
                let t = horizon * (1.0 - threshold);
                if t < 0.0 { None } else { Some(t) }
            }
        }
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &ForgettingModel {
        &self.model
    }
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self::ebbinghaus_with_half_life(100.0)
    }
}
