//! Long-term memory — compressed summary of all past experience.
//!
//! [`LongTermMemory`] keeps running statistics about observed outcomes keyed by
//! an arbitrary label (typically the action that produced the outcome), without
//! retaining the individual observations. This is the long-lived "what tends to
//! happen when I do X" half of an agent's memory.
//!
//! ## When to use this
//!
//! Use [`LongTermMemory`] when you want O(1) per-observation cost and a bounded
//! summary even after unbounded experience. Don't use it if you need to recall
//! individual past events — that's [`EpisodicMemory`](crate::EpisodicMemory) or
//! [`ShortTermMemory`](crate::ShortTermMemory).

/// A summary of accumulated experience across many decisions, keyed by label.
///
/// Statistics are maintained incrementally via Welford's online algorithm
/// (single-pass, numerically stable). See [`ExperienceSummary::observe`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExperienceSummary {
    /// Label for the experience category (typically the action name).
    pub label: String,
    /// Running count of observations.
    pub count: u64,
    /// Running mean outcome.
    pub mean_outcome: f64,
    /// Running sum of squared deviations from the mean (Welford's M₂).
    /// Divide by `count` (population) or `count − 1` (sample) to get variance.
    pub variance: f64,
    /// Best outcome observed.
    pub best_outcome: f64,
    /// Worst outcome observed.
    pub worst_outcome: f64,
}

impl ExperienceSummary {
    /// Create a new empty summary for the given label.
    ///
    /// `best_outcome` starts at `-inf` and `worst_outcome` at `+inf`; both
    /// collapse to the first observed value on the first call to
    /// [`observe`](Self::observe).
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            count: 0,
            mean_outcome: 0.0,
            variance: 0.0,
            best_outcome: f64::NEG_INFINITY,
            worst_outcome: f64::INFINITY,
        }
    }

    /// Observe a new outcome, updating running statistics via Welford's online
    /// algorithm (Welford, 1962).
    ///
    /// This is single-pass and numerically stable even for large counts and
    /// near-equal values, unlike the naive `Σx² − (Σx)²/n` formula.
    pub fn observe(&mut self, outcome: f64) {
        self.count += 1;
        let delta = outcome - self.mean_outcome;
        self.mean_outcome += delta / self.count as f64;
        if self.count > 1 {
            let delta2 = outcome - self.mean_outcome;
            self.variance += delta * delta2;
        }
        if outcome > self.best_outcome {
            self.best_outcome = outcome;
        }
        if outcome < self.worst_outcome {
            self.worst_outcome = outcome;
        }
    }

    /// Population variance: M₂ / count. Returns `0.0` for an empty summary.
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.variance / self.count as f64
        }
    }

    /// Sample variance (Bessel-corrected): M₂ / (count − 1). Returns `0.0`
    /// when fewer than two observations have been recorded.
    #[must_use]
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.variance / (self.count - 1) as f64
        }
    }

    /// Population standard deviation: `sqrt(population_variance)`.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// A simple confidence measure in `[0, 1)`: `1 − 1/(1 + √n)`.
    ///
    /// `n = 0  → 0`, `n = 1 → 0.5`, `n = 10 → ≈0.76`, `n = 100 → ≈0.91`,
    /// `n → ∞ → 1`. Use this to gate decisions on low-sample summaries.
    #[must_use]
    pub fn confidence(&self) -> f64 {
        1.0 - 1.0 / (1.0 + (self.count as f64).sqrt())
    }
}

/// Long-term memory: a collection of [`ExperienceSummary`]s keyed by label.
///
/// Construct with [`LongTermMemory::new`] and feed observations via
/// [`observe`](Self::observe). Each distinct label gets its own summary; use
/// [`get`](Self::get) / [`summaries`](Self::summaries) to inspect them.
#[derive(Debug, Clone)]
pub struct LongTermMemory {
    summaries: Vec<ExperienceSummary>,
}

impl LongTermMemory {
    /// Create empty long-term memory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            summaries: Vec::new(),
        }
    }

    /// Observe an outcome under a given label, creating the summary on first
    /// use. O(k) in the number of distinct labels (linear scan).
    pub fn observe(&mut self, label: &str, outcome: f64) {
        if let Some(summary) = self.summaries.iter_mut().find(|s| s.label == label) {
            summary.observe(outcome);
        } else {
            let mut summary = ExperienceSummary::new(label);
            summary.observe(outcome);
            self.summaries.push(summary);
        }
    }

    /// Get the summary for a label, if any.
    #[must_use]
    pub fn get(&self, label: &str) -> Option<&ExperienceSummary> {
        self.summaries.iter().find(|s| s.label == label)
    }

    /// Get the best-known label (highest mean outcome).
    ///
    /// Summaries whose mean is `NaN` are skipped — a `NaN` mean is treated as
    /// "no usable statistic" rather than silently winning the comparison.
    /// Returns `None` if there are no summaries or all means are `NaN`.
    #[must_use]
    pub fn best_label(&self) -> Option<&str> {
        self.summaries
            .iter()
            .filter(|s| !s.mean_outcome.is_nan())
            .max_by(|a, b| a.mean_outcome.total_cmp(&b.mean_outcome))
            .map(|s| s.label.as_str())
    }

    /// Number of distinct labels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.summaries.len()
    }

    /// Whether there are no summaries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty()
    }

    /// Iterate all summaries.
    pub fn summaries(&self) -> impl Iterator<Item = &ExperienceSummary> {
        self.summaries.iter()
    }

    /// Total observations across all summaries.
    #[must_use]
    pub fn total_observations(&self) -> u64 {
        self.summaries.iter().map(|s| s.count).sum()
    }

    /// Merge another long-term memory into this one.
    ///
    /// Counts are summed; means are combined as count-weighted averages; best
    /// and worst outcomes are the running max/min across both. Per-label
    /// variance is **not** reconstructed (it can't be from a summary alone), so
    /// the merged summary's variance is left as this side's value. Summaries
    /// for labels not already present are cloned verbatim.
    pub fn merge(&mut self, other: &LongTermMemory) {
        for summary in &other.summaries {
            if let Some(own) = self.summaries.iter_mut().find(|s| s.label == summary.label) {
                let total = own.count + summary.count;
                if total > 0 {
                    own.mean_outcome = (own.mean_outcome * own.count as f64
                        + summary.mean_outcome * summary.count as f64)
                        / total as f64;
                }
                own.count = total;
                own.best_outcome = own.best_outcome.max(summary.best_outcome);
                own.worst_outcome = own.worst_outcome.min(summary.worst_outcome);
            } else {
                self.summaries.push(summary.clone());
            }
        }
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observe_running_stats() {
        let mut s = ExperienceSummary::new("test");
        s.observe(2.0);
        s.observe(4.0);
        s.observe(6.0);
        assert_eq!(s.count, 3);
        assert!((s.mean_outcome - 4.0).abs() < 1e-9);
        assert!((s.best_outcome - 6.0).abs() < 1e-9);
        assert!((s.worst_outcome - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_population_variance() {
        let mut s = ExperienceSummary::new("v");
        s.observe(2.0);
        s.observe(4.0);
        s.observe(6.0);
        // Variance of [2,4,6] = ((2-4)^2 + (4-4)^2 + (6-4)^2)/3 = 8/3
        assert!((s.population_variance() - 8.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_variance_matches_bessel() {
        let mut s = ExperienceSummary::new("v");
        s.observe(2.0);
        s.observe(4.0);
        s.observe(6.0);
        // Sample variance: M2 / (n-1) = 8 / 2 = 4
        assert!((s.sample_variance() - 4.0).abs() < 1e-9);
        assert!((s.std_dev() - (8.0 / 3.0_f64).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_welford_matches_two_pass_on_tricky_input() {
        // Large nearly-equal values: the naive Σx²−(Σx)²/n formula loses
        // catastrophic precision here; Welford must not.
        let mut s = ExperienceSummary::new("big");
        let base = 1e9_f64;
        for dx in [0.0_f64, 1.0, 2.0, -1.0, 0.5] {
            s.observe(base + dx);
        }
        let mean = s.mean_outcome;
        let pop_var = s.population_variance();
        // Two-pass reference:
        let xs = [base, base + 1.0, base + 2.0, base - 1.0, base + 0.5];
        let ref_mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let ref_var = xs.iter().map(|x| (x - ref_mean).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(
            (mean - ref_mean).abs() < 1e-3,
            "mean drifted: {mean} vs {ref_mean}"
        );
        assert!(
            (pop_var - ref_var).abs() < 1e-3,
            "var drifted: {pop_var} vs {ref_var}"
        );
    }

    #[test]
    fn test_long_term_best_label() {
        let mut ltm = LongTermMemory::new();
        ltm.observe("explore", 0.3);
        ltm.observe("exploit", 0.9);
        ltm.observe("explore", 0.5);
        assert_eq!(ltm.best_label(), Some("exploit"));
    }

    #[test]
    fn test_best_label_ignores_nan() {
        // Regression: previously, partial_cmp().unwrap_or(Equal) silently let
        // a NaN-mean label "win" best_label. After the fix, NaN labels are
        // skipped and the real best label is returned.
        let mut ltm = LongTermMemory::new();
        ltm.observe("good", 0.9);
        ltm.observe("nan_label", f64::NAN);
        assert_eq!(ltm.best_label(), Some("good"));

        // If every label is NaN, there is no best.
        let mut ltm2 = LongTermMemory::new();
        ltm2.observe("only_nan", f64::NAN);
        assert_eq!(ltm2.best_label(), None);
    }

    #[test]
    fn test_merge() {
        let mut a = LongTermMemory::new();
        a.observe("x", 1.0);
        a.observe("x", 3.0);
        let mut b = LongTermMemory::new();
        b.observe("x", 5.0);
        a.merge(&b);
        assert_eq!(a.get("x").unwrap().count, 3);
        // mean = (1+3+5)/3 = 3
        assert!((a.get("x").unwrap().mean_outcome - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_increases_and_matches_formula() {
        // Formula: 1 - 1/(1 + sqrt(n)).
        let mut s = ExperienceSummary::new("c");
        assert!((s.confidence() - 0.0).abs() < 1e-12); // n=0
        s.observe(0.0);
        assert!((s.confidence() - 0.5).abs() < 1e-12); // n=1: 1 - 1/2
        let c1 = s.confidence();
        for i in 0..100 {
            s.observe(i as f64);
        }
        // n=101: 1 - 1/(1 + sqrt(101))
        let expected = 1.0 - 1.0 / (1.0 + (101.0_f64).sqrt());
        assert!((s.confidence() - expected).abs() < 1e-12);
        assert!(s.confidence() > c1);
    }

    #[test]
    fn test_empty_summary_extremes() {
        let s = ExperienceSummary::new("e");
        assert_eq!(s.count, 0);
        assert_eq!(s.population_variance(), 0.0);
        assert_eq!(s.sample_variance(), 0.0);
        assert_eq!(s.std_dev(), 0.0);
        assert!(s.best_outcome.is_infinite() && s.best_outcome.is_sign_negative());
        assert!(s.worst_outcome.is_infinite() && !s.worst_outcome.is_sign_negative());
    }

    #[test]
    fn test_single_observation_zero_variance() {
        let mut s = ExperienceSummary::new("one");
        s.observe(3.5);
        assert_eq!(s.count, 1);
        assert!((s.mean_outcome - 3.5).abs() < 1e-12);
        assert_eq!(s.population_variance(), 0.0);
        assert_eq!(s.sample_variance(), 0.0);
    }
}
