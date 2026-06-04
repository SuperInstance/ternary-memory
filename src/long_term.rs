//! Long-term memory — compressed summary of all past experience.

/// A summary of accumulated experience across many decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperienceSummary {
    /// Label for the experience category.
    pub label: String,
    /// Running count of observations.
    pub count: u64,
    /// Running mean outcome.
    pub mean_outcome: f64,
    /// Running variance (population).
    pub variance: f64,
    /// Best outcome observed.
    pub best_outcome: f64,
    /// Worst outcome observed.
    pub worst_outcome: f64,
}

impl ExperienceSummary {
    /// Create a new empty summary.
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

    /// Observe a new outcome, updating running statistics (Welford's online algorithm).
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

    /// Population variance.
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.variance / self.count as f64 }
    }

    /// Sample variance (unbiased).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.variance / (self.count - 1) as f64 }
    }

    /// Standard deviation (population).
    pub fn std_dev(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// Confidence: how many observations back this summary.
    pub fn confidence(&self) -> f64 {
        // Simple confidence measure: 1 - 1/(1 + sqrt(count))
        1.0 - 1.0 / (1.0 + (self.count as f64).sqrt())
    }
}

/// Long-term memory: a collection of experience summaries keyed by label.
#[derive(Debug, Clone)]
pub struct LongTermMemory {
    summaries: Vec<ExperienceSummary>,
}

impl LongTermMemory {
    /// Create empty long-term memory.
    pub fn new() -> Self {
        Self { summaries: Vec::new() }
    }

    /// Observe an outcome under a given label, creating the summary if needed.
    pub fn observe(&mut self, label: &str, outcome: f64) {
        if let Some(summary) = self.summaries.iter_mut().find(|s| s.label == label) {
            summary.observe(outcome);
        } else {
            let mut summary = ExperienceSummary::new(label);
            summary.observe(outcome);
            self.summaries.push(summary);
        }
    }

    /// Get a summary by label.
    pub fn get(&self, label: &str) -> Option<&ExperienceSummary> {
        self.summaries.iter().find(|s| s.label == label)
    }

    /// Get the best-known label (highest mean outcome).
    pub fn best_label(&self) -> Option<&str> {
        self.summaries
            .iter()
            .max_by(|a, b| a.mean_outcome.partial_cmp(&b.mean_outcome).unwrap_or(core::cmp::Ordering::Equal))
            .map(|s| s.label.as_str())
    }

    /// Number of distinct labels.
    pub fn len(&self) -> usize {
        self.summaries.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.summaries.is_empty()
    }

    /// Iterate all summaries.
    pub fn summaries(&self) -> impl Iterator<Item = &ExperienceSummary> {
        self.summaries.iter()
    }

    /// Total observations across all summaries.
    pub fn total_observations(&self) -> u64 {
        self.summaries.iter().map(|s| s.count).sum()
    }

    /// Merge another long-term memory into this one (summing counts, averaging means).
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
    fn test_long_term_best_label() {
        let mut ltm = LongTermMemory::new();
        ltm.observe("explore", 0.3);
        ltm.observe("exploit", 0.9);
        ltm.observe("explore", 0.5);
        assert_eq!(ltm.best_label(), Some("exploit"));
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
    fn test_confidence_increases() {
        let mut s = ExperienceSummary::new("c");
        let c1 = s.confidence();
        for i in 0..100 {
            s.observe(i as f64);
        }
        assert!(s.confidence() > c1);
    }
}
