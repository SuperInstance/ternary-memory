//! Episodic memory — store and recall specific important episodes.

/// The kind of episode — why it was important enough to store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeKind {
    /// A near-miss (almost bad outcome).
    NearMiss,
    /// A breakthrough (exceptionally good outcome).
    Breakthrough,
    /// A surprise (unexpected outcome).
    Surprise,
    /// User-defined kind.
    Custom(String),
}

impl core::fmt::Display for EpisodeKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EpisodeKind::NearMiss => write!(f, "near_miss"),
            EpisodeKind::Breakthrough => write!(f, "breakthrough"),
            EpisodeKind::Surprise => write!(f, "surprise"),
            EpisodeKind::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

/// A specific episodic memory.
#[derive(Debug, Clone)]
pub struct Episode {
    /// What happened (description).
    pub description: String,
    /// The kind of episode.
    pub kind: EpisodeKind,
    /// Tick when the episode occurred.
    pub tick: u64,
    /// Outcome score.
    pub outcome: f64,
    /// Context tags.
    pub tags: Vec<String>,
}

impl Episode {
    /// Create a new episode.
    pub fn new(description: impl Into<String>, kind: EpisodeKind, tick: u64, outcome: f64) -> Self {
        Self {
            description: description.into(),
            kind,
            tick,
            outcome,
            tags: Vec::new(),
        }
    }

    /// Add a context tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

/// Episodic memory stores and retrieves specific important episodes.
#[derive(Debug, Clone)]
pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    max_episodes: usize,
}

impl EpisodicMemory {
    /// Create episodic memory with a maximum capacity.
    /// When capacity is reached, the oldest episode is evicted.
    pub fn new(max_episodes: usize) -> Self {
        Self {
            episodes: Vec::with_capacity(max_episodes),
            max_episodes,
        }
    }

    /// Create with unlimited capacity.
    pub fn unlimited() -> Self {
        Self {
            episodes: Vec::new(),
            max_episodes: usize::MAX,
        }
    }

    /// Store an episode.
    pub fn store(&mut self, episode: Episode) {
        if self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
        }
        self.episodes.push(episode);
    }

    /// Recall all episodes, most recent first.
    pub fn recall_all(&self) -> &[Episode] {
        &self.episodes
    }

    /// Recall episodes of a specific kind.
    pub fn recall_by_kind(&self, kind: &EpisodeKind) -> Vec<&Episode> {
        self.episodes.iter().filter(|e| &e.kind == kind).collect()
    }

    /// Recall episodes matching any of the given tags.
    pub fn recall_by_tags(&self, tags: &[&str]) -> Vec<&Episode> {
        self.episodes
            .iter()
            .filter(|e| e.tags.iter().any(|t| tags.contains(&t.as_str())))
            .collect()
    }

    /// Recall the top-N episodes by outcome.
    pub fn recall_top(&self, n: usize) -> Vec<&Episode> {
        let mut refs: Vec<&Episode> = self.episodes.iter().collect();
        refs.sort_by(|a, b| b.outcome.partial_cmp(&a.outcome).unwrap_or(core::cmp::Ordering::Equal));
        refs.truncate(n);
        refs
    }

    /// Number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Clear all episodes.
    pub fn clear(&mut self) {
        self.episodes.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() {
        let mut em = EpisodicMemory::unlimited();
        em.store(Episode::new("almost fell", EpisodeKind::NearMiss, 10, -0.5));
        em.store(Episode::new("found gold", EpisodeKind::Breakthrough, 20, 1.0));
        assert_eq!(em.len(), 2);
    }

    #[test]
    fn test_recall_by_kind() {
        let mut em = EpisodicMemory::unlimited();
        em.store(Episode::new("a", EpisodeKind::NearMiss, 1, -0.5));
        em.store(Episode::new("b", EpisodeKind::Breakthrough, 2, 1.0));
        em.store(Episode::new("c", EpisodeKind::NearMiss, 3, -0.3));
        let nm = em.recall_by_kind(&EpisodeKind::NearMiss);
        assert_eq!(nm.len(), 2);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut em = EpisodicMemory::new(2);
        em.store(Episode::new("first", EpisodeKind::Surprise, 1, 0.0));
        em.store(Episode::new("second", EpisodeKind::Surprise, 2, 0.0));
        em.store(Episode::new("third", EpisodeKind::Surprise, 3, 0.0));
        assert_eq!(em.len(), 2);
        assert_eq!(em.recall_all()[0].description, "second");
    }

    #[test]
    fn test_recall_by_tags() {
        let mut em = EpisodicMemory::unlimited();
        em.store(Episode::new("a", EpisodeKind::NearMiss, 1, 0.0).with_tag("combat"));
        em.store(Episode::new("b", EpisodeKind::Breakthrough, 2, 0.0).with_tag("exploration"));
        let combat = em.recall_by_tags(&["combat"]);
        assert_eq!(combat.len(), 1);
        assert_eq!(combat[0].description, "a");
    }

    #[test]
    fn test_recall_top() {
        let mut em = EpisodicMemory::unlimited();
        em.store(Episode::new("low", EpisodeKind::Surprise, 1, 0.1));
        em.store(Episode::new("mid", EpisodeKind::Surprise, 2, 0.5));
        em.store(Episode::new("high", EpisodeKind::Surprise, 3, 0.9));
        let top = em.recall_top(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].description, "high");
        assert_eq!(top[1].description, "mid");
    }
}
