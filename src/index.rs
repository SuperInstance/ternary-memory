//! Memory index — index memories by context tags for fast retrieval.

/// A context tag used to index memories.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextTag {
    pub key: String,
    pub value: String,
}

impl ContextTag {
    /// Create a new context tag.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self { key: key.into(), value: value.into() }
    }
}

/// A generic indexed memory entry.
#[derive(Debug, Clone)]
pub struct IndexedMemory<T: Clone> {
    pub item: T,
    pub tags: Vec<ContextTag>,
    pub relevance: f64,
}

/// Memory index: maps context tags to items for fast retrieval.
#[derive(Debug, Clone)]
pub struct MemoryIndex<T: Clone> {
    entries: Vec<IndexedMemory<T>>,
}

impl<T: Clone> MemoryIndex<T> {
    /// Create an empty index.
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Insert an item with associated context tags.
    pub fn insert(&mut self, item: T, tags: Vec<ContextTag>, relevance: f64) {
        self.entries.push(IndexedMemory { item, tags, relevance });
    }

    /// Query items matching ALL given tags, sorted by relevance (descending).
    pub fn query_all(&self, tags: &[ContextTag]) -> Vec<&IndexedMemory<T>> {
        let mut results: Vec<&IndexedMemory<T>> = self.entries
            .iter()
            .filter(|e| {
                tags.iter().all(|t| {
                    e.tags.iter().any(|et| et == t)
                })
            })
            .collect();
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(core::cmp::Ordering::Equal));
        results
    }

    /// Query items matching ANY of the given tags, sorted by relevance.
    pub fn query_any(&self, tags: &[ContextTag]) -> Vec<&IndexedMemory<T>> {
        let mut results: Vec<&IndexedMemory<T>> = self.entries
            .iter()
            .filter(|e| {
                tags.iter().any(|t| {
                    e.tags.iter().any(|et| et == t)
                })
            })
            .collect();
        results.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(core::cmp::Ordering::Equal));
        results
    }

    /// Return all entries.
    pub fn all(&self) -> &[IndexedMemory<T>] {
        &self.entries
    }

    /// Number of indexed entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove entries where a predicate returns false.
    pub fn retain(&mut self, f: impl Fn(&IndexedMemory<T>) -> bool) {
        self.entries.retain(f);
    }
}

impl<T: Clone> Default for MemoryIndex<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_query_all() {
        let mut idx: MemoryIndex<String> = MemoryIndex::new();
        idx.insert("alpha".into(), vec![ContextTag::new("env", "forest"), ContextTag::new("type", "combat")], 0.9);
        idx.insert("beta".into(), vec![ContextTag::new("env", "desert")], 0.5);
        let results = idx.query_all(&[ContextTag::new("env", "forest")]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, "alpha");
    }

    #[test]
    fn test_query_any() {
        let mut idx: MemoryIndex<String> = MemoryIndex::new();
        idx.insert("a".into(), vec![ContextTag::new("x", "1")], 0.5);
        idx.insert("b".into(), vec![ContextTag::new("y", "2")], 0.8);
        let results = idx.query_any(&[ContextTag::new("x", "1"), ContextTag::new("y", "2")]);
        assert_eq!(results.len(), 2);
        // sorted by relevance desc
        assert_eq!(results[0].item, "b");
    }

    #[test]
    fn test_retain() {
        let mut idx: MemoryIndex<i32> = MemoryIndex::new();
        idx.insert(1, vec![ContextTag::new("k", "v")], 0.5);
        idx.insert(2, vec![ContextTag::new("k", "v")], 0.9);
        idx.retain(|e| e.relevance > 0.7);
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.all()[0].item, 2);
    }
}
