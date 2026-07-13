//! Memory index — index memories by context tags for fast retrieval.
//!
//! [`MemoryIndex<T>`] is a small generic in-memory tag index. Insert arbitrary
//! items along with a list of [`ContextTag`]s and a relevance score, then query
//! either with AND semantics ([`query_all`](MemoryIndex::query_all)) or OR
//! semantics ([`query_any`](MemoryIndex::query_any)).
//!
//! ## When to use this
//!
//! Use [`MemoryIndex<T>`] when you need to recall items by structured context
//! (env=forest, type=combat, …) rather than by recency or salience. It is
//! deliberately generic over `T` so you can index any type — strings,
//! [`Decision`](crate::Decision)s, [`Episode`](crate::Episode)s, or your own.

/// A context tag used to index memories: a `(key, value)` pair such as
/// `("env", "forest")`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ContextTag {
    /// Tag namespace / key (e.g. `"env"`).
    pub key: String,
    /// Tag value within the namespace (e.g. `"forest"`).
    pub value: String,
}

impl ContextTag {
    /// Create a new context tag.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A generic indexed memory entry.
#[derive(Debug, Clone)]
pub struct IndexedMemory<T: Clone> {
    /// The indexed item.
    pub item: T,
    /// Context tags associated with the item.
    pub tags: Vec<ContextTag>,
    /// Relevance score used to rank query results (higher = more relevant).
    pub relevance: f64,
}

/// Memory index: maps context tags to items for fast retrieval.
///
/// Construct with [`MemoryIndex::new`] (or [`Default::default`]) and populate
/// via [`insert`](Self::insert). Queries return references sorted by relevance
/// in descending order; entries whose relevance is `NaN` are skipped (a `NaN`
/// relevance is treated as "unrankable" rather than silently winning).
#[derive(Debug, Clone)]
pub struct MemoryIndex<T: Clone> {
    entries: Vec<IndexedMemory<T>>,
}

impl<T: Clone> MemoryIndex<T> {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert an item with associated context tags and a relevance score.
    pub fn insert(&mut self, item: T, tags: Vec<ContextTag>, relevance: f64) {
        self.entries.push(IndexedMemory {
            item,
            tags,
            relevance,
        });
    }

    /// Query items matching ALL given tags, sorted by relevance (descending).
    ///
    /// Entries whose `relevance` is `NaN` are skipped.
    pub fn query_all(&self, tags: &[ContextTag]) -> Vec<&IndexedMemory<T>> {
        let mut results: Vec<&IndexedMemory<T>> = self
            .entries
            .iter()
            .filter(|e| tags.iter().all(|t| e.tags.iter().any(|et| et == t)))
            .filter(|e| !e.relevance.is_nan())
            .collect();
        results.sort_by(|a, b| b.relevance.total_cmp(&a.relevance));
        results
    }

    /// Query items matching ANY of the given tags, sorted by relevance
    /// (descending). Entries whose `relevance` is `NaN` are skipped.
    pub fn query_any(&self, tags: &[ContextTag]) -> Vec<&IndexedMemory<T>> {
        let mut results: Vec<&IndexedMemory<T>> = self
            .entries
            .iter()
            .filter(|e| tags.iter().any(|t| e.tags.iter().any(|et| et == t)))
            .filter(|e| !e.relevance.is_nan())
            .collect();
        results.sort_by(|a, b| b.relevance.total_cmp(&a.relevance));
        results
    }

    /// Return all entries, in insertion order.
    #[must_use]
    pub fn all(&self) -> &[IndexedMemory<T>] {
        &self.entries
    }

    /// Number of indexed entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    #[must_use]
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
        idx.insert(
            "alpha".into(),
            vec![
                ContextTag::new("env", "forest"),
                ContextTag::new("type", "combat"),
            ],
            0.9,
        );
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

    #[test]
    fn test_query_skips_nan_relevance() {
        // Regression: previously NaN relevance silently won the sort via the
        // partial_cmp().unwrap_or(Equal) fallback. After the fix, NaN entries
        // are skipped so real numeric relevances win.
        let mut idx: MemoryIndex<&'static str> = MemoryIndex::new();
        idx.insert("real", vec![ContextTag::new("k", "v")], 0.5);
        idx.insert("nan", vec![ContextTag::new("k", "v")], f64::NAN);
        let results = idx.query_all(&[ContextTag::new("k", "v")]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item, "real");
    }

    #[test]
    fn test_empty_index_query_returns_empty() {
        let idx: MemoryIndex<i32> = MemoryIndex::new();
        assert!(idx.is_empty());
        assert!(idx.query_all(&[ContextTag::new("any", "any")]).is_empty());
        assert!(idx.query_any(&[]).is_empty());
    }
}
