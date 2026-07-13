//! # ternary-memory
//!
//! Multi-tier memory systems for ternary (and other) agents — short-term,
//! long-term, and episodic memory with context-tag indexing, configurable
//! forgetting curves, and periodic consolidation.
//!
//! ## When to use this crate
//!
//! Use `ternary-memory` when you're building an agent (game NPC, robot,
//! LLM-driven assistant, RL policy) that needs to **learn from experience**
//! — i.e. recall recent context, accumulate statistics about what works, and
//! remember noteworthy events. The three memory systems mirror the standard
//! cognitive-science taxonomy (Atkinson & Shiffrin, 1968; Tulving, 1972):
//!
//! | System | Rust type | Holds | Decays? |
//! |--------|-----------|-------|---------|
//! | Short-term | [`ShortTermMemory`] | Recent decisions (bounded ring) | Yes (forgetting curve) |
//! | Long-term | [`LongTermMemory`] | Per-label running statistics | No |
//! | Episodic | [`EpisodicMemory`] | Individual noteworthy events | No (FIFO evict) |
//!
//! Wire them together with [`MemoryConsolidation`], which drains STM into LTM
//! statistics and selects salient events for episodic storage. Index any of
//! them by structured context with [`MemoryIndex`].
//!
//! ## Quick start
//!
//! ```no_run
//! use ternary_memory::{
//!     Decision, Episode, EpisodeKind, EpisodicMemory, LongTermMemory,
//!     MemoryConsolidation, ShortTermMemory,
//! };
//!
//! // Short-term memory: bounded ring buffer with default Ebbinghaus decay.
//! let mut stm = ShortTermMemory::with_capacity(100);
//! stm.store(Decision::new("explore_north", 0.7, 1).with_tag("frontier"));
//! stm.store(Decision::new("attack_early", -0.3, 2).with_tag("combat"));
//!
//! // Long-term memory: per-label running statistics.
//! let mut ltm = LongTermMemory::new();
//! ltm.observe("explore_north", 0.7);
//! ltm.observe("explore_north", 0.5);
//! if let Some(summary) = ltm.get("explore_north") {
//!     println!("mean={:.2}, std={:.2}, n={}", summary.mean_outcome, summary.std_dev(), summary.count);
//! }
//!
//! // Episodic memory: noteworthy events (capacity-bounded).
//! let mut episodic = EpisodicMemory::new(1000);
//! episodic.store(Episode::new("Found gold mine!", EpisodeKind::Breakthrough, 5, 0.95));
//!
//! // Periodic consolidation: STM → LTM + Episodic.
//! let mut consolidation = MemoryConsolidation::new();
//! let result = consolidation.consolidate(&mut stm, &mut ltm, &mut episodic);
//! println!("Consolidated {} entries, found {} episodes",
//!     result.consolidated_count, result.new_episodes);
//! ```
//!
//! ## Architecture
//!
//! - **Short-term** ([`ShortTermMemory`]): Fixed-size ring buffer of recent
//!   decisions; retention computed lazily on recall via a
//!   [`ForgettingCurve`].
//! - **Long-term** ([`LongTermMemory`]): Compressed summary of all past
//!   experience per label (running mean, variance, extremes) using Welford's
//!   numerically stable online algorithm.
//! - **Episodic** ([`EpisodicMemory`]): Bounded FIFO store of salient
//!   individual events.
//! - **Index** ([`MemoryIndex`]): Generic tag-indexed retrieval with
//!   AND/OR query semantics.
//! - **Forgetting** ([`ForgettingCurve`]/[`ForgettingModel`]): Ebbinghaus,
//!   power-law, or linear decay.
//! - **Consolidation** ([`MemoryConsolidation`]): Periodic STM → LTM +
//!   episodic transfer.

mod consolidation;
mod episodic;
mod forgetting;
mod index;
mod long_term;
mod short_term;

pub use consolidation::{ConsolidationConfig, ConsolidationResult, MemoryConsolidation};
pub use episodic::{Episode, EpisodeKind, EpisodicMemory};
pub use forgetting::{ForgettingCurve, ForgettingModel};
pub use index::{ContextTag, IndexedMemory, MemoryIndex};
pub use long_term::{ExperienceSummary, LongTermMemory};
pub use short_term::{Decision, MemoryEntry, ShortTermMemory};
