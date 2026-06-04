//! # ternary-memory
//!
//! Memory systems for ternary agents — short-term, long-term, and episodic memory
//! that agents use to make better decisions.
//!
//! ## Architecture
//!
//! - **ShortTermMemory**: Fixed-size ring buffer of recent decisions with time-based decay
//! - **LongTermMemory**: Compressed summary of all past experience (running averages, counts)
//! - **EpisodicMemory**: Store and recall specific important episodes
//! - **MemoryIndex**: Index memories by context tags for fast retrieval
//! - **ForgettingCurve**: Configurable Ebbinghaus-style forgetting model
//! - **MemoryConsolidation**: Periodically consolidate short-term → long-term

mod forgetting;
mod short_term;
mod long_term;
mod episodic;
mod index;
mod consolidation;

pub use forgetting::{ForgettingCurve, ForgettingModel};
pub use short_term::{ShortTermMemory, MemoryEntry, Decision};
pub use long_term::{LongTermMemory, ExperienceSummary};
pub use episodic::{EpisodicMemory, Episode, EpisodeKind};
pub use index::{MemoryIndex, ContextTag};
pub use consolidation::{MemoryConsolidation, ConsolidationResult};
