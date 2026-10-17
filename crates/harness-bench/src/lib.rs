//! harness-bench: Benchmark task loading, running, and evaluation.
//!
//! Supports SWE-Bench Verified and Terminal-Bench 2.1 task formats, plus
//! trajectory analysis for failure-stage attribution (paper §C.1).

pub mod analysis;
pub mod runner;
pub mod swe_bench;
pub mod task;
pub mod terminal_bench;

pub use analysis::{
    analyze, AnalysisSummary, FailureStage, Phase, PhaseCounts, TrajectoryAnalysis,
};
pub use runner::{prepare_workspace, SweepReport, SweepSummary, SweepTimer, TaskOutcome};
pub use swe_bench::SweBench;
pub use task::{run_verification, Benchmark, BenchmarkResult, BenchmarkTask};
pub use terminal_bench::TerminalBench;
