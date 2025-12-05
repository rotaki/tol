use std::env;
use std::time::{Duration, Instant};

use rand::{rngs::StdRng, Rng, SeedableRng};

use tol::offset_value_coding::{OVC64Trait, OVCEntry, SentinelValue};
use tol::replacement_selection::RecordSize;
use tol::replacement_selection::ReplacementSelection;
use tol::replacement_selection_heap::ReplacementSelectionHeap;
use tol::replacement_selection_ovc::ReplacementSelectionOVC;

fn main() {
    let cfg = Config::from_env();
    println!(
        "Benchmarking replacement selection variants | items={} key_len={} workspace_items={} seed={}",
        cfg.num_items, cfg.key_len, cfg.workspace_items, cfg.seed
    );

    let data = generate_keys(cfg.num_items, cfg.key_len, cfg.seed);

    let tree_stats = bench_tree(&data, cfg.key_len, cfg.workspace_items);
    let ovc_stats = bench_ovc(&data, cfg.key_len, cfg.workspace_items);
    let heap_stats = bench_heap(&data, cfg.key_len, cfg.workspace_items);

    println!("\nResults (items/s):");
    println!(
        "  LoserTree RS     : {:.2} (runs={}, out={})",
        throughput(cfg.num_items, tree_stats.elapsed),
        tree_stats.runs,
        tree_stats.output_len
    );
    println!(
        "  OVC RS           : {:.2} (runs={}, out={})",
        throughput(cfg.num_items, ovc_stats.elapsed),
        ovc_stats.runs,
        ovc_stats.output_len
    );
    println!(
        "  Heap RS          : {:.2} (runs={}, out={})",
        throughput(cfg.num_items, heap_stats.elapsed),
        heap_stats.runs,
        heap_stats.output_len
    );
}

// -------------------------------------------------------------------------
// Benchmark harness
// -------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Stats {
    elapsed: Duration,
    runs: usize,
    output_len: usize,
}

fn bench_tree(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> Stats {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelection::<ByteRecord>::new(workspace_bytes);

    let start = Instant::now();

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(ByteRecord::from(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(ByteRecord::from(key.clone())));
    }
    output.extend(rs.drain());

    let elapsed = start.elapsed();
    let runs = count_runs(&output);

    Stats {
        elapsed,
        runs,
        output_len: output.len(),
    }
}

fn bench_ovc(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> Stats {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionOVC::<OVCEntry>::new(workspace_bytes);

    let start = Instant::now();

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(OVCEntry::new(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(OVCEntry::new(key.clone())));
    }
    output.extend(rs.drain());

    let elapsed = start.elapsed();
    let runs = count_runs_ovc(&output);

    Stats {
        elapsed,
        runs,
        output_len: output.len(),
    }
}

fn bench_heap(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> Stats {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionHeap::<ByteRecord>::new(workspace_bytes);

    let start = Instant::now();

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(ByteRecord::from(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(ByteRecord::from(key.clone())));
    }
    output.extend(rs.drain());

    let elapsed = start.elapsed();
    let runs = count_runs(&output);

    Stats {
        elapsed,
        runs,
        output_len: output.len(),
    }
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn throughput(items: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return f64::INFINITY;
    }
    items as f64 / elapsed.as_secs_f64()
}

fn generate_keys(count: usize, key_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut key = vec![0u8; key_len];
        rng.fill(&mut key[..]);
        out.push(key);
    }
    out
}

fn count_runs(records: &[ByteRecord]) -> usize {
    if records.is_empty() {
        return 0;
    }
    let mut runs = 1;
    for i in 1..records.len() {
        if records[i] < records[i - 1] {
            runs += 1;
        }
    }
    runs
}

fn count_runs_ovc(records: &[OVCEntry]) -> usize {
    if records.is_empty() {
        return 0;
    }
    let mut runs = 1;
    for i in 1..records.len() {
        if records[i].key() < records[i - 1].key() {
            runs += 1;
        }
    }
    runs
}

// Simple config parsing from env vars/args.
#[derive(Debug, Clone, Copy)]
struct Config {
    num_items: usize,
    key_len: usize,
    workspace_items: usize,
    seed: u64,
}

impl Config {
    fn from_env() -> Self {
        let mut args = env::args().skip(1);
        let num_items = args
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);
        let key_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
        let workspace_items = args
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);
        let seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(42);
        Self {
            num_items,
            key_len,
            workspace_items,
            seed,
        }
    }
}

// -------------------------------------------------------------------------
// Data type for non-OVC variants
// -------------------------------------------------------------------------

#[derive(Clone, Eq, PartialEq)]
struct ByteRecord {
    key: Vec<u8>,
    sentinel: SentinelKind,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum SentinelKind {
    Early,
    Data,
    Late,
}

impl From<Vec<u8>> for ByteRecord {
    fn from(key: Vec<u8>) -> Self {
        Self {
            key,
            sentinel: SentinelKind::Data,
        }
    }
}

impl Ord for ByteRecord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.sentinel, other.sentinel) {
            (SentinelKind::Early, SentinelKind::Early)
            | (SentinelKind::Late, SentinelKind::Late) => std::cmp::Ordering::Equal,
            (SentinelKind::Early, _) => std::cmp::Ordering::Less,
            (_, SentinelKind::Early) => std::cmp::Ordering::Greater,
            (SentinelKind::Late, _) => std::cmp::Ordering::Greater,
            (_, SentinelKind::Late) => std::cmp::Ordering::Less,
            (SentinelKind::Data, SentinelKind::Data) => self.key.cmp(&other.key),
        }
    }
}

impl PartialOrd for ByteRecord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl RecordSize for ByteRecord {
    fn size(&self) -> usize {
        if matches!(self.sentinel, SentinelKind::Data) {
            self.key.len()
        } else {
            0
        }
    }
}

impl SentinelValue for ByteRecord {
    fn early_fence() -> Self {
        Self {
            key: Vec::new(),
            sentinel: SentinelKind::Early,
        }
    }

    fn late_fence() -> Self {
        Self {
            key: Vec::new(),
            sentinel: SentinelKind::Late,
        }
    }

    fn is_early_fence(&self) -> bool {
        matches!(self.sentinel, SentinelKind::Early)
    }

    fn is_late_fence(&self) -> bool {
        matches!(self.sentinel, SentinelKind::Late)
    }
}
