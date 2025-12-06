use std::env;
use std::time::{Duration, Instant};

use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;

use tol::offset_value_coding::{OVC64Trait, OVCEntry, Sentineled};
use tol::replacement_selection_heap::ReplacementSelectionHeap;
use tol::replacement_selection_ovc::ReplacementSelectionOVC;
use tol::replacement_selection_tol::ReplacementSelectionToL;

fn main() {
    let cfg = Config::from_env();
    println!(
        "Benchmarking replacement selection variants | items={} key_len={} workspace_items={} seed={} warmup_runs={} benchmark_runs={}",
        cfg.num_items, cfg.key_len, cfg.workspace_items, cfg.seed, cfg.warmup_runs, cfg.bench_runs
    );

    let data = generate_keys(cfg.num_items, cfg.key_len, cfg.seed);
    let mut expected_keys = data.clone();
    expected_keys.sort_unstable();

    let (heap_summary, heap_runs, heap_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        cfg.num_items,
        || bench_heap(&data, cfg.key_len, cfg.workspace_items),
        |result| verify_runs("Heap RS", result, count_runs, &expected_keys, |v| v),
    );
    let (tree_summary, tree_runs, tree_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        cfg.num_items,
        || bench_tree(&data, cfg.key_len, cfg.workspace_items),
        |result| verify_runs("LoserTree RS", result, count_runs, &expected_keys, |v| v),
    );
    let (ovc_summary, ovc_runs, ovc_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        cfg.num_items,
        || bench_ovc(&data, cfg.key_len, cfg.workspace_items),
        |result| {
            verify_runs("OVC RS", result, count_runs_ovc, &expected_keys, |v| {
                v.key()
            })
        },
    );

    println!("\nResults (aligned; time ratios vs Heap):");
    println!(
        "{:<14} {:>27} {:>25} {:>9}",
        "Variant", "items/s (±)", "seconds (±)", "time x"
    );
    let baseline = heap_summary.avg_secs.max(f64::MIN_POSITIVE);
    print_row("Heap RS", heap_summary, heap_runs, heap_output, 1.0);
    print_row(
        "LoserTree RS",
        tree_summary,
        tree_runs,
        tree_output,
        tree_summary.avg_secs / baseline,
    );
    print_row(
        "OVC RS",
        ovc_summary,
        ovc_runs,
        ovc_output,
        ovc_summary.avg_secs / baseline,
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

#[derive(Debug, Clone, Copy)]
struct Summary {
    avg_tput: f64,
    stddev_tput: f64,
    avg_secs: f64,
    stddev_secs: f64,
}

#[derive(Debug)]
struct BenchResult<T> {
    stats: Stats,
    output: Vec<T>,
}

fn run_benches<F, T, V>(
    warmup_runs: usize,
    bench_runs: usize,
    items: usize,
    mut bench: F,
    mut verify: V,
) -> (Summary, usize, usize)
where
    F: FnMut() -> BenchResult<T>,
    V: FnMut(&BenchResult<T>),
{
    for _ in 0..warmup_runs {
        let result = bench();
        assert_eq!(
            result.stats.output_len, items,
            "warmup output length mismatch"
        );
        verify(&result);
    }

    let runs = bench_runs.max(1);
    let mut durations = Vec::with_capacity(runs);
    let mut last_output = 0;
    let mut last_runs = 0;

    for _ in 0..runs {
        let result = bench();
        assert_eq!(
            result.stats.output_len, items,
            "benchmark output length mismatch"
        );
        durations.push(result.stats.elapsed);
        last_output = result.stats.output_len;
        last_runs = result.stats.runs;
    }

    let secs: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
    let avg_secs = mean(&secs);
    let stddev_secs = stddev(&secs, avg_secs);

    let tputs: Vec<f64> = durations.iter().map(|d| throughput(items, *d)).collect();
    let avg_tput = mean(&tputs);
    let stddev_tput = stddev(&tputs, avg_tput);

    (
        Summary {
            avg_tput,
            stddev_tput,
            avg_secs,
            stddev_secs,
        },
        last_runs,
        last_output,
    )
}

fn print_row(label: &str, summary: Summary, runs: usize, output_len: usize, time_ratio: f64) {
    println!(
        "{:<14} {:>14.2} ± {:<12.2} {:>9.4}s ± {:<9.4}s {:>7.2}x  (runs={:>4}, out={})",
        label,
        summary.avg_tput,
        summary.stddev_tput,
        summary.avg_secs,
        summary.stddev_secs,
        time_ratio,
        runs,
        output_len
    );
}

fn verify_runs<T>(
    label: &str,
    result: &BenchResult<T>,
    run_counter: impl Fn(&[T]) -> usize,
    expected_sorted: &[Vec<u8>],
    key_fn: impl Fn(&T) -> &[u8],
) {
    let expected_runs = run_counter(&result.output);
    assert_eq!(
        result.stats.output_len,
        result.output.len(),
        "{} warmup output length mismatch",
        label
    );
    assert_eq!(
        expected_runs, result.stats.runs,
        "{} warmup run count mismatch",
        label
    );
    if result.output.len() != expected_sorted.len() {
        panic!(
            "{} warmup verification failed: length mismatch ({} vs {})",
            label,
            result.output.len(),
            expected_sorted.len()
        );
    }

    let mut keys: Vec<Vec<u8>> = result.output.iter().map(|v| key_fn(v).to_vec()).collect();
    keys.sort_unstable();

    if keys != expected_sorted {
        panic!("{} warmup verification failed: contents differ", label);
    }
}

fn bench_tree(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> BenchResult<Vec<u8>> {
    let data = black_box(data);
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionToL::<Sentineled<Vec<u8>>>::new(workspace_bytes);

    let start = Instant::now();

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(Sentineled::Normal(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(
            rs.absorb_record(Sentineled::Normal(key.clone()))
                .into_iter()
                .map(Sentineled::inner),
        );
    }
    output.extend(rs.drain().into_iter().map(Sentineled::inner));

    let elapsed = start.elapsed();
    let runs = count_runs(&output);
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
            runs,
            output_len,
        },
        output,
    }
}

fn bench_ovc(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> BenchResult<OVCEntry> {
    let data = black_box(data);
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
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
            runs,
            output_len,
        },
        output,
    }
}

fn bench_heap(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> BenchResult<Vec<u8>> {
    let data = black_box(data);
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionHeap::new(workspace_bytes);

    let start = Instant::now();

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(key.clone());
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(key.clone()));
    }
    output.extend(rs.drain());

    let elapsed = start.elapsed();
    let runs = count_runs(&output);
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
            runs,
            output_len,
        },
        output,
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

fn count_runs<T: Ord>(records: &[T]) -> usize {
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

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

// Simple config parsing from env vars/args.
#[derive(Debug, Clone, Copy)]
struct Config {
    num_items: usize,
    key_len: usize,
    workspace_items: usize,
    seed: u64,
    warmup_runs: usize,
    bench_runs: usize,
}

impl Config {
    fn from_env() -> Self {
        let mut args = env::args().skip(1);
        let num_items = args.next().and_then(|v| v.parse().ok()).unwrap_or(200_000);
        let key_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
        let workspace_items = args.next().and_then(|v| v.parse().ok()).unwrap_or(10_000);
        let seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(42);
        let warmup_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
        let bench_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);
        Self {
            num_items,
            key_len,
            workspace_items,
            seed,
            warmup_runs,
            bench_runs,
        }
    }
}
