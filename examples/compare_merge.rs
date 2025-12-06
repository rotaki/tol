use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::{rngs::StdRng, Rng, SeedableRng};

use tol::offset_value_coding::{
    encode_runs_with_ovc64, OVC64Trait, OVCEntry, SentinelValue, Sentineled,
};
use tol::tree_of_losers::LoserTree;
use tol::tree_of_losers_ovc::LoserTreeOVC;

fn main() {
    let cfg = Config::from_env();
    let runs = generate_runs(cfg.num_runs, cfg.run_len, cfg.key_len, cfg.seed);
    let mut expected_keys = runs.iter().flatten().cloned().collect::<Vec<_>>();
    expected_keys.sort_unstable();
    let encoded_runs = encode_runs_with_ovc64(&runs);
    let total_items: usize = runs.iter().map(|r| r.len()).sum();
    println!(
        "Benchmarking k-way merge variants | runs={} run_len={} key_len={} total_items={} seed={} warmup_runs={} benchmark_runs={}",
        cfg.num_runs,
        cfg.run_len,
        cfg.key_len,
        total_items,
        cfg.seed,
        cfg.warmup_runs,
        cfg.bench_runs
    );

    let (heap_summary, heap_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        total_items,
        || bench_binary_heap(&runs),
        |output| verify_sorted("BinaryHeap", output, &expected_keys),
    );
    let (tree_summary, tree_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        total_items,
        || bench_loser_tree(&runs),
        |output| verify_sorted("LoserTree", output, &expected_keys),
    );
    let (ovc_summary, ovc_output) = run_benches(
        cfg.warmup_runs,
        cfg.bench_runs,
        total_items,
        || bench_loser_tree_ovc(&encoded_runs),
        |output| verify_sorted_ovc("LoserTree+OVC", output, &expected_keys),
    );

    println!("\nResults (aligned; time ratios vs BinaryHeap):");
    println!(
        "{:<14} {:>27} {:>25} {:>9}",
        "Variant", "items/s (±)", "seconds (±)", "time x"
    );
    let baseline = heap_summary.avg_secs.max(f64::MIN_POSITIVE);
    print_row("BinaryHeap", heap_summary, heap_output, 1.0);
    print_row(
        "LoserTree",
        tree_summary,
        tree_output,
        tree_summary.avg_secs / baseline,
    );
    print_row(
        "LoserTree+OVC",
        ovc_summary,
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
) -> (Summary, usize)
where
    F: FnMut() -> BenchResult<T>,
    V: FnMut(&[T]),
{
    for _ in 0..warmup_runs {
        let result = bench();
        assert_eq!(
            result.stats.output_len, items,
            "warmup output length mismatch"
        );
        verify(&result.output);
    }

    let runs = bench_runs.max(1);
    let mut durations = Vec::with_capacity(runs);
    let mut last_output = 0;

    for _ in 0..runs {
        let result = bench();
        assert_eq!(
            result.stats.output_len, items,
            "benchmark output length mismatch"
        );
        durations.push(result.stats.elapsed);
        last_output = result.stats.output_len;
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
        last_output,
    )
}

fn print_row(label: &str, summary: Summary, output_len: usize, time_ratio: f64) {
    println!(
        "{:<14} {:>14.2} ± {:<12.2} {:>9.4}s ± {:<9.4}s {:>7.2}x  (out={})",
        label,
        summary.avg_tput,
        summary.stddev_tput,
        summary.avg_secs,
        summary.stddev_secs,
        time_ratio,
        output_len
    );
}

fn verify_sorted(label: &str, output: &[Vec<u8>], expected_sorted: &[Vec<u8>]) {
    if output.windows(2).any(|w| w[0] > w[1]) {
        panic!("{} warmup verification failed: output not sorted", label);
    }
    if output.len() != expected_sorted.len() {
        panic!(
            "{} warmup verification failed: length mismatch ({} vs {})",
            label,
            output.len(),
            expected_sorted.len()
        );
    }
    if !output.iter().zip(expected_sorted).all(|(o, e)| o == e) {
        panic!("{} warmup verification failed: contents differ", label);
    }
}

fn verify_sorted_ovc(label: &str, output: &[OVCEntry], expected_sorted: &[Vec<u8>]) {
    if output.windows(2).any(|w| w[0].key() > w[1].key()) {
        panic!("{} warmup verification failed: output not sorted", label);
    }
    if output.len() != expected_sorted.len() {
        panic!(
            "{} warmup verification failed: length mismatch ({} vs {})",
            label,
            output.len(),
            expected_sorted.len()
        );
    }
    if !output
        .iter()
        .zip(expected_sorted)
        .all(|(o, e)| o.key() == e)
    {
        panic!("{} warmup verification failed: contents differ", label);
    }
}

fn bench_binary_heap(runs: &[Vec<Vec<u8>>]) -> BenchResult<Vec<u8>> {
    let runs = black_box(runs);
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

    let start = Instant::now();

    let mut heap = BinaryHeap::new();
    for (run_id, iter) in iters.iter_mut().enumerate() {
        if let Some(value) = iter.next() {
            heap.push(HeapEntry { value, run_id });
        }
    }

    let mut output = Vec::new();
    while let Some(entry) = heap.pop() {
        output.push(entry.value);
        if let Some(next) = iters[entry.run_id].next() {
            heap.push(HeapEntry {
                value: next,
                run_id: entry.run_id,
            });
        }
    }

    let elapsed = start.elapsed();
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
            output_len,
        },
        output,
    }
}

fn bench_loser_tree(runs: &[Vec<Vec<u8>>]) -> BenchResult<Vec<u8>> {
    let runs = black_box(runs);
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

    let start = Instant::now();

    let mut initial = Vec::with_capacity(iters.len());
    for iter in iters.iter_mut() {
        if let Some(val) = iter.next() {
            initial.push(Sentineled::new(val));
        } else {
            initial.push(Sentineled::late_fence());
        }
    }

    let mut tree = LoserTree::new(initial);
    let mut output = Vec::new();

    while let Some((_, source_idx)) = tree.peek() {
        if let Some(next_val) = iters[source_idx].next() {
            let winner = tree.push(Sentineled::new(next_val));
            output.push(winner.inner());
        } else if let Some(winner) = tree.mark_current_exhausted() {
            output.push(winner.inner());
        }
    }

    let elapsed = start.elapsed();
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
            output_len,
        },
        output,
    }
}

fn bench_loser_tree_ovc(runs: &[Vec<OVCEntry>]) -> BenchResult<OVCEntry> {
    let runs = black_box(runs);
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

    let start = Instant::now();

    let mut initial = Vec::with_capacity(iters.len());
    for iter in iters.iter_mut() {
        if let Some(val) = iter.next() {
            initial.push(val.clone());
        } else {
            initial.push(OVCEntry::late_fence());
        }
    }

    let mut tree = LoserTreeOVC::new(initial);
    let mut output = Vec::new();

    while let Some((_, source_idx)) = tree.peek() {
        if let Some(next_val) = iters[source_idx].next() {
            let winner = tree.push(next_val);
            output.push(winner);
        } else if let Some(winner) = tree.mark_current_exhausted() {
            output.push(winner);
        }
    }

    let elapsed = start.elapsed();
    let output_len = output.len();
    let output = black_box(output);

    BenchResult {
        stats: Stats {
            elapsed,
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

fn generate_runs(num_runs: usize, run_len: usize, key_len: usize, seed: u64) -> Vec<Vec<Vec<u8>>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut runs = Vec::with_capacity(num_runs);
    for _ in 0..num_runs {
        let mut run = Vec::with_capacity(run_len);
        for _ in 0..run_len {
            let mut key = vec![0u8; key_len];
            rng.fill(&mut key[..]);
            run.push(key);
        }
        run.sort_unstable();
        runs.push(run);
    }
    runs
}

#[derive(Debug)]
struct HeapEntry<T> {
    value: T,
    run_id: usize,
}

impl<T: Ord> PartialEq for HeapEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.run_id == other.run_id
    }
}

impl<T: Ord> Eq for HeapEntry<T> {}

impl<T: Ord> PartialOrd for HeapEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Ord> Ord for HeapEntry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .value
            .cmp(&self.value)
            .then_with(|| other.run_id.cmp(&self.run_id))
    }
}

// Simple config parsing from env vars/args.
#[derive(Debug, Clone, Copy)]
struct Config {
    num_runs: usize,
    run_len: usize,
    key_len: usize,
    seed: u64,
    warmup_runs: usize,
    bench_runs: usize,
}

impl Config {
    fn from_env() -> Self {
        let mut args = env::args().skip(1);
        let num_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
        let run_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(50_000);
        let key_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
        let seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(42);
        let warmup_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(1);
        let bench_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);
        Self {
            num_runs,
            run_len,
            key_len,
            seed,
            warmup_runs,
            bench_runs,
        }
    }
}
