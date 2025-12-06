use std::cmp::Ordering;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use rand::{rngs::StdRng, Rng, SeedableRng};

use tol::offset_value_coding::{
    encode_runs_with_ovc64, OVC64Trait, OVCEntry, OVCEntryWithCounter, SentinelValue, Sentineled,
};
use tol::tree_of_losers::LoserTree;
use tol::tree_of_losers_ovc::LoserTreeOVC;

// Count byte-wise comparisons for plain loser-tree merge.
static PLAIN_BYTE_COMPARISONS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
struct CountingVec(Vec<u8>);

impl CountingVec {
    fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    fn reset_counts() {
        PLAIN_BYTE_COMPARISONS.store(0, AtomicOrdering::Relaxed);
    }

    fn take_counts() -> usize {
        PLAIN_BYTE_COMPARISONS.swap(0, AtomicOrdering::Relaxed)
    }

    #[inline]
    fn cmp_byte(a: u8, b: u8) -> Ordering {
        PLAIN_BYTE_COMPARISONS.fetch_add(1, AtomicOrdering::Relaxed);
        a.cmp(&b)
    }
}

impl PartialOrd for CountingVec {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CountingVec {
    fn cmp(&self, other: &Self) -> Ordering {
        let min_len = self.0.len().min(other.0.len());
        for i in 0..min_len {
            match Self::cmp_byte(self.0[i], other.0[i]) {
                Ordering::Equal => continue,
                non_eq => return non_eq,
            }
        }
        self.0.len().cmp(&other.0.len())
    }
}

fn main() {
    let cfg = Config::from_env();
    let runs = generate_runs(cfg.num_runs, cfg.run_len, cfg.key_len, cfg.seed);
    let encoded_runs: Vec<Vec<OVCEntry>> = encode_runs_with_ovc64(&runs);
    let encoded_runs_counter: Vec<Vec<OVCEntryWithCounter>> = encoded_runs
        .iter()
        .map(|run| run.iter().cloned().map(OVCEntryWithCounter::from).collect())
        .collect();
    let expected_len: usize = runs.iter().map(|r| r.len()).sum();

    println!(
        "Dataset: runs={} run_len={} key_len={} seed={}",
        cfg.num_runs, cfg.run_len, cfg.key_len, cfg.seed
    );

    // Plain loser-tree merge with counting bytes
    CountingVec::reset_counts();
    let plain_output = merge_with_loser_tree_counted(&runs);
    let plain_byte_comparisons = CountingVec::take_counts();

    // OVC loser-tree merge with counting bytes + OVC metadata comparisons
    OVCEntryWithCounter::reset_byte_comparisons();
    OVCEntryWithCounter::reset_ovc_comparisons();
    let ovc_output = merge_with_loser_tree_ovc_counted(&encoded_runs_counter);
    let ovc_byte_comparisons = OVCEntryWithCounter::take_byte_comparisons();
    let ovc_meta_comparisons = OVCEntryWithCounter::take_ovc_comparisons();

    // Verify correctness
    assert_eq!(plain_output.len(), expected_len, "plain output len mismatch");
    assert_eq!(ovc_output.len(), expected_len, "ovc output len mismatch");
    assert!(is_sorted_bytes(&plain_output));
    assert!(is_sorted_ovc(&ovc_output));

    println!("k-way merge comparison counts (no timing):");
    println!(
        "  LoserTree:     {:>12} byte comparisons",
        plain_byte_comparisons
    );
    println!(
        "  LoserTree+OVC: {:>12} byte comparisons, {:>12} ovc comparisons",
        ovc_byte_comparisons, ovc_meta_comparisons
    );
}

fn merge_with_loser_tree_counted(runs: &[Vec<Vec<u8>>]) -> Vec<Vec<u8>> {
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

    let mut initial = Vec::with_capacity(iters.len());
    for iter in iters.iter_mut() {
        if let Some(val) = iter.next() {
            initial.push(Sentineled::new(CountingVec::new(val)));
        } else {
            initial.push(Sentineled::late_fence());
        }
    }

    let mut tree = LoserTree::new(initial);
    let mut output = Vec::new();

    while let Some((_, source_idx)) = tree.peek() {
        if let Some(next_val) = iters[source_idx].next() {
            let winner = tree.push(Sentineled::new(CountingVec::new(next_val)));
            output.push(winner.inner().0);
        } else if let Some(winner) = tree.mark_current_exhausted() {
            output.push(winner.inner().0);
        }
    }

    output
}

fn merge_with_loser_tree_ovc_counted(runs: &[Vec<OVCEntryWithCounter>]) -> Vec<OVCEntryWithCounter> {
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

    let mut initial = Vec::with_capacity(iters.len());
    for iter in iters.iter_mut() {
        if let Some(val) = iter.next() {
            initial.push(val);
        } else {
            initial.push(OVCEntryWithCounter::late_fence());
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

    output
}

fn is_sorted_bytes(output: &[Vec<u8>]) -> bool {
    output.windows(2).all(|w| w[0] <= w[1])
}

fn is_sorted_ovc<T: OVC64Trait>(output: &[T]) -> bool {
    output.windows(2).all(|w| w[0].key() <= w[1].key())
}

fn generate_runs(
    num_runs: usize,
    run_len: usize,
    key_len: usize,
    seed: u64,
) -> Vec<Vec<Vec<u8>>> {
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

// Simple config parsing from env vars/args.
#[derive(Debug, Clone, Copy)]
struct Config {
    num_runs: usize,
    run_len: usize,
    key_len: usize,
    seed: u64,
}

impl Config {
    fn from_env() -> Self {
        let mut args = env::args().skip(1);
        let num_runs = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
        let run_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(50_000);
        let key_len = args.next().and_then(|v| v.parse().ok()).unwrap_or(16);
        let seed = args.next().and_then(|v| v.parse().ok()).unwrap_or(42);
        Self {
            num_runs,
            run_len,
            key_len,
            seed,
        }
    }
}
