use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tol::offset_value_coding::{OVCEntry, SentinelValue, Sentineled, encode_runs_with_ovc64};
use tol::tree_of_losers::LoserTree;
use tol::tree_of_losers_ovc::LoserTreeOVC;

fn bench_merge(c: &mut Criterion) {
    let runs = generate_runs(NUM_RUNS, RUN_LEN, KEY_LEN, SEED);
    let encoded_runs = encode_runs_with_ovc64(&runs);

    let mut group = c.benchmark_group("merge");

    group.bench_function("binary_heap", |b| {
        b.iter(|| {
            let out = merge_with_binary_heap(black_box(&runs));
            black_box(out.len());
        })
    });

    group.bench_function("loser_tree", |b| {
        b.iter(|| {
            let out = merge_with_loser_tree(black_box(&runs));
            black_box(out.len());
        })
    });

    group.bench_function("loser_tree_ovc", |b| {
        b.iter(|| {
            let out = merge_with_loser_tree_ovc(black_box(&encoded_runs));
            black_box(out.len());
        })
    });

    group.finish();
}

fn merge_with_binary_heap(runs: &[Vec<Vec<u8>>]) -> Vec<Vec<u8>> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

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

    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();
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
    output
}

fn merge_with_loser_tree(runs: &[Vec<Vec<u8>>]) -> Vec<Vec<u8>> {
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

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

    output
}

fn merge_with_loser_tree_ovc(runs: &[Vec<OVCEntry>]) -> Vec<OVCEntry> {
    let mut iters: Vec<_> = runs.iter().map(|run| run.clone().into_iter()).collect();

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

    output
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

const NUM_RUNS: usize = 100;
const RUN_LEN: usize = 10_000;
const KEY_LEN: usize = 100;
const SEED: u64 = 43;

criterion_group!(benches, bench_merge);
criterion_main!(benches);
