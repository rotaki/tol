use criterion::{Criterion, black_box, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use tol::offset_value_coding::{OVCEntry, Sentineled};
use tol::replacement_selection_heap::ReplacementSelectionHeap;
use tol::replacement_selection_ovc::ReplacementSelectionOVC;
use tol::replacement_selection_tol::ReplacementSelectionToL;

fn bench_replacement_selection(c: &mut Criterion) {
    let datasets = build_datasets();

    let mut group = c.benchmark_group("replacement_selection");

    for ds in datasets {
        group.bench_function(format!("heap/{}", ds.name), |b| {
            b.iter(|| {
                let len = rs_heap(black_box(&ds.data), KEY_LEN, WORKSPACE_ITEMS);
                black_box(len);
            })
        });

        group.bench_function(format!("loser_tree/{}", ds.name), |b| {
            b.iter(|| {
                let len = rs_loser_tree(black_box(&ds.data), KEY_LEN, WORKSPACE_ITEMS);
                black_box(len);
            })
        });

        group.bench_function(format!("ovc/{}", ds.name), |b| {
            b.iter(|| {
                let len = rs_ovc(black_box(&ds.data), KEY_LEN, WORKSPACE_ITEMS);
                black_box(len);
            })
        });
    }

    group.finish();
}

struct DataSet {
    name: &'static str,
    data: Vec<Vec<u8>>,
}

fn build_datasets() -> Vec<DataSet> {
    let random = generate_keys(NUM_ITEMS, KEY_LEN, SEED);

    let mut sorted = random.clone();
    sorted.sort_unstable();

    let shared_prefix = generate_shared_prefix_keys(NUM_ITEMS, KEY_LEN, PREFIX_LEN, SEED + 1);

    vec![
        DataSet {
            name: "random",
            data: random,
        },
        DataSet {
            name: "sorted",
            data: sorted,
        },
        DataSet {
            name: "shared_prefix",
            data: shared_prefix,
        },
    ]
}

fn rs_loser_tree(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> usize {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionToL::<Sentineled<Vec<u8>>>::new(workspace_bytes);

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

    output.len()
}

fn rs_ovc(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> usize {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionOVC::<OVCEntry>::new(workspace_bytes);

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(OVCEntry::new(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(OVCEntry::new(key.clone())));
    }
    output.extend(rs.drain());

    output.len()
}

fn rs_heap(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> usize {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionHeap::new(workspace_bytes);

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(key.clone());
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(key.clone()));
    }
    output.extend(rs.drain());

    output.len()
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

fn generate_shared_prefix_keys(
    count: usize,
    key_len: usize,
    prefix_len: usize,
    seed: u64,
) -> Vec<Vec<u8>> {
    assert!(prefix_len <= key_len, "prefix_len must not exceed key_len");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut key = vec![0u8; key_len];
        rng.fill(&mut key[prefix_len..]);
        out.push(key);
    }
    out
}

const NUM_ITEMS: usize = 200_000;
const KEY_LEN: usize = 100;
const WORKSPACE_ITEMS: usize = 1_000;
const SEED: u64 = 42;
const PREFIX_LEN: usize = 20;

criterion_group!(benches, bench_replacement_selection);
criterion_main!(benches);
