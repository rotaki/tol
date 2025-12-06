use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use tol::offset_value_coding::{OVCEntryWithCounter, OVC64Trait, Sentineled};
use tol::replacement_selection_ovc::ReplacementSelectionOVC;
use tol::replacement_selection_tol::ReplacementSelectionToL;
#[cfg(feature = "instrument_calls")]
use tol::tree_of_losers::reset_push_update_counts as reset_tol_counts;
#[cfg(feature = "instrument_calls")]
use tol::tree_of_losers::take_push_update_counts as take_tol_counts;
#[cfg(feature = "instrument_calls")]
use tol::tree_of_losers_ovc::reset_push_update_counts as reset_ovc_counts;
#[cfg(feature = "instrument_calls")]
use tol::tree_of_losers_ovc::take_push_update_counts as take_ovc_counts;

// Count byte-wise comparisons for the plain loser-tree RS.
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

impl AsRef<[u8]> for CountingVec {
    fn as_ref(&self) -> &[u8] {
        &self.0
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
    let num_items = 200_000;
    let key_len = 100;
    let workspace_items = 1_000;
    let seed = 42u64;

    println!(
        "Config: num_items={} key_len={} workspace_items={} seed={}",
        num_items, key_len, workspace_items, seed
    );

    let scenarios = vec![
        ("random", generate_random_keys(num_items, key_len, seed)),
        ("sorted", generate_sorted_keys(num_items, key_len)),
        (
            "shared_prefix",
            generate_shared_prefix_keys(num_items, key_len, 20, seed),
        ),
        (
            "variable_size",
            generate_variable_size_keys(num_items, key_len, seed, 8, 32),
        ),
    ];

    println!("Replacement Selection comparison counts:");
    for (label, data) in scenarios {
        let Stats {
            plain_runs,
            plain_bytes,
            ovc_runs,
            ovc_bytes,
            ovc_meta_bytes,
            plain_push_data,
            plain_push_late,
            plain_updates,
            ovc_push_data,
            ovc_push_late,
            ovc_updates,
        } = measure_scenario(&data, key_len, workspace_items);
        #[cfg(not(feature = "instrument_calls"))]
        let _ = (
            plain_push_data,
            plain_push_late,
            plain_updates,
            ovc_push_data,
            ovc_push_late,
            ovc_updates,
        );

        println!("Scenario: {}", label);
        println!(
            "  Plain loser-tree RS: {:>10} byte comparisons, runs={}",
            plain_bytes, plain_runs
        );
        println!(
            "  OVC RS:              {:>10} byte comparisons, {:>10} ovc comparisons, runs={}",
            ovc_bytes, ovc_meta_bytes, ovc_runs
        );
        #[cfg(feature = "instrument_calls")]
        {
            println!(
                "  Tree calls (plain): data_pushes={}, late_pushes={}, updates={}",
                plain_push_data, plain_push_late, plain_updates
            );
            println!(
                "  Tree calls (ovc):   data_pushes={}, late_pushes={}, updates={}",
                ovc_push_data, ovc_push_late, ovc_updates
            );
        }
    }
}

struct Stats {
    plain_runs: usize,
    plain_bytes: usize,
    ovc_runs: usize,
    ovc_bytes: usize,
    ovc_meta_bytes: usize,
    #[cfg(feature = "instrument_calls")]
    plain_push_data: usize,
    #[cfg(feature = "instrument_calls")]
    plain_push_late: usize,
    #[cfg(feature = "instrument_calls")]
    plain_updates: usize,
    #[cfg(feature = "instrument_calls")]
    ovc_push_data: usize,
    #[cfg(feature = "instrument_calls")]
    ovc_push_late: usize,
    #[cfg(feature = "instrument_calls")]
    ovc_updates: usize,
    #[cfg(not(feature = "instrument_calls"))]
    plain_push_data: (),
    #[cfg(not(feature = "instrument_calls"))]
    plain_push_late: (),
    #[cfg(not(feature = "instrument_calls"))]
    plain_updates: (),
    #[cfg(not(feature = "instrument_calls"))]
    ovc_push_data: (),
    #[cfg(not(feature = "instrument_calls"))]
    ovc_push_late: (),
    #[cfg(not(feature = "instrument_calls"))]
    ovc_updates: (),
}

fn measure_scenario(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> Stats {
    // Plain loser-tree RS (Sentineled<CountingVec>)
    CountingVec::reset_counts();
    #[cfg(feature = "instrument_calls")]
    reset_tol_counts();
    let plain_runs = run_plain_rs(data, key_len, workspace_items);
    let plain_bytes = CountingVec::take_counts();
    #[cfg(feature = "instrument_calls")]
    let (plain_push_data, plain_push_late, plain_updates) = take_tol_counts();

    // OVC RS with counting entry
    OVCEntryWithCounter::reset_byte_comparisons();
    OVCEntryWithCounter::reset_ovc_comparisons();
    #[cfg(feature = "instrument_calls")]
    reset_ovc_counts();
    let ovc_runs = run_ovc_rs(data, key_len, workspace_items);
    let ovc_bytes = OVCEntryWithCounter::take_byte_comparisons();
    let ovc_meta_bytes = OVCEntryWithCounter::take_ovc_comparisons();
    #[cfg(feature = "instrument_calls")]
    let (ovc_push_data, ovc_push_late, ovc_updates) = take_ovc_counts();

    Stats {
        plain_runs,
        plain_bytes,
        ovc_runs,
        ovc_bytes,
        ovc_meta_bytes,
        #[cfg(feature = "instrument_calls")]
        plain_push_data,
        #[cfg(feature = "instrument_calls")]
        plain_push_late,
        #[cfg(feature = "instrument_calls")]
        plain_updates,
        #[cfg(feature = "instrument_calls")]
        ovc_push_data,
        #[cfg(feature = "instrument_calls")]
        ovc_push_late,
        #[cfg(feature = "instrument_calls")]
        ovc_updates,
        #[cfg(not(feature = "instrument_calls"))]
        plain_push_data: (),
        #[cfg(not(feature = "instrument_calls"))]
        plain_push_late: (),
        #[cfg(not(feature = "instrument_calls"))]
        plain_updates: (),
        #[cfg(not(feature = "instrument_calls"))]
        ovc_push_data: (),
        #[cfg(not(feature = "instrument_calls"))]
        ovc_push_late: (),
        #[cfg(not(feature = "instrument_calls"))]
        ovc_updates: (),
    }
}

fn run_plain_rs(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> usize {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionToL::<Sentineled<CountingVec>>::new(workspace_bytes);

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(Sentineled::Normal(CountingVec::new(key.clone())));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(
            rs.absorb_record(Sentineled::Normal(CountingVec::new(key.clone())))
                .into_iter()
                .map(Sentineled::inner),
        );
    }
    output.extend(rs.drain().into_iter().map(Sentineled::inner));

    // Compute runs without affecting the comparison counter.
    let output_bytes: Vec<Vec<u8>> = output.iter().map(|c| c.0.clone()).collect();
    count_runs_bytes(&output_bytes)
}

fn run_ovc_rs(data: &[Vec<u8>], key_len: usize, workspace_items: usize) -> usize {
    let workspace_bytes = workspace_items * key_len;
    let mut rs = ReplacementSelectionOVC::<OVCEntryWithCounter>::new(workspace_bytes);

    for key in data.iter().take(workspace_items) {
        rs.insert_initial(OVCEntryWithCounter::new(key.clone()));
    }
    rs.build();

    let mut output = Vec::new();
    for key in data.iter().skip(workspace_items) {
        output.extend(rs.absorb_record(OVCEntryWithCounter::new(key.clone())));
    }
    output.extend(rs.drain());

    count_runs_ovc(&output)
}

fn count_runs_bytes(records: &[Vec<u8>]) -> usize {
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

fn count_runs_ovc<T: OVC64Trait>(records: &[T]) -> usize {
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

fn generate_random_keys(count: usize, key_len: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let mut key = vec![0u8; key_len];
        rng.fill(&mut key[..]);
        out.push(key);
    }
    out
}

fn generate_sorted_keys(count: usize, key_len: usize) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = (0..count)
        .map(|i| {
            let mut key = vec![0u8; key_len];
            key[key_len - 1] = (i % 256) as u8;
            key[key_len - 2] = ((i / 256) % 256) as u8;
            key
        })
        .collect();
    out.sort_unstable();
    out
}

fn generate_shared_prefix_keys(
    count: usize,
    key_len: usize,
    prefix_len: usize,
    seed: u64,
) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1234));
    let mut out = Vec::with_capacity(count);
    let mut key = vec![0u8; key_len];
    if prefix_len > 0 {
        for b in key.iter_mut().take(prefix_len.min(key_len)) {
            *b = 7; // fixed shared prefix
        }
    }
    for _ in 0..count {
        let mut k = key.clone();
        for b in k.iter_mut().skip(prefix_len.min(key_len)) {
            *b = rng.random();
        }
        out.push(k);
    }
    out
}

fn generate_variable_size_keys(
    count: usize,
    base_len: usize,
    seed: u64,
    min_extra: usize,
    max_extra: usize,
) -> Vec<Vec<u8>> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(9999));
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let extra = rng.random_range(min_extra..=max_extra);
        let len = base_len.saturating_sub(min_extra).saturating_add(extra);
        let mut key = vec![0u8; len.max(1)];
        rng.fill(&mut key[..]);
        out.push(key);
    }
    out
}
