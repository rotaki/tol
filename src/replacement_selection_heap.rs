use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::replacement_selection::RecordSize;

/// Replacement selection driven by a binary heap instead of a tournament tree.
///
/// The API mirrors `ReplacementSelection` so it can be used interchangeably in
/// experiments. Records that cannot belong to the current run are staged in
/// `next_run_buffer`; when the heap empties we rebuild it from that buffer.
pub struct ReplacementSelectionHeap<T: Ord + RecordSize> {
    /// Min-heap storing `(record, sequence_number)` to keep tie-breaking stable.
    heap: BinaryHeap<Reverse<(T, usize)>>,

    /// Buffer holding records for the next run.
    next_run_buffer: Vec<T>,

    /// Maximum workspace in bytes.
    workspace_size: usize,

    /// Bytes currently held across heap + buffer.
    used_space: usize,

    /// Whether `build` has been called.
    initialized: bool,

    /// Logical capacity (next power-of-two of the initial load).
    capacity: usize,

    /// Monotonic counter used to stabilize ordering of equal keys.
    seq: usize,
}

impl<T: Ord + RecordSize> ReplacementSelectionHeap<T> {
    /// Create a new heap-backed replacement selection structure.
    pub fn new(workspace_size: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_run_buffer: Vec::new(),
            workspace_size,
            used_space: 0,
            initialized: false,
            capacity: 0,
            seq: 0,
        }
    }

    /// Insert initial records before the first build.
    pub fn insert_initial(&mut self, record: T) {
        assert!(
            !self.initialized,
            "Cannot insert initial records after build is called"
        );
        self.used_space += record.size();
        self.next_run_buffer.push(record);
    }

    /// Build the initial heap from the staged records.
    pub fn build(&mut self) {
        assert!(!self.initialized, "build() may only be called once");
        self.initialize_heap_from_buffer();
        self.initialized = true;
    }

    /// Absorb a new record; may evict one or more records to respect workspace.
    pub fn absorb_record(&mut self, record: T) -> Vec<T> {
        assert!(self.initialized, "absorb_record called before build()");

        let mut output = Vec::new();
        let new_size = record.size();

        self.ensure_active_heap();

        if self.should_defer_to_next_run(&record) {
            self.add_to_next_run(record);
        } else {
            self.add_to_current_run(record, &mut output);
        }

        self.used_space += new_size;
        self.evict_until_space_available(&mut output);

        output
    }

    /// Drain all remaining records from the current and next runs.
    pub fn drain(&mut self) -> Vec<T> {
        let mut output = Vec::new();

        self.drain_current_heap(&mut output);
        self.switch_run();
        self.drain_current_heap(&mut output);

        output
    }

    /// Number of records staged for the next run.
    pub fn buffer_len(&self) -> usize {
        self.next_run_buffer.len()
    }

    /// ---------------------------------------------------------------------
    /// Internal helpers
    /// ---------------------------------------------------------------------

    fn initialize_heap_from_buffer(&mut self) {
        let num_real = self.next_run_buffer.len();
        self.capacity = if num_real == 0 {
            0
        } else {
            num_real.next_power_of_two()
        };

        self.heap = BinaryHeap::with_capacity(self.capacity);
        for record in std::mem::take(&mut self.next_run_buffer) {
            self.push_heap(record);
        }
    }

    fn ensure_active_heap(&mut self) {
        if self.heap.is_empty() {
            self.switch_run();
        }
    }

    fn should_defer_to_next_run(&self, record: &T) -> bool {
        if let Some(min) = self.peek_min() {
            record < min
        } else {
            false
        }
    }

    fn add_to_next_run(&mut self, record: T) {
        self.next_run_buffer.push(record);
    }

    fn add_to_current_run(&mut self, record: T, output: &mut Vec<T>) {
        let record_size = record.size();
        let will_exceed = self.used_space + record_size > self.workspace_size;

        // If we started empty (capacity 0), allow the first insertion to seed
        // the heap without an eviction.
        if self.capacity == 0 && self.heap.is_empty() {
            self.capacity = 1;
        }

        if !will_exceed && self.heap.len() < self.capacity {
            self.push_heap(record);
            return;
        }

        // Insert and evict the current minimum to keep heap size bounded.
        self.push_heap(record);
        if let Some(winner) = self.pop_min() {
            self.used_space = self.used_space.saturating_sub(winner.size());
            output.push(winner);
        }
    }

    fn evict_until_space_available(&mut self, output: &mut Vec<T>) {
        while self.used_space > self.workspace_size {
            self.ensure_active_heap();
            if self.heap.is_empty() {
                break;
            }

            if let Some(winner) = self.pop_min() {
                self.used_space = self.used_space.saturating_sub(winner.size());
                output.push(winner);
            } else {
                break;
            }
        }
    }

    fn switch_run(&mut self) {
        if self.next_run_buffer.is_empty() {
            return;
        }
        self.initialize_heap_from_buffer();
    }

    fn drain_current_heap(&mut self, output: &mut Vec<T>) {
        while let Some(val) = self.pop_min() {
            self.used_space = self.used_space.saturating_sub(val.size());
            output.push(val);
        }
    }

    fn push_heap(&mut self, record: T) {
        let seq = self.seq;
        self.seq = self.seq.wrapping_add(1);
        self.heap.push(Reverse((record, seq)));
    }

    fn pop_min(&mut self) -> Option<T> {
        self.heap.pop().map(|Reverse((val, _))| val)
    }

    fn peek_min(&self) -> Option<&T> {
        self.heap.peek().map(|Reverse((val, _))| val)
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, fmt::Debug};

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;

    #[derive(Clone)]
    pub struct TestRecord {
        pub val: i32,
        pub size: usize,
    }

    impl Ord for TestRecord {
        fn cmp(&self, other: &Self) -> Ordering {
            self.val.cmp(&other.val)
        }
    }
    impl PartialOrd for TestRecord {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl PartialEq for TestRecord {
        fn eq(&self, other: &Self) -> bool {
            self.val == other.val
        }
    }
    impl Eq for TestRecord {}

    impl TestRecord {
        pub fn new(val: i32, size: usize) -> Self {
            Self { val, size }
        }
    }

    impl RecordSize for TestRecord {
        fn size(&self) -> usize {
            self.size
        }
    }

    impl Debug for TestRecord {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "TestRecord(val: {}, size: {})", self.val, self.size)
        }
    }

    #[test]
    fn test_padding_optimization_usage() {
        let mut rs = ReplacementSelectionHeap::new(100);
        rs.insert_initial(TestRecord::new(10, 10));
        rs.insert_initial(TestRecord::new(20, 10));
        rs.insert_initial(TestRecord::new(30, 10));

        rs.build();

        let out = rs.absorb_record(TestRecord::new(15, 10));

        assert!(out.is_empty(), "Should use padding slot, not evict");
        assert_eq!(rs.used_space, 40);
        assert_eq!(rs.heap.len(), 4, "Heap should have used extra slot");
    }

    #[test]
    fn test_run_generation_logic() {
        let mut rs = ReplacementSelectionHeap::new(20);
        rs.insert_initial(TestRecord::new(10, 10));
        rs.insert_initial(TestRecord::new(20, 10));
        rs.build();

        let out = rs.absorb_record(TestRecord::new(5, 10));

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].val, 10, "Should evict current winner");
        assert_eq!(rs.buffer_len(), 1, "5 should be in next run buffer");
        assert_eq!(rs.used_space, 20);
    }

    #[test]
    fn test_variable_size_eviction() {
        let mut rs = ReplacementSelectionHeap::new(10);
        for i in 1..=5 {
            rs.insert_initial(TestRecord::new(i * 10, 2));
        }
        rs.build();

        let out = rs.absorb_record(TestRecord::new(100, 6));

        assert_eq!(out.len(), 3, "Should evict 3 small records");
        assert_eq!(out[0].val, 10);
        assert_eq!(out[1].val, 20);
        assert_eq!(out[2].val, 30);
        assert_eq!(rs.used_space, 10);
    }

    #[test]
    fn test_run_switch() {
        let mut rs = ReplacementSelectionHeap::new(19);
        rs.insert_initial(TestRecord::new(10, 10));
        rs.build();

        let out = rs.absorb_record(TestRecord::new(5, 10));
        assert_eq!(out[0].val, 10);
        assert_eq!(rs.buffer_len(), 1);

        let out2 = rs.absorb_record(TestRecord::new(20, 10));
        assert_eq!(out2[0].val, 5);
    }

    #[test]
    fn test_full_sort_experiment() {
        let mut rng = StdRng::seed_from_u64(42);

        let item_size = 10;
        let workspace_capacity_items = 10;
        let workspace_size = workspace_capacity_items * item_size;
        let num_elements = 400;

        let mut rs = ReplacementSelectionHeap::new(workspace_size);

        let mut data: Vec<TestRecord> = (0..num_elements)
            .map(|_| TestRecord::new(rng.random_range(0..1000), item_size))
            .collect();

        for _ in 0..workspace_capacity_items {
            if let Some(rec) = data.pop() {
                rs.insert_initial(rec);
            }
        }
        rs.build();

        let mut full_output = Vec::new();

        while let Some(rec) = data.pop() {
            let mut evicted = rs.absorb_record(rec);
            full_output.append(&mut evicted);
        }

        let mut drained = rs.drain();
        full_output.append(&mut drained);

        let mut run_count = 0;
        let mut run_start_indices = vec![0];

        for i in 1..full_output.len() {
            if full_output[i] < full_output[i - 1] {
                run_count += 1;
                run_start_indices.push(i);
            }
        }
        run_count += 1;
        run_start_indices.push(full_output.len());

        for r in 0..run_count {
            let start = run_start_indices[r];
            let end = run_start_indices[r + 1];
            let run_slice = &full_output[start..end];

            for i in 1..run_slice.len() {
                assert!(
                    run_slice[i] >= run_slice[i - 1],
                    "Run {} is unsorted at index {}! {:?} < {:?}",
                    r,
                    i,
                    run_slice[i],
                    run_slice[i - 1]
                );
            }
        }

        assert_eq!(full_output.len(), num_elements, "Output count mismatch");
        assert!(run_count > 1, "Should produce multiple runs given small workspace");
    }
}
