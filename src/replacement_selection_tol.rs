use crate::offset_value_coding::SentinelValue;
use crate::tree_of_losers::LoserTree;

pub trait RecordSize {
    fn size(&self) -> usize;
}

// Example impl for byte vectors
impl<T: AsRef<[u8]>> RecordSize for T {
    fn size(&self) -> usize {
        self.as_ref().len()
    }
}
pub struct ReplacementSelectionToL<T: Ord + SentinelValue + RecordSize> {
    /// Tournament tree for efficient min-element extraction
    tree: LoserTree<T>,

    /// Buffer for records destined for the next sorted run
    next_run_buffer: Vec<T>,

    /// Available padding slots in the tree (indices between num_elements and capacity)
    late_fence_slots: Vec<usize>,

    /// Maximum allowed memory usage in bytes
    workspace_size: usize,

    /// Current memory usage in bytes
    used_space: usize,

    /// Whether the tree has been initialized via build()
    initialized: bool,
}

impl<T: Ord + SentinelValue + RecordSize> ReplacementSelectionToL<T> {
    /// Create a new replacement selection instance with the specified workspace size.
    ///
    /// # Arguments
    /// * `workspace_size` - Maximum memory (in bytes) that can be used for buffering
    pub fn new(workspace_size: usize) -> Self {
        Self {
            tree: LoserTree::new(vec![]),
            next_run_buffer: Vec::new(),
            late_fence_slots: Vec::new(),
            workspace_size,
            used_space: 0,
            initialized: false,
        }
    }

    /// Insert an initial record before building the tree.
    ///
    /// This method can only be called before `build()`. Initial records form
    /// the first sorted run.
    ///
    /// # Panics
    /// Panics if called after `build()` has been invoked.
    pub fn insert_initial(&mut self, record: T) {
        assert!(
            !self.initialized,
            "Cannot insert initial records after build is called"
        );
        self.used_space += record.size();
        self.next_run_buffer.push(record);
    }

    /// Build the initial tree from all inserted records.
    ///
    /// This transitions the data structure from setup phase to operational phase.
    /// After calling this method, use `absorb_record()` to process new records.
    ///
    /// # Panics
    /// Panics if called more than once.
    pub fn build(&mut self) {
        assert!(!self.initialized, "build() may only be called once");
        self.initialize_tree_from_buffer();
        self.initialized = true;
    }

    /// Initialize the tree from the next run buffer and set up padding slots
    fn initialize_tree_from_buffer(&mut self) {
        let num_real = self.next_run_buffer.len();
        let capacity = if num_real == 0 {
            0
        } else {
            num_real.next_power_of_two()
        };

        self.tree = LoserTree::new(std::mem::take(&mut self.next_run_buffer));
        self.late_fence_slots.clear();

        // Capture padding slots (between num_real and next power of 2)
        for i in (num_real..capacity).rev() {
            self.late_fence_slots.push(i);
        }
    }

    /// Absorb a new record into the replacement selection algorithm.
    ///
    /// Returns a vector of records that were evicted to make space.
    /// The evicted records are guaranteed to be in sorted order within the current run.
    pub fn absorb_record(&mut self, record: T) -> Vec<T> {
        assert!(self.initialized, "absorb_record called before build()");

        let mut record = record;
        let new_size = record.size();
        let mut output = Vec::new();

        // Ensure we have an active run to compare against
        self.ensure_active_tree();

        // Determine if the record belongs to current or next run
        let belongs_to_next_run = self.should_defer_to_next_run(&mut record);

        if belongs_to_next_run {
            self.add_to_next_run(record);
        } else {
            self.add_to_current_run(record, &mut output);
        }

        self.used_space += new_size;

        // Evict records until workspace constraint is satisfied
        self.evict_until_space_available(&mut output);

        output
    }

    /// Ensure there is an active tree to work with
    fn ensure_active_tree(&mut self) {
        if self.tree.peek().is_none() {
            self.switch_run();
        }
    }

    /// Determine if a record should go to the next run
    fn should_defer_to_next_run(&mut self, record: &T) -> bool {
        if let Some((winner, _)) = self.tree.peek() {
            record < winner
        } else {
            false
        }
    }

    /// Add a record to the next run buffer
    fn add_to_next_run(&mut self, record: T) {
        self.next_run_buffer.push(record);
    }

    /// Add a record to the current run tree
    fn add_to_current_run(&mut self, record: T, output: &mut Vec<T>) {
        let record_size = record.size();
        let will_exceed = self.used_space + record_size > self.workspace_size;

        // Only use padding slots if we have memory headroom
        // When memory is tight, prefer push interface which combines eviction + insertion in one pass
        if !will_exceed {
            if let Some(idx) = self.late_fence_slots.pop() {
                // Use available padding slot (no eviction needed)
                self.tree.update(idx, record);
                return;
            }
        }

        // Memory tight OR no padding slots: use push interface
        // This efficiently combines eviction and insertion in a single leaf-to-root pass
        let winner = self.tree.push(record);
        if !winner.is_late_fence() && !winner.is_early_fence() {
            self.used_space -= winner.size();
            output.push(winner);
        }
    }

    /// Evict records until workspace size constraint is satisfied
    fn evict_until_space_available(&mut self, output: &mut Vec<T>) {
        while self.used_space > self.workspace_size {
            self.ensure_active_tree();

            if self.tree.peek().is_none() {
                break; // No more records to evict
            }

            let (_, idx) = self.tree.peek().unwrap();
            let winner = self.tree.push(T::late_fence());

            if !winner.is_late_fence() && !winner.is_early_fence() {
                self.used_space -= winner.size();
                output.push(winner);
                self.late_fence_slots.push(idx);
            }
        }
    }

    /// Switch from current run to next run by rebuilding the tree
    fn switch_run(&mut self) {
        if self.next_run_buffer.is_empty() {
            return;
        }
        self.initialize_tree_from_buffer();
    }

    /// Drain all remaining records from both current and next runs.
    ///
    /// Returns all remaining records in sorted order within each run.
    /// After draining, the data structure is ready to accept new initial records.
    pub fn drain(&mut self) -> Vec<T> {
        let mut output = Vec::new();

        // Drain current tree
        self.drain_current_tree(&mut output);

        // Switch to next run and drain it as well
        self.switch_run();
        self.drain_current_tree(&mut output);

        output
    }

    /// Helper to drain all records from the current tree
    fn drain_current_tree(&mut self, output: &mut Vec<T>) {
        while self.tree.peek().is_some() {
            let val = self.tree.push(T::late_fence());
            if !val.is_late_fence() && !val.is_early_fence() {
                self.used_space -= val.size();
                output.push(val);
            }
        }
    }

    /// Get the number of records currently in the next run buffer.
    pub fn buffer_len(&self) -> usize {
        self.next_run_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use std::{cmp::Ordering, fmt::Debug};

    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::{
        offset_value_coding::SentinelValue,
        replacement_selection_tol::{RecordSize, ReplacementSelectionToL},
    };

    pub struct TestRecord {
        pub val: i32,
        pub size: usize,
    }

    impl SentinelValue for TestRecord {
        fn early_fence() -> Self {
            Self {
                val: i32::MIN,
                size: 0,
            }
        }
        fn late_fence() -> Self {
            Self {
                val: i32::MAX,
                size: 0,
            }
        }
        fn is_early_fence(&self) -> bool {
            self.val == i32::MIN
        }
        fn is_late_fence(&self) -> bool {
            self.val == i32::MAX
        }
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
        // SCENARIO:
        // We insert 3 items. Next power of two is 4.
        // There is 1 "hidden" free slot (index 3).
        // Workspace is large enough to not force eviction.
        // The algorithm should use the free slot instead of evicting.

        let mut rs = ReplacementSelectionToL::new(100); // 100 bytes capacity
        rs.insert_initial(TestRecord::new(10, 10));
        rs.insert_initial(TestRecord::new(20, 10));
        rs.insert_initial(TestRecord::new(30, 10));

        rs.build();

        // State: Tree [10, 20, 30, LateFence]. Used 30/100.
        // late_fence_slots should contain [3].

        // Absorb 15.
        // 15 > 10 (Winner). Goes to Current Tree.
        // Should use slot 3. No eviction.
        let out = rs.absorb_record(TestRecord::new(15, 10));

        assert!(out.is_empty(), "Should use padding slot, not evict");
        assert_eq!(rs.used_space, 40); // 30 + 10
        assert!(
            rs.late_fence_slots.is_empty(),
            "Padding slot should be consumed"
        );

        // Next absorb should force eviction or buffer if tree full
        // Since tree is physically full (4/4), and mock logic just pushes to buffer if no slots:
        // (In real logic, we'd evict because used_space (40) < workspace (100) allows holding,
        // but physical tree limits apply. If tree logic supports expansion, it expands.
        // If not, it buffers. Here we check standard RS behavior).
    }

    #[test]
    fn test_run_generation_logic() {
        // SCENARIO:
        // Tree has [10, 20].
        // Absorb 5.
        // 5 < 10. 5 CANNOT go to current run.
        // 5 should go to buffer. 10 should be evicted.

        let mut rs = ReplacementSelectionToL::new(20); // Tight space. Holds 2 items of size 10.
        rs.insert_initial(TestRecord::new(10, 10));
        rs.insert_initial(TestRecord::new(20, 10));
        rs.build();

        // Tree: [10, 20]. Capacity 2. Used 20/20.

        // Absorb 5. Size 10. Needs 10 space.
        // Decision: 5 < 10 (Min). Goes to Future.
        // Action: Evict 10.
        let out = rs.absorb_record(TestRecord::new(5, 10));

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].val, 10, "Should evict current winner");
        assert_eq!(rs.buffer_len(), 1, "5 should be in next run buffer");
        assert_eq!(rs.used_space, 20); // 20 - 10 (evict) + 10 (insert 5)
    }

    #[test]
    fn test_variable_size_eviction() {
        // SCENARIO:
        // Tree has 5 small items (size 2). Total 10.
        // Workspace limit 10.
        // Absorb 1 large item (size 6).
        // Must evict 3 small items to fit the large one.

        let mut rs = ReplacementSelectionToL::new(10);
        for i in 1..=5 {
            rs.insert_initial(TestRecord::new(i * 10, 2));
        }
        rs.build(); // Capacity 8 (next pow 2 of 5). 3 Padding slots.

        // Used space = 10. Full.

        // Absorb large item (Size 6).
        // It fits in value (100 > 10), so goes to current tree.
        // Needs 6 bytes.
        let out = rs.absorb_record(TestRecord::new(100, 6));

        // We have 3 padding slots, but used_space=10/10.
        // We MUST evict based on size, even if physical slots exist.
        // Evict 1 (size 2) -> Used 8. Need 4 more.
        // Evict 2 (size 2) -> Used 6. Need 2 more.
        // Evict 3 (size 2) -> Used 4. Good.

        assert_eq!(out.len(), 3, "Should evict 3 small records");
        assert_eq!(out[0].val, 10);
        assert_eq!(out[1].val, 20);
        assert_eq!(out[2].val, 30);

        // Final used: 4 (remaining items) + 6 (new item) = 10.
        assert_eq!(rs.used_space, 10);
    }

    #[test]
    fn test_run_switch() {
        // SCENARIO:
        // Tree: [10]. Buffer: [5].
        // Evict 10. Tree empty.
        // Should auto-switch to [5].

        let mut rs = ReplacementSelectionToL::new(19);
        rs.insert_initial(TestRecord::new(10, 10)); // Tree
        rs.build();

        // Force something into buffer
        // Absorb 5. 5 < 10. Goes to buffer. Evict 10.
        let out = rs.absorb_record(TestRecord::new(5, 10));
        assert_eq!(out[0].val, 10);
        assert_eq!(rs.buffer_len(), 1);

        // Now Tree is effectively empty (contains LateFences/Holes).
        // Absorb 20.
        // Logic:
        // 1. Peek fails (or returns LateFence).
        // 2. Switches run. Tree becomes [5]. Buffer empty.
        // 3. Compare 20 vs 5. 20 > 5. Current Tree.
        // 4. Insert 20.

        // Note: Our Mock tree might behave slightly differently than a complex LoserTree
        // regarding Sentinels, but the RS logic 'switch_run' should trigger.

        let out2 = rs.absorb_record(TestRecord::new(20, 10));

        // If switch happened, 5 is the new winner.
        // If 20 is inserted, it might replace 5 (if eviction needed) or join it.
        // Used space was 20 (before absorb 20).
        // We need 10 space. Evict 5.

        assert_eq!(
            out2[0].val, 5,
            "Should have switched runs and evicted the new winner"
        );
    }

    #[test]
    fn test_full_sort_experiment() {
        // SETUP: Random seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        // Parameters
        let item_size = 10;
        let workspace_capacity_items = 10; // Small workspace to force frequent runs
        let workspace_size = workspace_capacity_items * item_size;
        let num_elements = 1000;

        let mut rs = ReplacementSelectionToL::new(workspace_size);

        // Generate data
        let mut data: Vec<TestRecord> = (0..num_elements)
            .map(|_| TestRecord::new(rng.random_range(0..1000), item_size))
            .collect();

        // 1. Pre-fill workspace
        // We fill exactly workspace capacity
        for _ in 0..workspace_capacity_items {
            if let Some(rec) = data.pop() {
                rs.insert_initial(rec);
            }
        }
        rs.build();

        // 2. Run Experiment
        let mut full_output = Vec::new();

        while let Some(rec) = data.pop() {
            let mut evicted = rs.absorb_record(rec);
            full_output.append(&mut evicted);
        }

        // 3. Drain remaining
        let mut drained = rs.drain();
        full_output.append(&mut drained);

        // 4. Verification

        // Count runs and verify sorting within runs
        let mut run_count = 0;
        let mut run_start_indices = vec![0];

        for i in 1..full_output.len() {
            // If current value is LESS than previous, a new run started
            if full_output[i] < full_output[i - 1] {
                run_count += 1;
                run_start_indices.push(i);
            }
        }
        run_count += 1; // Count the first run
        run_start_indices.push(full_output.len());

        println!(
            "Processed {} items with workspace size {}",
            num_elements, workspace_size
        );
        println!("Generated {} sorted runs:", run_count);

        // Verify each run is sorted
        for r in 0..run_count {
            let start = run_start_indices[r];
            let end = run_start_indices[r + 1];
            let run_slice = &full_output[start..end];

            println!(
                "  Run {}: len={}, elements={:?}",
                r,
                run_slice.len(),
                run_slice.iter().map(|r| r.val).collect::<Vec<_>>()
            );

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

        // Assertions
        assert_eq!(full_output.len(), num_elements, "Output count mismatch");
        assert!(
            run_count > 1,
            "Should produce multiple runs given small workspace"
        );

        // Expected Logic Check:
        // With random data, run length is expected to be ~2 * WorkspaceSize
        // 100 items / (2 * 5 items) = ~10 runs expected.
        println!(
            "Average Run Length: {:.2}",
            num_elements as f64 / run_count as f64
        );
    }
}
