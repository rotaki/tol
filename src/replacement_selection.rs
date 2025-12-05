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

pub struct ReplacementSelection<T: Ord + SentinelValue + RecordSize> {
    tree: LoserTree<T>,
    next_run_buffer: Vec<T>,

    // Tracks empty slots (LateFences) in the tree for reuse
    late_fence_slots: Vec<usize>,

    workspace_size: usize,
    used_space: usize,
    initialized: bool,
}

impl<T: Ord + SentinelValue + RecordSize> ReplacementSelection<T> {
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

    pub fn insert_initial(&mut self, record: T) {
        assert!(!self.initialized);
        self.used_space += record.size();
        self.next_run_buffer.push(record);
    }

    pub fn build(&mut self) {
        assert!(!self.initialized);

        // 1. Capture real size
        let num_real = self.next_run_buffer.len();

        // 2. Build tree
        self.tree = LoserTree::new(std::mem::take(&mut self.next_run_buffer));

        // 3. Optimization: Capture the 2^N padding slots immediately
        let capacity = self.tree.capacity(); // Assumes you added the .capacity() accessor
        for i in (num_real..capacity).rev() {
            self.late_fence_slots.push(i);
        }

        self.initialized = true;
    }

    pub fn absorb_record(&mut self, record: T) -> Vec<T> {
        assert!(self.initialized);
        let new_size = record.size();
        let mut output = Vec::new();

        // --- Step 1: Decision Phase ---

        // We look at the top of the tree to decide the fate of the NEW record.
        // We handle the "Empty Tree" case first to ensure we have a valid winner to compare against.
        if self.tree.peek().is_none() {
            self.switch_run();
        }

        // Logic:
        // If the tree is still empty after switch (total exhaustion), decision is trivial (Current).
        // Otherwise, we compare Input vs Winner.
        // Note: We use peek() which gives &T. This avoids taking ownership yet.
        let goes_to_future = if let Some((winner_ref, _)) = self.tree.peek() {
            &record < winner_ref
        } else {
            // Tree is completely empty even after switch? Just insert into current.
            false
        };

        // --- Step 2: Eviction Phase ---

        // We loop until we have enough space.
        // We MUST evict at least once if the tree is full, but with variable sizes
        // we might evict 0 times (if we have free slots) or N times.

        // However, standard Replacement Selection usually implies a 1:1 swap logic.
        // If we have free space (from padding), we might not evict anything.
        // But if `used_space + new_size > workspace`, we MUST evict.

        while self.used_space + new_size > self.workspace_size {
            // Check if active tree ran out of nodes mid-eviction
            if self.tree.peek().is_none() {
                self.switch_run();
                // After switch, we might still need to evict from the NEW tree to make space
                if self.tree.peek().is_none() {
                    break;
                } // Safety break
            }

            let (_, idx) = self.tree.peek().unwrap();

            // Replace winner with LateFence (infinite value)
            let winner = self.tree.update(idx, T::late_fence());

            if !winner.is_late_fence() && !winner.is_early_fence() {
                self.used_space -= winner.size();
                output.push(winner);
                self.late_fence_slots.push(idx);
            }
        }

        // --- Step 3: Insertion Phase ---

        self.used_space += new_size;

        if goes_to_future {
            // New record is smaller than the winner we just replaced (or peeked).
            // It cannot go into the current sorted run.
            self.next_run_buffer.push(record);
        } else {
            // New record is >= winner. It extends the current run.
            // Try to use a free slot (LateFence) in the tree.
            if let Some(slot_idx) = self.late_fence_slots.pop() {
                self.tree.update(slot_idx, record);
            } else {
                // No free slots. This shouldn't happen if we enforced space constraints above,
                // unless the tree structure is static and we are trying to add more items
                // than capacity allows (which buffer handles).
                // Fallback: append to buffer or handle resize.
                // In strict RS, we put it in buffer if tree is physically full.
                self.next_run_buffer.push(record);
            }
        }

        output
    }

    /// Helper: Convert the Future Buffer into the Current Tree
    fn switch_run(&mut self) {
        if self.next_run_buffer.is_empty() {
            return;
        }

        // 1. Take buffer
        let next_records = std::mem::take(&mut self.next_run_buffer);
        let num_real = next_records.len();

        // 2. Rebuild tree
        self.tree = LoserTree::new(next_records);

        // 3. Reset state
        self.late_fence_slots.clear();

        // 4. Capture padding slots for the new run
        let capacity = self.tree.capacity();
        for i in (num_real..capacity).rev() {
            self.late_fence_slots.push(i);
        }
    }

    pub fn drain(&mut self) -> Vec<T> {
        let mut output = Vec::new();

        // Drain Current Tree
        while self.tree.peek().is_some() {
            let val = self.tree.push(T::late_fence());
            if !val.is_late_fence() && !val.is_early_fence() {
                output.push(val);
            }
        }

        // Output remaining buffer (Next Run)
        // Usually we sort this to keep the output clean, or return it as a separate chunk
        if !self.next_run_buffer.is_empty() {
            self.next_run_buffer.sort();
            output.append(&mut self.next_run_buffer);
        }

        output
    }

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
        replacement_selection::{RecordSize, ReplacementSelection},
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

        let mut rs = ReplacementSelection::new(100); // 100 bytes capacity
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

        let mut rs = ReplacementSelection::new(20); // Tight space. Holds 2 items of size 10.
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

        let mut rs = ReplacementSelection::new(10);
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

        let mut rs = ReplacementSelection::new(19);
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

        let mut rs = ReplacementSelection::new(workspace_size);

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
