use crate::offset_value_coding::{OVC64Trait, OVCEntry, OVCU64};
use crate::replacement_selection::RecordSize;
use crate::tree_of_losers_ovc::LoserTreeOVC;

/// Replacement selection algorithm with Offset Value Coding (OVC) optimization.
///
/// This implementation extends the standard replacement selection algorithm by
/// maintaining and updating offset value codes during tree comparisons, enabling
/// efficient compression of sorted runs.
///
/// The algorithm maintains two logical partitions:
/// - Current run: Active tree from which sorted records are extracted
/// - Next run: Buffer collecting records that cannot fit in the current run
///
/// Key features:
/// - Automatic OVC updates during comparisons via `LoserTreeOVC`
/// - Memory-aware eviction based on workspace size constraints
/// - Padding slot optimization for power-of-two tree capacities
pub struct ReplacementSelectionOVC<T: OVC64Trait + RecordSize> {
    /// Tournament tree for efficient min-element extraction with OVC updates
    tree: LoserTreeOVC<T>,

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

impl RecordSize for OVCEntry {
    fn size(&self) -> usize {
        self.get_key().len()
    }
}

impl<T: OVC64Trait + RecordSize> ReplacementSelectionOVC<T> {
    /// Create a new replacement selection instance with the specified workspace size.
    ///
    /// # Arguments
    /// * `workspace_size` - Maximum memory (in bytes) that can be used for buffering
    pub fn new(workspace_size: usize) -> Self {
        Self {
            tree: LoserTreeOVC::new(vec![]),
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

        self.tree = LoserTreeOVC::new(std::mem::take(&mut self.next_run_buffer));
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
    fn should_defer_to_next_run(&mut self, record: &mut T) -> bool {
        if let Some((winner, _)) = self.tree.peek() {
            // Compare with current winner and update offset value codes
            // Returns true if record >= winner (can go in current run)
            // Returns false if record < winner (must go to next run)
            !record.derive_ovc_from(winner)
        } else {
            false
        }
    }

    /// Add a record to the next run buffer
    fn add_to_next_run(&mut self, mut record: T) {
        *record.ovc_mut() = OVCU64::initial_value();
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
    use rand::{Rng, SeedableRng, rngs::StdRng};

    use crate::offset_value_coding::is_ovc_consistent;

    use super::*;

    fn make_entry(val: u16, size: usize) -> OVCEntry {
        let mut key = vec![0; size];
        if size > 0 {
            key[0] = (val >> 8) as u8;
        }
        if size > 1 {
            key[1] = val as u8;
        }
        OVCEntry::new(key)
    }

    #[test]
    fn test_padding_optimization_usage() {
        let mut rs = ReplacementSelectionOVC::new(100);
        rs.insert_initial(make_entry(10, 10));
        rs.insert_initial(make_entry(20, 10));
        rs.insert_initial(make_entry(30, 10));

        rs.build();

        let out = rs.absorb_record(make_entry(15, 10));

        assert!(out.is_empty(), "Should use padding slot, not evict");
        assert_eq!(rs.used_space, 40);
        assert!(
            rs.late_fence_slots.is_empty(),
            "Padding slot should be consumed"
        );
    }

    #[test]
    fn test_run_generation_logic() {
        let mut rs = ReplacementSelectionOVC::new(20);
        rs.insert_initial(make_entry(10, 10));
        rs.insert_initial(make_entry(20, 10));
        rs.build();

        let out = rs.absorb_record(make_entry(5, 10));

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].get_key()[1], 10);
        assert_eq!(rs.buffer_len(), 1);
        assert_eq!(rs.used_space, 20);
    }

    #[test]
    fn test_variable_size_eviction() {
        let mut rs = ReplacementSelectionOVC::new(10);
        for i in 1..=5 {
            rs.insert_initial(make_entry(i * 10, 2));
        }
        rs.build();

        let out = rs.absorb_record(make_entry(100, 6));

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].get_key()[1], 10);
        assert_eq!(out[1].get_key()[1], 20);
        assert_eq!(out[2].get_key()[1], 30);
        assert_eq!(rs.used_space, 10);
    }

    #[test]
    fn test_run_switch() {
        let mut rs = ReplacementSelectionOVC::new(19);
        rs.insert_initial(make_entry(10, 10));
        rs.build();

        let out = rs.absorb_record(make_entry(5, 10));
        assert_eq!(out[0].get_key()[1], 10);
        assert_eq!(rs.buffer_len(), 1);

        let out2 = rs.absorb_record(make_entry(20, 10));
        assert_eq!(out2[0].get_key()[1], 5);
    }

    #[test]
    fn test_full_sort_experiment() {
        let mut rng = StdRng::seed_from_u64(40);
        let item_size = 10;
        let workspace_capacity_items = 10;
        let workspace_size = workspace_capacity_items * item_size;
        let num_elements = 300;

        let mut rs = ReplacementSelectionOVC::new(workspace_size);

        let mut data: Vec<OVCEntry> = (0..num_elements)
            .map(|_| make_entry(rng.random_range(0..1000), item_size))
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
            if full_output[i].key() < full_output[i - 1].key() {
                run_count += 1;
                run_start_indices.push(i);
            }
        }
        run_count += 1;
        run_start_indices.push(full_output.len());

        println!(
            "Processed {} items with workspace size {}",
            num_elements, workspace_size
        );
        println!("Generated {} sorted runs:", run_count);

        for r in 0..run_count {
            let start = run_start_indices[r];
            let end = run_start_indices[r + 1];
            let run_slice = &full_output[start..end];
            let consistent = is_ovc_consistent(run_slice);
            println!(
                "Run {} (len={}, consistent={}):",
                r,
                run_slice.len(),
                consistent
            );
            for entry in run_slice {
                println!("  ovc={:?}, key={:?}", entry.get_ovc(), entry.get_key());
            }
            for i in 1..run_slice.len() {
                assert!(
                    run_slice[i].key() >= run_slice[i - 1].key(),
                    "Run {} is unsorted at index {}! {:?} < {:?}",
                    r,
                    i,
                    run_slice[i].key(),
                    run_slice[i - 1].key()
                );
            }
        }

        assert_eq!(full_output.len(), num_elements);
        assert!(run_count > 1);

        // Expected Logic Check:
        // With random data, run length is expected to be ~2 * WorkspaceSize
        // 100 items / (2 * 5 items) = ~10 runs expected.
        println!(
            "Average Run Length: {:.2}",
            num_elements as f64 / run_count as f64
        );
    }
}
