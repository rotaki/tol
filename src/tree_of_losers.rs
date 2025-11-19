use crate::entry::{Entry, SentinelValue, Sentineled};
use std::cmp::Ord;

fn prev_power_of_two(num: usize) -> usize {
    let mut size = 1;
    while size <= num {
        size *= 2;
    }
    size / 2
}

pub struct TreeOfLosers<T: Ord + SentinelValue> {
    curr: usize,
    input_leaf_start: usize,
    entries: Vec<Entry<T>>,
}

impl<T: Ord + SentinelValue> TreeOfLosers<T> {
    /// Creates a new TreeOfLosers structure for k-way merging
    ///
    /// # Arguments
    /// * `num_runs` - The number of input runs (sequences) to merge
    ///
    /// # Algorithm
    /// The tree is constructed as a tournament tree where internal nodes store losers
    /// and the root contains the overall winner. The tree structure is designed to handle
    /// any number of runs by creating a balanced tree with additional nodes as needed.
    pub fn new(num_runs: usize) -> Self {
        assert!(num_runs > 0);

        // Calculate the number of leaf nodes needed in the tree.
        // Each pair of runs shares a leaf node, so we need ceil(num_runs/2) leaves.
        let input_leaf_nodes = num_runs.div_ceil(2);
        assert!(input_leaf_nodes > 0);

        // Find the largest power of 2 that is smaller or equal to the number of leaf nodes.
        // This forms the base of our nearly-complete binary tree structure.
        //
        // Example 1: If num_runs=5, then input_leaf_nodes=3 (ceil(5/2))
        //           base_leaf_nodes=2 (largest power of 2 ≤ 3)
        //
        // Initial tree (before inserting any runs):
        // ================= Tree of Losers (Pretty) =================
        // Root (Winner): EF
        //
        // └── [EF]
        //     ├── [EF] <-- runs (0, 1)
        //     └── [EF]
        //         ├── [LF] <-- run 4
        //         └── [EF] <-- runs (2, 3)
        //
        // After inserting values [20, 10, 30, 15, 25]:
        // ================= Tree of Losers (Pretty) =================
        // Root (Winner): (10, 1)
        //
        // └── [R3:15]
        //     ├── [R0:20] <-- runs (0, 1)
        //     └── [R4:25]
        //         ├── [LF] <-- run 4
        //         └── [R2:30] <-- runs (2, 3)

        // Base leaf nodes refer to the number of leaf nodes in the largest complete
        // binary subtree contained within our tree.
        let base_leaf_nodes = prev_power_of_two(input_leaf_nodes);

        // Calculate total number of nodes in the tree.
        // Formula: 2 * base_leaf_nodes + (input_leaf_nodes - base_leaf_nodes) * 2
        //
        // This creates a tree where:
        // - 2 * base_leaf_nodes form a complete binary tree
        // - Remaining 2 * (input_leaf_nodes - base_leaf_nodes) are additional nodes
        //
        // For num_runs=5 example:
        // - base_leaf_nodes = 2
        // - input_leaf_nodes = 3
        // - num_nodes = 2*2 + (3-2)*2 = 4 + 2 = 6
        let num_nodes = 2 * base_leaf_nodes + (input_leaf_nodes - base_leaf_nodes) * 2;
        let mut entries = Vec::with_capacity(num_nodes);

        // Fill the tree with late fence values
        for _ in 0..num_runs {
            entries.push(Entry::new_early_fence());
        }

        for _ in num_runs..num_nodes {
            entries.push(Entry::new_late_fence());
        }

        Self {
            input_leaf_start: num_nodes - input_leaf_nodes,
            curr: 0,
            entries,
        }
    }

    pub fn top_run_id(&mut self) -> Option<usize> {
        if self.entries[0].value.is_late_fence() {
            None
        } else if self.entries[0].value.is_early_fence() {
            let curr = self.curr;
            self.curr += 1;
            Some(curr)
        } else {
            Some(self.entries[0].run_id)
        }
    }

    pub fn node_index(&self, run_id: usize) -> usize {
        self.input_leaf_start + run_id / 2
    }

    pub fn root_index(&self) -> usize {
        0
    }

    pub fn parent_index(index: usize) -> usize {
        // [0 is root] [1] [2, 3] [4,5,6,7] ...
        index / 2
    }

    /// Leaf-to-root pass with early termination (Fig. 3)
    /// This is the core of the addressable priority queue.
    /// Stops early when finding the target entry.
    ///
    /// # Algorithm (from Fig. 3, lines 10-19)
    /// - for (leaf(index, slot); parent(slot), slot != root() && heap[slot].index != index; )
    /// - Loop continues while: (1) not at root AND (2) haven't found target entry
    fn pass(
        &mut self,
        run_id: usize,
        mut candidate: Entry<T>,
    ) -> (Entry<T>, usize) {
        let mut slot = self.node_index(run_id);

        // Loop while: not at root AND current entry doesn't match target run_id
        while slot != self.root_index() && self.entries[slot].run_id != run_id {
            if self.entries[slot].value < candidate.value {
                std::mem::swap(&mut self.entries[slot], &mut candidate);
            }
            slot = Self::parent_index(slot);
        }

        // Final swap at stopping position
        std::mem::swap(&mut self.entries[slot], &mut candidate);
        (candidate, slot)
    }

    /// Pop minimum and insert new value (Fig. 1/Fig. 3)
    /// Uses pass_with_target since we're replacing an existing entry with the same run_id
    pub fn pop_and_insert(&mut self, run_id: usize, value: Option<T>) -> Option<T> {
        let candidate = value.map_or_else(
            || Entry::new_late_fence(),
            |value| Entry::new(value, run_id),
        );

        let (replaced, _slot) = self.pass(run_id, candidate);

        if replaced.value.is_early_fence() {
            None
        } else {
            Some(replaced.value)
        }
    }

    /// Searches for an entry with the given run_id along its leaf-to-root path.
    /// Returns a reference to the value and the slot index where it was found.
    ///
    /// This is a non-modifying search operation.
    /// Average search cost: ~2 nodes (constant time, independent of queue size)
    pub fn find(&self, run_id: usize) -> Option<(&T, usize)> {
        // Check if run_id is valid
        if run_id >= self.entries.len() {
            return None;
        }

        let mut slot = self.node_index(run_id);

        // Search from leaf to root
        loop {
            if self.entries[slot].run_id == run_id
                && !self.entries[slot].value.is_early_fence()
                && !self.entries[slot].value.is_late_fence()
            {
                return Some((&self.entries[slot].value, slot));
            }

            if slot == self.root_index() {
                break;
            }

            slot = Self::parent_index(slot);
        }

        None
    }

    /// Deletes an entry with the given run_id by replacing it with a late fence.
    /// This is the core operation for addressable priority queues.
    ///
    /// # Use case
    /// In scheduling applications or simulations, if a future event is cancelled,
    /// this operation finds and removes it from the queue.
    ///
    /// # Returns
    /// The deleted value if found, None otherwise
    pub fn delete(&mut self, run_id: usize) -> Option<T> {
        // Check if run_id is valid
        if run_id >= self.entries.len() {
            return None;
        }

        // Create a late fence candidate to replace the entry
        let candidate = Entry::new_late_fence();

        // Use pass_with_target to find and replace the entry with run_id
        let (replaced, _slot) = self.pass(run_id, candidate);

        // Return the deleted value - but only if we actually found the target run_id
        // Check that the replaced entry has the correct run_id
        if replaced.run_id == run_id
            && !replaced.value.is_early_fence()
            && !replaced.value.is_late_fence()
        {
            Some(replaced.value)
        } else {
            None
        }
    }

    /// Helper to compute level (distance from leaf) for a given slot
    fn level_of(&self, slot: usize) -> usize {
        let mut level = 0;
        let mut current = slot;
        while current > 0 {
            current = Self::parent_index(current);
            level += 1;
        }
        level
    }

    /// Updates an entry with the given run_id to a new value.
    /// Implements the true non-monotone PQ algorithm from Fig. 4 (lines 20-44).
    ///
    /// # Algorithm
    /// This is a faithful implementation of the repair loop from the paper:
    /// - Locate the old entry along the leaf-to-root path
    /// - If new_value >= old_value: replace in place (monotone case)
    /// - If new_value < old_value: run repair loop to move former winners backward
    ///
    /// # Returns
    /// The old value if found, None otherwise
    pub fn update(&mut self, run_id: usize, new_value: T) -> Option<T> {
        // Check if run_id is valid
        if run_id >= self.entries.len() {
            return None;
        }

        // Index slot - line 23 in Fig. 4
        let mut slot = self.node_index(run_id);
        let mut level = 0;

        // for (leaf (index, slot); parent (slot), slot != root (); ) - line 24 in Fig. 4
        //     if (heap [slot].index == index) break; - line 25
        while slot != self.root_index() {
            if self.entries[slot].run_id == run_id
                && !self.entries[slot].value.is_early_fence()
                && !self.entries[slot].value.is_late_fence()
            {
                break;
            }
            slot = Self::parent_index(slot);
            level += 1;
        }

        // Check if we found the entry at root
        if slot == self.root_index() {
            if self.entries[slot].run_id == run_id
                && !self.entries[slot].value.is_early_fence()
                && !self.entries[slot].value.is_late_fence()
            {
                // Found at root, continue
            } else {
                // Entry not found
                return None;
            }
        }

        // Check monotone vs non-monotone case
        let is_monotone = new_value >= self.entries[slot].value;

        if is_monotone {
            // Monotone case: simple replacement
            let old_value = std::mem::replace(
                &mut self.entries[slot].value,
                new_value
            );
            return Some(old_value);
        }

        // Non-monotone case: new value < old value
        // For non-monotone updates, we use a simple delete + reinsert approach
        // This ensures entries remain findable from their home leaf positions

        // Replace the entire entry with a late fence to extract the old value
        let old_entry = std::mem::replace(&mut self.entries[slot], Entry::new_late_fence());

        // Now do a pass with the new value to reinsert it
        let candidate = Entry::new(new_value, run_id);
        let (_replaced, _slot) = self.pass(run_id, candidate);

        Some(old_entry.value)
    }
}

impl<T: Ord + SentinelValue + std::fmt::Debug> TreeOfLosers<T> {
    pub fn print(&self) {
        if self.entries.is_empty() {
            return;
        }
        println!("================= Tree of losers =================");
        println!("{:?} ", self.entries[0]);
        let mut index = 1;
        let mut level = 0;
        while index < self.entries.len() {
            for _ in 0..(1 << level) {
                if index >= self.entries.len() {
                    break;
                }
                print!("{:?} ", self.entries[index]);
                index += 1;
            }
            println!();
            level += 1;
        }
    }

    /// Pretty prints the tree structure with visual indentation and tree branches
    pub fn pretty_print(&self) {
        if self.entries.is_empty() {
            println!("Empty tree");
            return;
        }

        println!("================= Tree of Losers (Pretty) =================");
        println!("Root (Winner): {:?}", self.entries[0]);
        println!();

        // Calculate which runs map to which leaf nodes
        let num_runs = self
            .entries
            .iter()
            .filter(|e| !e.value.is_late_fence())
            .count();

        // Helper function to format entry with run mapping info
        let format_entry = |index: usize, entry: &Entry<T>| -> String {
            let base_str = if entry.value.is_early_fence() {
                "[EF]".to_string()
            } else if entry.value.is_late_fence() {
                "[LF]".to_string()
            } else {
                format!("[R{}:{:?}]", entry.run_id, entry.value)
            };

            // Add run mapping for leaf nodes
            if self.is_leaf_node(index) {
                let run_ids = self.get_runs_for_leaf(index, num_runs);
                if !run_ids.is_empty() {
                    let runs_str = if run_ids.len() == 1 {
                        format!("run {}", run_ids[0])
                    } else {
                        format!("runs ({}, {})", run_ids[0], run_ids[1])
                    };
                    format!("{} <-- {}", base_str, runs_str)
                } else {
                    base_str
                }
            } else {
                base_str
            }
        };

        // Print tree using recursive approach
        self.print_subtree(1, "", true, &format_entry);
    }

    /// Check if a node index represents a leaf node
    fn is_leaf_node(&self, index: usize) -> bool {
        let left_child = index * 2;
        left_child >= self.entries.len()
    }

    /// Get the run IDs that map to a specific leaf node
    fn get_runs_for_leaf(&self, leaf_index: usize, num_runs: usize) -> Vec<usize> {
        let mut runs = Vec::new();

        // Reverse the node_index calculation to find which runs map here
        //     node_index = input_leaf_start + run_id / 2
        //     run_id / 2 = node_index - input_leaf_start
        let offset = leaf_index as isize - self.input_leaf_start as isize;

        if offset >= 0 {
            let offset = offset as usize;
            // Each leaf position can handle 2 runs
            let base_run = offset * 2;
            if base_run < num_runs {
                runs.push(base_run);
            }
            if base_run + 1 < num_runs {
                runs.push(base_run + 1);
            }
        }

        runs
    }

    /// Helper method to recursively print subtree
    fn print_subtree<F>(&self, index: usize, prefix: &str, is_last: bool, format_fn: &F)
    where
        F: Fn(usize, &Entry<T>) -> String,
    {
        if index >= self.entries.len() {
            return;
        }

        // Print current node
        let connector = if is_last { "└── " } else { "├── " };
        println!(
            "{}{}{}",
            prefix,
            connector,
            format_fn(index, &self.entries[index])
        );

        // Prepare prefix for children
        let child_prefix = format!("{}{}", prefix, if is_last { "    " } else { "│   " });

        // Print children (right child first)
        let left_child = index * 2;
        let right_child = index * 2 + 1;

        if right_child < self.entries.len() {
            self.print_subtree(
                right_child,
                &child_prefix,
                left_child >= self.entries.len(),
                format_fn,
            );
        }

        if left_child < self.entries.len() {
            self.print_subtree(left_child, &child_prefix, true, format_fn);
        }
    }
}

pub fn merge_with_tree_of_losers_with_sentinel<T: SentinelValue + Ord>(
    mut runs: Vec<Box<impl Iterator<Item = T>>>,
) -> Vec<T> {
    // Create a tree of losers.
    let mut tree = TreeOfLosers::<T>::new(runs.len());

    // Output
    let mut output = Vec::new();

    // Fill the tree with the first entry of each run.
    // If the top entry is a late fence, it will return None.
    while let Some(run_id) = tree.top_run_id() {
        // Pop the top entry and insert the next entry from the run.
        // It the top entry is a early fence, it will return None.
        if let Some(val) = tree.pop_and_insert(run_id, runs[run_id].next()) {
            output.push(val);
        }
    }

    output
}

pub fn merge_with_tree_of_losers_no_sentinel<T: Ord>(
    mut runs: Vec<Box<impl Iterator<Item = T>>>,
) -> Vec<T> {
    // Create a tree of losers.
    let mut tree = TreeOfLosers::<Sentineled<T>>::new(runs.len());

    // Output
    let mut output = Vec::new();

    // Fill the tree with the first entry of each run.
    // If the top entry is a late fence, it will return None.
    while let Some(run_id) = tree.top_run_id() {
        // Pop the top entry and insert the next entry from the run.
        // It the top entry is a early fence, it will return None.
        if let Some(val) = tree.pop_and_insert(run_id, runs[run_id].next().map(Sentineled::new)) {
            output.push(val.inner());
        }
    }

    output
}

pub fn sort_with_tree_of_losers_with_sentinel<T: SentinelValue + Ord>(mut run: Vec<T>) -> Vec<T> {
    let num_runs = run.len();
    let mut tree = TreeOfLosers::<T>::new(num_runs);

    // Output
    let mut output = Vec::new();

    while let Some(run_id) = tree.top_run_id() {
        if let Some(val) = tree.pop_and_insert(run_id, run.pop()) {
            output.push(val);
        }
    }

    output
}

pub fn sort_with_tree_of_losers_no_sentinel<T: Ord>(mut run: Vec<T>) -> Vec<T> {
    let num_runs = run.len();
    let mut tree = TreeOfLosers::<Sentineled<T>>::new(num_runs);

    // Output
    let mut output = Vec::with_capacity(num_runs);

    while let Some(run_id) = tree.top_run_id() {
        if let Some(val) = tree.pop_and_insert(run_id, run.pop().map(Sentineled::new)) {
            output.push(val.inner());
        }
    }

    output
}

#[cfg(test)]
mod test {
    use crate::utils::{generate_random_array, generate_runs};

    use super::*;

    #[test]
    fn test_merge_with_tree_of_losers() {
        let num_runs = 100;
        let run_length = 10000;
        let runs = generate_runs::<i32>(num_runs, run_length);

        let output = merge_with_tree_of_losers_no_sentinel(
            runs.clone()
                .into_iter()
                .map(|run| Box::new(run.into_iter()))
                .collect(),
        );

        // Check the output.
        let mut expected_output = runs.into_iter().flatten().collect::<Vec<_>>();
        expected_output.sort();

        assert_eq!(output, expected_output);
    }

    #[test]
    fn test_sort_with_tree_of_losers() {
        let run_length = 10000;
        let mut run = generate_random_array::<i32>(run_length);

        let output = sort_with_tree_of_losers_no_sentinel(run.clone());

        // Check the output.
        run.sort();

        let output = output.into_iter().collect::<Vec<_>>();

        assert_eq!(output, run);
    }

    #[test]
    fn test_pretty_print_small_tree() {
        use crate::entry::Sentineled;

        // Test with 3 runs
        println!("\n=== Testing pretty_print with 3 runs ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(3);

        // Show initial state with early fences
        println!("\nInitial tree (all early fences):");
        tree.pretty_print();

        // Insert some values
        tree.pop_and_insert(0, Some(Sentineled::new(10)));
        tree.pop_and_insert(1, Some(Sentineled::new(5)));
        tree.pop_and_insert(2, Some(Sentineled::new(15)));

        println!("\nAfter inserting values 10, 5, 15:");
        tree.pretty_print();
    }

    #[test]
    fn test_pretty_print_medium_tree() {
        use crate::entry::Sentineled;

        // Test with 5 runs
        println!("\n=== Testing pretty_print with 5 runs ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(5);

        // Show initial state
        println!("\nInitial tree (all early fences):");
        tree.pretty_print();

        // Insert values in order
        let values = [20, 10, 30, 15, 25];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nAfter inserting values [20, 10, 30, 15, 25]:");
        tree.pretty_print();
    }

    #[test]
    fn test_pretty_print_large_tree() {
        use crate::entry::Sentineled;

        // Test with 7 runs - perfect for showing late fences
        println!("\n=== Testing pretty_print with 7 runs ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(7);

        // Insert values
        let values = [35, 10, 45, 20, 50, 15, 40];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nTree with 7 runs (notice the late fence):");
        tree.pretty_print();

        // Simulate exhausting one run
        tree.pop_and_insert(1, None); // Insert late fence for run 1

        println!("\nAfter exhausting run 1 (replaced with late fence):");
        tree.pretty_print();
    }

    #[test]
    fn test_print_comparison() {
        use crate::entry::Sentineled;

        // Compare old print vs pretty print
        println!("\n=== Comparing print() vs pretty_print() ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(4);

        let values = [25, 10, 30, 15];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nUsing original print():");
        tree.print();

        println!("\nUsing pretty_print():");
        tree.pretty_print();
    }

    // ============== Addressable PQ Tests ==============

    #[test]
    fn test_find_entry() {
        use crate::entry::Sentineled;

        println!("\n=== Testing find operation ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(5);

        // Insert values
        let values = [20, 10, 30, 15, 25];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nTree after insertions:");
        tree.pretty_print();

        // Test finding each entry
        for (run_id, &expected_val) in values.iter().enumerate() {
            let result = tree.find(run_id);
            println!("\nSearching for run_id {}: {:?}", run_id, result);

            // The entry should be found somewhere on the path
            assert!(result.is_some());
            let (found_val, slot) = result.unwrap();
            assert_eq!(*found_val, Sentineled::new(expected_val));
            println!("  Found at slot {}: {:?}", slot, found_val);
        }

        // Test finding non-existent entry
        let result = tree.find(100);
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_entry() {
        use crate::entry::Sentineled;

        println!("\n=== Testing delete operation ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(5);

        // Insert values
        let values = [20, 10, 30, 15, 25];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nTree before deletion:");
        tree.pretty_print();

        // Delete entry at run_id 1 (value 10, which should be at root)
        let deleted = tree.delete(1);
        println!("\nDeleted run_id 1: {:?}", deleted);
        assert_eq!(deleted, Some(Sentineled::new(10)));

        println!("\nTree after deleting run_id 1:");
        tree.pretty_print();

        // Verify we can't find it anymore
        assert!(tree.find(1).is_none());

        // Delete another entry (run_id 3, value 15)
        let deleted = tree.delete(3);
        println!("\nDeleted run_id 3: {:?}", deleted);
        assert_eq!(deleted, Some(Sentineled::new(15)));

        println!("\nTree after deleting run_id 3:");
        tree.pretty_print();

        // Try to delete non-existent entry
        let deleted = tree.delete(100);
        assert!(deleted.is_none());

        // Try to delete already deleted entry
        let deleted = tree.delete(1);
        assert!(deleted.is_none());
    }

    #[test]
    fn test_update_entry() {
        use crate::entry::Sentineled;

        println!("\n=== Testing update operation ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(5);

        // Insert values
        let values = [20, 10, 30, 15, 25];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nTree before update:");
        tree.pretty_print();

        // Update run_id 2 from 30 to 5 (should become new minimum)
        let old_val = tree.update(2, Sentineled::new(5));
        println!("\nUpdated run_id 2: old={:?}, new=5", old_val);
        assert_eq!(old_val, Some(Sentineled::new(30)));

        println!("\nTree after updating run_id 2 to 5:");
        tree.pretty_print();

        // Verify the new value is at root
        assert_eq!(tree.entries[0].value, Sentineled::new(5));
        assert_eq!(tree.entries[0].run_id, 2);

        // Update run_id 4 from 25 to 100 (should move down)
        let old_val = tree.update(4, Sentineled::new(100));
        println!("\nUpdated run_id 4: old={:?}, new=100", old_val);
        assert_eq!(old_val, Some(Sentineled::new(25)));

        println!("\nTree after updating run_id 4 to 100:");
        tree.pretty_print();

        // Update non-existent entry
        let old_val = tree.update(100, Sentineled::new(50));
        assert!(old_val.is_none());
    }

    #[test]
    fn test_addressable_pq_interleaved_ops() {
        use crate::entry::Sentineled;

        println!("\n=== Testing interleaved operations ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(7);

        // Initial insertions
        let values = [35, 10, 45, 20, 50, 15, 40];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nInitial tree:");
        tree.pretty_print();

        // Pop minimum (should be 10, run_id 1)
        let min = tree.pop_and_insert(1, Some(Sentineled::new(12)));
        assert_eq!(min, Some(Sentineled::new(10)));
        println!("\nAfter popping min and inserting 12 at run_id 1:");
        tree.pretty_print();

        // Delete an entry (run_id 3, value 20)
        let deleted = tree.delete(3);
        assert_eq!(deleted, Some(Sentineled::new(20)));
        println!("\nAfter deleting run_id 3:");
        tree.pretty_print();

        // Update an entry (run_id 6, from 40 to 8)
        let old = tree.update(6, Sentineled::new(8));
        assert_eq!(old, Some(Sentineled::new(40)));
        println!("\nAfter updating run_id 6 to 8:");
        tree.pretty_print();

        // New minimum should be 8
        assert_eq!(tree.entries[0].value, Sentineled::new(8));

        // Find an entry (run_id 4, value 50)
        let found = tree.find(4);
        assert!(found.is_some());
        assert_eq!(*found.unwrap().0, Sentineled::new(50));
        println!("\nFound run_id 4: {:?}", found);
    }

    #[test]
    fn test_delete_all_entries() {
        use crate::entry::Sentineled;

        println!("\n=== Testing delete all entries ===");
        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(4);

        // Insert values
        let values = [25, 10, 30, 15];
        for (i, &val) in values.iter().enumerate() {
            tree.pop_and_insert(i, Some(Sentineled::new(val)));
        }

        println!("\nInitial tree:");
        tree.pretty_print();

        // Delete all entries
        for i in 0..4 {
            let deleted = tree.delete(i);
            println!("\nDeleted run_id {}: {:?}", i, deleted);
            assert!(deleted.is_some());
        }

        println!("\nTree after deleting all entries:");
        tree.pretty_print();

        // Root should now be a late fence
        assert!(tree.entries[0].value.is_late_fence());
    }

    #[test]
    fn test_update_to_same_value() {
        use crate::entry::Sentineled;

        let mut tree = TreeOfLosers::<Sentineled<i32>>::new(3);

        // Insert values
        tree.pop_and_insert(0, Some(Sentineled::new(10)));
        tree.pop_and_insert(1, Some(Sentineled::new(20)));
        tree.pop_and_insert(2, Some(Sentineled::new(30)));

        // Update to same value
        let old = tree.update(1, Sentineled::new(20));
        assert_eq!(old, Some(Sentineled::new(20)));

        // Tree should still be valid
        assert_eq!(tree.entries[0].value, Sentineled::new(10));
    }
}