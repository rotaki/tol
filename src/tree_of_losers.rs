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

    /// Traditional leaf-to-root pass algorithm (Fig. 1)
    /// Implements: void PQ::pass(Index const index, Key const key)
    ///
    /// This follows the algorithm from the paper:
    /// 1. Create a candidate node with (run_id, value)
    /// 2. For each node from leaf to root:
    ///    - If heap[slot] < candidate, swap them (loser stays, winner advances)
    /// 3. Store final winner at root
    /// 4. Return the old root value (the popped element)
    pub fn pop_and_insert(&mut self, run_id: usize, value: Option<T>) -> Option<T> {
        // Create candidate Node(index, key) - line 3 in Fig. 1
        let mut candidate = value.map_or_else(
            || Entry::new_late_fence(),
            |value| Entry::new(value, run_id),
        );

        // Index slot - line 4 in Fig. 1
        // for (leaf(index, slot); parent(slot), slot != root(); ) - line 5 in Fig. 1
        let mut slot = self.node_index(run_id);

        // Traverse from leaf to root
        while slot != self.root_index() {
            // if (heap[slot].less(candidate)) - line 6 in Fig. 1
            //     heap[slot].swap(candidate) - line 7 in Fig. 1
            if self.entries[slot].value < candidate.value {
                std::mem::swap(&mut self.entries[slot], &mut candidate);
            }

            // Move to parent
            slot = Self::parent_index(slot);
        }

        // heap[root()] = candidate - line 8 in Fig. 1
        let root = self.root_index();
        std::mem::swap(&mut self.entries[root], &mut candidate);

        // Return the old root value (what was popped)
        if candidate.value.is_early_fence() {
            None
        } else {
            Some(candidate.value)
        }
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
}