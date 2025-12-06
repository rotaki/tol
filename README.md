# tol helpers

Common commands to reproduce the instrumentation and comparison outputs:

- Replacement selection counts (byte/OVC comparisons; add `--features instrument_calls` to include push/update breakdowns):  
  `cargo run --example compare_replacement_selection_counts --quiet`  
  `cargo run --example compare_replacement_selection_counts --features instrument_calls --quiet`

- Merge counts (byte and OVC comparisons for loser-tree merge):  
  `cargo run --example compare_merge_counts --quiet`

- Focused instrumentation test for padding-slot updates (requires feature flag):  
  `cargo test --features instrument_calls test_absorb_record_uses_update_with_padding_and_headroom -- --nocapture`

- Benchmarks (Criterion):  
  `cargo bench --bench merge_bench`  
  `cargo bench --bench replacement_selection_bench`

Notes:
- The `instrument_calls` feature enables push/update counters in both loser-tree variants and in the examples’ output.
- Examples use fixed seeds; adjust the source files if you want different data sizes or patterns.
