# Historical GPU experiment implementations

These are the actual experiment scripts archived from the study. The original server paths and task-manifest dependencies are preserved for provenance. They are readable historical implementations, not standalone commands for an arbitrary machine.

| File | Role |
| --- | --- |
| run_freq10_guided12_holdout.py | Frozen frequency-guidance candidate, zero-intervention check and paired outputs |
| weight_head_edit_round9.py | Selected attention-head weight edits and random controls |
| collect_output_head_distillation_round17.py | Teacher rollout collection and output-projection fitting |
| distill_output_head_helpers.py | Sufficient statistics, ridge fit and norm-matched random patches |
| evaluate_output_head_patches.py | Static patch evaluation and exact restoration checks |
| run_early_weight_patch_round20.py | First-12-step-only weight intervention |
| evaluate_art_general.py | Content and style proxy evaluation |

The frequency scripts reference the FreqEdit authors' utilities, which must be obtained from https://github.com/FreqEdit/FreqEdit under the applicable upstream terms. The large sufficient-statistics file is retained in the private research backup. The fitted 3.5 MiB projection artifact is included in `artifacts/`.

For a portable native/static-patch entry point, use `scripts/run_image_edit.py`. Its argument parsing and artifact compatibility checks have been checked locally; this newly extracted entry point has not been rerun on a GPU. Historical results come from the archived scripts above.
