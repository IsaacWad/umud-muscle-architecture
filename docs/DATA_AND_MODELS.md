# Data and models

Original competition data were organized locally under `data/raw/apo_imgs_v1`, `apo_masks_v1`, `fasc_imgs_v1`, `fasc_masks_v1`, and `test_images_v2/test_set_v2`. This package does not redistribute those source images or annotation sets. It preserves the final 309-row prediction export and associated project audits.

Four local checkpoints were identified from the saved inference setup:

| Role | Source checkpoint | Saved epoch |
| --- | --- | ---: |
| Aponeurosis model | `outputs/aponeurosis_model_v2/best_aponeurosis_model_v2.pt` | 17 |
| Orientation/concentration | `notebooks/fascicle_recovered_uncertainty_v1_run/best.pt` | 27 |
| Selection support | `support_manual_v1/best_support.pt` within the uncertainty run | 8 |
| Measurement support | `support_apo_detected_20261112_180130_634276/best_support.pt` within that run | 15 |

The full-precision tensor values are preserved exactly. Optimizer state and machine-specific training paths were omitted from the exported inference checkpoints. Large checkpoint payloads are divided into 16 MiB parts. Keep every part together in `models/`; `run_v8.py` checks the hashes and loads the combined bytes directly in memory. The parts are not standalone checkpoint files. No precision reduction or model quantization was applied.

`models/manifest.json` records source checkpoint paths, epochs, part sizes, and checksums. This establishes the provenance of the packaged models, not proof that every historical interactive session used exactly the same live state. A complete training replay and the unsaved final interactive cells are outside this snapshot.
