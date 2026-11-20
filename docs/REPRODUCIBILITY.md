# Reproducibility and final review

## Preserved evidence

The final submission, weighted PA audit, verified-ruler submission, and local source dependency chain are copied from the completed project. The final PA values agree with the per-image `weighted_pa_deg` audit values for all 309 images. The audit records one pair change: `IMG_00293.png`, from `0-1` to `0-2`. Compared with the verified-ruler export, this image is the only meaningful FL/MT change; several other CSV values differ only at floating-point serialization precision.

`rebuild_final_export.py` can replay these saved adjustments. Its adjustment table preserves final FL/MT strings wherever they differ from the baseline, including insignificant serialization differences. This is an export replay from archived measurements, not fresh inference or independent validation.

## Saved code versus final interactive changes

The source V8 wrapper uses dense support-weighted FL and local OCR calibration, but its PA implementation still calls the earlier unweighted sampler. It also does not embed all final reviewed ruler overrides or the final pair correction. The final interactive code was entered during competition work and was not recovered in an executable saved notebook. It has not been silently reconstructed or presented as the original implementation.

Consequently, the provided inference command is named `run_v8.py` and its output identifies the preserved local workflow. It should not overwrite `results/final_submission.csv` or be described as reproducing the 0.67556 result. The final PA algorithm described in the project discussion used a 28-pixel inward offset, a 6-pixel neighborhood radius, support times concentration within local orientation averaging, and a weighted median across samples. The audit preserves the outputs; exact sampling and edge-handling code remain unrecovered.

## Reviewed calibration decisions recorded during the competition

- Manually interpreted depth labels included `50` and `50/50F` as 50 mm where context established ruler depth.
- Depth-label overrides included `IMG_00013` at 70 mm, `IMG_00201`, `IMG_00202`, and `IMG_00247` at 40 mm.
- A flipped screenshot (`IMG_00032`) required calibration orientation handling.
- The final left-ruler review used approximately y=42 to y=784 for 50 mm on `IMG_00186` through `IMG_00195`.
- `IMG_00023` used 50 mm between approximately y=86.5 and y=695.5.
- `IMG_00307` used the reviewed full-depth/screenshot-edge treatment; clipping made its endpoint interpretation uncertain.
- Accepted horizontal-ruler fallback assumed 10 mm per tick interval and square pixels.

These notes describe historical decisions rather than a complete executable final override manifest. The included `initial_calibration_audit.csv` predates some corrections and must not be treated as the final calibration truth.

The copied calibration source contains historical special-case and endpoint logic. It is preserved unchanged to avoid altering the algorithm during publication; the inference output still requires review. A future release can consolidate reviewed overrides and recover the final PA sampler before claiming full end-to-end final-submission reproduction.

## Verification

`verification.json` records checks actually performed. Checkpoint tensor equality, model loading, source compilation, submission schema, audit agreement, and final export replay can be verified without inference. No training or full 309-image inference was run during repository preparation.
