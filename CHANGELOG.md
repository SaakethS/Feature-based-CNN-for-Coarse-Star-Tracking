# Changes from the uploaded project

- Extracted the nested Python archive into normal source files; omitted stale ZIP,
  build products and Git metadata from delivery.
- Expanded the active feature from 25 to 100 bins after a controlled comparison;
  retrained/exported the 70,321-parameter model and updated C/Python together.
- Imported 8,785 real Hipparcos stars with source, coordinate-epoch and scope notes.
- Added rendered training through a host bridge to the C detector. Fixed fractional
  PSF centers, unified flux zero points and made the analytic threshold effective.
- Added catalog validation and errors for nonexistent requested files.
- Stored camera, catalog hash, bins hash, simulation settings, seeds, counts,
  detected vectors, attitudes, training split and chosen epoch in artifacts.
- Selected the best validation-BCE epoch rather than automatically the final epoch.
- Added exact C selector evaluation, rejection-aware denominators, real low-star
  counts, high-confidence failure counts and policy comparisons.
- Added controlled feature comparison and isolated FOV experiments.
- Added a ground-only pair-angle/SVD matcher and measured assisted/full acquisition.
- Aligned minimum detections at four; reject blob-table/flood-fill overflow;
  clear pipeline outputs on failure; preserve equal-flux ordering; validate nonfinite
  selector/camera inputs; check bias pointers and nonfinite MLP input.
- Corrected C float export for integral-valued constants, added camera export and
  model/bin identity validation, and made cross-check subprocess errors fail loudly.
- Added 10 regression tests and a 30-image full C/Python inference comparison.

## Second doubling: 23,319 frames, measured epoch budget (2026-09-09)

- Regenerated the dataset at 24,000 attempts, giving 23,319 usable frames. Same
  seed, camera, catalog, noise model and simulator version as before.
- Chose the epoch budget by measurement instead of assumption: a 60-epoch survey
  run located the validation minimum at epoch 21, and the active model was then
  retrained with --epochs 21. Verified by SHA-256 per array that this produces
  bit-identical weights to those the 60-epoch run selected, since train.py
  restores best-epoch weights and the RNG stream is seeded deterministically.
- Retained every learning curve: gen/training_history_5819.json,
  gen/training_history_11643.json, gen/training_history_23319_ep60.json and the
  active gen/training_history.json. plot_loss.py now labels each run with whether
  its minimum was a genuine early stop or the epoch budget running out.
- Seed-999 evaluation: hit rate 95.9% -> 96.8% overall and 97.56% -> 98.47% on
  usable frames, mean shortlist 1.60 -> 1.27 cells, top-8 98.58% -> 98.98%,
  low-star 84.62% -> 90.38%, confident-but-wrong 17 of 895 -> 11 of 943.
- Documented that returns are diminishing on the headline metrics (hit rate gains
  halved with each doubling) and that validation loss is not comparable across
  dataset sizes, because each run holds out the tail of its own dataset.
- Feature ranking unchanged; 100 bins still wins at every dataset size. Matcher
  still shows no speedup, and the assisted penalty stopped improving (1.33x then
  1.37x, within host timing noise).

## First doubling: 5,819 to 11,643 frames (2026-09-09)

- Regenerated the rendered dataset at 12,000 attempts instead of 6,000, giving
  11,643 usable frames instead of 5,819. Same seed, camera, catalog, noise model
  and simulator version; only the frame count changed. Superseded by the run above.
- Retrained (120 epochs, seed 0) and re-exported gen/model_params.h, then rebuilt
  and re-ran the crosscheck, regression and end-to-end tests: all pass.
- train.py now records per-epoch training and validation loss to
  gen/training_history.json, and takes --history and --tag. Added
  python/plot_loss.py, which draws one or more of those runs into a standalone
  SVG with no plotting dependency. Retained the retrained 5,819-frame run so the
  curves can be compared (now gen/training_history_5819.json).
- Re-measured everything downstream of the model on the unchanged seed-999
  evaluation set: visible-cell hit rate 94.2% -> 95.9% overall and 95.83% ->
  97.56% on usable frames, mean shortlist 1.90 -> 1.60 cells, top-8 hit rate
  97.36% -> 98.58%, low-star hit rate 78.85% -> 84.62%, confident-but-wrong
  frames 31 of 855 -> 17 of 895. Best validation cross-entropy 0.07103 -> 0.06049.
- Re-ran the feature comparison and matcher benchmark on the new artifacts. The
  100-bin feature still wins by the same margin. The matcher still shows no
  speedup, though the assisted-with-fallback penalty fell from 2.31x to 1.33x.

No claim of real-camera validation, flight qualification or measured matcher
speedup is made. The original 25-bin artifacts remain in gen/legacy for reference.
