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

No claim of real-camera validation, flight qualification or measured matcher
speedup is made. The original 25-bin artifacts remain in gen/legacy for reference.
