# Personal project: Neural Star-Tracker Acquisition

- Developed a C/Python star-tracker pipeline combining subpixel centroiding, 100-bin angular features, and a 70,321-parameter neural network to shortlist 529 sky regions.
- Trained on 5,819 rendered frames using 8,785 Hipparcos stars; achieved a 94.2% visible-cell hit rate across 1,000 independent simulated frames, averaging 1.9 candidates per usable frame.
- Verified C/Python inference agreement within 1e-6 across 200 test cases and implemented geometric star matching, SVD attitude estimation, and full-search fallback benchmarking.

Use these after reviewing and understanding the implementation. The hit rate is
sky-cell localization in simulated images, not real-camera attitude accuracy.
The matcher benchmark did not demonstrate a speedup.
