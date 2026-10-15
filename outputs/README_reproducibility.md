# Lithography hotspot detection: CNN features + classical heads (ICCAD-12) -- reproducibility notes

* Notebook: this file. Data: ICCAD-12 image clips (archive SHA-256: 0d64e924eaa7aea97bd178ceb91f5735c40fbf4e0324841bff51c8543b74e499).
* Seeds: [0, 1, 2, 3, 4] (completed here: [0, 1, 2, 3, 4]). One seed controls the stratified 85/15 train/validation split, weight
  initialisation, augmentation, shuffling and the RF / DT random_state. GPU kernels are not forced deterministic
  (DETERMINISTIC=False), so identical seeds can differ slightly between runs; across-seed variance is reported.
* Protocol: each benchmark is treated independently (never merged). Model hyper-parameters (SVM C, DT depth/leaf), decision
  thresholds and early stopping use the VALIDATION split only; the official test set is scored once per trained model.
* Shared CNN training recipe for BOTH backbones: {"optimizer": "nadam", "lr": 0.001, "epochs": 20, "patience": 5, "reduce_lr_factor": 0.5, "reduce_lr_patience": 2, "min_lr": 1e-06, "batch_size": 32, "augment": true, "val_fraction": 0.15, "min_steps": 100, "es_start_epoch": 3}; focal loss {'gamma': 2.0, 'alpha': 0.75}; weighted BCE caps the positive class
  weight at 10.0.
* Grid: backbones ['ref', 'proposed'] x losses ['focal', 'bce'] x heads ['SVM', 'RF', 'LR', 'DT', 'ENS']; early/late/combined taps only for ('proposed', 'focal').
* Environment: {"python": "3.13.15", "platform": "Linux-6.18.48+-x86_64-with-glibc2.39", "tensorflow": "2.20.0", "numpy": "2.1.3", "pandas": "2.3.3", "sklearn": "1.6.1", "scipy": "1.16.3", "gpus": ["Tesla T4", "Tesla T4"], "n_cpu": 4, "deterministic": false}
* Outputs: outputs/raw (per-run scores .npz + metric rows .json), outputs/tables (CSV), outputs/figures (PNG), outputs/config.json.
* Re-run: set QUICK_PILOT=False, run cells 1-12 once per seed (RUN_SEEDS=[k]), then cells 14-23.
* Compute: wall-clock for each run is stored in the rows (train_s, fit_s); latency/FLOPs in raw/lat_*.json (first seed only).
