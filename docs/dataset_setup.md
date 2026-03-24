# Dataset Setup Guide

EmoKit expects datasets under a single root directory pointed to by the
`EMOKIT_DATA_ROOT` environment variable (or passed via `--root` /
`root=` in the config YAML).

```bash
export EMOKIT_DATA_ROOT=/path/to/datasets
```

---

## Directory Structure

```
$EMOKIT_DATA_ROOT/
├── DEAP/
│   ├── s01.dat          # Preprocessed pickle (recommended)
│   ├── s02.dat
│   ├── ...
│   └── s32.dat
│
├── SEED-V/
│   ├── 1/               # Subject directories (1-based)
│   │   ├── 1.mat        # Session 1
│   │   ├── 2.mat        # Session 2
│   │   └── 3.mat        # Session 3
│   ├── 2/
│   │   └── ...
│   └── 16/
│       └── ...
│
├── SEED/
│   ├── Preprocessed_EEG/
│   │   ├── 1_1.mat      # Subject 1, Session 1
│   │   ├── 1_2.mat
│   │   ├── 1_3.mat
│   │   ├── 2_1.mat
│   │   └── ...
│   └── (or subject directories like SEED-V)
│
├── MAHNOB-HCI/           # Optional
│   └── Sessions/
│       ├── Session_1/
│       │   ├── *.bdf
│       │   └── *.xml
│       └── ...
│
└── DREAMER/               # Optional
    └── DREAMER.mat
```

---

## DEAP

**Source**: <http://eecs.qmul.ac.uk/mmv/datasets/deap/>

Download the **preprocessed Python** data files (`data_preprocessed_python.zip`).
Unzip into `$EMOKIT_DATA_ROOT/DEAP/`. You should have 32 files: `s01.dat` through
`s32.dat`.

Each `.dat` is a Python 2 pickle containing:
- `data`: shape `(40, 40, 8064)` — 40 trials × 40 channels × 63s @128 Hz
- `labels`: shape `(40, 4)` — valence, arousal, dominance, liking (1–9 scale)

Channel mapping (40 channels):
- 0–31: EEG (Fp1, AF3, F3, F7, FC5, FC1, C3, T7, CP5, CP1, P3, P7,
  PO3, O1, Oz, Pz, Fp2, AF4, F4, F8, FC6, FC2, C4, T8, CP6, CP2,
  P4, P8, PO4, O2, Fz, Cz)
- 32: hEOG, 33: vEOG, 34: zEMG, 35: tEMG
- **36: GSR**, 37: Resp belt, **38: BVP (used as ECG proxy)**, 39: Temperature

Loader behaviour:
- Baseline removal: first 384 samples (3 s) are stripped
- 1–45 Hz Butterworth bandpass (order 5) applied to EEG channels
- Common-average re-referencing applied to EEG
- Labels binarised at threshold 5.0 (configurable)

**Verification**:
```bash
python -m emokit.scripts.verify_deap_pipeline --root $EMOKIT_DATA_ROOT/DEAP --subject 1
```

---

## SEED-V

**Source**: <https://bcmi.sjtu.edu.cn/home/seed/seed-v.html>

Download the pre-extracted DE features. Place session `.mat` files in
per-subject directories:

```
$EMOKIT_DATA_ROOT/SEED-V/
  1/1.mat   1/2.mat   1/3.mat
  2/1.mat   ...
```

Alternative layouts are also accepted:
- `sub1/session1.mat`
- `s01/sess01.mat`
- `1_1.mat` (flat)

Each `.mat` contains:
- `de_LDS`: either a numeric array `(62, 5, n_windows)` or a MATLAB cell
  array where each cell is one trial's DE features
- `label` / `labels`: integer labels 0–4

5-class emotion mapping: 0=happy, 1=sad, 2=neutral, 3=fear, 4=disgust

The loader handles both cell-array and numeric-array formats automatically.

**Verification**:
```bash
python -m emokit.scripts.verify_seedv_pipeline --root $EMOKIT_DATA_ROOT/SEED-V --subject 1
```

---

## SEED

**Source**: <https://bcmi.sjtu.edu.cn/home/seed/seed.html>

15 subjects × 3 sessions. DE features in `*_de_LDS.mat` files.

Preferred layout:
```
$EMOKIT_DATA_ROOT/SEED/
  Preprocessed_EEG/
    1_1.mat   1_2.mat   1_3.mat
    2_1.mat   ...
    15_3.mat
```

Or per-subject directories (same patterns as SEED-V).

3-class emotions: 0=negative, 1=neutral, 2=positive

**Verification**:
```bash
python -m emokit.scripts.verify_seed_pipeline --root $EMOKIT_DATA_ROOT/SEED --subject 1
```

---

## MAHNOB-HCI (Optional)

**Source**: <https://mahnob-db.eu/hci-tagging/>

Requires BDF raw files. Each session in its own directory with
a `.bdf` EEG file and `.xml` annotation.

---

## DREAMER (Optional)

**Source**: Request access from the authors.

Single `DREAMER.mat` file containing all 23 subjects.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `UnicodeDecodeError` on DEAP `.dat` | Missing `encoding='latin1'` — EmoKit handles this |
| `KeyError: 'de_LDS'` | Mat file doesn't contain pre-extracted DE; check the download |
| Shape `(40, 40, 8064)` vs `(40, 32, 7680)` | Raw vs baseline-removed; loader strips baseline |
| Label values > 4 for SEED-V | Continuous ratings not binarised; check label file |
| `FileNotFoundError` | Directory layout doesn't match patterns; see above |
