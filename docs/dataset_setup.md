# Dataset Setup Guide

EmoKit expects datasets under a single root directory pointed to by the
`EMOKIT_DATA_ROOT` environment variable (or passed via `--root` /
`root=` in the config YAML).

EmoKit does not redistribute DEAP, SEED, SEED-V, DREAMER, or MAHNOB-HCI;
users should download each dataset from its official provider and point EmoKit
to their local copies.

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

## MAHNOB-HCI

**Source**: <https://mahnob-db.eu/hci-tagging/>

Supports two on-disk layouts:

**Layout 1 — CSV** (preferred for local preprocessing):
```
$EMOKIT_DATA_ROOT/MAHNOB-HCI/
  eeg/{subject_id}/{trial_id}_eeg.csv    # (rows × 32 channels)
  ecg/{subject_id}/{trial_id}_ecg.csv
  gsr/{subject_id}/{trial_id}_gsr.csv
  valence_label.csv                       # rows = subjects, cols = trials
  arousal_label.csv
```

**Layout 2 — BDF** (original):
```
$EMOKIT_DATA_ROOT/MAHNOB-HCI/
  Subject1/session_dir/*.bdf
  Subject2/...
  labels.npy                              # optional label file
```

27 subjects, 20 trials each. 32-channel EEG + ECG + GSR.

**Verification**:
```bash
python scripts/verify_hci_pipeline.py --root $EMOKIT_DATA_ROOT/MAHNOB-HCI
```

---

## DREAMER

**Source**: [Zenodo](https://zenodo.org/record/546113) (Katsigiannis & Ramzan, 2018)

Single `DREAMER.mat` file containing all 23 subjects × 18 video clips.

**Default path**: `/data/ssd/xwt/DREAMER/DREAMER.mat`

```
$EMOKIT_DATA_ROOT/DREAMER/
  DREAMER.mat          # single file (~800 MB)
```

**Loading**:
```python
from scipy.io import loadmat
mat = loadmat('DREAMER.mat', squeeze_me=True, struct_as_record=False)
dreamer = mat['DREAMER']
```

**Internal structure**:
```
dreamer.Data[i]             # subject i (0-indexed, 23 subjects)
  .EEG.stimuli[k]          # video k EEG, shape (M, 14) at 128 Hz
  .EEG.baseline[k]         # baseline EEG, shape (M_base, 14)
  .ECG.stimuli[k]          # video k ECG, shape (M, 2) at 256 Hz
  .ECG.baseline[k]         # baseline ECG, shape (M_base, 2)
  .ScoreValence[k]         # 1–5 float rating
  .ScoreArousal[k]         # 1–5 float rating
```

**Important notes**:
- EEG shape is `(M, 14)` not `(14, M)` — samples × channels
- M varies across videos (different clip durations)
- 14 channels = Emotiv EPOC layout: AF3, F7, F3, FC5, T7, P7, O1,
  O2, P8, T8, FC6, F4, F8, AF4
- Labels binarised at threshold 3.0: `> 3` → 1, `<= 3` → 0
- Baseline correction: per-channel mean of baseline subtracted from stimulus

**Verification**:
```bash
python scripts/verify_dreamer_pipeline.py --root /data/ssd/xwt/DREAMER
```

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| `UnicodeDecodeError` on DEAP `.dat` | Missing `encoding='latin1'` — EmoKit handles this |
| `KeyError: 'de_LDS'` | Mat file doesn't contain pre-extracted DE; check the download |
| Shape `(40, 40, 8064)` vs `(40, 32, 7680)` | Raw vs baseline-removed; loader strips baseline |
| Label values > 4 for SEED-V | Continuous ratings not binarised; check label file |
| `FileNotFoundError` | Directory layout doesn't match patterns; see above |
