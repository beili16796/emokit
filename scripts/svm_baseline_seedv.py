#!/usr/bin/env python3
"""Quick SVM baseline for SEED-V LOSO to validate data quality."""
import numpy as np
from emokit.datasets import load_dataset
from emokit.features.eeg import EEGNormalizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

ds = load_dataset('SEED-V', root='/data/ssd/shared_data/SEED-Ⅴ', use_de_features=True)
sids = ds.get_subject_ids()

svm_accs = []
for test_sid in sids:
    train_X, train_y = [], []
    for sid in sids:
        if sid == test_sid:
            continue
        raw = ds.read_raw(sid)
        train_X.append(raw['eeg'])
        train_y.append(raw['labels'])
    test_raw = ds.read_raw(test_sid)
    train_X = np.concatenate(train_X)
    train_y = np.concatenate(train_y)
    test_X, test_y = test_raw['eeg'], test_raw['labels']

    norm = EEGNormalizer()
    norm.fit(train_X)
    trn = norm.transform(train_X).reshape(len(train_X), -1)
    tst = norm.transform(test_X).reshape(len(test_X), -1)

    clf = LinearSVC(max_iter=5000, random_state=42, C=0.1)
    clf.fit(trn, train_y)
    acc = accuracy_score(test_y, clf.predict(tst))
    svm_accs.append(acc)
    print(f'Subject {test_sid:2d}: {acc*100:.1f}%')

print(f'\nSVM LOSO mean: {np.mean(svm_accs)*100:.1f} +- {np.std(svm_accs)*100:.1f}%')
