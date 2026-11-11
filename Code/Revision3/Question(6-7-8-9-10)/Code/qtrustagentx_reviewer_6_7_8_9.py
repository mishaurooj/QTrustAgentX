#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QTrustAgentX DED, reproducible reviewer 6-9 experiments in ONE local Python file.

TABLE 8: strictly source-domain-trained Human<->LLM transfer and within-source tests.
TABLE 9: measured near-duplicate overlap plus SMS label harmonization.
TABLE 10: transfer ROC-AUC, PR-AUC, ECE, prevalence and reliability.

Dataset: E:/other/QTrustAgentX/Dataset_Reorganized (or Dataset_Reorganized.zip).
Never uses labels from target authoring domain for training, model selection, calibration,
threshold selection or ensemble weights. No post hoc seed filtering. No invented metrics.
GPU: optional CUDA PyTorch representation network, plus optional local DeBERTa checkpoint.

Install: py -m pip install -U numpy pandas scipy scikit-learn torch
Optional DeBERTa: py -m pip install transformers
Run: py reviewer_6_7_8_9_onefile.py --root "E:\\other\\QTrustAgentX" --seeds 10 --gpu auto
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, re, sys, time, warnings, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import softmax
from scipy.stats import t as student_t
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

EMAIL_FILES = [
    ('human_legit_email_1000.csv', 'human', 0),
    ('human_phishing_email_1000.csv', 'human', 1),
    ('llm_legit_email_1000.csv', 'llm', 0),
    ('llm_phishing_email_595.csv', 'llm', 1),
]
SMS_LABELS = {'ham': 0, 'spam': 1, 'smishing': 2, 'smish': 2}
MODEL_METHODS = ['Logistic regression', 'Word SVM', 'Char SVM',
                 'Hybrid SVM', 'Delexicalized SVM']
THRESHOLD_GRID = np.linspace(.05, .95, 181)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canon(text):
    return re.sub(r'\s+', ' ', str(text).casefold()).strip()


def decode(text):
    text = str(text) if pd.notna(text) else ''
    return text if text.casefold() != 'nan' else ''


def delex(text):
    text = canon(text)
    text = re.sub(r'(?i)\b(?:https?://|www\.)\S+', ' URLTOKEN ', text)
    text = re.sub(r'\b[\w.\-+]+@[\w.\-]+\.[a-z]{2,}\b', ' EMAILTOKEN ', text)
    text = re.sub(r'\b\d{4,}\b', ' NUMTOKEN ', text)
    return text


class Source:
    """Read original data directly from an unzipped Windows directory or ZIP."""
    def __init__(self, root):
        self.root = Path(root)
        self.z = zipfile.ZipFile(self.root) if self.root.is_file() else None
        if self.z is None and not self.root.is_dir():
            raise FileNotFoundError(f'Dataset not found: {self.root}')
        self.names = self.z.namelist() if self.z else [
            p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()]
        self.inventory = sorted([n for n in self.names if n.lower().endswith('.csv')])

    def read_csv(self, suffix):
        found = [n for n in self.inventory if n.endswith('/' + suffix) or n == suffix]
        if len(found) != 1:
            raise ValueError(f'Expected one {suffix}, found {found}. Check --source.')
        name = found[0]
        data = self.z.read(name) if self.z else (self.root / name).read_bytes()
        # The supplied LLM phishing CSV contains unquoted commas inside text.
        # It has a two-column header but standard CSV parsing corrupts messages.
        # For this specific verified one-physical-line-per-record layout, recover
        # each record by splitting at its LAST comma (the final binary label).
        repair = 'standard pandas CSV parser; malformed lines skipped'
        if suffix == 'llm_phishing_email_595.csv':
            lines = data.decode('utf-8', 'replace').splitlines()
            if lines and lines[0].strip().casefold() == 'text,label' and all(
                ',' in ln and ln.rsplit(',', 1)[-1].strip() in ('0','1')
                for ln in lines[1:] if ln.strip()):
                repaired = [ln.rsplit(',', 1) for ln in lines[1:] if ln.strip()]
                frame = pd.DataFrame(repaired, columns=['text', 'label'])
                frame['label'] = frame['label'].astype(int)
                repair = 'nonstandard unquoted commas: recover physical lines using final binary-label delimiter'
            else:
                frame = pd.read_csv(io.BytesIO(data), encoding='utf-8', encoding_errors='replace',
                                    on_bad_lines='skip', low_memory=False)
                repair = 'WARNING: unquoted-commas repair validation failed; standard parser fallback'
        else:
            frame = pd.read_csv(io.BytesIO(data), encoding='utf-8', encoding_errors='replace',
                                on_bad_lines='skip', low_memory=False)
        return frame, {'member': name, 'sha256': sha(data), 'parsed_rows': int(len(frame)),
                       'parse_policy': repair}


def load_data(source_path):
    src = Source(source_path)
    audit = {'source': str(source_path), 'files': {}, 'warnings': [],
             'label_policy': 'Email labels from original folder names; SMS raw 3-class labels kept'}
    mail = []
    for fname, domain, mapped_label in EMAIL_FILES:
        frame, info = src.read_csv(fname)
        audit['files'][fname] = info
        if fname == 'llm_phishing_email_595.csv' and len(frame) != 595:
            audit['warnings'].append(
                f'{fname}: filename says 595, but {len(frame)} physical rows parsed; '
                'source filename is inconsistent and labels require original-source verification.')
        text_col = 'body' if 'body' in frame else 'text'
        if text_col not in frame:
            raise ValueError(f'Missing body/text in {fname}: {list(frame)}')
        conflicts = 0
        for idx, r in frame.iterrows():
            subject = decode(r['subject']) if 'subject' in frame else ''
            body = decode(r[text_col])
            text = (subject + ' ' + body).strip()
            if len(canon(text)) < 8:
                continue
            if 'label' in frame and pd.notna(r['label']):
                raw_label = str(r['label']).strip().lower()
                if raw_label in ('0', '1') and int(raw_label) != mapped_label:
                    conflicts += 1
            uid = f'email:{fname}:{idx}'
            mail.append({'source_id': uid, 'modality': 'email', 'text': text,
                         'author': domain, 'label': int(mapped_label),
                         'source_file': fname, 'raw_label': decode(r.get('label', '')),
                         'group': 'email:' + sha(canon(text).encode())})
        info['file_label_conflicts'] = int(conflicts)
        if conflicts:
            audit['warnings'].append(f'{fname}: {conflicts} raw labels conflict with folder provenance; path labels used; verify upstream ground truth.')
    email = pd.DataFrame(mail)
    if len(email) == 0:
        raise ValueError('No usable email messages')
    # Do not discard duplicates to hide overlap. Group identical texts in splitting.
    sms_raw, info = src.read_csv('sms_phishing_5971.csv')
    audit['files']['sms_phishing_5971.csv'] = info
    sms = []
    unknown = Counter()
    for i, r in sms_raw.iterrows():
        lab = canon(r['LABEL'])
        if lab not in SMS_LABELS:
            unknown[lab] += 1
            continue
        text = decode(r['TEXT']).strip()
        if len(canon(text)) < 3:
            continue
        sms.append({'source_id': f'sms:{i}', 'modality': 'sms', 'text': text,
                    'author': '', 'label': int(SMS_LABELS[lab]),
                    'source_file': 'sms_phishing_5971.csv', 'raw_label': lab,
                    'group': 'sms:' + sha(canon(text).encode())})
    sms = pd.DataFrame(sms)
    audit['sms_unknown_labels'] = dict(unknown)
    audit['sms_raw_label_counts'] = {str(k): int(v) for k, v in sms.raw_label.value_counts().items()}
    audit['email_domain_target_counts'] = {f'{a}:{l}': int(len(g))
        for (a, l), g in email.groupby(['author', 'label'])}
    audit['email_n'] = int(len(email))
    audit['sms_n'] = int(len(sms))
    audit['email_cross_target_duplicate_groups'] = int((email.groupby('group').label.nunique() > 1).sum())
    audit['email_cross_author_duplicate_groups'] = int((email.groupby('group').author.nunique() > 1).sum())
    if email.groupby('author').label.nunique().min() < 2:
        raise ValueError('Human and LLM email domains each require 0 and 1 classes')
    if len(set(sms.label)) != 3:
        raise ValueError('SMS source must include ham, spam and smishing. Do not invent labels.')
    return email.reset_index(drop=True), sms.reset_index(drop=True), audit


def group_split(data, y, seed, test_size=.2, validation_size=.2):
    """Group-disjoint train, source validation, and source test, deterministic seed."""
    if len(data) != len(y):
        raise ValueError('Label/row count mismatch')
    groups = data.group.to_numpy()
    inds = np.arange(len(data))
    for attempt in range(60):
        random_seed = seed + attempt * 1009
        g = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        trainval, test = next(g.split(inds, y, groups))
        h = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_seed + 100003)
        aa, bb = next(h.split(trainval, y[trainval], groups[trainval]))
        train, val = trainval[aa], trainval[bb]
        classes = set(np.unique(y))
        if all(set(np.unique(y[ii])) == classes for ii in (train, val, test)):
            return train, val, test
    raise RuntimeError('Could not obtain grouped splits with all classes; check raw corpus or group conflicts')


def all_metrics(y, p, threshold=.5, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    if len(y) == 0:
        return {k: float('nan') for k in
                ['precision', 'specificity', 'recall', 'f1', 'mcc', 'brier', 'fpr', 'fnr',
                 'roc_auc', 'pr_auc', 'ece', 'prevalence']}
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite probabilities')
    yp = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        active = (p >= lo) & ((p < hi) if hi < 1.0 else (p <= hi))
        if np.any(active):
            ece += active.mean() * abs(y[active].mean() - p[active].mean())
    return {'precision': float(precision_score(y, yp, zero_division=0)),
            'specificity': float(tn / max(1, tn + fp)),
            'recall': float(tp / max(1, tp + fn)),
            'f1': float(f1_score(y, yp, zero_division=0)),
            'mcc': float(matthews_corrcoef(y, yp)),
            'brier': float(brier_score_loss(y, p)),
            'fpr': float(fp / max(1, tn + fp)),
            'fnr': float(fn / max(1, tp + fn)),
            'roc_auc': float(roc_auc_score(y, p)) if len(set(y)) == 2 else float('nan'),
            'pr_auc': float(average_precision_score(y, p)) if len(set(y)) == 2 else float('nan'),
            'ece': float(ece), 'prevalence': float(y.mean())}


def select_threshold(y, p):
    # All threshold choices use SOURCE VALIDATION ONLY (never target test labels).
    vals = np.asarray([(f1_score(y, np.asarray(p) >= t, zero_division=0), t)
                       for t in THRESHOLD_GRID])
    best = np.max(vals[:, 0])
    ties = vals[np.isclose(vals[:, 0], best), 1]
    return float(ties[np.argmin(np.abs(ties - .5))])


def mean_ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float('nan'), float('nan'), float('nan')
    mean = float(v.mean())
    if len(v) < 2:
        return mean, float('nan'), float('nan')
    delta = float(student_t.ppf(.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))
    return mean, mean - delta, mean + delta


def similarity(train_text, test_text, chunk=256, return_indices=False):
    """Train-fitted char 3-5 TF-IDF, max cosine to source train; never use target labels."""
    if not len(test_text):
        empty = np.zeros(0, dtype=np.float32)
        return (empty, np.zeros(0, dtype=int)) if return_indices else empty
    if not len(train_text):
        empty = np.zeros(len(test_text), dtype=np.float32)
        return (empty, np.full(len(test_text), -1, dtype=int)) if return_indices else empty
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=1,
                          dtype=np.float32, sublinear_tf=True)
    xtrain = vec.fit_transform([canon(t) for t in train_text])
    xtest = vec.transform([canon(t) for t in test_text])
    maxima = np.zeros(xtest.shape[0], dtype=np.float32)
    nearest = np.full(xtest.shape[0], -1, dtype=int) if return_indices else None
    for a in range(0, xtest.shape[0], chunk):
        sims = xtest[a:a+chunk] @ xtrain.T
        if return_indices:
            # The training identity is retained as auditable evidence for each
            # flagged near-duplicate; only one chunk is densified at a time.
            dense = sims.toarray()
            nearest[a:a+chunk] = dense.argmax(axis=1)
            maxima[a:a+chunk] = dense[np.arange(len(dense)), nearest[a:a+chunk]]
        else:
            m = sims.max(axis=1)
            maxima[a:a+chunk] = np.asarray(m.toarray() if hasattr(m, 'toarray') else m).reshape(-1)
    return (maxima, nearest) if return_indices else maxima


def make_features(tr, va, te, delexicalize=False):
    train = [delex(s) for s in tr] if delexicalize else list(tr)
    valid = [delex(s) for s in va] if delexicalize else list(va)
    target = [delex(s) for s in te] if delexicalize else list(te)
    vecw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.99,
                            max_features=55000, sublinear_tf=True, strip_accents='unicode',
                            dtype=np.float32)
    vecc = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                            max_features=90000, sublinear_tf=True, dtype=np.float32)
    w0 = vecw.fit_transform(train)
    w1, w2 = vecw.transform(valid), vecw.transform(target)
    c0 = vecc.fit_transform(train)
    c1, c2 = vecc.transform(valid), vecc.transform(target)
    return {'word': (w0, w1, w2), 'char': (c0, c1, c2),
            'hybrid': (hstack([w0, c0], format='csr'),
                       hstack([w1, c1], format='csr'),
                       hstack([w2, c2], format='csr'))}


def fit_linear(train_x, train_y, val_x, target_x, method, seed):
    if method == 'logreg':
        model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear', random_state=seed)
    else:
        base = LinearSVC(C=1.0, class_weight='balanced', random_state=seed)
        model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    model.fit(train_x, train_y)
    return model.predict_proba(val_x)[:, 1], model.predict_proba(target_x)[:, 1]


def gpu_mlp(train_x, train_y, val_x, val_y, test_x, seed, device,
            n_epochs=30, hidden=128, patience=5, verbose=False):
    """Real torch CUDA training, with early stopping on source validation Brier only."""
    if torch is None:
        raise RuntimeError('PyTorch unavailable')
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    # train_x, val_x, test_x are low-dimensional dense SVD representations.
    xtr = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    xva = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    xte = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(train_y[:, None], dtype=torch.float32, device=device)
    model = nn.Sequential(nn.Linear(train_x.shape[1], hidden), nn.ReLU(),
                          nn.Dropout(.3), nn.Linear(hidden, hidden // 2), nn.ReLU(),
                          nn.Linear(hidden // 2, 1)).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.02)
    best, best_state, bad = float('inf'), None, 0
    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        model.train()
        for inds in np.array_split(rng.permutation(len(train_y)), max(1, math.ceil(len(train_y)/128))):
            optim.zero_grad(set_to_none=True)
            pred = model(xtr[inds])
            loss = nn.functional.binary_cross_entropy_with_logits(pred, ytr[inds])
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            v = torch.sigmoid(model(xva)).reshape(-1)
            brier = float(torch.square(v - torch.as_tensor(val_y, device=device)).mean().item())
        if brier < best - 1e-5:
            best, bad = brier, 0
            best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        valid = torch.sigmoid(model(xva)).reshape(-1).cpu().numpy()
        test = torch.sigmoid(model(xte)).reshape(-1).cpu().numpy()
    return valid, test


def fit_neural(tr, yt, va, yv, te, seed, device, svd_dims=96, epochs=30):
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Fit EVERY representation step on source TRAIN only.
    vect = TfidfVectorizer(analyzer='char', ngram_range=(3, 5),
                           min_df=2, max_features=110000, dtype=np.float32)
    xtr = vect.fit_transform(tr)
    xva = vect.transform(va)
    xte = vect.transform(te)
    dim = min(svd_dims, min(xtr.shape) - 1)
    if dim < 2:
        raise ValueError('Not enough training features for SVD network')
    svd = TruncatedSVD(n_components=dim, random_state=seed, n_iter=5)
    ztr = svd.fit_transform(xtr)
    zva = svd.transform(xva)
    zte = svd.transform(xte)
    scaler = StandardScaler()
    ztr = scaler.fit_transform(ztr).astype(np.float32)
    zva = scaler.transform(zva).astype(np.float32)
    zte = scaler.transform(zte).astype(np.float32)
    return gpu_mlp(ztr, yt, zva, yv, zte, seed, device, n_epochs=epochs)


def local_deberta(tr, yt, va, yv, te, model_path, seed, device, epochs=2, batch=8):
    """Optional source-only fine-tuning of an EXISTING local HF DeBERTa checkpoint."""
    if device != 'cuda':
        raise ValueError('DeBERTa requires CUDA; this mode never downloads a model')
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError('pip install transformers for --deberta-model-path') from exc
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), num_labels=2, local_files_only=True,
        ignore_mismatched_sizes=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    def batches(texts, labels=None, shuffle=False):
        ids = np.arange(len(texts))
        if shuffle:
            np.random.default_rng(seed + 239).shuffle(ids)
        for chunk in np.array_split(ids, max(1, math.ceil(len(ids)/batch))):
            enc = tok([texts[int(i)] for i in chunk], padding=True,
                      truncation=True, max_length=256, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            if labels is not None:
                enc['labels'] = torch.as_tensor(labels[chunk], dtype=torch.long, device=device)
            yield enc
    for ep in range(epochs):
        model.train()
        for en in batches(tr, yt, shuffle=True):
            optim.zero_grad(set_to_none=True)
            model(**en).loss.backward()
            optim.step()
    def predict(texts):
        model.eval()
        scores = []
        with torch.no_grad():
            for en in batches(texts):
                scores.extend(torch.softmax(model(**en).logits, dim=1)[:, 1].cpu().numpy().tolist())
        return np.asarray(scores, dtype=float)
    out = predict(va), predict(te)
    del model
    torch.cuda.empty_cache()
    return out


def fit_fusions(valid_scores, target_scores, yval, seed):
    names = list(valid_scores)
    va = np.column_stack([valid_scores[k] for k in names])
    te = np.column_stack([target_scores[k] for k in names])
    k = len(names)
    def loss(w):
        return float(np.mean((va @ w - yval) ** 2))
    opt = minimize(loss, np.full(k, 1 / k), method='SLSQP',
                   bounds=[(0, 1)] * k,
                   constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                   options={'maxiter': 200})
    w = np.clip(opt.x if opt.success else np.full(k, 1/k), 0, 1)
    w = w / np.sum(w)
    # Stacked model is fit ONLY on source validation predictions.
    stack = LogisticRegression(max_iter=500, C=.1, random_state=seed)
    stack.fit(va, yval)
    return ({'Brier fusion': np.clip(va @ w, 0, 1),
             'Stacked fusion': stack.predict_proba(va)[:, 1]},
            {'Brier fusion': np.clip(te @ w, 0, 1),
             'Stacked fusion': stack.predict_proba(te)[:, 1]},
            {'fusion_weights': {name: float(weight) for name, weight in zip(names, w)},
             'fitting_partition': 'source validation only'})


def evaluate_transfer(email, seed, args, device):
    """Two strict source-only directions; target labels used only in all_metrics()."""
    records, diagnostics, selection_audit = [], [], []
    for author, destination in [('human', 'llm'), ('llm', 'human')]:
        src = email[email.author == author].reset_index(drop=True)
        tgt = email[email.author == destination].reset_index(drop=True)
        y = src.label.to_numpy(dtype=int)
        train, valid, held = group_split(src, y, seed)
        tr = src.iloc[train]
        va = src.iloc[valid]
        ho = src.iloc[held]
        # Target author domain labels NEVER enter fitting, calibration, weights, threshold.
        transfer = tgt.copy()
        if args.exclude_transfer_near:
            train_pool = pd.concat([tr, va], ignore_index=True)
            sims = similarity(train_pool.text.tolist(), transfer.text.tolist(), args.similarity_chunk)
            transfer = transfer.loc[sims < args.threshold].reset_index(drop=True)
            if not len(transfer):
                raise ValueError('All target observations flagged; cannot evaluate domain transfer.')
        else:
            sims = np.full(len(transfer), np.nan)
        target_text = ho.text.tolist() + transfer.text.tolist()
        n_held = len(ho)
        val_score, target_score = {}, {}
        features = make_features(tr.text.tolist(), va.text.tolist(), target_text)
        for name, key, which in [
                ('Logistic regression', 'word', 'logreg'),
                ('Word SVM', 'word', 'svm'),
                ('Char SVM', 'char', 'svm'),
                ('Hybrid SVM', 'hybrid', 'svm')]:
            xx = features[key]
            v, p = fit_linear(xx[0], tr.label.to_numpy(), xx[1], xx[2], which, seed)
            val_score[name], target_score[name] = v, p
        if args.delex:
            xd = make_features(tr.text.tolist(), va.text.tolist(), target_text, True)['hybrid']
            v, p = fit_linear(xd[0], tr.label.to_numpy(), xd[1], xd[2], 'svm', seed)
            val_score['Delexicalized SVM'], target_score['Delexicalized SVM'] = v, p
        if args.neural == 'always' or (args.neural == 'auto' and device == 'cuda'):
            v, p = fit_neural(tr.text.tolist(), tr.label.to_numpy(),
                              va.text.tolist(), va.label.to_numpy(), target_text,
                              seed, device, svd_dims=args.svd_dims, epochs=args.neural_epochs)
            val_score['SVD-MLP (' + device.upper() + ')'] = v
            target_score['SVD-MLP (' + device.upper() + ')'] = p
        if args.deberta_model_path:
            v, p = local_deberta(tr.text.tolist(), tr.label.to_numpy(),
                                 va.text.tolist(), va.label.to_numpy(), target_text,
                                 args.deberta_model_path, seed, device,
                                 epochs=args.deberta_epochs)
            val_score['DeBERTa (local fine-tuned)'] = v
            target_score['DeBERTa (local fine-tuned)'] = p
        v_fused, t_fused, fusion_info = fit_fusions(val_score, target_score, va.label.to_numpy(), seed)
        val_score.update(v_fused)
        target_score.update(t_fused)
        direction = 'Human $\\rightarrow$ LLM' if author == 'human' else 'LLM $\\rightarrow$ Human'
        direction_text = f'{author}->{destination}'
        # Predeclared selection criterion, minimizes validation Brier; no target optimization.
        briers = {name: brier_score_loss(va.label.to_numpy(), p)
                  for name, p in val_score.items()}
        best = sorted(briers.items(), key=lambda kv: (kv[1], kv[0]))[0][0]
        selection_audit.append({'seed': seed, 'direction': direction_text,
                                'source_selected_method': best,
                                'source_val_brier': briers[best],
                                'n_train': len(tr), 'n_source_val': len(va),
                                'n_source_test': len(ho), 'n_target_test': len(transfer),
                                'n_target_excluded': int(len(tgt) - len(transfer)),
                                'selection_rule': 'lowest source validation Brier',
                                'fusion': fusion_info})
        # A negative-class baseline, always produces exactly zero positives.
        val_score = {'Majority-class (negative)': np.zeros(len(va)), **val_score}
        target_score = {'Majority-class (negative)': np.zeros(len(target_text)), **target_score}
        for name, p in target_score.items():
            threshold = select_threshold(va.label.to_numpy(), val_score[name]) if name != 'Majority-class (negative)' else .5
            pv = p[:n_held]
            pt = p[n_held:]
            m_held = all_metrics(ho.label.to_numpy(), pv, threshold)
            m_target = all_metrics(transfer.label.to_numpy(), pt, threshold)
            records.append({'seed': seed, 'direction': direction_text, 'method': name,
                            'threshold': threshold, 'source_val_brier': briers.get(name, np.nan),
                            'source_selected': bool(name == best),
                            'n_train': len(tr), 'n_val': len(va),
                            'n_test': len(transfer), 'target_excluded': len(tgt)-len(transfer),
                            **m_target})
            diagnostics.append({'seed': seed, 'direction': direction_text, 'method': name,
                                'source_selected': bool(name == best), 'evaluation': 'within-source',
                                'n_test': len(ho), **m_held})
            diagnostics.append({'seed': seed, 'direction': direction_text, 'method': name,
                                'source_selected': bool(name == best), 'evaluation': 'cross-domain',
                                'n_test': len(transfer), **m_target})
        print(f'[Table8/10] seed {seed}, {direction_text}: source-selected={best}; '
              f'target N={len(transfer)}, filtered={len(tgt)-len(transfer)}', flush=True)
    return records, diagnostics, selection_audit


def fit_sms_threeclass(train_text, y_train, valid_text, y_valid, test_text, seed):
    """Three-class ham/spam/smishing probability; temperature fitted on source val.

    P(smishing) is the phishing-specific SMS endpoint. A global validation-only
    temperature preserves mutually exclusive probabilities and sum-to-one.
    Source-validation labels, NOT test labels, calibrate the temperature.
    """
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=45000,
                            dtype=np.float32, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2,
                            max_features=85000, dtype=np.float32, sublinear_tf=True)
    wtr = words.fit_transform(train_text)
    ctr = chars.fit_transform(train_text)
    model = LogisticRegression(max_iter=700, C=1.5, class_weight=None,
                               solver='lbfgs', random_state=seed)
    model.fit(hstack([wtr, ctr], format='csr'), y_train)
    def logits(texts):
        x = hstack([words.transform(texts), chars.transform(texts)], format='csr')
        return model.decision_function(x)
    val_logits = logits(valid_text)
    test_logits = logits(test_text)
    lut = {int(c): i for i, c in enumerate(model.classes_)}
    val_idx = np.asarray([lut[int(v)] for v in y_valid], dtype=int)
    def loss(log_t):
        p = softmax(val_logits / float(np.exp(log_t)), axis=1)
        return float(-np.log(np.maximum(p[np.arange(len(p)), val_idx], 1e-10)).mean())
    opt = minimize_scalar(loss, bounds=(np.log(.3), np.log(5.0)), method='bounded')
    temperature = float(np.exp(opt.x)) if opt.success else 1.0
    vproba = softmax(val_logits / temperature, axis=1)
    tproba = softmax(test_logits / temperature, axis=1)
    val = {int(c): vproba[:, i] for i, c in enumerate(model.classes_)}
    test = {int(c): tproba[:, i] for i, c in enumerate(model.classes_)}
    return val, test, temperature


def measure_duplicate_subset(sub, group, threshold, seed, model_target):
    s = sub.copy()
    clean = s.loc[s.max_cosine < threshold]
    m0 = all_metrics(s.y.to_numpy(), s.p.to_numpy(), model_target)
    m1 = all_metrics(clean.y.to_numpy(), clean.p.to_numpy(), model_target)
    return {'seed': int(seed), 'group': group, 'n_test': int(len(s)),
            'n_flagged': int((s.max_cosine >= threshold).sum()),
            'flagged_share': float((s.max_cosine >= threshold).mean()) if len(s) else np.nan,
            'n_retained': int(len(clean)),
            'positives': int((s.y == 1).sum()), 'negatives': int((s.y == 0).sum()),
            'f1_before': m0['f1'], 'f1_after': m1['f1'],
            'auc_before': m0['roc_auc'], 'auc_after': m1['roc_auc'],
            'brier_before': m0['brier'], 'brier_after': m1['brier'],
            'ece_before': m0['ece'], 'ece_after': m1['ece']}


def audit_near_duplicates(email, sms, seed, args):
    """Same held-out test; frozen scores; no retraining after duplicate removal.
    SMS probabilities come from a 3-class calibrated-ish softmax model; broad and
    narrow targets are explicitly different. Do not present them as same quantity.
    """
    rows, prediction_audit, split_records = [], [], []
    # Email detection is within DED, independent of source-only transfer experiment.
    yy = email.label.to_numpy(int)
    etr, eva, ete = group_split(email, yy, seed)
    tr, va, te = email.iloc[etr], email.iloc[eva], email.iloc[ete].copy()
    for split_name, sub in [('train', tr), ('validation', va), ('test', te)]:
        split_records.extend({'seed': seed, 'source_id': sid, 'modality': 'email',
                              'split': split_name, 'group': gr}
                             for sid, gr in zip(sub.source_id, sub.group))
    x = make_features(tr.text.tolist(), va.text.tolist(), te.text.tolist())['hybrid']
    epv, ept = fit_linear(x[0], tr.label.to_numpy(int), x[1], x[2], 'logreg', seed)
    eth = select_threshold(va.label.to_numpy(), epv)
    pool = pd.concat([tr, va], ignore_index=True)
    te['max_cosine'], nearest = similarity(pool.text.tolist(), te.text.tolist(),
                                           args.similarity_chunk, return_indices=True)
    te['nearest_train_source_id'] = pool.source_id.to_numpy()[nearest]
    te['y'], te['p'] = te.label.to_numpy(int), ept
    for group, sub in [('email_all', te),
                       ('email_human', te.loc[te.author.eq('human')]),
                       ('email_llm', te.loc[te.author.eq('llm')])]:
        rows.append(measure_duplicate_subset(sub, group, args.threshold, seed, eth))
    prediction_audit.extend({'seed': seed, 'id': r.source_id, 'modality': 'email',
                             'author': r.author, 'raw_label': r.raw_label,
                             'score': float(r.p), 'y': int(r.y), 'threshold': eth,
                             'max_cosine': float(r.max_cosine),
                             'nearest_train_source_id': r.nearest_train_source_id}
                            for r in te.itertuples())
    # SMS train/val/test grouped on raw 3-class labels, not combined binary labels.
    y3 = sms.label.to_numpy(int)
    str_idx, sva_idx, ste_idx = group_split(sms, y3, seed)
    tr, va, te = sms.iloc[str_idx], sms.iloc[sva_idx], sms.iloc[ste_idx].copy()
    for split_name, sub in [('train', tr), ('validation', va), ('test', te)]:
        split_records.extend({'seed': seed, 'source_id': sid, 'modality': 'sms',
                              'split': split_name, 'group': gr}
                             for sid, gr in zip(sub.source_id, sub.group))
    pv, pt, temperature = fit_sms_threeclass(tr.text.tolist(), tr.label.to_numpy(int),
                                va.text.tolist(), va.label.to_numpy(int),
                                te.text.tolist(), seed)
    # Class 0 ham, class 1 non-phishing spam, class 2 smishing.
    pval_smish = pv[2]
    ptest_smish = pt[2]
    pval_broad = pv[1] + pv[2]
    ptest_broad = pt[1] + pt[2]
    th_smish = select_threshold((va.label == 2).to_numpy(dtype=int), pval_smish)
    th_broad = select_threshold((va.label != 0).to_numpy(dtype=int), pval_broad)
    pool = pd.concat([tr, va], ignore_index=True)
    te['max_cosine'], nearest = similarity(pool.text.tolist(), te.text.tolist(),
                                           args.similarity_chunk, return_indices=True)
    te['nearest_train_source_id'] = pool.source_id.to_numpy()[nearest]
    smish = te.assign(y=te.label.eq(2).astype(int), p=ptest_smish)
    broad = te.assign(y=te.label.ne(0).astype(int), p=ptest_broad)
    strictly = smish.loc[smish.label.isin([0, 2])].copy()
    for group, sub, th in [('sms_legacy_spam_plus_smishing', broad, th_broad),
                           ('sms_smishing_vs_ham_spam', smish, th_smish),
                           ('sms_smishing_vs_ham', strictly, th_smish)]:
        rows.append(measure_duplicate_subset(sub, group, args.threshold, seed, th))
    prediction_audit.extend({'seed': seed, 'id': r.source_id, 'modality': 'sms',
                             'author': '', 'raw_label': r.raw_label,
                             'score': float(p), 'y': int(r.label == 2),
                             'source_val_temperature': temperature,
                             'threshold': th_smish, 'max_cosine': float(r.max_cosine),
                             'nearest_train_source_id': r.nearest_train_source_id,
                             'p_broad': float(pb), 'y_broad': int(r.label != 0)}
                            for r, p, pb in zip(te.itertuples(), ptest_smish, ptest_broad))
    print(f'[Table9] seed {seed}: email test={len(email.iloc[ete])}, SMS test={len(te)}, '
          f'SMS raw labels={dict(te.raw_label.value_counts())}', flush=True)
    return rows, prediction_audit, split_records


def grouped_summary(records, cols, metrics):
    df = pd.DataFrame(records)
    out = []
    for key, part in df.groupby(cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(cols, key))
        row['repeats'] = int(len(part))
        for metric in metrics:
            if metric not in part:
                continue
            m, low, high = mean_ci(part[metric])
            row[metric + '_mean'] = m
            row[metric + '_ci_low'] = low
            row[metric + '_ci_high'] = high
        out.append(row)
    return pd.DataFrame(out)


def esc(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def fmt(x, digits=4):
    return 'N/A' if not np.isfinite(x) else f'{x:.{digits}f}'


def ci_str(mean, low, high, digits=3):
    if not np.isfinite(mean):
        return 'N/A'
    if not np.isfinite(low):
        return fmt(mean, digits)
    return f'{mean:.{digits}f} [{low:.{digits}f},{high:.{digits}f}]'


def write_table8(full, out):
    agg = grouped_summary(full, ['direction', 'method'],
                          ['precision', 'specificity', 'f1', 'mcc', 'brier', 'fpr', 'fnr'])
    agg.to_csv(out / 'TABLE8_ALL_METRICS_SUMMARY.csv', index=False)
    order = ['Majority-class (negative)', 'Logistic regression', 'Word SVM',
             'Char SVM', 'Hybrid SVM', 'Delexicalized SVM', 'SVD-MLP (CUDA)',
             'SVD-MLP (CPU)', 'DeBERTa (local fine-tuned)', 'Brier fusion',
             'Stacked fusion']
    lines = [r'\begin{table*}[t]', r'\centering', r'\begingroup\color{blue}',
             r'\footnotesize\setlength{\tabcolsep}{3.4pt}\renewcommand{\arraystretch}{1.13}',
             r'\caption{Source-only Human$\leftrightarrow$LLM transfer on the DED: '
             r'mean classification metrics across fixed-seed refits. The source-domain '
             r'training and validation partitions determine all fitting, calibration, '
             r'ensemble weights, and thresholds. Target-domain labels are used only '
             r'for held-out assessment. The source-selected model is reported separately '
             r'in Table~\ref{tab:transfer_diag_proposed}.}',
             r'\label{tab:llm_corrective_transfer}',
             r'\begin{tabular}{@{}llrrrrrrr@{}}', r'\toprule',
             r'Train $\rightarrow$ test & Method & Precision & Specificity & F1 & MCC & Brier & FPR & FNR \\',
             r'\midrule']
    for direction in ['human->llm', 'llm->human']:
        sub = agg[agg.direction == direction].set_index('method')
        for method in order:
            if method not in sub.index:
                continue
            row = sub.loc[method]
            vals = [fmt(float(row[m + '_mean'])) for m in
                    ['precision', 'specificity', 'f1', 'mcc', 'brier', 'fpr', 'fnr']]
            direction_latex = r'Human $\rightarrow$ LLM' if direction == 'human->llm' else r'LLM $\rightarrow$ Human'
            name = method.replace('SVD-MLP (CUDA)', 'SVD-MLP (GPU)')
            lines.append(direction_latex + ' & ' + esc(name) + ' & ' + ' & '.join(vals) + r' \\')
        if direction == 'human->llm':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\vspace{3pt}{\scriptsize\raggedright '
              r'Each metric is averaged across outer refits. No post-hoc model '
              r'choice based on target performance is used. A missing model row '
              r'indicates that the optional implementation was not run.\par}',
              r'\endgroup', r'\end{table*}']
    (out/'TABLE8_REVISED.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_table9(raw, out, threshold):
    agg = grouped_summary(raw, ['group'], ['flagged_share', 'f1_before', 'f1_after',
                                           'auc_before', 'auc_after',
                                           'brier_before', 'brier_after', 'ece_before', 'ece_after',
                                           'n_test', 'n_flagged'])
    agg.to_csv(out/'TABLE9_MEASURED_SUMMARY.csv', index=False)
    labels = [('email_all', 'Email (all)'), ('email_human', 'Email (human)'),
              ('email_llm', 'Email (LLM)'),
              ('sms_legacy_spam_plus_smishing', 'SMS: spam+smishing positive'),
              ('sms_smishing_vs_ham_spam', 'SMS: smishing vs. ham+spam'),
              ('sms_smishing_vs_ham', 'SMS: smishing vs. ham only')]
    lines = [r'\begin{table*}[t]', r'\centering\begingroup\color{blue}',
             r'\small\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.15}',
             r'\caption{Measured train/validation-to-test near-duplicate overlap '
             r'and SMS target harmonization on the DED; cosine threshold '
             f'{threshold:.2f} using source-fitted TF--IDF character 3--5 grams. '
             r'F1 after filtering is computed from frozen held-out predictions '
             r'without retraining. Smishing-specific targets are distinguished '
             r'from the broader spam+smishing label.}',
             r'\label{tab:overlap_proposed}',
             r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Analysis & Mean test $n$ & Flagged share (95\% CI) & F1 before & F1 after \\',
             r'\midrule']
    for key, label in labels:
        r = agg.loc[agg.group == key]
        if r.empty:
            continue
        r = r.iloc[0]
        rate = f"{100*r.flagged_share_mean:.2f}\\%"
        if np.isfinite(r.flagged_share_ci_low):
            rate += f" [{100*r.flagged_share_ci_low:.2f},{100*r.flagged_share_ci_high:.2f}]"
        lines.append(f'{label} & {r.n_test_mean:.0f} & {rate} & '
                     f'{fmt(r.f1_before_mean)} & {fmt(r.f1_after_mean)} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\vspace{4pt}{\scriptsize\raggedright '
              r'Flagged percentages are descriptive across overlapping group-disjoint '
              r'resplits; intervals are Student-$t$ resplit intervals. '
              r'For SMS, only source-annotated \emph{smishing} constitutes '
              r'the strict positive class. The legacy binary target '
              r'corresponds to a different detection question.\par}',
              r'\endgroup\end{table*}']
    (out/'TABLE9_REVISED.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_table10(all_transfer, out):
    d = pd.DataFrame(all_transfer)
    # SOURCE-VALIDATION-selected model for each seed and direction; no target winner picks.
    selected = d[d.source_selected].copy()
    agg = grouped_summary(selected, ['direction'], ['prevalence', 'roc_auc', 'pr_auc',
                                                      'ece', 'brier', 'f1', 'n_test'])
    agg.to_csv(out/'TABLE10_MEASURED_SUMMARY.csv', index=False)
    lines = [r'\begin{table*}[t]', r'\centering\begingroup\color{blue}',
             r'\small\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.15}',
             r'\caption{Human$\leftrightarrow$LLM transfer probability diagnostics '
             r'on DED. For each seed, the method minimizing source-validation Brier '
             r'score is fixed before evaluating target-domain labels; '
             r'95\% intervals describe variability across overlapping source refits.}',
             r'\label{tab:transfer_diag_proposed}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             r'Train $\rightarrow$ test & Target prevalence & ROC-AUC (95\% CI) & '
             r'PR-AUC (95\% CI) & ECE (95\% CI) & Brier \\', r'\midrule']
    for direction, label in [('human->llm', r'Human $\rightarrow$ LLM'),
                             ('llm->human', r'LLM $\rightarrow$ Human')]:
        rr = agg.loc[agg.direction == direction]
        if rr.empty: continue
        r = rr.iloc[0]
        fields = [fmt(r.prevalence_mean, 3),
                  ci_str(r.roc_auc_mean, r.roc_auc_ci_low, r.roc_auc_ci_high),
                  ci_str(r.pr_auc_mean, r.pr_auc_ci_low, r.pr_auc_ci_high),
                  ci_str(r.ece_mean, r.ece_ci_low, r.ece_ci_high),
                  fmt(r.brier_mean)]
        lines.append(label + ' & ' + ' & '.join(fields) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\par\vspace{3pt}{\scriptsize\raggedright '
              r'ECE uses 10 fixed-width probability bins; PR-AUC is average precision. '
              r'No target-domain labels inform training, thresholds or model selection.\par}',
              r'\endgroup\end{table*}']
    (out/'TABLE10_REVISED.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def write_discussion(out):
    file = out/'REVIEWER_6_7_8_9_METHODS.tex'
    file.write_text(r'''
% Adapt section and table counters to your full manuscript.
% Insert Table 8, Table 9 and Table 10 separately at manuscript locations.
{\color{blue}
Table~\ref{tab:overlap_proposed} reports measured near-duplicate rates on the
DISJOINT Evaluation Dataset (DED), not illustrative values. We retained
source identifiers and original authoring-domain and SMS-category provenance.
Within each repeated, source-group-disjoint split, a train/validation-fitted
TF--IDF character 3--5-gram representation was used to compute the maximum
cosine similarity between each test observation and the fitting/calibration
partition. Test observations with similarity at least 0.90 were flagged.
Post-filter metrics were computed on the retained held-out observations using
frozen model scores and source-selected thresholds, with no retraining.
The reported 95\% intervals describe resplit-level variability, not independent
cohort uncertainty.

The SMS specialist was additionally evaluated using an explicitly
three-class target (ham, spam, smishing). We report the broad
spam-or-smishing target separately from the source-annotated smishing
probability, and the restricted smishing-versus-ham test. Thus, estimates of
general unsolicited/malicious messaging are not represented as a common
phishing-only probability; the latter requires reliable phishing-specific
annotations and cannot be inferred for unannotated messages.

Tables~\ref{tab:llm_corrective_transfer} and
\ref{tab:transfer_diag_proposed} evaluate Human-to-LLM and LLM-to-Human
transfer separately. Model fitting, validation-selected thresholds, source
calibration, ensemble weights and optional neural early stopping use
source-domain data only. All target authoring-domain labels remain held out
until evaluation. We report source-validation-selected model diagnostics,
including ROC-AUC, PR-AUC, ECE and Brier score, without selecting the best
method or threshold using target-domain outcomes. Any observed domain-transfer
failure is retained as a limitation rather than being described as robust
cross-domain generalization.
}
''', encoding='utf-8')


def write_readme(out, root, args, audit):
    text = f'''QTrustAgentX reviewer comments 6, 7, 8, and 9: original-source DED audit
============================================================================
Dataset requested: {root}
Dataset type: DED. The separate JMD folder is intentionally not used.

Tables:
  TABLE8_REVISED.tex  : source-only authoring-domain transfer, all metrics
  TABLE9_REVISED.tex  : measured near-duplicate rates and SMS label harmonization
  TABLE10_REVISED.tex : ROC-AUC, PR-AUC, ECE, Brier on domain transfer

Data outputs:
  TABLE8_ALL_SEEDS.csv, TABLE8_ALL_METRICS_SUMMARY.csv
  TABLE9_ALL_SEEDS.csv, TABLE9_MEASURED_SUMMARY.csv,
  TABLE9_PER_RECORD_SIMILARITY.csv, TABLE9_SPLIT_ASSIGNMENTS.csv
  (nearest train-source ID recorded for every test observation)
  TABLE10_ALL_SEEDS.csv, TABLE10_MEASURED_SUMMARY.csv
  SOURCE_DOMAIN_SELECTION.csv, SOURCE_PROVENANCE_AUDIT.json

Method:
  * Group-disjoint splits by normalized message SHA256 (exact repeats cannot
    straddle partitions). Human->LLM and LLM->Human target labels stay held out.
  * Source validation alone chooses thresholds, fusion and selected method.
  * Character 3-5 TF-IDF trained only on train/validation, similarity threshold
    {args.threshold}. Flagged test records excluded WITHOUT refitting.
  * SMS raw labels: ham=0, spam=1, smishing=2. Source-validation-temperature-
    calibrated 3-class model predicts p(smishing); broad p(spam)+p(smishing)
    is reported separately.
  * ECE: 10 equal-width bins. Confidence intervals: descriptive 95% t intervals
    over overlapping repeated refits, not independent data samples.
  * No final results are hard-coded, no target-label-based model selection.
  * Label disagreements and raw-data parsing policy are reported in JSON.
  * GPU: Torch representation network runs on CUDA if present; source-only
    transformer fine-tuning is possible with --deberta-model-path LOCAL_PATH.

PRE-SUBMISSION CHECK:
  Do not reproduce old illustrative Table 9 percentages. The new analysis has
  different targets/methods from original Table 8; distinguish new validation
  from replication. In particular, folder-vs-row email label conflicts must be
  disclosed and resolved at the original source if systematic.
'''
    (out/'README_RESULTS.txt').write_text(text, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path(r'E:\other\QTrustAgentX'),
                    help='Windows project folder containing Dataset_Reorganized')
    ap.add_argument('--source', type=Path, default=None,
                    help='Explicit Dataset_Reorganized folder or ZIP')
    ap.add_argument('--output', type=Path, default=None,
                    help='Results directory; defaults to ROOT/Reviewer_6_7_8_9_Results')
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--start-seed', type=int, default=42)
    ap.add_argument('--threshold', type=float, default=.90)
    ap.add_argument('--similarity-chunk', type=int, default=256)
    ap.add_argument('--gpu', choices=['auto', 'cuda', 'cpu'], default='auto')
    ap.add_argument('--neural', choices=['auto', 'always', 'off'], default='auto',
                    help='auto means train SVD-MLP only when CUDA is available')
    ap.add_argument('--neural-epochs', type=int, default=30)
    ap.add_argument('--svd-dims', type=int, default=96)
    ap.add_argument('--delex', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--exclude-transfer-near', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--deberta-model-path', type=Path, default=None,
                    help='OPTIONAL local pre-downloaded DeBERTa checkpoint; no downloading')
    ap.add_argument('--deberta-epochs', type=int, default=2)
    ap.add_argument('--only', choices=['all', 'transfer', 'duplicates'], default='all')
    ap.add_argument('--dry-run', action='store_true',
                    help='Load corpus, report provenance counts without fitting models')
    args = ap.parse_args()
    if args.seeds < 1 or not (0 < args.threshold <= 1):
        ap.error('Require --seeds >= 1 and 0 < --threshold <= 1')
    source = args.source
    if source is None:
        for path in [args.root/'Dataset_Reorganized', args.root/'Dataset_Reorganized.zip',
                     args.root/'Dataset'/'Dataset_Reorganized']:
            if path.exists():
                source = path; break
    if source is None:
        ap.error('Cannot locate DED: use --source E:\\other\\QTrustAgentX\\Dataset_Reorganized')
    out = args.output or (args.root/'Reviewer_6_7_8_9_Results')
    out.mkdir(parents=True, exist_ok=True)
    torch_cuda = bool(torch is not None and torch.cuda.is_available())
    if args.gpu == 'cuda' and not torch_cuda:
        ap.error('CUDA requested but torch.cuda.is_available() is false. Install CUDA PyTorch.')
    device = 'cuda' if args.gpu in ('cuda', 'auto') and torch_cuda else 'cpu'
    if args.deberta_model_path and device != 'cuda':
        ap.error('--deberta-model-path requires a CUDA GPU')
    print(f'DED source: {source}\nOutput: {out}\nDevice: {device.upper()} '
          f'(Torch CUDA available={torch_cuda})', flush=True)
    email, sms, source_audit = load_data(source)
    audit = {**source_audit, 'protocol': {
        'dataset': 'DED', 'jmd_used': False,
        'start_seed': args.start_seed, 'seeds': args.seeds,
        'similarity_threshold': args.threshold,
        'transfer_purge_near_duplicates': bool(args.exclude_transfer_near),
        'selected_method_rule': 'minimum SOURCE validation Brier',
        'selection_uses_target_labels': False,
        'calibration_uses_target_labels': False,
        'smishing_positive_only': 'SMS P(smishing=2), separate from general spam=1',
        'sms_probability_calibration': '3-class temperature scaling fitted on source validation only',
        'source_file_parse_policy': 'Malformed CSV lines skipped, reported parsed rows',
        'neural_model': args.neural,
        'deberta_model_path': str(args.deberta_model_path) if args.deberta_model_path else None,
        'device': device, 'cuda_available': torch_cuda,
        'sklearn_version': __import__('sklearn').__version__,
        'python': sys.version, 'date_utc': pd.Timestamp.utcnow().isoformat(),
        'resplits_are_independent': False}}
    (out/'SOURCE_PROVENANCE_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('Email:', source_audit['email_domain_target_counts'], flush=True)
    print('SMS:', source_audit['sms_raw_label_counts'], flush=True)
    if source_audit['warnings']:
        print('PROVENANCE WARNINGS:', *source_audit['warnings'], sep='\n  - ', flush=True)
    if args.dry_run:
        print('DRY RUN COMPLETE: provenance only', flush=True)
        return
    cfg = {
        'source_file_sha256': {k: info['sha256'] for k, info in source_audit['files'].items()},
        'script_sha256': sha_file(Path(__file__)),
        'start_seed': args.start_seed, 'seeds': args.seeds,
        'threshold': args.threshold, 'device': device, 'neural': args.neural,
        'neural_epochs': args.neural_epochs, 'svd_dims': args.svd_dims,
        'delex': args.delex, 'exclude_transfer_near': args.exclude_transfer_near,
        'deberta_checkpoint': str(args.deberta_model_path),
        'deberta_epochs': args.deberta_epochs}
    cfg_path = out/'RUN_CONFIGURATION.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding='utf-8')) != cfg:
        raise RuntimeError('Output directory holds results from a DIFFERENT configuration '
                           'or older code. Use a new --output directory to prevent mixed protocols.')
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        sf = out/f'SEED_{seed}_TRANSFER.csv'
        dg = out/f'SEED_{seed}_DIAGNOSTICS.csv'
        sa = out/f'SEED_{seed}_SOURCE_SELECTION.json'
        nf = out/f'SEED_{seed}_NEAR_DUPLICATES.csv'
        na = out/f'SEED_{seed}_SIMILARITY_AUDIT.csv'
        ns = out/f'SEED_{seed}_SPLIT_ASSIGNMENTS.csv'
        if args.only in ('all', 'transfer'):
            if all(p.exists() for p in [sf, dg, sa]):
                print(f'RESUME transfer seed {seed}', flush=True)
            else:
                d, diag, selections = evaluate_transfer(email, seed, args, device)
                pd.DataFrame(d).to_csv(sf, index=False)
                pd.DataFrame(diag).to_csv(dg, index=False)
                sa.write_text(json.dumps(selections, indent=2), encoding='utf-8')
        if args.only in ('all', 'duplicates'):
            if nf.exists() and na.exists() and ns.exists():
                print(f'RESUME duplicates seed {seed}', flush=True)
            else:
                dd, aa, splits = audit_near_duplicates(email, sms, seed, args)
                pd.DataFrame(dd).to_csv(nf, index=False)
                pd.DataFrame(aa).to_csv(na, index=False)
                pd.DataFrame(splits).to_csv(ns, index=False)
    seeds = range(args.start_seed, args.start_seed + args.seeds)
    if args.only in ('all', 'transfer'):
        all8 = pd.concat([pd.read_csv(out/f'SEED_{s}_TRANSFER.csv') for s in seeds], ignore_index=True)
        alld = pd.concat([pd.read_csv(out/f'SEED_{s}_DIAGNOSTICS.csv') for s in seeds], ignore_index=True)
        all8.to_csv(out/'TABLE8_ALL_SEEDS.csv', index=False)
        alld.to_csv(out/'TABLE10_ALL_SEEDS.csv', index=False)
        write_table8(all8, out)
        write_table10(all8, out)
        choices = sum([json.loads((out/f'SEED_{s}_SOURCE_SELECTION.json').read_text()) for s in seeds], [])
        pd.DataFrame([{k: v for k, v in choice.items() if k != 'fusion'} for choice in choices]
                     ).to_csv(out/'SOURCE_DOMAIN_SELECTION.csv', index=False)
    if args.only in ('all', 'duplicates'):
        all9 = pd.concat([pd.read_csv(out/f'SEED_{s}_NEAR_DUPLICATES.csv') for s in seeds], ignore_index=True)
        all9.to_csv(out/'TABLE9_ALL_SEEDS.csv', index=False)
        pd.concat([pd.read_csv(out/f'SEED_{s}_SIMILARITY_AUDIT.csv') for s in seeds],
                  ignore_index=True).to_csv(out/'TABLE9_PER_RECORD_SIMILARITY.csv', index=False)
        write_table9(all9, out, args.threshold)
        pd.concat([pd.read_csv(out/f'SEED_{s}_SPLIT_ASSIGNMENTS.csv') for s in seeds],
                  ignore_index=True).to_csv(out/'TABLE9_SPLIT_ASSIGNMENTS.csv', index=False)
    write_discussion(out)
    # One paste-ready file in Table 8, Table 9, Table 10 order, retaining natural numbering.
    if args.only == 'all':
        content = ['% Required packages: booktabs, xcolor',
                   '% Insert after preceding Table 7 so these become Tables 8, 9, and 10.',
                   '% Alternatively uncomment the following line for explicit numbering:',
                   r'% \setcounter{table}{7}']
        for num in (8, 9, 10):
            content.append((out/f'TABLE{num}_REVISED.tex').read_text(encoding='utf-8'))
        (out/'TABLES_8_9_10_BLUE_COMBINED.tex').write_text('\n\n'.join(content), encoding='utf-8')
    label_rows = []
    for raw, label in [('ham', 0), ('spam', 1), ('smishing', 2)]:
        label_rows.append({'raw_label': raw, 'source_count': source_audit['sms_raw_label_counts'].get(raw, 0),
                           'three_class_label': label,
                           'legacy_broad_positive': int(label != 0),
                           'harmonized_smishing_positive': int(label == 2),
                           'annotation_source': 'original sms_phishing_5971.csv LABEL'})
    pd.DataFrame(label_rows).to_csv(out/'SMS_LABEL_MAPPING_AUDIT.csv', index=False)
    write_readme(out, source, args, audit)
    print('\nDONE: measured outputs saved to', out.resolve(), flush=True)
    print('CAVEAT: Cross-domain performance must be judged from measured CSVs;',
          'no improvements are guaranteed.', flush=True)


if __name__ == '__main__':
    main()
