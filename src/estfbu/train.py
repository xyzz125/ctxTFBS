"""Parameterized version of the paper's context-model training scripts
(0_train_K562_TFBS_context_model_168bp.py /
1_train_HepG2_vs_K562_TFBS_context_model_168bp.py), for any TF/cell-type
instead of hardcoded TF_list=['GATA2'] + hardcoded .cuda() calls.

The paper's scripts assume a CUDA GPU; this uses MPS (Apple Silicon GPU)
automatically when available, measured ~11x faster than CPU for this
exact architecture (0.2s/batch vs 2.2s/batch), falling back to CPU
otherwise -- see README.md's "System requirements" section."""
import collections
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset

from . import data_prep
from .models import build_model, _import_seq_regression_model


class _SeqDataset(Dataset):
    def __init__(self, seq, label):
        self.seq, self.label = seq, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.seq[idx], self.label[idx]


def get_or_train_model(tf: str, cell_type: str, cfg, comparative_with: str = None):
    """Entry point that respects the 'pretrained: true' flag in config for
    cell types where the paper's own Zenodo weights should be used instead
    of training from scratch (currently: HepG2 only). Returns a loaded
    model in eval mode."""
    # torch.load needs the pickled class's module importable by its exact
    # original name ("SeqRegressionModel") at unpickle time, whether we're
    # loading the paper's pretrained weights or one we trained ourselves.
    _import_seq_regression_model(cfg.deeptfbu_repo)

    if comparative_with is None and cfg.cell_type(cell_type).pretrained:
        weights = Path(cfg.hepg2_pretrained_weights_dir) / f"test_denselstm_mc_0.001_mask_168_{tf}.pth"
        if not weights.exists():
            raise FileNotFoundError(
                f"'{cell_type}' is marked pretrained in config but no weight file found for "
                f"'{tf}' at {weights}. Either this TF isn't among the paper's 198 published "
                f"weights, or the path is wrong."
            )
        model = torch.load(weights, map_location="cpu", weights_only=False)
    else:
        path = train_context_model(tf, cell_type, cfg, comparative_with=comparative_with)
        model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def _model_path(cfg, tf: str, cell_type: str, comparative_with: str = None) -> "Path":
    if comparative_with:
        d = cfg.work_dir_path("models", f"{cell_type}_vs_{comparative_with}")
    else:
        d = cfg.work_dir_path("models", cell_type)
    return d / f"{tf}.pth"


def _load_h5_pair(h5_path, tf):
    with h5py.File(h5_path) as f:
        return f[f"pos_{tf}"][:], f[f"neg_{tf}"][:]


def train_context_model(tf: str, cell_type: str, cfg, comparative_with: str = None, force: bool = False):
    """Train (or reuse a cached) TFBS-context model for tf.

    If comparative_with is None: trains a single-cell-type model (e.g. K562
    GATA2: real ChIP-seq-supported context vs. not).
    If comparative_with is given: trains a comparative model (e.g. HepG2 vs
    K562 GATA2: is this context's binding evidence from cell_type or from
    comparative_with).
    """
    out_path = _model_path(cfg, tf, cell_type, comparative_with)
    if out_path.exists() and not force:
        return out_path

    if comparative_with:
        pos, neg = _comparative_h5(tf, cell_type, comparative_with, cfg)
    else:
        h5 = data_prep.prepare(tf, cell_type, cfg)
        pos, neg = _load_h5_pair(h5, tf)

    t = cfg.training
    rng = np.random.RandomState(t.random_seed)
    rng.shuffle(neg)
    rng2 = np.random.RandomState(t.random_seed)
    rng2.shuffle(pos)

    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    n_train = int(n * t.train_val_test_split[0])
    n_val = int(n * sum(t.train_val_test_split[:2]))

    train_seq = np.concatenate([neg[:n_train], pos[:n_train]])
    train_y = np.concatenate([np.zeros(n_train), np.ones(n_train)])
    val_seq = np.concatenate([neg[n_train:n_val], pos[n_train:n_val]])
    val_y = np.concatenate([np.zeros(n_val - n_train), np.ones(n_val - n_train)])

    torch.set_num_threads(t.cpu_threads)
    # MPS (Apple Silicon GPU) measured ~11x faster than CPU for this
    # architecture (0.2s/batch vs 2.2s/batch, direct benchmark) -- the
    # BiLSTM layers are the CPU bottleneck. Falls back to CPU elsewhere.
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=t.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # batch order draws from PyTorch's global RNG unless given its own
    # generator -- explicit here so training is deterministic given the
    # same seed regardless of what else has consumed the global RNG.
    shuffle_generator = torch.Generator().manual_seed(t.random_seed)
    train_loader = DataLoader(_SeqDataset(train_seq, train_y), batch_size=t.batch_size, shuffle=True,
                               generator=shuffle_generator)
    val_loader = DataLoader(_SeqDataset(val_seq, val_y), batch_size=t.batch_size, shuffle=False)

    best_auc = -1.0
    patience_counter = 0
    best_state = None

    for epoch in range(t.max_epochs):
        model.train()
        for x, y in train_loader:
            # cast to MPS-supported dtypes (float32/int64) BEFORE moving --
            # MPS doesn't support float64, which is what these numpy-backed
            # tensors are by default
            x, y = x.float().to(device), y.long().to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.float().to(device)
                p = F.softmax(model(x), dim=1).cpu().numpy()[:, 1]
                preds.extend(p)
                labels.extend(y.numpy())
        fpr, tpr, _ = roc_curve(labels, preds)
        val_auc = auc(fpr, tpr)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= t.early_stopping_patience:
            break

    model.load_state_dict(best_state)
    model = model.to("cpu")
    torch.save(model, out_path)
    return out_path


def _comparative_h5(tf, cell_type_a, cell_type_b, cfg):
    """Build (or reuse) the A-vs-B comparative dataset: cell_type_a's
    positives = label 1, cell_type_b's positives = label 0 (GC/histone
    matched, same convention as the paper's HepG2-vs-K562 setup)."""
    d = cfg.work_dir_path("data_prep", f"{cell_type_a}_vs_{cell_type_b}")
    out_h5 = d / f"step0_{tf}.h5"
    if out_h5.exists():
        with h5py.File(out_h5) as f:
            return f[f"pos_{tf}"][:], f[f"neg_{tf}"][:]

    h5_a = data_prep.prepare(tf, cell_type_a, cfg)
    h5_b = data_prep.prepare(tf, cell_type_b, cfg)
    pos_a, _ = _load_h5_pair(h5_a, tf)
    pos_b, _ = _load_h5_pair(h5_b, tf)

    n = min(len(pos_a), len(pos_b))
    rng = np.random.RandomState(cfg.training.random_seed)
    pos_a = pos_a[rng.permutation(len(pos_a))[:n]]
    pos_b = pos_b[rng.permutation(len(pos_b))[:n]]

    with h5py.File(out_h5, "w") as f:
        f.create_dataset(f"pos_{tf}", data=pos_a)  # "positive" = cell_type_a
        f.create_dataset(f"neg_{tf}", data=pos_b)  # "negative" = cell_type_b
    return pos_a, pos_b


def held_out_test_split(tf: str, cell_type: str, cfg):
    """Reproduces train_context_model's exact positive/negative held-out
    TEST split -- same seeds, same shuffle order, same truncation to
    n=min(len(pos),len(neg)) -- so this is precisely the 10% slice that
    training carves out via cfg.training.train_val_test_split but never
    evaluates against (the val split is used for early stopping/model
    selection; the test split's n_val:n slice was previously only ever
    consumed as GA seed material in design._seed_population, never
    scored). Scoring a TF's ranking on this instead of its full positive
    set avoids the in-sample problem: for a pretrained TF, ALL positives
    are exactly the windows the paper's own weights were trained on, so a
    full-positive-set score is measuring memorization, not
    generalization. Caveat: for the pretrained case, this held-out split
    is computed against OUR locally-regenerated h5 file, which isn't
    guaranteed to be pixel-identical to the paper's own original
    train/test partition -- still a much closer approximation to genuine
    held-out performance than 100% in-sample, just not a perfect
    reproduction of the paper's own split for that case."""
    h5 = data_prep.prepare(tf, cell_type, cfg)
    pos, neg = _load_h5_pair(h5, tf)
    t = cfg.training
    rng = np.random.RandomState(t.random_seed)
    rng.shuffle(neg)
    rng2 = np.random.RandomState(t.random_seed)
    rng2.shuffle(pos)
    n = min(len(pos), len(neg))
    pos, neg = pos[:n], neg[:n]
    n_val = int(n * sum(t.train_val_test_split[:2]))
    return pos[n_val:], neg[n_val:]


def score_and_auc(model, pos_test, neg_test):
    """Mean model score on held-out positives (for ranking/display,
    comparable to the previous in-sample number) plus this model's
    held-out AUC (positives vs negatives) -- a real, defensible
    generalization number instead of a heuristic."""
    from .design import _score_batch  # deferred: design imports this module at load time
    pos_scores = _score_batch(model, pos_test)
    neg_scores = _score_batch(model, neg_test)
    preds = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = roc_curve(labels, preds)
    return float(np.mean(pos_scores)), float(auc(fpr, tpr))


def rank_chipseq_tfs(cfg, tfs, cell_types, target):
    """The has-ChIP-seq branch's 'no TF specified' path: ranks each TF in
    `tfs` (already ChIP-seq-available + blacklist-filtered by the caller)
    by its own trained/pretrained model's held-out test-split score (see
    held_out_test_split/score_and_auc) -- the has-ChIP-seq equivalent of
    the no-ChIP-seq branch's screen_tfs.screen_all_tfs. Unlike that pure-
    motif-math screen, this needs a real trained model per candidate TF
    per cell type, so it's slow for any TF/cell-type pairing that isn't
    pretrained.

    Pure function, no prompting -- this used to live only inside
    run_pipeline.py's interactive run_chipseq_screen, wrapped around
    inline ask() calls for target/n_top, which made it both unreachable
    from the CLI and structurally untestable (a test can't answer an
    input() prompt). Callers (the CLI's --action screen and the
    interactive driver) decide target/n_top themselves and handle their
    own output (CSV, manifest, bubble plot); this just does the ranking.

    Returns a list of dicts (tf, score_a, score_b, auc_a, auc_b, tpm_a,
    tpm_b), sorted best-first (score_a - score_b for specificity, score_a
    for activity)."""
    from .expression import get_tpm

    results = []
    for tf in tfs:
        pos_test_a, neg_test_a = held_out_test_split(tf, cell_types[0], cfg)
        model_a = get_or_train_model(tf, cell_types[0], cfg)
        score_a, auc_a = score_and_auc(model_a, pos_test_a, neg_test_a)
        tpm_a = get_tpm(tf, cell_types[0], cfg)

        score_b, auc_b, tpm_b = None, None, None
        if target == "specificity":
            pos_test_b, neg_test_b = held_out_test_split(tf, cell_types[1], cfg)
            model_b = get_or_train_model(tf, cell_types[1], cfg)
            score_b, auc_b = score_and_auc(model_b, pos_test_b, neg_test_b)
            tpm_b = get_tpm(tf, cell_types[1], cfg)

        results.append({"tf": tf, "score_a": score_a, "score_b": score_b,
                         "auc_a": auc_a, "auc_b": auc_b,
                         "tpm_a": tpm_a, "tpm_b": tpm_b})

    rank_key = (lambda r: r["score_a"] - r["score_b"]) if target == "specificity" else (lambda r: r["score_a"])
    results.sort(key=rank_key, reverse=True)
    return results
