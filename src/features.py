"""Feature-matrix helpers: stacking fingerprints, fusing them, and rebalancing.

Everything here returns **new** arrays or new DataFrames. Nothing modifies the frame
you pass in, so the optional experiments in the notebook cannot disturb the main
training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def stack_fingerprint(df, column):
    """One fingerprint column -> a (n_molecules, n_bits) uint8 matrix.

    uint8 rather than float32 to keep the memory down: counts never exceed 24 in these
    fingerprints, so a byte is enough where float32 spends four. On the full training file
    that is 0.77 GB instead of 3.08 GB per fingerprint. LightGBM converts internally either
    way, so predictions are unchanged.
    """
    return np.stack(df[column].to_numpy()).astype(np.uint8)


def fuse_fingerprints(df, columns):
    """Lay several fingerprint columns side by side into one wider matrix.

    Each fingerprint describes the same molecule in a different way, so joining them
    gives the model more to work with. The cost is width: the matrix gets as wide as
    the sum of its parts, and training time grows with it.
    """
    if isinstance(columns, str):
        columns = [columns]
    return np.hstack([stack_fingerprint(df, c) for c in columns])


def fingerprint_widths(df, columns):
    """How many bits each fingerprint column contributes."""
    return {c: int(stack_fingerprint(df, c).shape[1]) for c in columns}


def fusion_summary(df, combinations):
    """Table of shape and relative cost for each combination of fingerprints."""
    rows = []
    base = None
    for combo in combinations:
        combo = [combo] if isinstance(combo, str) else list(combo)
        width = sum(fingerprint_widths(df, combo).values())
        base = width if base is None else base
        rows.append({
            "fingerprints": " + ".join(combo),
            "columns": len(combo),
            "features": width,
            "vs first row": f"{width / base:.2f}x",
        })
    return pd.DataFrame(rows)


def resample_negatives(df, ratio, label_col="LABEL", random_state=42, warn=True):
    """Return a COPY of `df` holding `ratio` negatives for every positive.

    `ratio` is negatives per positive: 1.0 is balanced, 5.0 is five inactives for
    each active, 0.5 is half as many inactives as actives.

    Every positive is always kept - actives are the scarce, expensive part of the
    data and you would not throw one away. Only the negatives are sampled.

    If the frame does not contain enough negatives for the ratio you asked for,
    all available negatives are used and a warning is printed, so the result is
    simply the original data rather than a silently different ratio.

    The input frame is never modified.
    """
    positives = df[df[label_col] == 1]
    negatives = df[df[label_col] == 0]

    wanted = int(round(float(ratio) * len(positives)))
    available = len(negatives)

    if wanted >= available:
        if warn and wanted > available:
            print(f"  ! ratio {ratio:g} needs {wanted} negatives but only {available} "
                  f"exist - using all of them (actual ratio "
                  f"{available / max(len(positives), 1):.2f})")
        kept = negatives
    else:
        kept = negatives.sample(n=wanted, random_state=random_state)

    out = pd.concat([positives, kept]).sample(frac=1.0, random_state=random_state)
    return out.reset_index(drop=True)


def balance_summary(df, ratios, label_col="LABEL", random_state=42):
    """Table of what each requested ratio does to the training set.

    Ratios that ask for more negatives than exist are reported as capped, with the
    actual ratio you end up with.
    """
    n_available = int((df[label_col] == 0).sum())
    n_positive = int((df[label_col] == 1).sum())

    rows = []
    for ratio in ratios:
        sub = resample_negatives(df, ratio=ratio, label_col=label_col,
                                 random_state=random_state, warn=False)
        pos = int(sub[label_col].sum())
        neg = len(sub) - pos
        capped = int(round(float(ratio) * n_positive)) > n_available
        rows.append({
            "requested ratio": ratio,
            "negatives kept": neg,
            "positives": pos,
            "total rows": len(sub),
            "actual ratio": round(neg / pos, 2) if pos else float("nan"),
            "% active": f"{pos / len(sub):.1%}",
            "note": "capped - not enough negatives, used all" if capped else "",
        })
    return pd.DataFrame(rows)
