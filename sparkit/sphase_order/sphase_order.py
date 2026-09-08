#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
import warnings


def _preset_threads():
    value = None
    for i, arg in enumerate(sys.argv):
        if arg == "--threads" and i + 1 < len(sys.argv):
            value = sys.argv[i + 1]
        elif arg.startswith("--threads="):
            value = arg.split("=", 1)[1]
    if value and str(value).isdigit() and int(value) > 0:
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(key, str(value))


_preset_threads()

import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, eigsh

np.seterr(all="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)

START = time.time()
WARNINGS: list[str] = []
EPS = np.float32(1e-6)


def log(msg):
    print(f"[{time.time() - START:8.1f}s] {msg}", flush=True)


def warn(msg):
    WARNINGS.append(str(msg))
    print("[WARN] " + str(msg), file=sys.stderr, flush=True)


def die(msg):
    print("[ERROR] " + str(msg), file=sys.stderr, flush=True)
    raise SystemExit(2)


def opener(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def fmt(value, digits=4):
    try:
        value = float(value)
        return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"
    except Exception:
        return "NA"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "yes", "1", "on"}:
        return True
    if token in {"false", "no", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def rank01(values, tiebreak=None):
    values = np.asarray(values, float)
    n = values.size
    if n <= 1:
        return np.zeros(n, float)
    keys = [np.arange(n)]
    if tiebreak is not None:
        keys.append(np.asarray(tiebreak, float))
    keys.append(values)
    out = np.empty(n, float)
    out[np.lexsort(tuple(keys))] = np.arange(n, dtype=float)
    return out / float(n - 1)


def rankdata_avg(values):
    values = np.asarray(values, float)
    n = values.size
    if n == 0:
        return np.zeros(0, float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 5:
        return np.nan
    ra = rankdata_avg(a[good]); ra -= ra.mean()
    rb = rankdata_avg(b[good]); rb -= rb.mean()
    denom = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / denom) if denom > 0 else np.nan


def qq_map(x, y, min_pairs=20):
    x, y = np.asarray(x, float), np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(good.sum()) < int(min_pairs):
        return lambda q: np.asarray(q, float).ravel().copy()
    xs, ys = np.sort(x[good]), np.sort(y[good])
    uniq, first = np.unique(xs, return_index=True)
    ym = np.add.reduceat(ys, first) / np.diff(np.append(first, xs.size))
    if uniq.size < 2:
        return lambda q: np.full(np.asarray(q, float).ravel().size, float(ym[0]))
    lo = (ym[1] - ym[0]) / max(uniq[1] - uniq[0], 1e-12)
    hi = (ym[-1] - ym[-2]) / max(uniq[-1] - uniq[-2], 1e-12)

    def mapper(q):
        q = np.asarray(q, float).ravel()
        out = np.interp(q, uniq, ym)
        left, right = q < uniq[0], q > uniq[-1]
        out[left] = ym[0] + lo * (q[left] - uniq[0])
        out[right] = ym[-1] + hi * (q[right] - uniq[-1])
        out[~np.isfinite(q)] = np.nan
        return out

    return mapper


def robust_mad(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    median = float(np.median(values))
    for candidate in (1.4826 * float(np.median(np.abs(values - median))),
                      float(np.std(values)), 1e-9):
        if np.isfinite(candidate) and candidate > 0:
            return median, float(candidate)
    return median, 1e-9


def gaussian_smooth(values, sigma):
    values = np.asarray(values, float)
    if sigma is None or sigma <= 0:
        return values.copy()
    radius = int(math.ceil(3.0 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / float(sigma)) ** 2)
    return np.convolve(values, kernel / kernel.sum(), mode="same")


def nan_reduce(stack, lo=10, hi=90):
    stack = np.asarray(stack, float)
    if stack.ndim != 2 or stack.size == 0:
        m = stack.shape[1] if stack.ndim == 2 else 0
        z = np.full(m, np.nan)
        return np.zeros(m), z, z.copy(), z.copy()
    quantile = np.nanpercentile(stack, [lo, hi], axis=0)
    return (np.isfinite(stack).sum(axis=0).astype(float), np.nanstd(stack, axis=0),
            quantile[0], quantile[1])


def binned_baseline(statistic, support, n_bins=20, min_n=25):
    statistic = np.asarray(statistic, float).ravel()
    support = np.asarray(support, float).ravel()
    good = np.isfinite(statistic) & np.isfinite(support) & (support > 0)
    if good.sum() < max(3 * min_n, 60):
        m, s = robust_mad(statistic[good])
        return (lambda q: np.full(np.size(q), m)), (lambda q: np.full(np.size(q), s))

    x, y = np.log10(support[good]), statistic[good]
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    n_bins = max(3, min(int(n_bins), x.size // min_n))
    cuts = np.linspace(0, x.size, n_bins + 1).astype(int)
    centres, means, sds = [], [], []
    for left, right in zip(cuts[:-1], cuts[1:]):
        if right - left < 5:
            continue
        m, s = robust_mad(y[left:right])
        centres.append(float(np.median(x[left:right]))); means.append(m); sds.append(s)
    if len(centres) < 2:
        m, s = robust_mad(y)
        return (lambda q: np.full(np.size(q), m)), (lambda q: np.full(np.size(q), s))
    centres, means, sds = map(np.asarray, (centres, means, sds))

    def mean_fn(query):
        q = np.log10(np.maximum(np.asarray(query, float), 1.0))
        return np.interp(q, centres, means, left=means[0], right=means[-1])

    def sd_fn(query):
        q = np.log10(np.maximum(np.asarray(query, float), 1.0))
        return np.maximum(np.interp(q, centres, sds, left=sds[0], right=sds[-1]), 1e-9)

    return mean_fn, sd_fn


def chrom_core(chrom):
    chrom = str(chrom)
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def chrom_sort_key(chrom):
    core = chrom_core(chrom)
    if core.isdigit():
        return 0, int(core)
    return {"X": (1, 0), "Y": (2, 0)}.get(core.upper(), (3, 0))


def is_autosome(chrom):
    return chrom_core(chrom).isdigit()


def chrom_alias_map(chroms):
    result = {}
    for index, chrom in enumerate(chroms):
        result[chrom] = index
        alias = chrom_core(chrom) if chrom.lower().startswith("chr") else "chr" + chrom
        result.setdefault(alias, index)
    return result


def load_chrom_sizes(path, specification):
    sizes, order = {}, []
    with opener(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            try:
                sizes[fields[0]] = int(fields[1]); order.append(fields[0])
            except Exception:
                continue
    if not sizes:
        die("no usable chromosome sizes")

    spec = specification.strip().lower()
    if spec in {"auto", "autosomes"}:
        keep = [c for c in order if is_autosome(c)]
    elif spec in {"auto+x", "autosomes+x", "auto_x"}:
        keep = [c for c in order if is_autosome(c) or chrom_core(c).upper() == "X"]
    else:
        aliases = {}
        for chrom in sizes:
            aliases[chrom] = chrom
            alias = (chrom_core(chrom) if chrom.lower().startswith("chr")
                     else "chr" + chrom)
            aliases.setdefault(alias, chrom)
        keep = []
        for token in specification.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                left, right = token.split("-", 1)
                prefix = "chr" if left.lower().startswith("chr") else ""
                ln = left.lower().replace("chr", "")
                rn = right.lower().replace("chr", "")
                if ln.isdigit() and rn.isdigit():
                    for number in range(int(ln), int(rn) + 1):
                        name = f"{prefix}{number}"
                        keep.append(aliases.get(name, name))
                    continue
            keep.append(aliases.get(token, token))
        missing = [c for c in keep if c not in sizes]
        if missing:
            die("chromosomes absent from --chrom-sizes: " + ",".join(missing))

    keep = sorted({c for c in keep if sizes.get(c, 0) > 1_000_000}, key=chrom_sort_key)
    if len(keep) < 4:
        die("at least four usable chromosomes are required")
    return keep, {c: sizes[c] for c in keep}


@dataclass
class Grid:
    chroms: list
    sizes: dict
    bin_size: int

    def __post_init__(self):
        self.bin_size = int(self.bin_size)
        self.nbin = np.asarray(
            [max(1, int(math.ceil(self.sizes[c] / float(self.bin_size))))
             for c in self.chroms], dtype=np.int64)
        self.offsets = np.concatenate(
            [np.asarray([0], np.int64), np.cumsum(self.nbin)[:-1]])
        self.total = int(self.nbin.sum())
        self.chrom_of = np.repeat(np.arange(len(self.chroms), dtype=np.int32), self.nbin)
        self.start_of = (np.arange(self.total, dtype=np.int64)
                         - self.offsets[self.chrom_of]) * self.bin_size

    def gbin(self, chrom_id, position):
        chrom_id = np.asarray(chrom_id, np.int64)
        local = np.clip(np.asarray(position, np.int64) // self.bin_size, 0, None)
        return self.offsets[chrom_id] + np.minimum(local, self.nbin[chrom_id] - 1)


def load_blacklist(path, chroms):
    if not path:
        return {}
    chrom_map = chrom_alias_map(chroms)
    raw = {}
    with opener(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3 or fields[0] not in chrom_map:
                continue
            try:
                start, end = int(fields[1]), int(fields[2])
            except Exception:
                continue
            if end > start:
                raw.setdefault(chrom_map[fields[0]], []).append((start, end))

    result = {}
    for chrom_id, intervals in raw.items():
        intervals.sort()
        starts, ends = [], []
        for start, end in intervals:
            if starts and start <= ends[-1]:
                ends[-1] = max(ends[-1], end)
            else:
                starts.append(start); ends.append(end)
        result[chrom_id] = (np.asarray(starts, np.int64), np.asarray(ends, np.int64))
    return result


def blacklist_hits(chrom_id, position, blacklist):
    hits = np.zeros(np.size(position), bool)
    if not blacklist or hits.size == 0:
        return hits
    chrom_id, position = np.asarray(chrom_id), np.asarray(position)
    for current, (starts, ends) in blacklist.items():
        mask = chrom_id == current
        if not mask.any():
            continue
        pos = position[mask]
        index = np.searchsorted(starts, pos, side="right") - 1
        valid = index >= 0
        inside = np.zeros(pos.size, bool)
        inside[valid] = pos[valid] < ends[index[valid]]
        hits[mask] = inside
    return hits


def blacklist_bin_fraction(grid, blacklist):
    fraction = np.zeros(grid.total, float)
    if not blacklist:
        return fraction
    bin_size = grid.bin_size
    for chrom_id, (starts, ends) in blacklist.items():
        offset, nbin = int(grid.offsets[chrom_id]), int(grid.nbin[chrom_id])
        for start, end in zip(starts, ends):
            first = max(0, start // bin_size)
            last = min(nbin - 1, (end - 1) // bin_size)
            if last < first:
                continue
            index = np.arange(first, last + 1, dtype=np.int64)
            bin_start = index * bin_size
            overlap = np.maximum(np.minimum(end, bin_start + bin_size)
                                 - np.maximum(start, bin_start), 0)
            np.add.at(fraction, offset + index, overlap / float(bin_size))
    return np.minimum(fraction, 1.0)


def bedgraph_to_bins(path, grid):
    chrom_map = chrom_alias_map(grid.chroms)
    numerator = np.zeros(grid.total, float)
    denominator = np.zeros(grid.total, float)
    bin_size = grid.bin_size
    with opener(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 4 or fields[0] not in chrom_map:
                continue
            try:
                chrom_id = chrom_map[fields[0]]
                start, end, value = int(fields[1]), int(fields[2]), float(fields[3])
            except Exception:
                continue
            if end <= start or not np.isfinite(value):
                continue
            offset, nbin = int(grid.offsets[chrom_id]), int(grid.nbin[chrom_id])
            first = max(0, start // bin_size)
            last = min(nbin - 1, (end - 1) // bin_size)
            if last < first:
                continue
            index = np.arange(first, last + 1, dtype=np.int64)
            bin_start = index * bin_size
            overlap = np.maximum(np.minimum(end, bin_start + bin_size)
                                 - np.maximum(start, bin_start), 0).astype(float)
            np.add.at(numerator, offset + index, value * overlap)
            np.add.at(denominator, offset + index, overlap)
    result = np.full(grid.total, np.nan)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def parse_region(value, chroms):
    if not value:
        return None
    try:
        chrom_name, interval = value.replace(",", "").split(":")
        start, end = (int(v) for v in interval.split("-"))
    except Exception:
        warn("cannot parse --plot-region; expected chr10:80000000-110000000")
        return None
    chrom_map = chrom_alias_map(chroms)
    if chrom_name not in chrom_map or end <= start:
        warn("--plot-region is outside the analysed chromosomes")
        return None
    return int(chrom_map[chrom_name]), start, end


def sniff_fragments(path):
    with opener(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                die("fragment file requires >=4 columns: chrom start end barcode")
            try:
                int(fields[1]); int(fields[2])
                return len(fields), False
            except Exception:
                return len(fields), True
    die("empty fragment file")


def _accumulate(existing, addition):
    if addition.size > existing.size:
        existing = np.concatenate(
            [existing, np.zeros(addition.size - existing.size, dtype=existing.dtype)])
    existing[:addition.size] += addition
    return existing


def read_fragments(path, chroms, blacklist, min_length=0, max_length=0,
                   chunk_size=4_000_000):
    n_columns, has_header = sniff_fragments(path)
    names = ["chrom", "start", "end", "barcode"] + [
        f"extra_{i}" for i in range(4, n_columns)]
    chrom_map = chrom_alias_map(chroms)
    barcode_map = {}
    total_by_cell = np.zeros(0, np.int64)
    black_by_cell = np.zeros(0, np.int64)
    cells, chrom_ids, midpoints = [], [], []
    n_lines = 0

    def map_barcodes(strings):
        output = np.empty(len(strings), np.int32)
        for i, barcode in enumerate(strings):
            index = barcode_map.get(barcode)
            if index is None:
                index = len(barcode_map); barcode_map[barcode] = index
            output[i] = index
        return output

    reader = pd.read_csv(path, sep="\t", header=0 if has_header else None, names=names,
                         usecols=["chrom", "start", "end", "barcode"], comment="#",
                         chunksize=chunk_size, engine="c")
    for frame in reader:
        n_lines += len(frame)
        cat_chrom = pd.Categorical(frame["chrom"].astype(str))
        cat_map = np.asarray([chrom_map.get(str(x), -1) for x in cat_chrom.categories],
                             np.int32)
        cat_map = np.concatenate([cat_map, np.asarray([-1], np.int32)])
        codes = np.asarray(cat_chrom.codes, np.int64)
        chrom_id = cat_map[np.where(codes >= 0, codes, cat_map.size - 1)]

        starts = pd.to_numeric(frame["start"], errors="coerce").to_numpy()
        ends = pd.to_numeric(frame["end"], errors="coerce").to_numpy()
        numeric = np.isfinite(starts) & np.isfinite(ends)
        starts = np.nan_to_num(starts, nan=-1).astype(np.int64)
        ends = np.nan_to_num(ends, nan=-1).astype(np.int64)

        length_ok = np.ones(ends.size, bool)
        if min_length > 0:
            length_ok &= (ends - starts) >= min_length
        if max_length > 0:
            length_ok &= (ends - starts) <= max_length
        selected = numeric & (chrom_id >= 0) & (ends > starts) & length_ok
        if not selected.any():
            continue

        cat_barcode = pd.Categorical(frame["barcode"].astype(str).to_numpy()[selected])
        global_barcode = map_barcodes([str(x) for x in cat_barcode.categories])
        cell_id = global_barcode[np.asarray(cat_barcode.codes, np.int64)]
        n_barcodes = len(barcode_map)
        total_by_cell = _accumulate(total_by_cell,
                                    np.bincount(cell_id, minlength=n_barcodes))
        midpoint = ((starts[selected] + ends[selected]) // 2).astype(np.int64)
        selected_chrom = chrom_id[selected].astype(np.int16)
        black = blacklist_hits(selected_chrom, midpoint, blacklist)
        black_by_cell = _accumulate(black_by_cell,
                                    np.bincount(cell_id[black], minlength=n_barcodes))
        keep = ~black
        cells.append(cell_id[keep].astype(np.int32))
        chrom_ids.append(selected_chrom[keep].astype(np.int16))
        midpoints.append(midpoint[keep].astype(np.int32))

    if not cells:
        die("no fragments remain after filtering")
    cell = np.concatenate(cells)
    chrom = np.concatenate(chrom_ids)
    midpoint = np.concatenate(midpoints)
    n_barcodes = len(barcode_map)
    if total_by_cell.size < n_barcodes:
        total_by_cell = np.pad(total_by_cell, (0, n_barcodes - total_by_cell.size))
    if black_by_cell.size < n_barcodes:
        black_by_cell = np.pad(black_by_cell, (0, n_barcodes - black_by_cell.size))
    barcodes = [None] * n_barcodes
    for barcode, index in barcode_map.items():
        barcodes[index] = barcode
    log(f"  {n_lines:,} input lines; {cell.size:,} retained events; "
        f"{n_barcodes:,} barcodes")
    return {"cell": cell, "chrom": chrom, "midpoint": midpoint, "barcodes": barcodes,
            "n_total": total_by_cell, "n_black": black_by_cell}


def pooled_adjacent_gaps(cell, chrom, position, n_cells, seed=1, n_sample_cells=400,
                         max_gap=1_000_000):
    rng = np.random.default_rng(seed)
    populated = np.flatnonzero(np.bincount(cell, minlength=n_cells) > 0)
    if populated.size == 0:
        return np.zeros(0, np.int64)
    chosen = rng.choice(populated, size=min(n_sample_cells, populated.size),
                        replace=False)
    selected = np.isin(cell, chosen)
    c, h, p = cell[selected], chrom[selected], position[selected]
    order = np.lexsort((p, h, c))
    c, h, p = c[order], h[order], p[order]
    same = (c[1:] == c[:-1]) & (h[1:] == h[:-1])
    gaps = (p[1:].astype(np.int64) - p[:-1].astype(np.int64))[same]
    return gaps[(gaps > 0) & (gaps < max_gap)]


def auto_collapse_distance(gaps, maximum=500):
    gaps = np.asarray(gaps, float)
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if gaps.size < 1000:
        return 0, {"mode": "too_few_gaps"}
    short_fraction = float(np.mean(gaps < 300))
    if short_fraction < 0.03:
        return 0, {"mode": "no_short_gap_peak", "short_fraction": short_fraction}
    edges = np.arange(0.0, 6.0 + 0.025, 0.025)
    histogram, _ = np.histogram(np.log10(gaps), bins=edges)
    smoothed = gaussian_smooth(histogram.astype(float), 2.0)
    centres = 0.5 * (edges[:-1] + edges[1:])
    short = centres < math.log10(300)
    long = (centres > math.log10(300)) & (centres < math.log10(5000))
    if not short.any() or not long.any():
        return min(200, maximum), {"mode": "fallback", "short_fraction": short_fraction}
    first = centres[short][np.argmax(smoothed[short])]
    second = centres[long][np.argmax(smoothed[long])]
    between = (centres > first) & (centres < second)
    if not between.any():
        return min(200, maximum), {"mode": "fallback", "short_fraction": short_fraction}
    antimode = centres[between][np.argmin(smoothed[between])]
    return int(np.clip(10 ** antimode, 50, maximum)), {
        "mode": "antimode", "short_fraction": short_fraction,
        "antimode_bp": float(10 ** antimode)}


def collapse_events(cell, chrom, position, chroms, sizes, distance):
    if distance <= 1:
        return cell, chrom, position
    grid = Grid(chroms, sizes, distance)
    key = (cell.astype(np.int64) * np.int64(grid.total)
           + grid.gbin(chrom.astype(np.int64), position.astype(np.int64)))
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    first = np.ones(sorted_key.size, bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    keep = np.sort(order[first])
    log(f"  collapse distance={distance} bp: {cell.size:,} -> {keep.size:,} events")
    return (cell[keep].astype(np.int32), chrom[keep].astype(np.int16),
            position[keep].astype(np.int32))


def fit_two_lognormal(values, iterations=200, tolerance=1e-7):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 50:
        return None
    q25, q75 = np.quantile(values, [0.25, 0.75])
    mean1, mean2 = float(q25), float(q75)
    spread = max(float(np.std(values)), 0.2)
    sd1, sd2 = max(0.6 * spread, 0.1), max(0.8 * spread, 0.1)
    weight = 0.6
    for _ in range(iterations):
        c1 = weight * np.exp(-0.5 * ((values - mean1) / sd1) ** 2) / sd1
        c2 = (1 - weight) * np.exp(-0.5 * ((values - mean2) / sd2) ** 2) / sd2
        r = c1 / np.maximum(c1 + c2, 1e-300)
        n1 = float(r.sum())
        if n1 < 5 or values.size - n1 < 5:
            break
        m1 = float(np.sum(r * values) / n1)
        m2 = float(np.sum((1 - r) * values) / (values.size - n1))
        s1 = math.sqrt(max(float(np.sum(r * (values - m1) ** 2) / n1), 0.01))
        s2 = math.sqrt(max(float(np.sum((1 - r) * (values - m2) ** 2)
                                 / (values.size - n1)), 0.01))
        w = float(np.clip(n1 / values.size, 0.02, 0.98))
        if m1 > m2:
            m1, m2, s1, s2, w = m2, m1, s2, s1, 1.0 - w
        delta = max(abs(m1 - mean1), abs(m2 - mean2), abs(w - weight))
        mean1, mean2, sd1, sd2, weight = m1, m2, max(s1, 0.1), max(s2, 0.1), w
        if delta < tolerance:
            break
    if not (np.isfinite([mean1, mean2, sd1, sd2, weight]).all() and mean2 > mean1):
        return None
    return mean1, mean2, sd1, sd2, weight


def infer_foreground(cell, chrom, position, n_cells, threshold=0.5, shrink_n0=60.0,
                     min_events=50, seed=1):
    n_events_per_cell = np.bincount(cell, minlength=n_cells).astype(float)
    order = np.lexsort((position, chrom, cell))
    sc, sh, sp_ = cell[order], chrom[order], position[order]
    n_events = sc.size

    gaps = np.full(n_events, np.nan)
    if n_events > 1:
        same = (sc[1:] == sc[:-1]) & (sh[1:] == sh[:-1])
        difference = (sp_[1:].astype(np.int64) - sp_[:-1].astype(np.int64)).astype(float)
        valid = same & (difference > 0)
        gaps[1:][valid] = difference[valid]
    has_gap = np.isfinite(gaps) & (gaps > 0)
    if has_gap.sum() < 500:
        die("too few adjacent gaps for foreground inference")

    log_gap = np.log(gaps[has_gap])
    rng = np.random.default_rng(seed)
    fitted = fit_two_lognormal(rng.choice(log_gap, 500_000, replace=False)
                               if log_gap.size > 500_000 else log_gap)
    if fitted is None:
        die("pooled two-lognormal gap mixture failed")
    mean1, mean2, sd1, sd2, pooled = fitted
    log(f"  gap mixture: short median={math.exp(mean1):.0f} bp; "
        f"long median={math.exp(mean2):.0f} bp; pooled short weight={pooled:.3f}")

    gap_cell = sc[has_gap]
    d1 = np.exp(-0.5 * ((log_gap - mean1) / sd1) ** 2) / sd1
    d2 = np.exp(-0.5 * ((log_gap - mean2) / sd2) ** 2) / sd2
    n_gap = np.bincount(gap_cell, minlength=n_cells).astype(float)
    weight = np.full(n_cells, pooled)
    for _ in range(40):
        w = weight[gap_cell]
        r = (w * d1) / np.maximum(w * d1 + (1.0 - w) * d2, 1e-300)
        raw = np.bincount(gap_cell, weights=r, minlength=n_cells) / np.maximum(n_gap, 1.)
        blend = n_gap / (n_gap + float(shrink_n0))
        new = np.clip(blend * raw + (1.0 - blend) * pooled, 0.01, 0.99)
        if np.max(np.abs(new - weight)) < 1e-6:
            weight = new
            break
        weight = new

    w = weight[gap_cell]
    gap_posterior = np.zeros(n_events)
    gap_posterior[has_gap] = (w * d1) / np.maximum(w * d1 + (1.0 - w) * d2, 1e-300)
    event_posterior = gap_posterior.copy()
    if n_events > 1:
        same_next = (sc[:-1] == sc[1:]) & (sh[:-1] == sh[1:])
        right = np.zeros(n_events - 1)
        right[same_next] = gap_posterior[1:][same_next]
        event_posterior[:-1] = np.maximum(event_posterior[:-1], right)

    foreground = np.zeros(n_events, bool)
    foreground[order] = event_posterior > threshold
    n_signal = np.bincount(cell[foreground], minlength=n_cells).astype(float)
    fallback = (n_signal < float(min_events)) & (n_events_per_cell >= float(min_events))
    if fallback.any():
        foreground |= fallback[cell]
        n_signal = np.bincount(cell[foreground], minlength=n_cells).astype(float)
        warn(f"{int(fallback.sum())} cells kept ALL events because the foreground "
             f"model would have emptied them (typical for very early S)")
    frac_signal = n_signal / np.maximum(n_events_per_cell, 1.0)
    log(f"  foreground: median frac_signal={np.nanmedian(frac_signal):.3f} "
        f"(p05/p95={np.nanpercentile(frac_signal, 5):.3f}/"
        f"{np.nanpercentile(frac_signal, 95):.3f})")
    return foreground, {"n_signal": n_signal, "frac_signal": frac_signal,
                        "n_fallback_cells": int(fallback.sum()),
                        "short_median_bp": float(math.exp(mean1)),
                        "long_median_bp": float(math.exp(mean2)),
                        "pooled_short_weight": float(pooled)}


def build_matrix(cell, chrom, position, event_mask, row_of_cell, grid, n_rows,
                 binarize=False):
    selected = (event_mask & (row_of_cell[cell] >= 0)) if event_mask is not None \
        else (row_of_cell[cell] >= 0)
    if not np.any(selected):
        return sp.csr_matrix((n_rows, grid.total), dtype=np.float32)
    rows = row_of_cell[cell[selected]].astype(np.int64)
    columns = grid.gbin(chrom[selected], position[selected]).astype(np.int64)
    matrix = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, columns)),
                           shape=(n_rows, grid.total)).tocsr()
    matrix.sum_duplicates()
    if binarize:
        matrix.data = np.ones_like(matrix.data, np.float32)
    return matrix


def dilate_columns(matrix, chrom_of_column, radius):
    matrix = matrix.tocsr().copy()
    matrix.data = np.ones_like(matrix.data, np.float32)
    radius = int(max(0, radius))
    if radius == 0:
        return matrix
    n = matrix.shape[1]
    chrom_of_column = np.asarray(chrom_of_column)
    diagonals, offsets = [np.ones(n, np.float32)], [0]
    for shift in range(1, radius + 1):
        if shift >= n:
            break
        same = (chrom_of_column[:-shift] == chrom_of_column[shift:]).astype(np.float32)
        diagonals.extend([same, same]); offsets.extend([shift, -shift])
    operator = sp.diags(diagonals, offsets, shape=(n, n), format="csr", dtype=np.float32)
    dilated = (matrix @ operator).tocsr()
    dilated.data = np.minimum(dilated.data, 1.0)
    dilated.eliminate_zeros()
    return dilated


def circular_shift(occupancy, chrom_of_column, rng):
    matrix = occupancy.tocsr()
    n_cells, n_columns = matrix.shape
    chroms = np.unique(chrom_of_column)
    compact = np.searchsorted(chroms, chrom_of_column)
    lengths = np.bincount(compact, minlength=chroms.size).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    within = np.arange(n_columns, dtype=np.int64) - starts[compact]
    rows = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(matrix.indptr))
    columns = matrix.indices.astype(np.int64)
    which = compact[columns]
    offsets = rng.integers(0, np.maximum(lengths, 1), size=(n_cells, chroms.size))
    new_columns = starts[which] + (within[columns]
                                   - offsets[rows, which]) % lengths[which]
    output = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, new_columns)),
                           shape=(n_cells, n_columns)).tocsr()
    output.data = np.ones_like(output.data, np.float32)
    return output


def make_blocks(chrom_of_column, target, budget, min_bins=64):
    chrom_of_column = np.asarray(chrom_of_column)
    present, counts = np.unique(chrom_of_column, return_counts=True)
    budget = int(max(4, budget))
    if present.size > budget:
        loads = np.zeros(budget)
        assign = {}
        for i in np.argsort(-counts):
            b = int(np.argmin(loads)); assign[int(present[i])] = b; loads[b] += counts[i]
        warn(f"only {budget} bootstrap blocks fit in --gram-budget-gb")
        return np.asarray([assign[int(c)] for c in chrom_of_column], np.int32), budget

    seg = max(counts.sum() / float(np.clip(target, present.size, budget)),
              float(min_bins))
    while True:
        n_seg = np.minimum(np.maximum(1, np.rint(counts / seg).astype(int)),
                           np.maximum(1, counts // int(min_bins)))
        if int(n_seg.sum()) <= budget:
            break
        seg *= 1.25
    block = np.zeros(chrom_of_column.size, np.int32)
    nxt = 0
    for chrom, count, k in zip(present, counts, n_seg):
        index = np.flatnonzero(chrom_of_column == chrom)
        bounds = np.linspace(0, int(count), int(k) + 1).astype(int)
        for j in range(int(k)):
            block[index[bounds[j]:bounds[j + 1]]] = nxt; nxt += 1
    if nxt < 8:
        warn(f"only {nxt} bootstrap blocks; uncertainty will be coarse")
    return block, int(nxt)


def block_information(occupancy, block_of_column, n_blocks):
    n_col = occupancy.shape[1]
    indicator = sp.csr_matrix(
        (np.ones(n_col, np.float32), (np.arange(n_col, dtype=np.int64),
                                      np.asarray(block_of_column, np.int64))),
        shape=(n_col, int(n_blocks)))
    counts = np.asarray((occupancy @ indicator).todense(), np.float32)
    n_bins = np.asarray(indicator.sum(axis=0)).ravel().astype(np.float32)
    return np.minimum(counts, n_bins[None, :] - counts)


def _dense_blocks(occupancy, block_of_column, n_blocks):
    columns = occupancy.tocsc()
    for b in range(int(n_blocks)):
        take = np.flatnonzero(block_of_column == b)
        if take.size:
            yield np.ascontiguousarray(columns[:, take].toarray(), dtype=np.float32)


def _phi_block(dense):
    n_bins = np.float32(dense.shape[1])
    k = dense.sum(axis=1, dtype=np.float32)
    p = (k / n_bins).astype(np.float32)
    sd = np.sqrt(np.maximum(p * (1.0 - p), np.float32(1e-12))).astype(np.float32)
    m = np.minimum(k, n_bins - k).astype(np.float32)
    gram = dense @ dense.T
    gram /= n_bins
    gram -= np.outer(p, p)
    gram /= np.outer(sd, sd)
    np.clip(gram, -1.0, 1.0, out=gram)
    np.nan_to_num(gram, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    dead = m <= 0
    gram[dead, :] = 0.0
    gram[:, dead] = 0.0
    return 0.5 * (gram + gram.T), m


def _finalise(num, den):
    similarity = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    np.clip(similarity, -1.0, 1.0, out=similarity)
    if similarity.shape[0] == similarity.shape[1]:
        np.fill_diagonal(similarity, np.nan)
    return similarity


class BlockPhi:
    def __init__(self, occupancy, block_of_column, n_blocks):
        self.n = occupancy.shape[0]
        self.phi, m_list = [], []
        for dense in _dense_blocks(occupancy, block_of_column, n_blocks):
            gram, m = _phi_block(dense)
            self.phi.append(gram); m_list.append(m)
        self.m = np.vstack(m_list) if m_list else np.zeros((0, self.n), np.float32)
        self.nb = len(self.phi)
        if self.nb < 2:
            die("fewer than two usable genomic blocks; lower --graph-bin")

    def combine(self, rows, weights=None):
        rows = np.asarray(rows, np.int64)
        grid = np.ix_(rows, rows)
        num = np.zeros((rows.size, rows.size), np.float32)
        den = np.zeros_like(num)
        tmp = np.empty_like(num)
        for b in range(self.nb):
            w = 1.0 if weights is None else float(weights[b])
            if w <= 0:
                continue
            mb = self.m[b][rows]
            weight = np.minimum.outer(mb, mb)
            if w != 1.0:
                weight *= np.float32(w)
            np.multiply(self.phi[b][grid], weight, out=tmp)
            num += tmp; den += weight
        return _finalise(num, den)


def phi_matrix(occupancy, block_of_column, n_blocks):
    n = occupancy.shape[0]
    num = np.zeros((n, n), np.float32)
    den = np.zeros_like(num)
    for dense in _dense_blocks(occupancy, block_of_column, n_blocks):
        gram, m = _phi_block(dense)
        weight = np.minimum.outer(m, m)
        num += gram * weight; den += weight
    return _finalise(num, den)


def topk_rows(similarity, k):
    n = similarity.shape[1]
    work = np.where(np.isfinite(similarity), similarity, -np.inf).astype(np.float32)
    k = int(min(max(1, k), n))
    index = np.argpartition(-work, k - 1, axis=1)[:, :k]
    value = np.take_along_axis(work, index, axis=1)
    order = np.argsort(-value, axis=1, kind="stable")
    return (np.take_along_axis(index, order, axis=1),
            np.take_along_axis(value, order, axis=1))


def build_graph(similarity, k):
    n = similarity.shape[0]
    if n < 4:
        return sp.csr_matrix((n, n)), np.zeros(n, bool)
    k = int(min(max(3, k), n - 1))
    index, value = topk_rows(similarity, k)
    weight = np.tile(np.arange(k, 0, -1, dtype=np.float64) / float(k), (n, 1))
    weight[~np.isfinite(value)] = 0.0
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    A = sp.coo_matrix((weight.ravel(), (rows, index.ravel().astype(np.int64))),
                      shape=(n, n)).tocsr()
    graph = (A + A.T) * 0.5
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    n_comp, label = connected_components(graph, directed=False)
    main = (np.ones(n, bool) if n_comp == 1
            else label == int(np.argmax(np.bincount(label))))
    main &= np.asarray(graph.sum(axis=1)).ravel() > 0
    return graph, main


def fiedler(graph):
    n = graph.shape[0]
    degree = np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1e-12)
    inverse_root = 1.0 / np.sqrt(degree)
    normalised = (sp.diags(inverse_root) @ graph @ sp.diags(inverse_root)).tocsr()
    v0 = np.sqrt(degree); v0 /= np.linalg.norm(v0)
    vector = None
    if n > 400:
        start = np.sin(np.arange(1, n + 1) * (math.pi / (n + 1)))
        start -= v0 * float(v0 @ start)
        norm = np.linalg.norm(start)
        start = start / norm if norm > 0 else np.ones(n) / math.sqrt(n)
        try:
            operator = LinearOperator(
                (n, n), dtype=np.float64,
                matvec=lambda x: normalised @ np.asarray(x, float).ravel()
                - 2.0 * v0 * float(v0 @ np.asarray(x, float).ravel()))
            values, vectors = eigsh(operator, k=min(3, n - 2), which="LA", v0=start,
                                    tol=1e-10, maxiter=50000)
            vector = vectors[:, int(np.argmax(values))]
        except Exception as error:
            warn(f"eigsh failed ({error}); using a dense solver")
    if vector is None:
        dense = normalised.toarray().astype(float) - 2.0 * np.outer(v0, v0)
        vector = np.linalg.eigh(dense)[1][:, -1]
    vector = inverse_root * vector
    return -vector if vector[int(np.argmax(np.abs(vector)))] < 0 else vector


def robinson_pairs(n, max_pairs=300_000):
    if n < 12:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    if n * (n - 1) // 2 <= max_pairs:
        return np.triu_indices(n, 1)
    offsets = np.unique(np.rint(np.geomspace(
        1, n - 1, int(max(8, min(128, max_pairs // n))))).astype(np.int64))
    a = np.tile(np.arange(n, dtype=np.int64), offsets.size)
    b = (a + np.repeat(offsets, n)) % n
    keep = a != b
    return a[keep], b[keep]


def _robinson_scorer(similarity, index, pairs):
    a, b = pairs
    if a.size == 0:
        return lambda coordinate: np.nan
    remap = np.full(similarity.shape[0], -1, np.int64)
    remap[index] = np.arange(index.size)
    pa, pb = remap[a], remap[b]
    keep = (pa >= 0) & (pb >= 0)
    pa, pb = pa[keep], pb[keep]
    value = np.asarray(similarity[index[pa], index[pb]], float)
    ok = np.isfinite(value)
    pa, pb, value = pa[ok], pb[ok], value[ok]
    if pa.size < 50:
        return lambda coordinate: np.nan
    return lambda coordinate: spearman(-np.abs(coordinate[pa] - coordinate[pb]), value)


def seriate(similarity, k, refine_rounds, pairs):
    n = similarity.shape[0]
    coordinate = np.full(n, np.nan)
    graph, main = build_graph(similarity, k)
    index = np.flatnonzero(main)
    if index.size < 20:
        return coordinate, main, np.nan
    subgraph = graph[index][:, index].tocsr()
    score = _robinson_scorer(similarity, index, pairs)

    best = rank01(fiedler(subgraph))
    best_score = score(best)
    row_sum = np.maximum(np.asarray(subgraph.sum(axis=1)).ravel(), 1e-12)
    smoother = sp.diags(1.0 / row_sum) @ subgraph
    for _ in range(int(max(0, refine_rounds))):
        candidate = rank01(0.5 * best + 0.5 * np.asarray(smoother @ best).ravel(),
                           tiebreak=best)
        candidate_score = score(candidate)
        if not (np.isfinite(candidate_score) and np.isfinite(best_score)
                and candidate_score > best_score + 1e-12):
            break
        best, best_score = candidate, candidate_score
    coordinate[index] = best
    return coordinate, main, best_score


def neighbour_contrast(similarity, k):
    n = similarity.shape[0]
    k = int(min(max(3, k), max(1, n - 1)))
    finite = np.isfinite(similarity)
    count = finite.sum(axis=1)
    baseline = np.where(finite, similarity, 0.0).sum(axis=1) / np.maximum(count, 1)
    _, value = topk_rows(similarity, k)
    good = np.isfinite(value)
    top = np.where(good, value, 0.0).sum(axis=1) / np.maximum(good.sum(axis=1), 1)
    contrast = top - baseline
    contrast[count == 0] = np.nan
    return contrast


def order_cells(bank, rows, args, rng, k):
    rows = np.asarray(rows, np.int64)
    similarity = bank.combine(rows)
    pairs = robinson_pairs(rows.size)
    coordinate, main, robinson = seriate(similarity, k, args.refine_rounds, pairs)
    contrast = neighbour_contrast(similarity, k)

    base = np.isfinite(coordinate)
    replicates, abs_rho, n_failed = [], [], 0
    for _ in range(int(max(0, args.n_bootstrap))):
        weights = rng.multinomial(bank.nb, np.full(bank.nb, 1.0 / bank.nb))
        if int((weights > 0).sum()) < 2:
            n_failed += 1
            continue
        boot = bank.combine(rows, weights)
        boot_coordinate, _, _ = seriate(boot, k, args.refine_rounds, pairs)
        del boot
        ok = base & np.isfinite(boot_coordinate)
        rho = spearman(boot_coordinate[ok], coordinate[ok]) if ok.any() else np.nan
        if (ok.sum() < 0.9 * max(base.sum(), 1) or not np.isfinite(rho)
                or abs(rho) < float(args.boot_min_abs_rho)):
            n_failed += 1
            continue
        replicate = np.full(rows.size, np.nan)
        replicate[ok] = rank01(boot_coordinate[ok] if rho > 0 else -boot_coordinate[ok])
        replicates.append(replicate); abs_rho.append(abs(rho))

    stack = np.vstack(replicates) if replicates else np.zeros((0, rows.size))
    seen, rank_sd, rank_lo, rank_hi = nan_reduce(stack)
    if replicates:
        enough = seen >= max(4, 0.6 * len(replicates))
        rank_sd, rank_lo, rank_hi = (np.where(enough, x, np.nan)
                                     for x in (rank_sd, rank_lo, rank_hi))

    split_a = np.full(rows.size, np.nan)
    split_b = np.full(rows.size, np.nan)
    split_rho = np.nan
    if bank.nb >= 4:
        odd = np.asarray([1.0 if b % 2 else 0.0 for b in range(bank.nb)])
        split_a, _, _ = seriate(bank.combine(rows, odd), k, args.refine_rounds, pairs)
        split_b, _, _ = seriate(bank.combine(rows, 1.0 - odd), k, args.refine_rounds,
                                pairs)
        split_rho = spearman(split_a, split_b)
        if np.isfinite(split_rho) and split_rho < 0:
            split_b, split_rho = 1.0 - split_b, -split_rho

    return {"similarity": similarity, "coordinate": coordinate, "main": main,
            "rank_sd": rank_sd, "rank_lo": rank_lo, "rank_hi": rank_hi,
            "n_boot_used": seen, "boot_stack": stack, "contrast": contrast,
            "robinson": robinson, "split_a": split_a, "split_b": split_b,
            "split_rho": split_rho, "n_boot_failed": int(n_failed),
            "boot_abs_rho_median": float(np.median(abs_rho)) if abs_rho else np.nan}



def choose_landmarks(support, tiebreak, cap):
    n = int(np.size(support))
    if cap <= 0 or n <= cap:
        return np.arange(n)
    order = np.lexsort((np.asarray(tiebreak), np.asarray(support, float)))
    pick = np.unique(np.rint(np.linspace(0, n - 1, int(cap))).astype(np.int64))
    return np.sort(order[np.unique(np.concatenate([[0, n - 1], pick]))])


def _landmark_blocks(occupancy, land_rows, block_of_column, n_blocks):
    columns = occupancy.tocsc()
    blocks = []
    for b in range(int(n_blocks)):
        take = np.flatnonzero(block_of_column == b)
        if take.size == 0:
            continue
        block = columns[:, take].tocsr()
        land_T = np.ascontiguousarray(
            block[land_rows].toarray().astype(np.float32).T)
        n_bins = np.float32(take.size)
        k = land_T.sum(axis=0, dtype=np.float32)
        p = (k / n_bins).astype(np.float32)
        sd = np.sqrt(np.maximum(p * (1.0 - p), np.float32(1e-12))).astype(np.float32)
        blocks.append((block, n_bins, land_T, p, sd,
                       np.minimum(k, n_bins - k).astype(np.float32)))
    if not blocks:
        die("no usable genomic block for the extension")
    return blocks


def _phi_rows(blocks, rows):
    n_land = blocks[0][2].shape[1]
    num = np.zeros((rows.size, n_land), np.float32)
    den = np.zeros_like(num)
    for block, n_bins, land_T, p_land, sd_land, m_land in blocks:
        dense = np.ascontiguousarray(block[rows].toarray(), dtype=np.float32)
        k = dense.sum(axis=1, dtype=np.float32)
        p = (k / n_bins).astype(np.float32)
        sd = np.sqrt(np.maximum(p * (1.0 - p), np.float32(1e-12))).astype(np.float32)
        phi = dense @ land_T
        phi /= n_bins
        phi -= np.outer(p, p_land)
        phi /= np.outer(sd, sd_land)
        np.clip(phi, -1.0, 1.0, out=phi)
        np.nan_to_num(phi, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        weight = np.minimum.outer(np.minimum(k, n_bins - k).astype(np.float32), m_land)
        num += phi * weight
        den += weight
    return _finalise(num, den)


def _moving_mean(matrix, half):
    n_row, n_col = matrix.shape
    ok = np.isfinite(matrix)
    value = np.zeros((n_row, n_col + 1))
    value[:, 1:] = np.cumsum(np.where(ok, matrix, 0.0), axis=1)
    count = np.zeros((n_row, n_col + 1))
    count[:, 1:] = np.cumsum(ok, axis=1)
    i = np.arange(n_col)
    lo, hi = np.clip(i - half, 0, n_col), np.clip(i + half + 1, 0, n_col)
    den = count[:, hi] - count[:, lo]
    return np.where(den > 0, (value[:, hi] - value[:, lo]) / np.maximum(den, 1.0),
                    np.nan)


def _centroid(value, index, t):
    good = np.isfinite(value)
    floor = np.where(good, value, np.inf).min(axis=1, keepdims=True)
    weight = np.where(good & np.isfinite(floor), value - floor + EPS, 0.0)
    total = weight.sum(axis=1)
    ok = total > 0
    out = np.full(value.shape[0], np.nan)
    out[ok] = (weight[ok] * t[index[ok]]).sum(axis=1) / total[ok]
    return out, weight, ok


def extend_to_anchors(occupancy, anchor_rows, t, boot_t, block_of_column, n_blocks,
                      target_rows, k, chunk=512):
    n_cells = occupancy.shape[0]
    n_anchor = int(anchor_rows.size)
    blocks = _landmark_blocks(occupancy, anchor_rows, block_of_column, n_blocks)
    k_use = int(min(max(3, k), n_anchor - 1))
    half = int(max(1, min(k_use // 3, max(1, n_anchor // 20))))

    hat = np.full(n_cells, np.nan)
    secondary = np.full(n_cells, np.nan)
    contrast = np.full(n_cells, np.nan)
    n_boot = int(np.shape(boot_t)[0])
    boot_hat = np.full((n_boot, n_cells), np.nan, np.float32)
    position = np.full(n_cells, -1, np.int64)
    position[anchor_rows] = np.arange(n_anchor)
    t = np.asarray(t, float)
    target_rows = np.asarray(target_rows, np.int64)

    for start in range(0, target_rows.size, int(chunk)):
        rows = target_rows[start:start + int(chunk)]
        phi = _phi_rows(blocks, rows)
        own = position[rows]
        self_row = np.flatnonzero(own >= 0)
        if self_row.size:
            phi[self_row, own[self_row]] = np.nan

        finite = np.isfinite(phi)
        count = finite.sum(axis=1)
        baseline = np.where(finite, phi, 0.0).sum(axis=1) / np.maximum(count, 1)
        index, value = topk_rows(phi, k_use)
        good = np.isfinite(value)
        top = np.where(good, value, 0.0).sum(axis=1) / np.maximum(good.sum(axis=1), 1)
        chunk_contrast = top - baseline
        chunk_contrast[count == 0] = np.nan
        contrast[rows] = chunk_contrast

        secondary[rows] = _centroid(value, index, t)[0]

        smooth = _moving_mean(phi, half)
        s_index, s_value = topk_rows(smooth, k_use)
        centre, weight, ok = _centroid(s_value, s_index, t)
        hat[rows] = centre
        if n_boot and ok.any():
            gathered = boot_t[:, s_index[ok]]
            w = weight[ok][None, :, :]
            valid = np.isfinite(gathered) & (w > 0)
            numerator = np.where(valid, gathered * w, 0.0).sum(axis=2)
            denominator = np.where(valid, w, 0.0).sum(axis=2)
            boot_hat[:, rows[ok]] = np.divide(
                numerator, denominator, out=np.full_like(numerator, np.nan),
                where=denominator > 0)
    del blocks
    return {"hat": hat, "secondary": secondary, "contrast": contrast,
            "boot_hat": boot_hat}


def structure_null(occupancy, chrom_of_column, block_of_column, n_blocks, k, n_shifts,
                   seed):
    rng = np.random.default_rng(int(seed) + 7717)
    statistics, supports = [], []
    support = np.diff(occupancy.tocsr().indptr).astype(float)
    for _ in range(int(max(1, n_shifts))):
        shifted = circular_shift(occupancy, chrom_of_column, rng)
        similarity = phi_matrix(shifted, block_of_column, n_blocks)
        statistics.append(neighbour_contrast(similarity, k)); supports.append(support)
        del similarity, shifted
    return binned_baseline(np.concatenate(statistics), np.concatenate(supports))


def orientation_flip(progression, cell_anchor, anchor_high, flip_requested):
    rho = spearman(progression, cell_anchor)
    if not np.isfinite(rho):
        warn("the cell-level anchor (GC) carries no signal; the early/late "
             "direction is NOT established. Check --anchor-bedgraph")
        flip = False
    else:
        flip = (rho > 0) if anchor_high == "early" else (rho < 0)
    if flip_requested:
        flip = not flip
    if np.isfinite(rho) and abs(rho) < 0.25:
        warn(f"|spearman(progression, cell GC)| = {abs(rho):.3f} is weak; the "
             f"early/late orientation is not well supported by the anchor")
    return bool(flip), float(rho)


def infer_rt(signal_rt, progression, chrom_of_bin, start_of_bin, window_bp, shrink=3.0):
    binary = signal_rt.tocsr().copy()
    binary.data = np.ones_like(binary.data, np.float32)
    numerator = np.asarray(binary.T.dot(progression)).ravel()
    denominator = np.asarray(binary.sum(axis=0)).ravel()
    prior = float(np.mean(progression))
    raw = (numerator + shrink * prior) / np.maximum(denominator + shrink, 1e-12)
    raw[~np.isfinite(raw)] = prior
    output = raw.copy()
    if window_bp > 0:
        for chrom_id in np.unique(chrom_of_bin):
            index = np.flatnonzero(chrom_of_bin == chrom_id)
            starts = start_of_bin[index].astype(np.int64)
            left = np.searchsorted(starts, starts - int(window_bp), side="left")
            right = np.searchsorted(starts, starts + int(window_bp), side="right")
            values = raw[index]
            for j in range(index.size):
                output[index[j]] = np.median(values[left[j]:right[j]])
    return output, denominator


_PLT = None


def get_plt():
    global _PLT
    if _PLT is False:
        return None
    if _PLT is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            matplotlib.rcParams.update({"image.interpolation": "nearest",
                                        "image.resample": False,
                                        "pdf.fonttype": 42, "ps.fonttype": 42,
                                        "axes.unicode_minus": False})
            _PLT = plt
        except Exception:
            warn("matplotlib unavailable; plots skipped")
            _PLT = False
            return None
    return _PLT


def plot_qc_gates(path, metric, thresholds, passed, inferred_rt, anchor_rt):
    plt = get_plt()
    if plt is None:
        return
    log_depth, breadth = metric["log_depth"], metric["breadth"]
    figure, axes = plt.subplots(2, 3, figsize=(19, 11))

    finite = log_depth[np.isfinite(log_depth)]
    edges = np.linspace(finite.min(), finite.max(), 90) if finite.size > 1 else 10
    axes[0, 0].hist(finite, bins=edges, color="0.78", label="all")
    axes[0, 0].hist(log_depth[passed], bins=edges, color="tab:blue", alpha=.85,
                    label="pass")
    for value in (thresholds["depth_low"], thresholds["depth_high"]):
        axes[0, 0].axvline(value, color="red", ls="--", lw=1.4)
    axes[0, 0].set_title("Gate A  depth"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xlabel("log10(total fragments)")

    edges = np.linspace(0.0, max(1.0, float(np.nanmax(breadth))), 90)
    axes[0, 1].hist(breadth, bins=edges, color="0.78", label="all")
    axes[0, 1].hist(breadth[passed], bins=edges, color="tab:blue", alpha=.85,
                    label="pass")
    for key, colour in (("breadth_hard_floor", "red"), ("breadth_cut", "orange"),
                        ("breadth_max", "purple")):
        axes[0, 1].axvline(thresholds[key], color=colour, ls="--", lw=1.4, label=key)
    axes[0, 1].set_title("Gate B  genome breadth\nboth ends of S phase live here, "
                         "check for truncation")
    axes[0, 1].legend(fontsize=7)

    support, excess = metric["support"], metric["excess"]
    good = np.isfinite(excess) & np.isfinite(support)
    if good.any():
        x = np.log10(np.maximum(support[good], 1.0))
        y, keep = excess[good], passed[good]
        axes[0, 2].scatter(x[~keep], y[~keep], s=5, c="0.75", label="removed")
        axes[0, 2].scatter(x[keep], y[keep], s=5, c="tab:blue", label="pass")
        axes[0, 2].axhline(thresholds["min_structure_excess"], color="red", ls="--")
        axes[0, 2].axhline(0.0, color="green", ls=":", label="null")
        axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_title("Gate C  contrast above a circular-shift null\n"
                         "single evaluation, no refit loop")
    axes[0, 2].set_xlabel("log10(occupied graph bins)")

    rank_sd = metric["rank_sd"]
    finite_sd = rank_sd[np.isfinite(rank_sd)]
    if finite_sd.size > 10:
        bins = np.linspace(0.0, max(0.32, float(finite_sd.max())), 70)
        axes[1, 0].hist(finite_sd, bins=bins, color="0.78", label="ordered")
        axes[1, 0].hist(rank_sd[passed & np.isfinite(rank_sd)], bins=bins,
                        color="tab:blue", alpha=.85, label="pass")
        axes[1, 0].axvline(thresholds["max_rank_sd"], color="red", ls="--", lw=1.6)
        axes[1, 0].axvline(1.0 / math.sqrt(12.0), color="black", ls=":",
                           label="uniform = 0.289")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("block-bootstrap rank sd (FLAG ONLY)\n"
                         "never fed back into the ordering")

    frac_signal = metric["frac_signal"]
    finite_fs = frac_signal[np.isfinite(frac_signal)]
    if finite_fs.size > 10:
        bins = np.linspace(0, 1, 60)
        axes[1, 1].hist(finite_fs, bins=bins, color="0.78", label="all")
        axes[1, 1].hist(frac_signal[passed & np.isfinite(frac_signal)], bins=bins,
                        color="tab:blue", alpha=.85, label="pass")
        axes[1, 1].axvline(thresholds["min_frac_signal"], color="red", ls="--")
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Gate D  tract fraction (frac_signal)")

    good = np.isfinite(inferred_rt) & np.isfinite(anchor_rt)
    if good.sum() > 50:
        axes[1, 2].scatter(anchor_rt[good], inferred_rt[good], s=2, alpha=.25,
                           c="tab:purple")
    axes[1, 2].set_title("Validation (not used for ordering)\nrho="
                         f"{fmt(spearman(inferred_rt, anchor_rt), 3)}")
    axes[1, 2].set_xlabel("anchor (GC) per RT bin")
    axes[1, 2].set_ylabel("inferred RT")

    figure.suptitle("QC gates")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def plot_order_diagnostics(path, similarity, d):
    plt = get_plt()
    if plt is None:
        return
    figure, axes = plt.subplots(2, 3, figsize=(19, 11))
    progression = d["progression"]
    n = progression.size
    ordered = np.argsort(progression, kind="stable")
    x = (np.arange(n) + 0.5) / n

    land = d["land_progression"]
    show = np.argsort(land, kind="stable")
    if show.size > 1200:
        show = show[np.linspace(0, show.size - 1, 1200).astype(int)]
    block = similarity[np.ix_(show, show)]
    vmax = float(np.nanpercentile(block, 99.0)) if np.isfinite(block).any() else 1.0
    image = axes[0, 0].imshow(block, aspect="auto", interpolation="nearest",
                              cmap="magma", vmin=0.0, vmax=max(vmax, 1e-3),
                              extent=[0, 1, 1, 0])
    figure.colorbar(image, ax=axes[0, 0], fraction=0.045, label="phi")
    axes[0, 0].set_title("Landmark similarity in final order\nRobinson monotonicity = "
                         f"{fmt(d['robinson'], 3)} (+1 is perfect)")

    sel = show[::max(1, show.size // 600)]
    if sel.size > 20:
        distance = np.abs(land[sel][:, None] - land[sel][None, :])
        value = similarity[np.ix_(sel, sel)]
        edges = np.linspace(0, 1, 41)
        which = np.clip(np.searchsorted(edges, distance.ravel()) - 1, 0, 39)
        flat = value.ravel()
        means = np.array([np.nanmean(flat[which == j]) if np.any(which == j) else np.nan
                          for j in range(40)])
        axes[0, 1].plot(0.5 * (edges[:-1] + edges[1:]), means, "-o", ms=3)
    axes[0, 1].axhline(0.0, color="0.7", lw=.8)
    axes[0, 1].set_title("Similarity decay vs rank distance\n"
                         "monotone decay means the order is the true 1D coordinate")

    axes[0, 2].fill_between(x, d["rank_lo"][ordered], d["rank_hi"][ordered],
                            color="tab:blue", alpha=.25, label="bootstrap 10-90%")
    axes[0, 2].plot(x, progression[ordered], "-", lw=1, color="tab:blue")
    twin = axes[0, 2].twinx()
    twin.plot(x, d["rank_sd"][ordered], ".", ms=2, color="tab:red")
    twin.axhline(d["max_rank_sd"], color="red", ls="--", lw=1.0)
    twin.set_ylabel("rank_sd (red)")
    axes[0, 2].set_title("Ordinal coordinate and honest uncertainty\n"
                         "end uncertainty shows as a WIDER band, never as collapse")
    axes[0, 2].legend(fontsize=8, loc="upper left")

    axes[1, 0].plot(progression[ordered], d["cell_anchor"][ordered], ".", ms=2,
                    color="tab:green")
    axes[1, 0].set_title("Orientation evidence (post hoc)\n"
                         f"spearman(progression, cell GC) = {fmt(d['anchor_rho'], 3)}")

    axes[1, 1].semilogy(x, np.maximum(d["events"][ordered], 1.0), ".", ms=2,
                        color="tab:blue")
    twin = axes[1, 1].twinx()
    twin.plot(x, d["breadth"][ordered], ".", ms=2, color="tab:brown")
    twin.plot(x, d["contrast"][ordered], ".", ms=2, color="tab:purple")
    twin.set_ylabel("breadth (brown) / contrast (purple)")
    axes[1, 1].set_title("Support along the order")

    axes[1, 2].plot(d["split_a"], d["split_b"], ".", ms=2, color="tab:red")
    axes[1, 2].plot([0, 1], [0, 1], "-", lw=.8, color="0.7")
    axes[1, 2].set_title("Disjoint halves of the genome (same estimator)\n"
                         f"odd vs even blocks, spearman = {fmt(d['split_rho'], 4)}")

    figure.suptitle("Ordering diagnostics")
    figure.tight_layout(); figure.savefig(path, dpi=140); plt.close(figure)


def region_matrix(cell, chrom, position, row_of_cell, chrom_id, start, end, bin_size,
                  n_rows, event_mask=None):
    selected = ((chrom == chrom_id) & (position >= start) & (position < end)
                & (row_of_cell[cell] >= 0))
    if event_mask is not None:
        selected &= event_mask
    n_columns = int(math.ceil((end - start) / float(bin_size)))
    if not selected.any():
        return sp.csr_matrix((n_rows, n_columns), dtype=np.float32)
    rows = row_of_cell[cell[selected]].astype(np.int64)
    columns = np.clip(((position[selected] - start) // bin_size).astype(np.int64),
                      0, n_columns - 1)
    matrix = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, columns)),
                           shape=(n_rows, n_columns)).tocsr()
    matrix.sum_duplicates()
    return matrix


def plot_region_heatmap(path, region, ordered_rows, chrom_name, start, end, bin_size,
                        vmax=1.0, dpi=300, fig_size=(8.0, 5.5)):
    plt = get_plt()
    if plt is None:
        return None
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter

    matrix = np.log1p(np.asarray(region[ordered_rows].todense(), float))
    peak = matrix.max(axis=1, keepdims=True)
    matrix = np.divide(matrix, peak, out=np.zeros_like(matrix), where=peak > 0)
    empty = int((peak.ravel() <= 0).sum())

    cmap = LinearSegmentedColormap.from_list(
        "scedu", [(0.0, "#FFFFFF"), (0.3, "#2b3fd6"), (1.0, "#00008B")])
    cmap.set_bad("#FFFFFF")

    figure, axis = plt.subplots(figsize=fig_size)
    x_edges = start + np.arange(matrix.shape[1] + 1, dtype=float) * bin_size
    y_edges = np.linspace(0.0, 1.0, matrix.shape[0] + 1)
    mesh = axis.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(matrix), cmap=cmap,
                           vmin=0.0, vmax=float(vmax), shading="flat", linewidth=0,
                           antialiased=False, rasterized=True)
    axis.set_ylim(1, 0)
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.set_xlim(x_edges[0], x_edges[-1])
    axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    axis.set_xlabel("Chromosome position (Mb)")
    axis.set_ylabel(f"S-phase progression\n(n = {matrix.shape[0]} cells)")
    axis.set_title(f"{chrom_name}:{int(start):,}-{int(end):,}  "
                   f"(bin = {bin_size / 1000:g} kb)", pad=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    bar = figure.colorbar(mesh, ax=axis, fraction=0.03, pad=0.02)
    bar.set_label("max-normalised log counts", fontsize=9)
    bar.ax.tick_params(labelsize=8)

    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"n_cells": int(matrix.shape[0]), "n_bins": int(matrix.shape[1]),
            "n_cells_without_signal": empty}


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=("Deterministic single-cell S-phase ORDER reconstruction: spectral "
                     "seriation of a two-sided rank-affinity graph on an "
                     "information-weighted block phi similarity; block bootstrap for "
                     "uncertainty only; monotonically calibrated landmark extension."))

    group = parser.add_argument_group("Required I/O")
    group.add_argument("--fragments", default=None)
    group.add_argument("--chrom-sizes", required=True)
    group.add_argument("--anchor-bedgraph", required=True, help="per-bin anchor (GC). Used ONLY after ordering")
    group.add_argument("--outdir", required=True)
    group.add_argument("--blacklist", default=None)
    group.add_argument("--sample", default="sample")
    group.add_argument("--chroms", default="auto+x")
    group.add_argument("--anchor-high", choices=("early", "late"), default="early")
    group.add_argument("--flip-order", action="store_true")

    group = parser.add_argument_group("Gate A: depth")
    group.add_argument("--min-fragments", type=int, default=1000)
    group.add_argument("--depth-lower-quantile", type=float, default=0.01)
    group.add_argument("--depth-upper-quantile", type=float, default=0.99)
    group.add_argument("--minimum-cells", type=int, default=100)

    group = parser.add_argument_group("Gate B: breadth and events")
    group.add_argument("--breadth-hard-floor", type=float, default=0.1)
    group.add_argument("--breadth-min-floor", type=float, default=0.1)
    group.add_argument("--breadth-mad-k", type=float, default=3.0)
    group.add_argument("--breadth-max", type=float, default=0.7)
    group.add_argument("--min-profile-events", type=int, default=200)

    group = parser.add_argument_group("Gate C: replication structure")
    group.add_argument("--min-structure-excess", type=float, default=0.02)
    group.add_argument("--min-structure-z", type=float, default=0.0)
    group.add_argument("--null-shifts", type=int, default=4)

    group = parser.add_argument_group("Gate D: dispersed / ambient signal")
    group.add_argument("--min-frac-signal", type=float, default=0.1)

    group = parser.add_argument_group("Reliability (flag only)")
    group.add_argument("--max-rank-sd", type=float, default=0.10)
    group.add_argument("--n-bootstrap", type=int, default=40)
    group.add_argument("--boot-min-abs-rho", type=float, default=0.30)
    group.add_argument("--drop-low-confidence", action="store_true",
                       help="remove rank_sd > --max-rank-sd cells AFTER the order is "
                            "fixed (cannot change anybody's relative position)")

    group = parser.add_argument_group("Event preprocessing")
    group.add_argument("--collapse-distance", default="auto")
    group.add_argument("--collapse-max-bp", type=int, default=1000)
    group.add_argument("--fg-posterior", type=float, default=0.5)
    group.add_argument("--fg-min-events", type=int, default=50)
    group.add_argument("--min-frag-len", type=int, default=0)
    group.add_argument("--max-frag-len", type=int, default=0)

    group = parser.add_argument_group("Similarity graph")
    group.add_argument("--graph-bin", type=int, default=100000)
    group.add_argument("--graph-smooth-bp", type=int, default=200000)
    group.add_argument("--graph-autosomes-only", type=parse_bool, default=True)
    group.add_argument("--n-blocks", type=int, default=24)
    group.add_argument("--block-min-bins", type=int, default=64)
    group.add_argument("--min-valid-blocks", type=int, default=4)
    group.add_argument("--gram-budget-gb", type=float, default=4.0)
    group.add_argument("--max-order-cells", type=int, default=5000)
    group.add_argument("--knn", type=int, default=0,help="0 = auto (L/25, 10..50). Fixed once for the whole run")
    group.add_argument("--refine-rounds", type=int, default=2)
    group.add_argument("--extend-chunk", type=int, default=512)

    group = parser.add_argument_group("Outputs")
    group.add_argument("--rt-bin", type=int, default=200000)
    group.add_argument("--rt-smooth-bp", type=int, default=400000)
    group.add_argument("--usable-bin-lower-quantile", type=float, default=0.01)
    group.add_argument("--usable-bin-upper-quantile", type=float, default=0.999)
    group.add_argument("--n-bins", type=int, default=16)
    group.add_argument("--plot-region", default=None,help="region for the single S-phase heatmap")
    group.add_argument("--plot-bin", type=int, default=50000)
    group.add_argument("--plot-vmax", type=float, default=1.0)
    group.add_argument("--plot-dpi", type=float, default=300)
    group.add_argument("--no-plots", action="store_true")
    group.add_argument("--threads", type=int, default=16)
    group.add_argument("--seed", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    if args.fragments is None:
        args.fragments = args.sample + ".fragments.txt"

    rng = np.random.default_rng(args.seed)

    if not (0 < args.depth_lower_quantile < args.depth_upper_quantile < 1):
        die("invalid depth quantiles")
    if not (0.0 < args.breadth_hard_floor < args.breadth_max <= 1.0):
        die("require 0 < --breadth-hard-floor < --breadth-max <= 1")
    if args.graph_bin <= 0 or args.rt_bin <= 0:
        die("--graph-bin and --rt-bin must be positive")
    os.makedirs(os.path.join(args.outdir, "qc"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "bins"), exist_ok=True)
    prefix = os.path.join(args.outdir, args.sample)

    log("STEP 0  genome, blacklist and anchor")
    chroms, sizes = load_chrom_sizes(args.chrom_sizes, args.chroms)
    blacklist = load_blacklist(args.blacklist, chroms)
    rt_grid = Grid(chroms, sizes, args.rt_bin)
    graph_grid = Grid(chroms, sizes, args.graph_bin)
    bad_rt = blacklist_bin_fraction(rt_grid, blacklist) > 0.5
    bad_graph = blacklist_bin_fraction(graph_grid, blacklist) > 0.5
    anchor_all = bedgraph_to_bins(args.anchor_bedgraph, rt_grid)
    if np.isfinite(anchor_all).sum() < 200:
        die("insufficient anchor coverage")
    region = parse_region(args.plot_region, chroms)
    log(f"  {len(chroms)} chromosomes; {sum(sizes.values()) / 1e9:.2f} Gb; "
        f"{rt_grid.total} RT bins; {graph_grid.total} graph bins")

    log("STEP 1  reading fragments")
    fragments = read_fragments(args.fragments, chroms, blacklist, args.min_frag_len,
                               args.max_frag_len)
    raw_cell, raw_chrom, raw_position = (fragments["cell"], fragments["chrom"],
                                         fragments["midpoint"])
    barcodes = fragments["barcodes"]
    n_barcodes = len(barcodes)
    raw_depth = np.bincount(raw_cell, minlength=n_barcodes).astype(float)
    blacklist_fraction = fragments["n_black"] / np.maximum(fragments["n_total"], 1)
    metric_bc = np.flatnonzero(raw_depth > 0)
    n_metric = metric_bc.size
    if n_metric < args.minimum_cells:
        die("too few barcodes with fragments")
    row_of_metric = np.full(n_barcodes, -1, np.int64)
    row_of_metric[metric_bc] = np.arange(n_metric)
    barcode_key = np.empty(n_barcodes, np.int64)
    barcode_key[np.argsort(np.asarray(barcodes, dtype=object).astype(str),
                           kind="stable")] = np.arange(n_barcodes)
    log(f"  barcodes carrying fragments: {n_metric}/{n_barcodes}")

    log("STEP 2  near-duplicate collapse (modelling only)")
    token = str(args.collapse_distance).lower()
    if token in {"off", "none", "0"}:
        collapse_distance, collapse_info = 0, {"mode": "off"}
    elif token == "auto":
        collapse_distance, collapse_info = auto_collapse_distance(
            pooled_adjacent_gaps(raw_cell, raw_chrom, raw_position, n_barcodes,
                                 seed=args.seed), maximum=args.collapse_max_bp)
    else:
        collapse_distance, collapse_info = int(args.collapse_distance), {"mode": "user"}
    model_cell, model_chrom, model_position = collapse_events(
        raw_cell, raw_chrom, raw_position, chroms, sizes, collapse_distance)

    log("STEP 3  foreground (tract vs dispersed) gap mixture")
    signal_mask, foreground = infer_foreground(
        model_cell, model_chrom, model_position, n_barcodes,
        threshold=args.fg_posterior, min_events=args.fg_min_events, seed=args.seed)
    n_signal, frac_signal = foreground["n_signal"], foreground["frac_signal"]

    log("STEP 4  usable RT bins, breadth, per-cell anchor")
    pooled_rt = np.bincount(
        rt_grid.gbin(raw_chrom.astype(np.int64), raw_position.astype(np.int64)),
        minlength=rt_grid.total).astype(float)
    available = (~bad_rt) & (pooled_rt > 0) & np.isfinite(anchor_all)
    if available.sum() < 300:
        die("too few available RT bins")
    low, high = np.quantile(pooled_rt[available], [args.usable_bin_lower_quantile,
                                                   args.usable_bin_upper_quantile])
    rt_bins = np.flatnonzero(available & (pooled_rt >= low) & (pooled_rt <= high))
    if rt_bins.size < 300:
        die("too few usable RT bins")

    raw_rt = build_matrix(raw_cell, raw_chrom, raw_position, None, row_of_metric,
                          rt_grid, n_metric, binarize=True)[:, rt_bins].tocsr()
    breadth_raw = np.diff(raw_rt.indptr).astype(float) / rt_bins.size
    del raw_rt
    signal_rt_all = build_matrix(model_cell, model_chrom, model_position, signal_mask,
                                 row_of_metric, rt_grid, n_metric)[:, rt_bins].tocsr()
    profile_events = np.asarray(signal_rt_all.sum(axis=1)).ravel()
    breadth_signal = np.diff(signal_rt_all.indptr).astype(float) / rt_bins.size
    bin_chrom, bin_start = rt_grid.chrom_of[rt_bins], rt_grid.start_of[rt_bins]
    anchor_rt = anchor_all[rt_bins]
    cell_anchor_all = np.full(n_metric, np.nan)
    weight = np.asarray(signal_rt_all.sum(axis=1)).ravel()
    good = weight > 0
    cell_anchor_all[good] = np.asarray(
        signal_rt_all.dot(anchor_rt)).ravel()[good] / weight[good]
    log(f"  usable RT bins={rt_bins.size}; median breadth_raw="
        f"{np.median(breadth_raw):.4f}; median foreground events="
        f"{np.median(profile_events):.0f}")

    log("STEP 5  gates A (depth), B (breadth + events), D (dispersed signal)")
    log_depth = np.log10(np.maximum(raw_depth[metric_bc], 1.0))
    depth_low, depth_high = np.quantile(log_depth, [args.depth_lower_quantile,
                                                    args.depth_upper_quantile])
    depth_ok = (log_depth >= depth_low) & (log_depth <= depth_high)
    if args.min_fragments > 0:
        depth_ok &= raw_depth[metric_bc] >= args.min_fragments
    oversaturated = breadth_raw >= args.breadth_max
    pool = breadth_raw[(breadth_raw >= args.breadth_hard_floor) & (~oversaturated)
                       & depth_ok]
    if pool.size < 50:
        pool = breadth_raw[breadth_raw > 0]
    breadth_median, breadth_mad = robust_mad(pool)
    breadth_cut = max(args.breadth_min_floor,
                      breadth_median - args.breadth_mad_k * breadth_mad)
    empty = breadth_raw < args.breadth_hard_floor
    status = np.full(n_metric, "PASS", dtype=object)
    status[~depth_ok] = "DEPTH_FAIL"
    status[(~empty) & (breadth_raw < breadth_cut)] = "BREADTH_SPARSE_FAIL"
    status[empty] = "QC_FAIL_EMPTY"
    status[oversaturated] = "BREADTH_OVERSAT_FAIL"
    status[(status == "PASS")
           & (profile_events < float(args.min_profile_events))] = "LOW_EVENTS_FAIL"
    status[(status == "PASS")
           & (frac_signal[metric_bc] < float(args.min_frac_signal))] = "DISPERSED_FAIL"
    keep_rows = np.flatnonzero(status == "PASS")
    log(f"  gate A [{depth_low:.3f}, {depth_high:.3f}]; gate B cut={breadth_cut:.4f}")
    log(f"  after gates A/B/D: {keep_rows.size}/{n_metric} cells")
    if keep_rows.size < args.minimum_cells:
        die("too few cells after gates A/B/D")

    log("STEP 6  co-occupancy graph, blocks, landmarks")
    graph_columns = ~bad_graph
    if args.graph_autosomes_only:
        autosome = np.asarray([is_autosome(c) for c in chroms], bool)
        if autosome.sum() >= 4:
            graph_columns &= autosome[graph_grid.chrom_of]
    graph_columns = np.flatnonzero(graph_columns)
    occupancy = build_matrix(model_cell, model_chrom, model_position, signal_mask,
                             row_of_metric, graph_grid, n_metric,
                             binarize=True)[:, graph_columns].tocsr()[keep_rows].tocsr()
    column_chrom = graph_grid.chrom_of[graph_columns]
    radius = int(round(args.graph_smooth_bp / float(args.graph_bin)))
    occupancy = dilate_columns(occupancy, column_chrom, radius)
    occupancy.eliminate_zeros()

    n_keep = keep_rows.size
    support = np.diff(occupancy.indptr).astype(float)
    keep_key = barcode_key[metric_bc[keep_rows]]
    land_rows = choose_landmarks(support, keep_key, args.max_order_cells)
    n_land = int(land_rows.size)
    occ_land = occupancy[land_rows].tocsr()

    budget = int(args.gram_budget_gb * 1e9 / max(n_land ** 2 * 4.0, 1.0)) - 2
    block_of_column, n_blocks = make_blocks(column_chrom, args.n_blocks, budget,
                                            args.block_min_bins)
    information = block_information(occupancy, block_of_column, n_blocks)
    n_valid_blocks = (information > 0).sum(axis=1).astype(float)
    info_weight = information.sum(axis=1).astype(float)
    min_valid = int(np.clip(args.min_valid_blocks, 1, max(1, n_blocks // 2)))
    knn = int(args.knn) if args.knn > 0 else int(np.clip(n_land // 25, 10, 50))
    knn = int(min(knn, max(3, n_land - 1)))
    log(f"  graph: {graph_columns.size} bins, dilation +/-{radius} bins; {n_blocks} "
        f"blocks; median occupied bins={np.median(support):.0f}")
    log(f"  mode: {'exact (all cells)' if n_land == n_keep else 'landmark+extension'}; "
        f"L={n_land}, extended={n_keep - n_land}; fixed k={knn}; "
        f"min valid blocks={min_valid}")

    log("STEP 7  gate C null (circular shifts; same estimator, same k)")
    null_mean_fn, null_sd_fn = structure_null(occ_land, column_chrom, block_of_column,
                                              n_blocks, knn, args.null_shifts,
                                              args.seed)
    bank = BlockPhi(occ_land, block_of_column, n_blocks)
    log(f"  cached {bank.nb} landmark phi blocks "
        f"({bank.nb * n_land ** 2 * 4 / 1e9:.2f} GB)")
    del occ_land

    contrast = np.full(n_keep, np.nan)
    contrast[land_rows] = neighbour_contrast(bank.combine(np.arange(n_land)), knn)
    excess = contrast - null_mean_fn(support)
    z_score = excess / np.maximum(null_sd_fn(support), 1e-9)

    def structure_fail(rows):
        fail = np.zeros(np.size(rows), bool)
        if args.min_structure_excess > 0:
            fail |= ~(excess[rows] >= float(args.min_structure_excess))
        if args.min_structure_z > 0:
            fail |= ~(z_score[rows] >= float(args.min_structure_z))
        return fail

    fail_c = structure_fail(land_rows)
    fail_info = (n_valid_blocks[land_rows] < min_valid) & ~fail_c
    drop = fail_c | fail_info
    if drop.any() and int((~drop).sum()) < min(args.minimum_cells, n_land):
        warn("gates C/INFO would leave too few landmarks; flagged landmarks are KEPT")
        drop[:] = False
    status[keep_rows[land_rows[fail_c & drop]]] = "NO_STRUCTURE_FAIL"
    status[keep_rows[land_rows[fail_info & drop]]] = "LOW_INFO_FAIL"
    bank_keep = np.flatnonzero(~drop)
    log(f"  landmark gates: {int(fail_c.sum())} no structure, "
        f"{int(fail_info.sum())} low information; L kept={bank_keep.size}")
    if bank_keep.size < 20:
        die("fewer than 20 usable landmarks; the order cannot be established")

    log("STEP 8  landmark seriation (single deterministic pass) + bootstrap")
    ordering = order_cells(bank, bank_keep, args, rng, knn)
    if ordering["n_boot_failed"] > 0.25 * max(args.n_bootstrap, 1):
        warn(f"{ordering['n_boot_failed']}/{args.n_bootstrap} bootstrap replicates "
             f"failed to reproduce a 1D order; the whole ordering is fragile")
    del bank

    coordinate = np.full(n_keep, np.nan)
    secondary = np.full(n_keep, np.nan)
    split_a = np.full(n_keep, np.nan)
    split_b = np.full(n_keep, np.nan)
    is_landmark = np.zeros(n_keep, bool)
    is_landmark[land_rows] = True
    seriated = land_rows[bank_keep]
    coordinate[seriated] = ordering["coordinate"]
    split_a[seriated] = ordering["split_a"]
    split_b[seriated] = ordering["split_b"]
    replicate = np.full((int(ordering["boot_stack"].shape[0]), n_keep), np.nan)
    if replicate.shape[0]:
        replicate[:, seriated] = ordering["boot_stack"]
    n_boot = replicate.shape[0]

    anchor = seriated[np.isfinite(coordinate[seriated])]
    anchor = anchor[np.argsort(coordinate[anchor], kind="stable")]
    pending = np.setdiff1d(np.flatnonzero(status[keep_rows] == "PASS"), anchor)
    extension_rho = extension_contrast_rho = np.nan
    if pending.size:
        log(f"STEP 8b  attaching {pending.size} cells (same phi estimator, same "
            f"k={knn}; smoothed local centroid + monotone calibration)")
        t_anchor = coordinate[anchor]
        boot_anchor = (replicate[:, anchor] if n_boot
                       else np.zeros((0, anchor.size), np.float32))
        targets = np.union1d(pending, anchor)   # anchors need a LOO estimate as well
        placement = extend_to_anchors(occupancy, anchor, t_anchor, boot_anchor,
                                      block_of_column, n_blocks, targets, knn,
                                      chunk=args.extend_chunk)
        hat = placement["hat"]
        extension_rho = spearman(hat[anchor], t_anchor)
        extension_contrast_rho = spearman(placement["contrast"][anchor],
                                          contrast[anchor])
        log(f"  extension fidelity: spearman(LOO estimate, exact order) = "
            f"{fmt(extension_rho, 4)}; contrast agreement = "
            f"{fmt(extension_contrast_rho, 4)}")
        if np.isfinite(extension_rho) and extension_rho < 0.95:
            warn(f"the extension reproduces the exact landmark order only at rho="
                 f"{extension_rho:.3f}; raise --max-order-cells or coarsen --graph-bin")

        mapper = qq_map(hat[anchor], t_anchor)
        coordinate[pending] = mapper(hat[pending])
        secondary[targets] = placement["secondary"][targets]
        contrast[pending] = placement["contrast"][pending]
        excess = contrast - null_mean_fn(support)
        z_score = excess / np.maximum(null_sd_fn(support), 1e-9)
        for b in range(n_boot):
            replicate[b, pending] = qq_map(placement["boot_hat"][b, anchor],
                                           boot_anchor[b])(
                placement["boot_hat"][b, pending])
        del placement

        fail_c = structure_fail(pending)
        fail_info = (n_valid_blocks[pending] < min_valid) & ~fail_c
        fail_place = ~np.isfinite(coordinate[pending]) & ~(fail_c | fail_info)
        status[keep_rows[pending[fail_c]]] = "NO_STRUCTURE_FAIL"
        status[keep_rows[pending[fail_info]]] = "LOW_INFO_FAIL"
        status[keep_rows[pending[fail_place]]] = "UNPLACEABLE_FAIL"
        log(f"  extension gates: {int(fail_c.sum())} no structure, "
            f"{int(fail_info.sum())} low information, "
            f"{int(fail_place.sum())} unplaceable")

    log("STEP 9  total order, uncertainty, orientation, inferred RT")
    final = np.flatnonzero(status[keep_rows] == "PASS")
    bad = final[~np.isfinite(coordinate[final])]
    if bad.size:
        status[keep_rows[bad]] = "UNPLACEABLE_FAIL"
        final = np.flatnonzero(status[keep_rows] == "PASS")
    if final.size < args.minimum_cells:
        die("too few cells survive the gates")

    primary = coordinate[final]
    second = np.where(np.isfinite(secondary[final]), secondary[final], primary)
    n_final = final.size
    progression = np.empty(n_final)
    progression[np.lexsort((barcode_key[metric_bc[keep_rows[final]]], second,
                            primary))] = (np.arange(n_final) + 0.5) / n_final

    to_progression = qq_map(primary, progression, min_pairs=5)
    replicate_final = (np.vstack([to_progression(replicate[b, final])
                                  for b in range(n_boot)]) if n_boot
                       else np.zeros((0, n_final)))
    boot_used, rank_sd, rank_lo, rank_hi = nan_reduce(replicate_final)

    cell_anchor = cell_anchor_all[keep_rows[final]]
    flipped, anchor_rho_raw = orientation_flip(progression, cell_anchor,
                                               args.anchor_high, args.flip_order)
    if flipped:
        progression = 1.0 - progression
        coordinate = 1.0 - coordinate
        rank_lo, rank_hi = 1.0 - rank_hi, 1.0 - rank_lo
        split_a, split_b = 1.0 - split_a, 1.0 - split_b
    anchor_rho = -anchor_rho_raw if flipped else anchor_rho_raw

    low_confidence = np.isfinite(rank_sd) & (rank_sd > float(args.max_rank_sd))
    if args.drop_low_confidence and low_confidence.any():
        survive = ~low_confidence
        status[keep_rows[final[low_confidence]]] = "UNPLACEABLE_FAIL"
        log(f"  --drop-low-confidence removed {int(low_confidence.sum())} cells AFTER "
            f"ordering (relative order of the rest is unchanged)")
        final = final[survive]
        progression, rank_sd, rank_lo, rank_hi, boot_used, low_confidence = (
            x[survive] for x in (progression, rank_sd, rank_lo, rank_hi, boot_used,
                                 low_confidence))
        n_final = final.size
        cell_anchor = cell_anchor_all[keep_rows[final]]
        progression = (rankdata_avg(progression) - 0.5) / n_final

    rows_bc = keep_rows[final]
    kept_bc = metric_bc[rows_bc]
    signal_rt = signal_rt_all[rows_bc].tocsr()
    inferred_rt, rt_support = infer_rt(signal_rt, progression, bin_chrom, bin_start,
                                       args.rt_smooth_bp)
    rt_anchor_rho = spearman(inferred_rt, anchor_rt)
    log(f"  orientation: spearman(progression, cell anchor)={anchor_rho:+.3f} "
        f"(axis {'flipped' if flipped else 'retained'}); "
        f"spearman(inferred RT, anchor)={fmt(rt_anchor_rho, 3)}")
    if np.isfinite(rt_anchor_rho) and abs(rt_anchor_rho) < 0.3:
        warn(f"inferred RT correlates only {rt_anchor_rho:+.3f} with the anchor")

    ordered_rows = np.argsort(progression, kind="stable")
    global_order = np.empty(n_final, np.int64)
    global_order[ordered_rows] = np.arange(1, n_final + 1)
    pseudotime_bin = np.zeros(n_final, int)
    for index, block in enumerate(np.array_split(ordered_rows, max(1, args.n_bins)),
                                  start=1):
        pseudotime_bin[block] = index

    log("STEP 10  writing outputs")
    row_of_keep = np.full(n_barcodes, -1, np.int64)
    row_of_keep[kept_bc] = np.arange(n_final)

    def to_barcode(values, index, fill=np.nan, dtype=float):
        output = np.full(n_barcodes, fill, dtype=dtype)
        output[index] = values
        return output

    def keep_to_barcode(values):
        output = np.full(n_metric, np.nan)
        output[keep_rows] = values
        return to_barcode(output, metric_bc)

    final_status = np.full(n_barcodes, "NO_FRAGMENTS", dtype=object)
    final_status[metric_bc] = status
    pass_barcode = np.zeros(n_barcodes, bool)
    pass_barcode[kept_bc] = True
    reason_map = {
        "DEPTH_FAIL": "depth_outside_window_or_below_min_fragments",
        "BREADTH_SPARSE_FAIL": "sparse_genome_coverage",
        "QC_FAIL_EMPTY": "breadth_below_hard_floor_empty_or_broken_nucleus",
        "BREADTH_OVERSAT_FAIL": "near_uniform_coverage_ambient_doublet_or_G2",
        "LOW_EVENTS_FAIL": "too_few_foreground_events",
        "DISPERSED_FAIL": "tract_fraction_below_min_frac_signal",
        "NO_STRUCTURE_FAIL": "similarity_contrast_not_above_circular_shift_null",
        "LOW_INFO_FAIL": "too_few_non_degenerate_genomic_blocks",
        "UNPLACEABLE_FAIL": "no_finite_similarity_path_or_dropped_by_user_flag",
        "NO_FRAGMENTS": "no_usable_fragments"}

    qc = pd.DataFrame({
        "barcode": barcodes,
        "final_qc_status": final_status,
        "filter_reason": [reason_map.get(s, "") for s in final_status],
        "pass": pass_barcode.astype(int),
        "n_fragments_raw": raw_depth.astype(np.int64),
        "log10_fragments": to_barcode(log_depth, metric_bc),
        "frac_blacklist": blacklist_fraction,
        "n_signal_events": n_signal.astype(np.int64),
        "frac_signal": frac_signal,
        "breadth_raw": to_barcode(breadth_raw, metric_bc),
        "breadth_signal": to_barcode(breadth_signal, metric_bc),
        "profile_events": to_barcode(profile_events, metric_bc),
        "cell_anchor": to_barcode(cell_anchor_all, metric_bc),
        "similarity_contrast": keep_to_barcode(contrast),
        "structure_excess": keep_to_barcode(excess),
        "structure_z": keep_to_barcode(z_score),
        "n_valid_blocks": keep_to_barcode(n_valid_blocks),
        "info_weight": keep_to_barcode(info_weight),
        "oversaturated": to_barcode(oversaturated.astype(float), metric_bc,
                                    0.0).astype(int)})
    for name, values in {
        "progression": progression,
        "point_coordinate": coordinate[final],
        "rank_sd": rank_sd, "rank_lo10": rank_lo, "rank_hi90": rank_hi,
        "n_bootstrap_used": boot_used,
        "low_confidence": low_confidence.astype(float),
        "order_odd_blocks": split_a[final], "order_even_blocks": split_b[final],
        "is_landmark": is_landmark[final].astype(float)}.items():
        qc[name] = to_barcode(np.asarray(values, float), kept_bc)
    qc["global_order"] = to_barcode(global_order, kept_bc, -1, np.int64)
    qc["pseudotime_bin"] = to_barcode(pseudotime_bin, kept_bc, -1, np.int64)
    qc.to_csv(prefix + ".qc_metrics.tsv", sep="\t", index=False, na_rep="NA",
              float_format="%.6g")
    qc.loc[~pass_barcode, ["barcode", "final_qc_status", "filter_reason",
                           "n_fragments_raw", "breadth_raw", "profile_events",
                           "frac_signal", "similarity_contrast", "structure_excess",
                           "structure_z", "n_valid_blocks"]].to_csv(
        prefix + ".dropped_cells.tsv", sep="\t", index=False, na_rep="NA",
        float_format="%.6g")

    order_columns = ["barcode", "global_order", "progression", "rank_sd", "rank_lo10",
                     "rank_hi90", "low_confidence", "pseudotime_bin", "is_landmark",
                     "point_coordinate", "order_odd_blocks", "order_even_blocks",
                     "similarity_contrast", "structure_excess", "structure_z",
                     "n_bootstrap_used", "n_valid_blocks", "info_weight",
                     "breadth_raw", "breadth_signal", "profile_events", "frac_signal",
                     "cell_anchor", "n_fragments_raw", "n_signal_events"]
    with open(prefix + ".cell_order.tsv", "w") as handle:
        handle.write(
            f"# spearman(progression, cell anchor) = {anchor_rho:+.4f} "
            f"(axis {'flipped' if flipped else 'retained'})\n"
            f"# Robinson monotonicity = {fmt(ordering['robinson'], 4)}; "
            f"split-half spearman = {fmt(ordering['split_rho'], 4)}; "
            f"extension LOO spearman = {fmt(extension_rho, 4)}\n")
        qc.loc[pass_barcode, order_columns].sort_values("global_order").to_csv(
            handle, sep="\t", index=False, na_rep="NA", float_format="%.6g")

    for index, block in enumerate(np.array_split(ordered_rows, max(1, args.n_bins)),
                                  start=1):
        with open(os.path.join(args.outdir, "bins",
                               f"{args.sample}.bin{index:02d}.cells.txt"), "w") as h:
            for row in block:
                h.write(barcodes[kept_bc[row]] + "\n")

    with open(prefix + ".inferred_RT.bedgraph", "w") as handle:
        handle.write(f'track type=bedGraph name="{args.sample}_RT"\n')
        for index, global_bin in enumerate(rt_bins):
            value = inferred_rt[index]
            if not np.isfinite(value) or rt_support[index] <= 0:
                continue
            chrom_name = chroms[int(rt_grid.chrom_of[global_bin])]
            start = int(rt_grid.start_of[global_bin])
            handle.write(f"{chrom_name}\t{start}\t"
                         f"{min(start + rt_grid.bin_size, int(sizes[chrom_name]))}\t"
                         f"{value:.6g}\n")

    region_counts = None
    if region is not None:
        chrom_id, region_start, region_end = region
        region_counts = region_matrix(raw_cell, raw_chrom, raw_position, row_of_keep,
                                      chrom_id, region_start, region_end,
                                      args.plot_bin, n_final)
        region_events = np.asarray(region_counts.sum(axis=1)).ravel()
        pd.DataFrame({
            "barcode": [barcodes[kept_bc[row]] for row in ordered_rows],
            "global_order": global_order[ordered_rows],
            "progression": progression[ordered_rows],
            "rank_sd": rank_sd[ordered_rows],
            "is_landmark": is_landmark[final][ordered_rows].astype(int),
            "breadth_raw": breadth_raw[rows_bc][ordered_rows],
            "profile_events": profile_events[rows_bc][ordered_rows],
            "structure_excess": excess[final][ordered_rows],
            "region_events": region_events[ordered_rows],
        }).to_csv(prefix + ".region_support.tsv", sep="\t", index=False, na_rep="NA",
                  float_format="%.6g")

    status_names = ("PASS", "DEPTH_FAIL", "BREADTH_SPARSE_FAIL", "QC_FAIL_EMPTY",
                    "BREADTH_OVERSAT_FAIL", "LOW_EVENTS_FAIL", "DISPERSED_FAIL",
                    "NO_STRUCTURE_FAIL", "LOW_INFO_FAIL", "UNPLACEABLE_FAIL")
    counts = {s: int(np.sum(status == s)) for s in status_names}
    thresholds = {"depth_low": float(depth_low), "depth_high": float(depth_high),
                  "breadth_hard_floor": float(args.breadth_hard_floor),
                  "breadth_cut": float(breadth_cut),
                  "breadth_max": float(args.breadth_max),
                  "min_profile_events": int(args.min_profile_events),
                  "min_frac_signal": float(args.min_frac_signal),
                  "min_structure_excess": float(args.min_structure_excess),
                  "min_structure_z": float(args.min_structure_z),
                  "min_valid_blocks": int(min_valid),
                  "max_rank_sd": float(args.max_rank_sd)}
    summary = {
        "sample": args.sample, "params": vars(args),
        "applied_thresholds": thresholds, "n_barcodes": int(n_barcodes),
        "n_barcodes_with_fragments": int(n_metric), "n_pass": int(n_final),
        "status_counts": counts, "collapse_distance": int(collapse_distance),
        "collapse_info": collapse_info,
        "foreground": {k: foreground[k] for k in
                       ("short_median_bp", "long_median_bp", "pooled_short_weight",
                        "n_fallback_cells")},
        "median_frac_signal": float(np.nanmedian(frac_signal)),
        "usable_rt_bins": int(rt_bins.size),
        "graph": {"bin_bp": int(args.graph_bin), "dilation_bins": int(radius),
                  "n_columns": int(graph_columns.size), "n_blocks": int(n_blocks),
                  "knn_fixed": int(knn),
                  "edge_weight": "two-sided rank affinity (A+A^T)/2",
                  "block_weight": "min(informative bins), continuous",
                  "median_occupied_bins": float(np.median(support))},
        "landmarks": {"mode": ("exact" if n_land == n_keep
                               else "landmark+calibrated_extension"),
                      "n_landmarks": int(n_land),
                      "n_landmarks_kept": int(bank_keep.size),
                      "n_extended": int(max(0, n_keep - n_land)),
                      "extension_loo_spearman": float(extension_rho),
                      "extension_contrast_spearman": float(extension_contrast_rho)},
        "ordering": {
            "coordinate_source": "single deterministic point estimate; bootstrap only "
                                 "for uncertainty; no refit loop",
            "robinson_monotonicity": float(ordering["robinson"]),
            "split_half_spearman": float(ordering["split_rho"]),
            "median_rank_sd": float(np.nanmedian(rank_sd)),
            "n_low_confidence": int(low_confidence.sum()),
            "n_bootstrap_requested": int(args.n_bootstrap),
            "n_bootstrap_failed": int(ordering["n_boot_failed"]),
            "bootstrap_abs_rho_median": float(ordering["boot_abs_rho_median"])},
        "structure_test": {
            "median_excess": float(np.nanmedian(excess[final])),
            "p05_excess": float(np.nanpercentile(excess[final], 5)),
            "median_z": float(np.nanmedian(z_score[final]))},
        "orientation": {"anchor_high": args.anchor_high,
                        "spearman_progression_vs_cell_anchor": float(anchor_rho),
                        "axis_flipped": bool(flipped),
                        "spearman_inferred_rt_vs_anchor": float(rt_anchor_rho)},
        "confounding_checks": {
            "spearman_depth_vs_progression": float(
                spearman(log_depth[rows_bc], progression)),
            "spearman_breadth_vs_progression": float(
                spearman(breadth_raw[rows_bc], progression)),
            "spearman_events_vs_progression": float(
                spearman(profile_events[rows_bc], progression)),
            "spearman_frac_signal_vs_progression": float(
                spearman(frac_signal[kept_bc], progression))},
        "warnings": WARNINGS}

    if np.isfinite(ordering["split_rho"]) and ordering["split_rho"] < 0.7:
        warn(f"independent halves of the genome agree only at rho="
             f"{ordering['split_rho']:.3f}; coarsen --graph-bin, raise "
             f"--graph-smooth-bp, or raise --min-profile-events")
    if np.isfinite(ordering["robinson"]) and ordering["robinson"] < 0.3:
        warn(f"Robinson monotonicity is only {ordering['robinson']:.3f}; the "
             f"similarity matrix is not band-structured, so a 1D order may not "
             f"describe this population")

    # ------------------------------------------------------------------ STEP 11
    region_report = None
    if not args.no_plots:
        log("STEP 11  plots")
        passed = np.zeros(n_metric, bool); passed[rows_bc] = True
        metric = {"log_depth": log_depth, "breadth": breadth_raw,
                  "frac_signal": frac_signal[metric_bc],
                  "support": np.full(n_metric, np.nan),
                  "excess": np.full(n_metric, np.nan),
                  "rank_sd": np.full(n_metric, np.nan)}
        metric["support"][keep_rows] = support
        metric["excess"][keep_rows] = excess
        metric["rank_sd"][rows_bc] = rank_sd
        plot_qc_gates(os.path.join(args.outdir, "qc", args.sample + ".qc_gates.png"),
                      metric, thresholds, passed, inferred_rt, anchor_rt)

        position = np.full(n_keep, -1, np.int64)
        position[final] = np.arange(n_final)
        located = position[seriated]
        sim_rows = np.flatnonzero(located >= 0)
        plot_order_diagnostics(
            os.path.join(args.outdir, "qc", args.sample + ".order_diagnostics.png"),
            np.ascontiguousarray(ordering["similarity"][np.ix_(sim_rows, sim_rows)]),
            {"land_progression": progression[located[sim_rows]],
             "progression": progression, "rank_sd": rank_sd, "rank_lo": rank_lo,
             "rank_hi": rank_hi, "cell_anchor": cell_anchor,
             "events": profile_events[rows_bc], "breadth": breadth_raw[rows_bc],
             "contrast": contrast[final], "split_a": split_a[seriated][sim_rows],
             "split_b": split_b[seriated][sim_rows],
             "split_rho": ordering["split_rho"], "robinson": ordering["robinson"],
             "anchor_rho": anchor_rho, "max_rank_sd": args.max_rank_sd})

        if region_counts is not None:
            chrom_id, region_start, region_end = region
            region_report = plot_region_heatmap(
                os.path.join(args.outdir, "qc", args.sample + ".region_heatmap.png"),
                region_counts, ordered_rows, chroms[chrom_id], region_start,
                region_end, args.plot_bin, vmax=args.plot_vmax, dpi=args.plot_dpi)

    summary["region_heatmap"] = region_report

    def sanitise(obj):
        if isinstance(obj, dict):
            return {k: sanitise(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitise(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            return None if not np.isfinite(float(obj)) else float(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        return obj if isinstance(obj, (str, int, type(None))) else str(obj)

    with open(prefix + ".summary.json", "w") as handle:
        json.dump(sanitise(summary), handle, indent=2)

    log("=" * 74)
    log("QC summary")
    log(f"  input cells: {n_metric}")
    for key in status_names[1:]:
        log(f"  {key}: {counts[key]} ({100.0 * counts[key] / max(n_metric, 1):.1f}%)")
    log(f"  retained: {n_final} ({100.0 * n_final / max(n_metric, 1):.1f}%)")
    log(f"  ordering mode: {summary['landmarks']['mode']} "
        f"(landmarks={n_land}, extended={max(0, n_keep - n_land)}, fixed k={knn})")
    log("  ordering quality:")
    log(f"    Robinson monotonicity   = {fmt(ordering['robinson'], 4)}  (best +1)")
    log(f"    odd/even block agreement= {fmt(ordering['split_rho'], 4)}  "
        f"(>0.85 good, <0.7 needs tuning)")
    log(f"    extension fidelity (LOO)= {fmt(extension_rho, 4)}  "
        f"(>0.95 good; otherwise raise --max-order-cells)")
    log(f"    median rank_sd          = {fmt(np.nanmedian(rank_sd), 4)}  "
        f"(uniform = 0.289)")
    log(f"    failed bootstrap reps   = {ordering['n_boot_failed']}/{args.n_bootstrap}")
    log(f"    low-confidence cells    = {int(low_confidence.sum())} (flag only)")
    log(f"    depth vs order (~0)     = "
        f"{fmt(summary['confounding_checks']['spearman_depth_vs_progression'], 4)}")
    log(f"    orientation (cell GC)   = {anchor_rho:+.4f}  "
        f"axis {'flipped' if flipped else 'retained'}")
    log(f"    inferred RT vs anchor   = {fmt(rt_anchor_rho, 4)}")
    log("=" * 74)
    log(f"DONE: {n_final} cells ordered")


if __name__ == "__main__":
    main()