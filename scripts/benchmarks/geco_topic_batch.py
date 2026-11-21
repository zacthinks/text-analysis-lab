#!/usr/bin/env python3
"""Persisted TeAL topic-provider benchmark. No network, user projects, or UI required.

Run with Python 3.11 and the same installed dependencies for each source revision.
The driver intentionally lives outside the baseline source tree, so the old
revision need not implement the new optional batch API.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from unittest.mock import patch


SEED = 20261121
ROWS = 10000
FEATURES = 24
WARM_SAMPLES = 24
BASE = "13b93d9c7c8598254e5d694bf5a4ef90e6e4cd0c"


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def load_teal(source):
    sys.path.insert(0, str(Path(source).resolve()))
    import text_analysis_lab as teal
    from text_analysis_lab.integrations.geco import TeALGeCoProvider
    assert Path(teal.__file__).resolve().is_relative_to(Path(source).resolve()), teal.__file__
    return teal, TeALGeCoProvider


def prepare(args):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    teal, _ = load_teal(args.source)
    from text_analysis_lab.core.writer import create_artifact_writer
    from text_analysis_lab.translators.argmax import ArgMax

    rng = np.random.default_rng(SEED)
    dense = rng.random((ROWS, FEATURES), dtype=np.float32)
    dense /= dense.sum(axis=1, keepdims=True)
    indices = rng.integers(0, FEATURES, size=(ROWS, 3))
    scores = rng.random((ROWS, 3), dtype=np.float32)
    sparse_values = sparse.csr_matrix(
        (scores.ravel(), (np.repeat(np.arange(ROWS), 3), indices.ravel())),
        shape=(ROWS, FEATURES),
    )
    description = rng.random((12, 30), dtype=np.float32)

    def seed(project, artifact_id, values, columns):
        kind = "sparse_matrix" if sparse.issparse(values) else "dense_matrix"
        writer = create_artifact_writer(
            artifact_type=kind,
            artifact_dir=project.storage.artifact_dir(artifact_id),
            artifact_id=artifact_id,
            label=artifact_id,
            lineage_mode="new_key",
            basis_artifact_ids=(),
        )
        writer.write({
            "keys": pd.DataFrame({"row_id": np.arange(values.shape[0], dtype=np.int64)}),
            "data": {"values": values, "columns": list(columns)},
        })
        writer.finalize()
        project.catalog.register_artifact(
            artifact_id=artifact_id, artifact_type=kind, label=artifact_id,
            lineage_mode="new_key", status="complete", basis_artifact_ids=(),
        )
        return project.get_artifact(artifact_id)

    with teal.Project.create(args.project, name="topic_batch_benchmark") as project:
        cols = [f"topic_{i:02d}" for i in range(FEATURES)]
        dense_art = seed(project, "bench_dense_membership", dense, cols)
        sparse_art = seed(project, "bench_sparse_membership", sparse_values, cols)
        description_art = seed(
            project, "bench_description", description,
            [f"term_{i:02d}" for i in range(30)],
        )
        dense_argmax = project.translate(ArgMax(), dense_art)["output"]
        sparse_argmax = project.translate(ArgMax(), sparse_art)["output"]
        fixture = {
            "seed": SEED, "rows": ROWS, "features": FEATURES,
            "description_shape": list(description.shape),
            "artifacts": {
                "dense": {"membership": dense_art.artifact_id, "argmax": dense_argmax.artifact_id},
                "sparse": {"membership": sparse_art.artifact_id, "argmax": sparse_argmax.artifact_id},
                "description": description_art.artifact_id,
            },
        }
    write_json(args.fixture, fixture)
    print(json.dumps({"event": "fixture_created", "fixture": fixture}, sort_keys=True))


def keys_for(size):
    import numpy as np
    if size == 1:
        return [{"row_id": 7, "extra_geco_key": "ignored"}]
    rng = np.random.default_rng(SEED + size)
    sampled = rng.choice(ROWS, size=size, replace=False).tolist()
    # Missing keys and repeated keys are intentional and position-sensitive.
    sampled[4] = sampled[3]
    sampled[9] = ROWS + 9
    if size >= 5000:
        for i in range(47, size, 101):
            sampled[i] = ROWS + i
        for i in range(69, size, 137):
            sampled[i] = sampled[i - 1]
    return [{"row_id": int(v), "extra_geco_key": "ignored"} for v in sampled]


def packet_json(packet):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    values = packet.values
    if sparse.issparse(values):
        payload = {"format": "sparse", "values": values.toarray().tolist()}
    elif isinstance(values, pd.DataFrame):
        payload = {
            "format": "table", "columns": list(values.columns),
            "values": [[native_scalar(v) for v in row] for row in values.itertuples(index=False, name=None)],
        }
    else:
        payload = {"format": "dense", "values": np.asarray(values).tolist()}
    return {
        "kind": packet.kind, "columns": list(packet.columns),
        "positions": packet.positions, "row_ids": packet.row_ids,
        "payload": payload,
    }


def native_scalar(v):
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    return v


def digest_packets(packets):
    data = [packet_json(p) for p in packets]
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def assert_packets_equal(expected, actual):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    assert len(expected) == len(actual) == 3
    for first, second in zip(expected, actual, strict=True):
        assert first.kind == second.kind
        assert list(first.columns) == list(second.columns)
        assert first.positions == second.positions
        assert first.row_ids == second.row_ids
        if isinstance(first.values, pd.DataFrame):
            pd.testing.assert_frame_equal(first.values, second.values, check_exact=True)
        elif sparse.issparse(first.values):
            assert sparse.issparse(second.values)
            assert (first.values != second.values).nnz == 0
        else:
            np.testing.assert_array_equal(first.values, second.values)


def output_bytes(packets):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    total = 0
    for p in packets:
        v = p.values
        if sparse.issparse(v):
            total += v.data.nbytes + v.indices.nbytes + v.indptr.nbytes
        elif isinstance(v, pd.DataFrame):
            total += int(v.memory_usage(index=True, deep=True).sum())
        else:
            total += np.asarray(v).nbytes
    return int(total)


def detached_test(provider, requests, results, variant):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    assert all(p.positions is not None or p.row_ids is not None for p in results)
    # Mutate only a copy of the detached output; the fresh provider read must
    # be unaffected. Retained objects remain usable after each session closes.
    first = results[0]
    if sparse.issparse(first.values):
        before = first.values.copy()
        if first.values.nnz:
            first.values.data[0] = -1234
        fresh = provider.read_artifact(*requests[0])
        assert (fresh.values != before).nnz == 0
    else:
        before = np.asarray(first.values).copy()
        if first.values.size:
            first.values[0, 0] = -1234
        fresh = provider.read_artifact(*requests[0])
        np.testing.assert_array_equal(fresh.values, before)
    # Verify original output can still be accessed after session closure.
    assert results[1].values.shape[0] == len(results[1].positions)
    assert results[2].values.shape[0] == len(results[2].row_ids)


def probe(provider, requests, operation):
    from text_analysis_lab.core.artifact_base import BaseArtifact
    from text_analysis_lab.core.artifact_subclasses import DenseMatrixArtifact, SparseMatrixArtifact
    from text_analysis_lab.core.query import QueryEngine

    counts = Counter()
    real_session = provider._project_session
    real_fetch = QueryEngine._fetch_by_keys
    real_query = BaseArtifact.query
    real_dense = DenseMatrixArtifact.get_matrix
    real_sparse = SparseMatrixArtifact.get_matrix

    @contextmanager
    def session_spy():
        with real_session() as project:
            counts["session_open"] += 1
            try:
                yield project
            finally:
                counts["session_close"] += 1

    def fetch_spy(self, artifact, keys, **kwargs):
        counts["key_lookup_calls"] += 1
        counts["key_lookup_input_rows"] += len(keys)
        return real_fetch(self, artifact, keys, **kwargs)

    def query_spy(self, *a, **kw):
        counts["artifact_query_calls"] += 1
        return real_query(self, *a, **kw)

    def dense_spy(self, *a, **kw):
        counts["dense_get_matrix_calls"] += 1
        return real_dense(self, *a, **kw)

    def sparse_spy(self, *a, **kw):
        counts["sparse_get_matrix_calls"] += 1
        return real_sparse(self, *a, **kw)

    with patch.object(provider, "_project_session", session_spy), \
         patch.object(QueryEngine, "_fetch_by_keys", fetch_spy), \
         patch.object(BaseArtifact, "query", query_spy), \
         patch.object(DenseMatrixArtifact, "get_matrix", dense_spy), \
         patch.object(SparseMatrixArtifact, "get_matrix", sparse_spy):
        packets = operation()
    counts["result_bytes"] = output_bytes(packets)
    counts["result_rows"] = sum(int(p.values.shape[0]) for p in packets)
    return dict(counts)


def percentiles(values):
    import numpy as np
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)), "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "mean_ms": float(np.mean(x)),
        "stdev_ms": float(np.std(x, ddof=1)) if len(x) > 1 else 0.,
        "min_ms": float(x.min()), "max_ms": float(x.max()),
        "cv": float(x.std(ddof=1)/x.mean()) if len(x) > 1 and x.mean() else None,
    }


def bench(args):
    teal, Provider = load_teal(args.source)
    import numpy as np
    from scipy import sparse
    fixture = json.loads(Path(args.fixture).read_text())
    provider_project = teal.Project.open(args.project)
    try:
        provider = Provider(provider_project)
        ids = fixture["artifacts"]
        requested_keys = keys_for(args.size)
        requests = [
            (ids[args.kind]["membership"], requested_keys),
            (ids[args.kind]["argmax"], requested_keys),
            (ids["description"], None),
        ]
        operations = {"sequential": lambda: [provider.read_artifact(*r) for r in requests]}
        if args.role == "candidate":
            operations["batch"] = lambda: provider.read_artifacts(requests)
        elif hasattr(provider, "read_artifacts"):
            raise AssertionError("Historical baseline unexpectedly exposes batch API")

        if args.first:
            assert args.variant in operations
            operation = operations[args.variant]
            start = time.perf_counter_ns()
            packets = operation()
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            result = {
                "stage": "first_fresh_process", "round": args.first_round,
                "variant": args.variant, "elapsed_ms": elapsed_ms,
                "digest": digest_packets(packets),
                "result_bytes": output_bytes(packets),
                "result_rows": [int(p.values.shape[0]) for p in packets],
            }
        else:
            assert args.warm >= 20
            reference = operations["sequential"]()
            if "batch" in operations:
                paired = operations["batch"]()
                assert_packets_equal(reference, paired)
                assert digest_packets(reference) == digest_packets(paired)
            digest = digest_packets(reference)
            # Preserve a copy of the unmutated reference before mutation test.
            detached_test(provider, requests, operations["sequential"](), "sequential")
            if "batch" in operations:
                detached_test(provider, requests, operations["batch"](), "batch")
            samples = {name: [] for name in operations}
            rounds = {name: [[], []] for name in operations}
            for i in range(args.warm):
                # Swapped A/B order in two rounds, so one always runs last only
                # within a round; each method gets at least 20 independent samples.
                names = list(operations)
                if len(names) == 2 and i >= args.warm // 2:
                    names.reverse()
                for name in names:
                    start = time.perf_counter_ns()
                    packets = operations[name]()
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    samples[name].append(elapsed)
                    rounds[name][0 if i < args.warm // 2 else 1].append(elapsed)

            probes = {name: probe(provider, requests, fn)
                      for name, fn in operations.items()}
            allocations = {}
            for name, fn in operations.items():
                tracemalloc.start()
                tracemalloc.reset_peak()
                fn()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                allocations[name] = {"tracemalloc_peak_bytes": int(peak),
                                     "tracemalloc_current_bytes": int(current)}
            result = {
                "stage": "warm", "digest": digest,
                "result_bytes": output_bytes(reference),
                "result_rows": [int(p.values.shape[0]) for p in reference],
                "samples_ms": samples,
                "round_samples_ms": rounds,
                "summary": {name: percentiles(values) for name, values in samples.items()},
                "probes": probes, "allocations": allocations,
            }

        row_ids = [r["row_id"] for r in requested_keys]
        meta = {
            "role": args.role, "kind": args.kind, "size": args.size,
            "source_path": str(Path(teal.__file__).resolve()), 
            "revision": os.environ.get("TEAL_REVISION", "unspecified"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "seed": fixture["seed"],
            "fixture_ids": ids,
            "requested_keys_sha256": hashlib.sha256(
                json.dumps(requested_keys, sort_keys=True).encode()).hexdigest(),
            "missing_key_count": sum(k >= ROWS for k in row_ids),
            "duplicate_key_count": len(row_ids) - len(set(row_ids)),
            "rss_peak_kib_process_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rss_note": "Process cumulative high-water RSS, not per-operation peak; use allocations for operation comparison",
        }
        write_json(args.output, {"meta": meta, "result": result})
        print(json.dumps({"event": "bench", "role": args.role,
                          "kind": args.kind, "size": args.size,
                          "stage": result["stage"],
                          "summary": result.get("summary"),
                          "first_ms": result.get("elapsed_ms")}, sort_keys=True))
    finally:
        provider_project.close()


def report(args):
    import csv
    root = Path(args.results)
    warm_files = sorted(root.glob("warm_*.json"))
    first_files = sorted(root.glob("first_*.json"))
    warm = [json.loads(p.read_text()) for p in warm_files]
    first = [json.loads(p.read_text()) for p in first_files]
    assert len(warm) == 12, len(warm)  # baseline + candidate, 2 kinds x 3 sizes
    assert len(first) == 54, len(first)  # 3 rounds x 3 methods x 2 kinds x 3 sizes
    rows, raw = [], []
    for kind in ("dense", "sparse"):
        for size in (1, 100, 5000):
            entries = {x["meta"]["role"]: x for x in warm
                       if x["meta"]["kind"] == kind and x["meta"]["size"] == size}
            assert set(entries) == {"baseline", "candidate"}, entries.keys()
            prior, later = entries["baseline"], entries["candidate"]
            assert prior["result"]["digest"] == later["result"]["digest"], (kind, size, "baseline drift")
            assert prior["meta"]["requested_keys_sha256"] == later["meta"]["requested_keys_sha256"]
            assert prior["result"]["result_rows"] == later["result"]["result_rows"]
            c = later["result"]["summary"]
            old = prior["result"]["summary"]["sequential"]
            a, b = c["sequential"], c["batch"]
            probes = later["result"]["probes"]
            assert probes["sequential"]["session_open"] == 3
            assert probes["batch"]["session_open"] == 1
            assert probes["sequential"]["session_close"] == 3
            assert probes["batch"]["session_close"] == 1
            assert probes["sequential"]["key_lookup_calls"] == probes["batch"]["key_lookup_calls"] == 2
            cases = [x for x in first if x["meta"]["kind"] == kind and x["meta"]["size"] == size]
            assert len(cases) == 9
            assert all(x["result"]["digest"] == later["result"]["digest"] for x in cases)
            cold = {}
            for role, variant in (("baseline", "sequential"), ("candidate", "sequential"), ("candidate", "batch")):
                selected = [x["result"]["elapsed_ms"] for x in cases
                            if x["meta"]["role"] == role and x["result"]["variant"] == variant]
                assert len(selected) == 3
                cold[f"{role}_{variant}"] = selected
            row = {
                "membership": kind, "keys": size, "matched_matrix_rows": later["result"]["result_rows"][0],
                "baseline_old_p50_ms": old["p50_ms"],
                "baseline_old_p95_ms": old["p95_ms"],
                "candidate_old_p50_ms": a["p50_ms"],
                "candidate_old_p95_ms": a["p95_ms"],
                "candidate_batch_p50_ms": b["p50_ms"],
                "candidate_batch_p95_ms": b["p95_ms"],
                "delta_p50_ms": a["p50_ms"] - b["p50_ms"],
                "speedup": a["p50_ms"] / b["p50_ms"] if b["p50_ms"] else None,
                "candidate_old_cv": a["cv"], "candidate_batch_cv": b["cv"],
                "sessions_old": 3, "sessions_batch": 1,
                "key_fetches_old": probes["sequential"]["key_lookup_calls"],
                "key_fetches_batch": probes["batch"]["key_lookup_calls"],
                "matrix_calls_old": (probes["sequential"].get("dense_get_matrix_calls", 0) +
                                     probes["sequential"].get("sparse_get_matrix_calls", 0)),
                "matrix_calls_batch": (probes["batch"].get("dense_get_matrix_calls", 0) +
                                       probes["batch"].get("sparse_get_matrix_calls", 0)),
                "tracemalloc_old_peak_bytes": later["result"]["allocations"]["sequential"]["tracemalloc_peak_bytes"],
                "tracemalloc_batch_peak_bytes": later["result"]["allocations"]["batch"]["tracemalloc_peak_bytes"],
                "result_bytes": later["result"]["result_bytes"],
                "fresh_process_first_ms": cold,
            }
            rows.append(row)
            for x in (prior, later):
                for variant, samples in x["result"]["samples_ms"].items():
                    for i, ms in enumerate(samples):
                        raw.append({
                            "role": x["meta"]["role"], "membership": kind, "keys": size,
                            "variant": variant, "sample": i + 1,
                            "round": 1 if i < len(samples) // 2 else 2,
                            "elapsed_ms": ms,
                        })
    write_json(root / "summary.json", {
        "baseline_sha": BASE,
        "candidate_sha": next(x["meta"]["revision"] for x in warm if x["meta"]["role"] == "candidate"),
        "fixture": next(x["meta"]["fixture_ids"] for x in warm),
        "rows": rows,
        "cold_label": "first provider call in a fresh Python process; OS filesystem cache may be warm",
        "warm_samples_per_variant_case": WARM_SAMPLES,
        "rss_note": "Process cumulative high-water RSS is in raw files; operation-specific peak uses tracemalloc",
    })
    with (root / "samples.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    for row in rows:
        print(json.dumps({"event": "SUMMARY", **row}, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["prepare", "run", "report"], required=True)
    p.add_argument("--source")
    p.add_argument("--project")
    p.add_argument("--fixture")
    p.add_argument("--output")
    p.add_argument("--results")
    p.add_argument("--role", choices=["baseline", "candidate"])
    p.add_argument("--kind", choices=["dense", "sparse"])
    p.add_argument("--size", type=int, choices=[1, 100, 5000])
    p.add_argument("--warm", type=int, default=WARM_SAMPLES)
    p.add_argument("--first", action="store_true")
    p.add_argument("--variant", choices=["sequential", "batch"], default="sequential")
    p.add_argument("--first-round", type=int, default=0)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare(a)
    elif a.mode == "run":
        bench(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
