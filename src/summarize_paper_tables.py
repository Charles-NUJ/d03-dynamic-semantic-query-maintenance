"""Summarize paper tables from the controlled R trajectory summary."""
from collections import defaultdict
from pathlib import Path
import json
root = Path(__file__).resolve().parents[1]
records = json.loads((root / "raw_results/controlled_r_v2/summary.json").read_text(encoding="utf-8-sig"))
if len(records) != 396:
    raise ValueError(f"Expected 396 trajectory summaries; found {len(records)}.")
groups = defaultdict(list)
for row in records:
    groups[(row["query"], row["fee_cap"], row["write_cap"], row["method"])].append(row)
print("# Fee cap 80: service and mean sequence cost")
for query, caps in (("filter", (32, 96, 512)), ("join", (1200, 2400, 8192))):
    print(f"\n## {query}")
    print("| Write cap | Method | Served | Mean fee | Mean writes |")
    print("|---:|---|---:|---:|---:|")
    for cap in caps:
        for method in ("CF", "SC", "CR", "WQCR"):
            rows = groups[(query, 80, cap, method)]
            if len(rows) != 3 or len({r["stream"] for r in rows}) != 3:
                raise ValueError(f"Expected three distinct sequences: {query}, {cap}, {method}")
            n = len(rows)
            served = sum(r["served"] for r in rows)
            nonempty = sum(r["nonempty"] for r in rows)
            if nonempty <= 0:
                raise ValueError("Service denominator must be positive.")
            fee = sum(r["fee"] for r in rows) / n
            writes = sum(r["writes"] for r in rows) / n
            print(f"| {cap} | {method} | {served}/{nonempty} | {fee:.3f} | {writes:.1f} |")
print("\n# Matched-history branch counts across all fee and write caps")
print("| Query | WCR | WQCR | WCR/WQCR |")
print("|---|---:|---:|---:|")
for query in ("filter", "join"):
    counts = {}
    for method in ("WCR", "WQCR"):
        rows = [r for r in records if r["query"] == query and r["method"] == method]
        if len(rows) != 18:
            raise ValueError(f"Expected 18 trajectories for {query}/{method}.")
        counts[method] = sum(r["branch_tests"] for r in rows)
    if counts["WQCR"] <= 0:
        raise ValueError("Branch-count denominator must be positive.")
    print("| {} | {} | {} | {:.2f} |".format(query, counts["WCR"], counts["WQCR"], counts["WCR"]/counts["WQCR"]))