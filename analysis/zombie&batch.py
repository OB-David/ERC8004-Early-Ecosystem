#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the association between zombie agents and bulk-registered agents.

Inputs:
    ERC8004/data/all_agent.csv

Outputs:
    ERC8004/zombie&batch/zombie_bulk_agent_records.csv
    ERC8004/zombie&batch/zombie_bulk_agent_crosstab.csv
    ERC8004/zombie&batch/agent_distribution_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "zombie&batch"

ALL_AGENT_CSV = DATA_DIR / "all_agent.csv"
RECORDS_CSV = RESULT_DIR / "zombie_bulk_agent_records.csv"
CROSSTAB_CSV = RESULT_DIR / "zombie_bulk_agent_crosstab.csv"
AGENT_DISTRIBUTION_CSV = RESULT_DIR / "agent_distribution_summary.csv"

CONFIRMATION_BLOCKS = 72000
DEFAULT_BULK_THRESHOLD = 10


def norm_text(value: object) -> str:
    return str(value or "").strip()


def norm_addr(value: object) -> str:
    return norm_text(value).lower()


def safe_int(value: object, default: int = 0) -> int:
    text = norm_text(value)
    if not text:
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def read_all_agent(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    records: List[Dict[str, object]] = []
    for row in rows:
        if not norm_text(row.get("agent_id")):
            continue
        records.append(
            {
                "agent_id": safe_int(row.get("agent_id")),
                "block_stamp": safe_int(row.get("block_stamp")),
                "client_count": safe_int(row.get("client_count")),
                "owner_wallet": norm_addr(row.get("owner_wallet")),
                "agent_wallet": norm_addr(row.get("agent_wallet")),
            }
        )
    return records


def is_zombie(record: Dict[str, object]) -> bool:
    client_count = int(record["client_count"])
    agent_wallet = str(record["agent_wallet"])
    return client_count == 0 or (client_count == 1 and not agent_wallet)


def chi_square_2x2(a: int, b: int, c: int, d: int) -> Dict[str, float | str]:
    """Return Pearson chi-square statistics for [[a, b], [c, d]]."""
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    expected = [
        row1 * col1 / n,
        row1 * col2 / n,
        row2 * col1 / n,
        row2 * col2 / n,
    ]
    observed = [a, b, c, d]
    chi2 = sum((obs - exp) ** 2 / exp for obs, exp in zip(observed, expected) if exp)
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    phi = (a * d - b * c) / math.sqrt(row1 * row2 * col1 * col2) if row1 * row2 * col1 * col2 else 0.0

    if 0 in observed:
        odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
        odds_ratio_note = "haldane_anscombe_adjusted"
    else:
        odds_ratio = (a * d) / (b * c)
        odds_ratio_note = "unadjusted"

    return {
        "n": float(n),
        "chi2": chi2,
        "df": 1.0,
        "p_value": p_value,
        "phi": phi,
        "odds_ratio": odds_ratio,
        "odds_ratio_note": odds_ratio_note,
    }


def owner_agent_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if 2 <= value <= 10:
        return "2-10"
    if 11 <= value <= 50:
        return "11-50"
    if 51 <= value <= 250:
        return "51-250"
    if 251 <= value <= 1250:
        return "251-1250"
    return ">1250"


def client_count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 4:
        return str(value)
    return ">4"


def write_dicts(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_agent_distribution_summary(records: List[Dict[str, object]], path: Path) -> None:
    total = len(records)
    owner_counts = Counter(str(row["owner_wallet"]) for row in records if row["owner_wallet"])
    owner_order = ["1", "2-10", "11-50", "51-250", "251-1250", ">1250"]
    client_order = ["0", "1", "2", "3", "4", ">4"]
    owner_distribution = Counter({bucket: 0 for bucket in owner_order})
    client_distribution = Counter({bucket: 0 for bucket in client_order})

    for row in records:
        owner_wallet = str(row["owner_wallet"])
        owner_agent_count = int(owner_counts.get(owner_wallet, 0)) if owner_wallet else 1
        owner_distribution[owner_agent_bucket(owner_agent_count)] += 1
        client_distribution[client_count_bucket(int(row["client_count"]))] += 1

    rows: List[Dict[str, object]] = []
    for bucket in owner_order:
        count = int(owner_distribution[bucket])
        rows.append(
            {
                "distribution": "owner_held_agent_count",
                "variable": "owner_agent_count",
                "bucket": bucket,
                "agent_count": count,
                "agent_share": f"{(count / total if total else 0.0):.6f}",
                "total_agents": total,
            }
        )
    for bucket in client_order:
        count = int(client_distribution[bucket])
        rows.append(
            {
                "distribution": "client_count",
                "variable": "client_count",
                "bucket": bucket,
                "agent_count": count,
                "agent_share": f"{(count / total if total else 0.0):.6f}",
                "total_agents": total,
            }
        )

    write_dicts(
        path,
        rows,
        ["distribution", "variable", "bucket", "agent_count", "agent_share", "total_agents"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 2x2 chi-square test for zombie agents and bulk registrations."
    )
    parser.add_argument(
        "--end-block",
        type=int,
        default=None,
        help="Block height used as the observation endpoint. Defaults to max(block_stamp) in all_agent.csv.",
    )
    parser.add_argument(
        "--bulk-threshold",
        type=int,
        default=DEFAULT_BULK_THRESHOLD,
        help=(
            "Owner holding threshold for bulk registration. "
            "An agent is bulk-registered when owner_agent_count is greater than this value."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bulk_threshold < 1:
        raise SystemExit("--bulk-threshold must be at least 1")

    records = read_all_agent(ALL_AGENT_CSV)
    if not records:
        raise SystemExit(f"No records found in {ALL_AGENT_CSV}")

    observed_end_block = max(int(row["block_stamp"]) for row in records)
    end_block = args.end_block if args.end_block is not None else observed_end_block
    cutoff_block = end_block - CONFIRMATION_BLOCKS
    study_rows = [row for row in records if int(row["block_stamp"]) <= cutoff_block]
    if not study_rows:
        raise SystemExit(
            f"No agents satisfy block_stamp <= {cutoff_block}. "
            "Try passing a larger --end-block."
        )

    owner_counts = Counter(str(row["owner_wallet"]) for row in study_rows if row["owner_wallet"])
    analysis_rows: List[Dict[str, object]] = []
    table = {
        "zombie_bulk": 0,
        "zombie_not_bulk": 0,
        "not_zombie_bulk": 0,
        "not_zombie_not_bulk": 0,
    }

    for row in sorted(study_rows, key=lambda item: int(item["agent_id"])):
        owner_wallet = str(row["owner_wallet"])
        owner_agent_count = int(owner_counts.get(owner_wallet, 0)) if owner_wallet else 0
        zombie = is_zombie(row)
        bulk = owner_agent_count > args.bulk_threshold

        if zombie and bulk:
            table["zombie_bulk"] += 1
        elif zombie:
            table["zombie_not_bulk"] += 1
        elif bulk:
            table["not_zombie_bulk"] += 1
        else:
            table["not_zombie_not_bulk"] += 1

        analysis_rows.append(
            {
                "agent_id": row["agent_id"],
                "block_stamp": row["block_stamp"],
                "client_count": row["client_count"],
                "owner_wallet": owner_wallet,
                "agent_wallet": row["agent_wallet"],
                "owner_agent_count": owner_agent_count,
                "is_zombie_agent": int(zombie),
                "is_bulk_registered_agent": int(bulk),
            }
        )

    stats = chi_square_2x2(
        table["zombie_bulk"],
        table["zombie_not_bulk"],
        table["not_zombie_bulk"],
        table["not_zombie_not_bulk"],
    )

    write_dicts(
        RECORDS_CSV,
        analysis_rows,
        [
            "agent_id",
            "block_stamp",
            "client_count",
            "owner_wallet",
            "agent_wallet",
            "owner_agent_count",
            "is_zombie_agent",
            "is_bulk_registered_agent",
        ],
    )
    write_dicts(
        CROSSTAB_CSV,
        [
            {"is_zombie_agent": 1, "is_bulk_registered_agent": 1, "count": table["zombie_bulk"]},
            {"is_zombie_agent": 1, "is_bulk_registered_agent": 0, "count": table["zombie_not_bulk"]},
            {"is_zombie_agent": 0, "is_bulk_registered_agent": 1, "count": table["not_zombie_bulk"]},
            {"is_zombie_agent": 0, "is_bulk_registered_agent": 0, "count": table["not_zombie_not_bulk"]},
        ],
        ["is_zombie_agent", "is_bulk_registered_agent", "count"],
    )
    write_agent_distribution_summary(records, AGENT_DISTRIBUTION_CSV)

    print(f"study_agents={len(study_rows)}")
    print(f"cutoff_block={cutoff_block}")
    print(f"chi2={float(stats['chi2']):.4f}")
    print(f"p_value={float(stats['p_value']):.6g}")
    print(f"phi={float(stats['phi']):.4f}")
    print(f"records={RECORDS_CSV}")
    print(f"crosstab={CROSSTAB_CSV}")
    print(f"distribution={AGENT_DISTRIBUTION_CSV}")


if __name__ == "__main__":
    main()
