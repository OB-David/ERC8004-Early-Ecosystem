#!/usr/bin/env python3
"""Clean agent reputation feedback values in-place.

Default behavior:
    python3 ERC8004/scripts/reputation_wash.py

The script reads ERC8004/data/agent_reputation.csv, converts feedback_value into a
comparable 0-100 reputation score, drops rows that are not reputation scores,
and overwrites ERC8004/data/agent_reputation.csv with the cleaned rows.
"""

import argparse
import csv
import os
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_REPUTATION_CSV = PACKAGE_ROOT / "data" / "agent_reputation.csv"

FIELDNAMES = [
    "agent_id",
    "feedback_tx",
    "feedback_client",
    "feedback_client_type",
    "feedback_type",
    "feedback_value",
]

# =========================================================
# Reputation Washing Rules
# =========================================================
#
# The raw ERC-8004 feedback table mixes several value semantics in the same
# feedback_value column. A direct average is misleading, so every row is first
# converted into a comparable 0-100 score, or dropped if the row is not a score.
#
# Rule 1: Tiny positive numeric values are removed
#   Applies before all other conversions.
#   - 0 < value < 0.5 is treated as an invalid/noisy reputation signal and
#     removed from the cleaned CSV.
#   - value = 0 is meaningful and is kept as a real zero score.
#   Example: nixon-president=0.000001 is dropped, reachable-bad=0 is kept.
#
# Rule 2: Boolean reachability / verification / liveness values
#   Applies to feedback types containing "reachable", "ownerVerified", or
#   starting with "liveness".
#   - value > 0 means the agent is reachable/live/verified, so score = 100.
#   - value = 0 means the check failed, so score = 0.
#   Example: reachable-web=1 becomes 100, reachable-bad=0 stays 0.
#
# Rule 3: Non-reputation metric rows are removed
#   Applies to economic/count/volume fields such as "revenues", "revenue",
#   "volume", "amount", "count", "balance", "price", "fee".
#   These numbers are useful metrics, but they are not user reputation scores.
#   Example: revenues=1000 is dropped instead of being averaged as 1000 points.
#
# Rule 4: Known 5-point service-quality rows are scaled to 0-100
#   Applies to feedback types like "health-check-service_quality" and
#   "review-service_quality" when the raw value is between 0 and 5.
#   - score = value * 20.
#   Example: health-check-service_quality=4 becomes 80.
#
# Rule 5: 10-point score rows are scaled to 0-100
#   Applies when the raw value is > 0 and <= 10 and the type looks like a rating,
#   score, review, quality, success, satisfaction, helpfulness, reliability, or
#   excellence signal.
#   - score = value * 10.
#   Example: x402-excellent=9.5 becomes 95.
#
# Rule 6: Normal percentage-like score rows are clamped to 0-100
#   For ordinary score/quality/starred/rating rows, keep the numeric value but
#   cap outliers into the valid reputation range.
#   Example: responseTime-Instant=200 becomes 100.
#
# Rule 7: Unparseable values are removed
#   Rows whose feedback_value cannot be parsed as a number are dropped because
#   they cannot safely contribute to a numeric reputation score.

DROP_TYPE_TOKENS = {
    "revenue",
    "revenues",
    "volume",
    "amount",
    "balance",
    "price",
    "fee",
    "fees",
    "payment_amount",
}

# A bare "count" token is dangerous because "account" contains count-like text
# in some datasets; use tokenized matching for count-like words instead.
DROP_TYPE_WORDS = {"count", "counts", "total", "quantity"}

BOOLEAN_TYPE_TOKENS = {
    "reachable",
    "ownerverified",
    "owner_verified",
    "verified-owner",
}

FIVE_POINT_TYPE_TOKENS = {
    "health-check-service_quality",
    "review-service_quality",
}

TEN_POINT_HINTS = {
    "excellent",
    "excellence",
    "rating",
    "score",
    "quality",
    "review",
    "success",
    "satisfaction",
    "helpful",
    "reliable",
    "reliability",
    "service",
    "good",
    "fast",
    "useful",
    "accurate",
    "trust",
    "performance",
}


def normalize_type(feedback_type: object) -> str:
    return str(feedback_type or "").strip().lower()


def type_words(feedback_type: str) -> List[str]:
    separators = "-_ /.,:;|()[]{}"
    normalized = feedback_type
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    return [word for word in normalized.split() if word]


def parse_decimal(value: object) -> Optional[Decimal]:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def clamp_0_100(value: Decimal) -> Decimal:
    if value < Decimal(0):
        return Decimal(0)
    if value > Decimal(100):
        return Decimal(100)
    return value


def decimal_to_csv(value: Decimal) -> str:
    value = value.normalize()
    if value == value.to_integral():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def is_drop_metric(feedback_type: str) -> bool:
    words = set(type_words(feedback_type))
    if words & DROP_TYPE_WORDS:
        return True
    return any(token in feedback_type for token in DROP_TYPE_TOKENS)


def is_boolean_metric(feedback_type: str) -> bool:
    if feedback_type.startswith("liveness"):
        return True
    return any(token in feedback_type for token in BOOLEAN_TYPE_TOKENS)


def looks_like_ten_point_score(feedback_type: str) -> bool:
    return any(token in feedback_type for token in TEN_POINT_HINTS)


def normalize_feedback_value(feedback_type: object, raw_value: object) -> Tuple[Optional[Decimal], str]:
    """Return (cleaned_score, rule_name). cleaned_score=None means drop row."""
    feedback_type_norm = normalize_type(feedback_type)
    value = parse_decimal(raw_value)
    if value is None:
        return None, "drop_unparseable"

    if Decimal(0) < value < Decimal("0.5"):
        return None, "drop_below_half"

    if is_drop_metric(feedback_type_norm):
        return None, "drop_non_reputation_metric"

    if is_boolean_metric(feedback_type_norm):
        return (Decimal(100) if value > 0 else Decimal(0)), "boolean_to_percent"

    if feedback_type_norm in FIVE_POINT_TYPE_TOKENS and Decimal(0) <= value <= Decimal(5):
        return value * Decimal(20), "five_point_to_percent"

    if Decimal(0) < value <= Decimal(10) and looks_like_ten_point_score(feedback_type_norm):
        return value * Decimal(10), "ten_point_to_percent"

    return clamp_0_100(value), "clamp_percent"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in FIELDNAMES if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        return [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]


def write_rows_atomic(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def wash_rows(rows: Iterable[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    cleaned: List[Dict[str, str]] = []
    stats: Dict[str, int] = {
        "input_rows": 0,
        "output_rows": 0,
        "drop_unparseable": 0,
        "drop_below_half": 0,
        "drop_non_reputation_metric": 0,
        "boolean_to_percent": 0,
        "five_point_to_percent": 0,
        "ten_point_to_percent": 0,
        "clamp_percent": 0,
    }

    for row in rows:
        stats["input_rows"] += 1
        cleaned_value, rule_name = normalize_feedback_value(row.get("feedback_type"), row.get("feedback_value"))
        stats[rule_name] = stats.get(rule_name, 0) + 1
        if cleaned_value is None:
            continue

        cleaned_row = dict(row)
        cleaned_row["feedback_value"] = decimal_to_csv(cleaned_value)
        cleaned.append(cleaned_row)

    stats["output_rows"] = len(cleaned)
    return cleaned, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean ERC8004/data/agent_reputation.csv in-place into comparable 0-100 scores."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_REPUTATION_CSV,
        help="CSV file to clean. Default: ERC8004/data/agent_reputation.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cleaning statistics without overwriting the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path.resolve()
    rows = read_rows(path)
    cleaned_rows, stats = wash_rows(rows)

    if not args.dry_run:
        write_rows_atomic(path, cleaned_rows)

    action = "would write" if args.dry_run else "wrote"
    print(f"[reputation_wash] {action} {stats['output_rows']} rows to {path}")
    for key in sorted(stats):
        print(f"[reputation_wash] {key}={stats[key]}")


if __name__ == "__main__":
    main()
