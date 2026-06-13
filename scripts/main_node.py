'''
Fetch all data from your Ethereum node.
This may be slow without an address-to-transaction index.
Use main_etherscan.py for faster collection."""
'''

import csv
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests
from requests import HTTPError
from web3 import Web3

from reputation_wash import wash_rows


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)


def load_local_env_file(path: str) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env_file(os.path.join(REPO_ROOT, ".env"))
load_local_env_file(os.path.join(PACKAGE_ROOT, ".env"))
load_local_env_file(os.path.join(SCRIPT_DIR, ".env"))


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# =========================================================
# Configuration
# =========================================================

DEFAULT_RPC_URL = "https://ethereum-rpc.publicnode.com"
RPC_URL = os.environ.get("ETHEREUM_ARCHIVE_RPC") or DEFAULT_RPC_URL

LOCAL_DATA_DIR = os.path.join(PACKAGE_ROOT, "data")

IDENTITY_REGISTRY = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
REPUTATION_REGISTRY = "0x8004BAa17C55a88189AE136b182e5fdA19dE9b63"

START_BLOCK = int(os.environ.get("START_BLOCK", "24339925"))
OBSERVATION_BLOCK = int(os.environ.get("OBSERVATION_BLOCK", "25277687"))

TARGET_AGENT_ID_MIN = int(os.environ.get("TARGET_AGENT_ID_MIN", "0"))
TARGET_AGENT_ID_MAX = int(os.environ.get("TARGET_AGENT_ID_MAX", "999999"))
TARGET_AGENT_COUNT = int(os.environ.get("TARGET_AGENT_COUNT", "1000000"))
TOP_AGENT_COUNT = int(os.environ.get("TOP_AGENT_COUNT", "200"))
MIN_SELECTED_CLIENT_COUNT = int(os.environ.get("MIN_SELECTED_CLIENT_COUNT", "2"))

SCAN_BLOCK_WINDOW = int(os.environ.get("SCAN_BLOCK_WINDOW", "500"))
PIPELINE_BATCH_SIZE = int(os.environ.get("PIPELINE_BATCH_SIZE", "50"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))
REPUTATION_MAX_WORKERS = int(os.environ.get("REPUTATION_MAX_WORKERS", "16"))
PROGRESS_INTERVAL = int(os.environ.get("PROGRESS_INTERVAL", "100"))
RPC_MAX_RETRIES = int(os.environ.get("RPC_MAX_RETRIES", "8"))
RPC_RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RPC_RETRY_BASE_DELAY_SECONDS", "0.5"))
RPC_BACKOFF_JITTER_SECONDS = float(os.environ.get("RPC_BACKOFF_JITTER_SECONDS", "0.25"))
RPC_MAX_IN_FLIGHT = int(os.environ.get("RPC_MAX_IN_FLIGHT", "32"))
RPC_MIN_INTERVAL_SECONDS = float(os.environ.get("RPC_MIN_INTERVAL_SECONDS", "0.05"))
FAIL_ON_INCOMPLETE_SNAPSHOT = env_bool("FAIL_ON_INCOMPLETE_SNAPSHOT", True)
RERUN_ALL_AGENT = env_bool("RERUN_ALL_AGENT", False)
RERUN_REPUTATION = env_bool("RERUN_REPUTATION", True)
RERUN_TRANSACTION = env_bool("RERUN_TRANSACTION", True)
USE_ETHERSCAN = env_bool("USE_ETHERSCAN", True)
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY")
ETHERSCAN_BASE_URL = os.environ.get("ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api")
ETHERSCAN_CHAIN_ID = int(os.environ.get("ETHERSCAN_CHAIN_ID", "1"))
ETHERSCAN_PAGE_SIZE = int(os.environ.get("ETHERSCAN_PAGE_SIZE", "1000"))
ETHERSCAN_TIMEOUT = int(os.environ.get("ETHERSCAN_TIMEOUT", "30"))
ETHERSCAN_MAX_RETRIES = int(os.environ.get("ETHERSCAN_MAX_RETRIES", "5"))
ETHERSCAN_RETRY_DELAY_SECONDS = float(os.environ.get("ETHERSCAN_RETRY_DELAY_SECONDS", "1"))
ETHERSCAN_TRUST_ENV = env_bool("ETHERSCAN_TRUST_ENV", True)
ETHERSCAN_PROXY = os.environ.get("ETHERSCAN_PROXY")


@dataclass(frozen=True)
class PipelineConfig:
    start_block: int = START_BLOCK
    observation_block: int = OBSERVATION_BLOCK
    target_agent_id_min: int = TARGET_AGENT_ID_MIN
    target_agent_id_max: int = TARGET_AGENT_ID_MAX
    target_agent_count: int = TARGET_AGENT_COUNT
    top_agent_count: int = TOP_AGENT_COUNT
    min_selected_client_count: int = MIN_SELECTED_CLIENT_COUNT
    scan_block_window: int = SCAN_BLOCK_WINDOW
    pipeline_batch_size: int = PIPELINE_BATCH_SIZE
    max_workers: int = MAX_WORKERS
    reputation_max_workers: int = REPUTATION_MAX_WORKERS
    progress_interval: int = PROGRESS_INTERVAL
    rpc_max_retries: int = RPC_MAX_RETRIES
    rpc_retry_base_delay_seconds: float = RPC_RETRY_BASE_DELAY_SECONDS
    rpc_backoff_jitter_seconds: float = RPC_BACKOFF_JITTER_SECONDS
    rpc_max_in_flight: int = RPC_MAX_IN_FLIGHT
    rpc_min_interval_seconds: float = RPC_MIN_INTERVAL_SECONDS
    fail_on_incomplete_snapshot: bool = FAIL_ON_INCOMPLETE_SNAPSHOT
    rerun_all_agent: bool = RERUN_ALL_AGENT
    rerun_reputation: bool = RERUN_REPUTATION
    rerun_transaction: bool = RERUN_TRANSACTION
    use_etherscan: bool = USE_ETHERSCAN
    etherscan_api_key: Optional[str] = ETHERSCAN_API_KEY
    etherscan_base_url: str = ETHERSCAN_BASE_URL
    etherscan_chain_id: int = ETHERSCAN_CHAIN_ID
    etherscan_page_size: int = ETHERSCAN_PAGE_SIZE
    etherscan_timeout: int = ETHERSCAN_TIMEOUT
    etherscan_max_retries: int = ETHERSCAN_MAX_RETRIES
    etherscan_retry_delay_seconds: float = ETHERSCAN_RETRY_DELAY_SECONDS
    etherscan_trust_env: bool = ETHERSCAN_TRUST_ENV
    etherscan_proxy: Optional[str] = ETHERSCAN_PROXY


w3 = Web3(Web3.HTTPProvider(RPC_URL))
RPC_SEMAPHORE = threading.BoundedSemaphore(max(1, RPC_MAX_IN_FLIGHT))
RPC_THROTTLE_LOCK = threading.Lock()
RPC_NEXT_ALLOWED_AT = 0.0


# =========================================================
# Helpers
# =========================================================

TRANSFER_TOPIC = w3.keccak(text="Transfer(address,address,uint256)")
ZERO_TOPIC_BYTES32 = b"\x00" * 32
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

TABLE_FIELDS = {
    "all_agent": [
        "agent_id",
        "block_stamp",
        "client_count",
        "owner_wallet",
        "agent_wallet",
    ],
    "agent_core": [
        "agent_id",
        "block_stamp",
        "client_count",
        "owner_wallet",
        "owner_agent_count",
        "agent_wallet",
    ],
    "agent_reputation": [
        "agent_id",
        "feedback_tx",
        "feedback_client",
        "feedback_client_type",
        "feedback_type",
        "feedback_value",
    ],
    "agent_transaction": [
        "agent_id",
        "agent_wallet",
        "block_stamp",
        "tx_hash",
        "tx_type",
    ],
    "agent_statistic": [
        "agent_id",
        "block_stamp",
        "owner_wallet",
        "owner_agent_count",
        "agent_wallet",
        "reputation",
        "feedback_count",
        "client_count",
        "agent_count",
        "contract_count",
        "eoa_count",
        "tx_count",
        "identity_operation_count",
        "ecosystem_operation_count",
        "other_operation_count",
    ],
}


def norm_addr(addr: Optional[str]) -> Optional[str]:
    return str(addr).lower() if addr else None


def norm_tx_hash(txh: Optional[str]) -> Optional[str]:
    return str(txh).lower() if txh else None


def topic_to_address(topic) -> str:
    topic_bytes = bytes(topic)
    return f"0x{topic_bytes[-20:].hex()}".lower()


def checksum(addr: str) -> str:
    return Web3.to_checksum_address(addr)


def ensure_local_data_dir() -> None:
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)


def csv_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return value


def decimal_value(value: object) -> Decimal:
    if value is None or value == "":
        return Decimal(0)
    return Decimal(str(value))


def normalize_summary_value(raw_value: object, decimals: object) -> Decimal:
    raw_decimal = Decimal(int(raw_value))
    decimal_places = int(decimals or 0)
    if decimal_places <= 0:
        return raw_decimal
    return raw_decimal / (Decimal(10) ** decimal_places)


def write_csv_table(table_name: str, rows: Sequence[Dict[str, object]]) -> None:
    ensure_local_data_dir()
    path = os.path.join(LOCAL_DATA_DIR, f"{table_name}.csv")
    fieldnames = TABLE_FIELDS[table_name]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def read_csv_table(table_name: str) -> List[Dict[str, str]]:
    path = os.path.join(LOCAL_DATA_DIR, f"{table_name}.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def chunked(items: Sequence[object], size: int) -> Iterable[List[object]]:
    for index in range(0, len(items), size):
        yield list(items[index:index + size])


def _is_retryable_rpc_exception(exc: Exception) -> bool:
    if isinstance(exc, HTTPError) and exc.response is not None:
        return int(exc.response.status_code) in {429, 500, 502, 503, 504}
    return False


def rpc_call_with_retry(callable_fn, config: PipelineConfig, context: str):
    global RPC_NEXT_ALLOWED_AT

    attempts = max(1, int(config.rpc_max_retries))
    delay_seconds = max(0.0, float(config.rpc_retry_base_delay_seconds))
    jitter_seconds = max(0.0, float(config.rpc_backoff_jitter_seconds))

    for attempt in range(1, attempts + 1):
        try:
            with RPC_SEMAPHORE:
                with RPC_THROTTLE_LOCK:
                    now = time.monotonic()
                    wait_for = RPC_NEXT_ALLOWED_AT - now
                    if wait_for > 0:
                        time.sleep(wait_for)
                    RPC_NEXT_ALLOWED_AT = time.monotonic() + max(
                        0.0,
                        config.rpc_min_interval_seconds,
                    )
                return callable_fn()
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_rpc_exception(exc):
                raise
            backoff = delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, jitter_seconds)
            print(
                f"[rpc] {context} failed on attempt {attempt}/{attempts}: "
                f"{repr(exc)}. Sleeping {backoff:.2f}s."
            )
            time.sleep(backoff)


# =========================================================
# ABI
# =========================================================

IDENTITY_ABI = [
    {
        "inputs": [{"internalType": "uint256", "name": "tokenId", "type": "uint256"}],
        "name": "ownerOf",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "name": "getAgentWallet",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

REPUTATION_ABI = [
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "int128", "name": "value", "type": "int128"},
            {"internalType": "uint8", "name": "valueDecimals", "type": "uint8"},
            {"internalType": "string", "name": "tag1", "type": "string"},
            {"internalType": "string", "name": "tag2", "type": "string"},
            {"internalType": "string", "name": "clientURI", "type": "string"},
            {"internalType": "string", "name": "serverURI", "type": "string"},
            {"internalType": "bytes32", "name": "requestHash", "type": "bytes32"},
        ],
        "name": "giveFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "uint64", "name": "feedbackIndex", "type": "uint64"},
        ],
        "name": "revokeFeedback",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "uint256", "name": "agentId", "type": "uint256"}],
        "name": "getClients",
        "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "address", "name": "clientAddress", "type": "address"},
        ],
        "name": "getLastIndex",
        "outputs": [{"internalType": "uint64", "name": "", "type": "uint64"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "address", "name": "clientAddress", "type": "address"},
            {"internalType": "uint64", "name": "feedbackIndex", "type": "uint64"},
        ],
        "name": "readFeedback",
        "outputs": [
            {"internalType": "int128", "name": "value", "type": "int128"},
            {"internalType": "uint8", "name": "valueDecimals", "type": "uint8"},
            {"internalType": "string", "name": "tag1", "type": "string"},
            {"internalType": "string", "name": "tag2", "type": "string"},
            {"internalType": "bool", "name": "isRevoked", "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "agentId", "type": "uint256"},
            {"internalType": "address[]", "name": "clientAddresses", "type": "address[]"},
            {"internalType": "string", "name": "tag1", "type": "string"},
            {"internalType": "string", "name": "tag2", "type": "string"},
        ],
        "name": "getSummary",
        "outputs": [
            {"internalType": "uint64", "name": "count", "type": "uint64"},
            {"internalType": "int128", "name": "summaryValue", "type": "int128"},
            {"internalType": "uint8", "name": "summaryValueDecimals", "type": "uint8"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

identity_contract = w3.eth.contract(address=checksum(IDENTITY_REGISTRY), abi=IDENTITY_ABI)
rep_contract = w3.eth.contract(address=checksum(REPUTATION_REGISTRY), abi=REPUTATION_ABI)


# =========================================================
# Discovery and Identity
# =========================================================

def fetch_identity_logs(start_block: int, end_block: int) -> List[dict]:
    return w3.eth.get_logs(
        {
            "fromBlock": int(start_block),
            "toBlock": int(end_block),
            "address": checksum(IDENTITY_REGISTRY),
            "topics": [TRANSFER_TOPIC],
        }
    )


def discover_target_agents(config: PipelineConfig) -> List[Dict[str, object]]:
    current_block = config.start_block
    current_window = config.scan_block_window
    discovered: Dict[int, Dict[str, object]] = {}

    print(
        f"[discovery] scanning from block {config.start_block} "
        f"to {config.observation_block}"
    )

    while current_block <= config.observation_block:
        chunk_end = min(current_block + current_window - 1, config.observation_block)
        try:
            logs = fetch_identity_logs(current_block, chunk_end)
        except Exception as exc:
            if current_window > 1:
                current_window = max(1, current_window // 2)
                print(
                    f"[discovery] get_logs failed for {current_block}-{chunk_end}: "
                    f"{repr(exc)}. Retrying with window={current_window}."
                )
                continue
            raise

        for log in logs:
            topics = log.get("topics", [])
            if len(topics) != 4 or topics[0] != TRANSFER_TOPIC:
                continue
            if bytes(topics[1]) != ZERO_TOPIC_BYTES32:
                continue

            agent_id = int.from_bytes(bytes(topics[3]), byteorder="big")
            if agent_id < config.target_agent_id_min or agent_id > config.target_agent_id_max:
                continue
            if agent_id in discovered:
                continue

            discovered[agent_id] = {
                "agent_id": agent_id,
                "block_stamp": int(log["blockNumber"]),
            }

        print(
            f"[discovery] blocks {current_block}-{chunk_end} "
            f"discovered={len(discovered)} window={current_window}"
        )

        if len(discovered) >= config.target_agent_count:
            break
        if current_window < config.scan_block_window:
            current_window = min(config.scan_block_window, current_window * 2)
        current_block = chunk_end + 1

    ordered = [discovered[agent_id] for agent_id in sorted(discovered)]
    return ordered[:config.target_agent_count]


def fetch_identity_state(agent_seed: Dict[str, object], config: PipelineConfig) -> Dict[str, object]:
    agent_id = int(agent_seed["agent_id"])

    owner_wallet = norm_addr(
        rpc_call_with_retry(
            lambda: identity_contract.functions.ownerOf(agent_id).call(
                block_identifier=config.observation_block
            ),
            config,
            f"ownerOf(agent_id={agent_id})",
        )
    )
    agent_wallet = norm_addr(
        rpc_call_with_retry(
            lambda: identity_contract.functions.getAgentWallet(agent_id).call(
                block_identifier=config.observation_block
            ),
            config,
            f"getAgentWallet(agent_id={agent_id})",
        )
    )
    if agent_wallet == ZERO_ADDRESS:
        agent_wallet = None

    return {
        "agent_id": agent_id,
        "block_stamp": int(agent_seed.get("block_stamp") or config.start_block),
        "owner_wallet": owner_wallet,
        "agent_wallet": agent_wallet,
    }


def run_identity_stage(
    agent_seeds: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    successes: List[Dict[str, object]] = []
    failed_agent_ids: List[int] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {
            executor.submit(fetch_identity_state, seed, config): seed
            for seed in agent_seeds
        }
        for future in as_completed(future_map):
            seed = future_map[future]
            agent_id = int(seed["agent_id"])
            try:
                successes.append(future.result())
            except Exception as exc:
                print(f"[identity] failed agent_id={agent_id}: {repr(exc)}")
                failed_agent_ids.append(agent_id)

    successes = sorted(successes, key=lambda row: int(row["agent_id"]))
    return successes, {
        "success": len(successes),
        "failed": len(failed_agent_ids),
        "failed_agent_ids": failed_agent_ids,
    }


# =========================================================
# Preselection
# =========================================================

def fetch_client_count(agent_id: int, config: PipelineConfig) -> int:
    clients = rpc_call_with_retry(
        lambda: rep_contract.functions.getClients(agent_id).call(
            block_identifier=config.observation_block
        ),
        config,
        f"getClients(agent_id={agent_id}) for preselection",
    )
    return len({norm_addr(client) for client in clients if client})


def build_preselection_record(
    identity_record: Dict[str, object],
    config: PipelineConfig,
) -> Dict[str, object]:
    agent_id = int(identity_record["agent_id"])
    client_count = fetch_client_count(agent_id, config)
    return {
        "agent_id": agent_id,
        "block_stamp": int(identity_record.get("block_stamp") or config.start_block),
        "client_count": client_count,
        "owner_wallet": identity_record["owner_wallet"],
        "agent_wallet": identity_record["agent_wallet"],
    }


def load_all_agent_records() -> List[Dict[str, object]]:
    rows = read_csv_table("all_agent")
    records = []
    for row in rows:
        records.append(
            {
                "agent_id": int(row["agent_id"]),
                "block_stamp": int(row.get("block_stamp") or START_BLOCK),
                "client_count": int(row.get("client_count") or 0),
                "owner_wallet": norm_addr(row.get("owner_wallet")),
                "agent_wallet": norm_addr(row.get("agent_wallet")) if row.get("agent_wallet") else None,
            }
        )
    return sorted(
        records,
        key=lambda row: (
            -int(row.get("client_count") or 0),
            int(row["agent_id"]),
        ),
    )


def select_identity_records_from_pre(
    pre_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> List[Dict[str, object]]:
    owner_agent_count_by_wallet: Dict[str, int] = {}
    for row in pre_records:
        owner_wallet = norm_addr(row.get("owner_wallet")) or ""
        if owner_wallet:
            owner_agent_count_by_wallet[owner_wallet] = owner_agent_count_by_wallet.get(owner_wallet, 0) + 1

    selected_pre = [
        row
        for row in pre_records
        if int(row.get("client_count") or 0) >= config.min_selected_client_count
    ][:config.top_agent_count]
    return [
        {
            "agent_id": int(row["agent_id"]),
            "block_stamp": int(row.get("block_stamp") or config.start_block),
            "client_count": int(row.get("client_count") or 0),
            "owner_wallet": row["owner_wallet"],
            "owner_agent_count": owner_agent_count_by_wallet.get(norm_addr(row.get("owner_wallet")) or "", 1),
            "agent_wallet": row["agent_wallet"],
        }
        for row in selected_pre
    ]


def run_preselection_stage(
    identity_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    pre_records: List[Dict[str, object]] = []
    failed_agent_ids: List[int] = []

    total = len(identity_records)
    completed = 0
    progress_interval = max(1, int(config.progress_interval))

    with ThreadPoolExecutor(max_workers=config.reputation_max_workers) as executor:
        future_map = {
            executor.submit(build_preselection_record, record, config): record
            for record in identity_records
        }
        for future in as_completed(future_map):
            record = future_map[future]
            agent_id = int(record["agent_id"])
            try:
                pre_records.append(future.result())
            except Exception as exc:
                print(f"[preselection] failed agent_id={agent_id}: {repr(exc)}")
                failed_agent_ids.append(agent_id)

            completed += 1
            if completed == total or completed % progress_interval == 0:
                print(
                    f"[preselection] reputation progress "
                    f"{completed}/{total} success={len(pre_records)} "
                    f"failed={len(failed_agent_ids)}"
                )

    pre_records = sorted(
        pre_records,
        key=lambda row: (
            -int(row.get("client_count") or 0),
            int(row["agent_id"]),
        ),
    )
    write_csv_table("all_agent", pre_records)

    # The preselection stage only writes all_agent.csv.
    # Later branches decide which agents need reputation or transaction reruns.
    return pre_records, [], {
        "success": len(pre_records),
        "failed": len(failed_agent_ids),
        "failed_agent_ids": failed_agent_ids,
    }


# =========================================================
# Reputation
# =========================================================

def build_known_agent_addresses(identity_records: Sequence[Dict[str, object]]) -> Set[str]:
    addresses: Set[str] = set()
    for record in identity_records:
        address = norm_addr(record.get("agent_wallet"))
        if address:
            addresses.add(address)
    return addresses


def classify_client_address(
    client_address: str,
    known_agent_addresses: Set[str],
    config: PipelineConfig,
) -> str:
    client_address = norm_addr(client_address)
    if not client_address:
        return "eoa"
    if client_address in known_agent_addresses:
        return "agent"

    code = rpc_call_with_retry(
        lambda: w3.eth.get_code(checksum(client_address), block_identifier=config.observation_block),
        config,
        f"get_code(client={client_address})",
    )
    code_bytes = bytes(code)
    if not code_bytes:
        return "eoa"

    # EIP-7702 delegated EOAs expose a small designator code beginning with 0xef0100.
    # They are still externally owned accounts for this dataset's client-type split.
    if code_bytes.startswith(bytes.fromhex("ef0100")):
        return "eoa"

    return "contract"


def make_feedback_type(tag1: object, tag2: object) -> str:
    left = str(tag1 or "").strip()
    right = str(tag2 or "").strip()
    if left and right:
        return f"{left}-{right}"
    return left or right


def is_reputation_feedback_type(feedback_type: object) -> bool:
    value = str(feedback_type or "").strip().lower()
    return any(token in value for token in ("score", "rating", "reputation"))


def tx_input_hex(tx) -> str:
    input_data = tx.get("input") or tx.get("data") or "0x"
    if isinstance(input_data, bytes):
        return "0x" + input_data.hex()
    return str(input_data)


def decode_reputation_transaction(tx) -> Tuple[Optional[str], Dict[str, object]]:
    try:
        function, args = rep_contract.decode_function_input(tx_input_hex(tx))
    except Exception:
        return None, {}
    return getattr(function, "fn_name", None), dict(args)


def build_feedback_tx_index(
    identity_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> Dict[Tuple[int, str, int], str]:
    selected_agent_ids = {int(record["agent_id"]) for record in identity_records}
    if not selected_agent_ids:
        return {}

    start_by_agent = {
        int(record["agent_id"]): agent_start_block(record, config)
        for record in identity_records
    }
    current_block = min(start_by_agent.values())
    current_window = config.scan_block_window
    seen_tx_hashes: Set[str] = set()
    feedback_tx_by_key: Dict[Tuple[int, str, int], str] = {}
    next_index_by_pair: Dict[Tuple[int, str], int] = {}

    print(
        f"[reputation] scanning reputation tx logs from {current_block} "
        f"to {config.observation_block}"
    )

    while current_block <= config.observation_block:
        chunk_end = min(current_block + current_window - 1, config.observation_block)
        try:
            logs = fetch_contract_logs(REPUTATION_REGISTRY, current_block, chunk_end)
        except Exception as exc:
            if current_window > 1:
                current_window = max(1, current_window // 2)
                print(
                    f"[reputation] tx log scan failed for {current_block}-{chunk_end}: "
                    f"{repr(exc)}. Retrying with window={current_window}."
                )
                continue
            raise

        for log in logs:
            tx_hash = norm_tx_hash(log["transactionHash"].hex())
            if not tx_hash or tx_hash in seen_tx_hashes:
                continue
            seen_tx_hashes.add(tx_hash)

            tx = rpc_call_with_retry(
                lambda txh=tx_hash: w3.eth.get_transaction(txh),
                config,
                f"get_transaction(tx_hash={tx_hash}) for feedback tx index",
            )
            function_name, args = decode_reputation_transaction(tx)
            if function_name != "giveFeedback":
                continue

            agent_id = int(args.get("agentId") or args.get("agent_id") or 0)
            if agent_id not in selected_agent_ids:
                continue
            block_number = tx_block_number(tx)
            if block_number < start_by_agent[agent_id]:
                continue

            client = norm_addr(tx.get("from"))
            if not client:
                continue
            pair = (agent_id, client)
            feedback_index = next_index_by_pair.get(pair, 0) + 1
            next_index_by_pair[pair] = feedback_index
            feedback_tx_by_key[(agent_id, client, feedback_index)] = tx_hash

        print(f"[reputation] tx logs {current_block}-{chunk_end}")
        if current_window < config.scan_block_window:
            current_window = min(config.scan_block_window, current_window * 2)
        current_block = chunk_end + 1

    return feedback_tx_by_key


def fetch_reputation(
    identity_record: Dict[str, object],
    known_agent_addresses: Set[str],
    feedback_tx_by_key: Dict[Tuple[int, str, int], str],
    config: PipelineConfig,
) -> Dict[str, object]:
    agent_id = int(identity_record["agent_id"])
    clients = rpc_call_with_retry(
        lambda: rep_contract.functions.getClients(agent_id).call(
            block_identifier=config.observation_block
        ),
        config,
        f"getClients(agent_id={agent_id})",
    )
    unique_clients = sorted({norm_addr(client) for client in clients if client})

    client_types: Dict[str, str] = {}
    for client in unique_clients:
        client_types[client] = classify_client_address(client, known_agent_addresses, config)

    feedback_rows: List[Dict[str, object]] = []
    score_values: List[Decimal] = []
    active_clients: Set[str] = set()

    for client in unique_clients:
        client_checksum = checksum(client)
        last_index = int(
            rpc_call_with_retry(
                lambda: rep_contract.functions.getLastIndex(agent_id, client_checksum).call(
                    block_identifier=config.observation_block
                ),
                config,
                f"getLastIndex(agent_id={agent_id}, client={client})",
            )
        )

        for feedback_index in range(1, last_index + 1):
            value_raw, value_decimals, tag1, tag2, is_revoked = rpc_call_with_retry(
                lambda: rep_contract.functions.readFeedback(
                    agent_id,
                    client_checksum,
                    feedback_index,
                ).call(block_identifier=config.observation_block),
                config,
                f"readFeedback(agent_id={agent_id}, client={client}, index={feedback_index})",
            )
            if bool(is_revoked):
                continue

            feedback_type = make_feedback_type(tag1, tag2)
            feedback_value = normalize_summary_value(value_raw, value_decimals)
            active_clients.add(client)
            if is_reputation_feedback_type(feedback_type):
                score_values.append(feedback_value)

            feedback_rows.append(
                {
                    "agent_id": agent_id,
                    "feedback_tx": feedback_tx_by_key.get((agent_id, client, feedback_index)),
                    "feedback_client": client,
                    "feedback_client_type": client_types[client],
                    "feedback_type": feedback_type,
                    "feedback_value": feedback_value,
                }
            )

    active_client_types = [client_types[client] for client in active_clients]
    reputation = Decimal(0)
    if score_values:
        reputation = sum(score_values, Decimal(0)) / Decimal(len(score_values))

    return {
        "agent_id": agent_id,
        "feedback_rows": feedback_rows,
        "summary": {
            "agent_id": agent_id,
            "reputation": reputation,
            "feedback_count": len(feedback_rows),
            "client_count": len(active_clients),
            "agent_count": active_client_types.count("agent"),
            "contract_count": active_client_types.count("contract"),
            "eoa_count": active_client_types.count("eoa"),
        },
    }


def run_reputation_stage(
    identity_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    known_agent_addresses = build_known_agent_addresses(identity_records)
    feedback_tx_by_key = build_feedback_tx_index(identity_records, config)
    feedback_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    failed_agent_ids: List[int] = []

    total = len(identity_records)
    completed = 0
    progress_interval = max(1, int(config.progress_interval))

    with ThreadPoolExecutor(max_workers=config.reputation_max_workers) as executor:
        future_map = {
            executor.submit(
                fetch_reputation,
                record,
                known_agent_addresses,
                feedback_tx_by_key,
                config,
            ): record
            for record in identity_records
        }
        for future in as_completed(future_map):
            record = future_map[future]
            agent_id = int(record["agent_id"])
            try:
                result = future.result()
                feedback_rows.extend(result["feedback_rows"])
                summaries.append(result["summary"])
            except Exception as exc:
                print(f"[reputation] failed agent_id={agent_id}: {repr(exc)}")
                failed_agent_ids.append(agent_id)

            completed += 1
            if completed == total or completed % progress_interval == 0:
                print(
                    f"[reputation] progress {completed}/{total} "
                    f"success={len(summaries)} failed={len(failed_agent_ids)}"
                )

    feedback_rows = sorted(
        feedback_rows,
        key=lambda row: (
            int(row["agent_id"]),
            str(row.get("feedback_client") or ""),
            str(row.get("feedback_tx") or ""),
            str(row.get("feedback_type") or ""),
        ),
    )

    # Normalize feedback values to 0-100 reputation scores before writing agent_reputation.csv.
    # Non-reputation metric rows are dropped by reputation_wash.
    feedback_rows, wash_stats = wash_rows(feedback_rows)
    feedback_rows = sorted(
        feedback_rows,
        key=lambda row: (
            int(row["agent_id"]),
            str(row.get("feedback_client") or ""),
            str(row.get("feedback_tx") or ""),
            str(row.get("feedback_type") or ""),
        ),
    )

    # Recompute reputation summaries from cleaned feedback rows.
    summaries = summarize_reputation_rows(feedback_rows)
    write_csv_table("agent_reputation", feedback_rows)
    print(
        f"[reputation_wash] cleaned feedback rows "
        f"{wash_stats['input_rows']} -> {wash_stats['output_rows']}"
    )
    for key in sorted(wash_stats):
        print(f"[reputation_wash] {key}={wash_stats[key]}")

    return feedback_rows, summaries, {
        "success": len(summaries),
        "failed": len(failed_agent_ids),
        "failed_agent_ids": failed_agent_ids,
        "wash_stats": wash_stats,
    }


# =========================================================
# Transactions
# =========================================================

TX_TYPE_PRIORITY = {
    "other_operation": 1,
    "ecosystem_operation": 2,
    "identity_operation": 3,
}


def load_signature_map() -> Dict[str, str]:
    path = os.path.join(SCRIPT_DIR, "signature_map.csv")
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        return {
            row["selector"].strip().lower(): row["transaction_group"].strip()
            for row in rows
            if row.get("selector") and row.get("transaction_group")
        }


def normalize_tx_type(transaction_group: Optional[str]) -> str:
    value = str(transaction_group or "").strip()
    aliases = {
        "fixed_agent_operation": "identity_operation",
        "agent_ecosystem_operation": "ecosystem_operation",
    }
    value = aliases.get(value, value)
    if value in {
        "identity_operation",
        "ecosystem_operation",
        "other_operation",
    }:
        return value
    return "other_operation"


def normalize_client_type(client_type: Optional[str]) -> str:
    value = str(client_type or "").strip()
    if value in {"agent", "contract", "eoa"}:
        return value
    return "eoa"


def transaction_selector(tx) -> str:
    input_data = tx.get("input") or tx.get("data") or "0x"
    if isinstance(input_data, bytes):
        input_hex = "0x" + input_data.hex()
    else:
        input_hex = str(input_data)
    input_hex = input_hex.lower()
    return input_hex[:10] if len(input_hex) >= 10 else "0x"


def agent_start_block(identity_record: Dict[str, object], config: PipelineConfig) -> int:
    return max(config.start_block, int(identity_record.get("block_stamp") or config.start_block))


def etherscan_row_block(row: Dict[str, object]) -> int:
    return int(row.get("blockNumber") or row.get("block_number") or 0)


def tx_block_number(tx) -> int:
    return int(tx.get("blockNumber") or 0)


def wallet_agent_index(identity_records: Sequence[Dict[str, object]]) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = {}
    for record in identity_records:
        agent_wallet = norm_addr(record.get("agent_wallet"))
        if not agent_wallet:
            continue
        index.setdefault(agent_wallet, []).append(int(record["agent_id"]))
    return index


def fetch_contract_logs(address: str, start_block: int, end_block: int) -> List[dict]:
    return w3.eth.get_logs(
        {
            "fromBlock": int(start_block),
            "toBlock": int(end_block),
            "address": checksum(address),
        }
    )


def etherscan_tx_selector(tx: Dict[str, object]) -> str:
    method_id = str(tx.get("methodId") or "").lower()
    if method_id.startswith("0x") and len(method_id) == 10:
        return method_id

    input_data = str(tx.get("input") or "0x").lower()
    return input_data[:10] if len(input_data) >= 10 else "0x"


def fetch_etherscan_wallet_transactions(
    wallet: str,
    start_block: int,
    config: PipelineConfig,
) -> List[Dict[str, object]]:
    if not config.etherscan_api_key:
        raise RuntimeError("ETHERSCAN_API_KEY is required when USE_ETHERSCAN=true")

    page = 1
    all_rows: List[Dict[str, object]] = []
    session = requests.Session()
    session.trust_env = config.etherscan_trust_env
    proxies = None
    if config.etherscan_proxy:
        proxies = {
            "http": config.etherscan_proxy,
            "https": config.etherscan_proxy,
        }

    while True:
        params = {
            "chainid": config.etherscan_chain_id,
            "module": "account",
            "action": "txlist",
            "address": wallet,
            "startblock": start_block,
            "endblock": config.observation_block,
            "page": page,
            "offset": config.etherscan_page_size,
            "sort": "asc",
            "apikey": config.etherscan_api_key,
        }
        payload = None
        for attempt in range(1, config.etherscan_max_retries + 1):
            try:
                response = session.get(
                    config.etherscan_base_url,
                    params=params,
                    timeout=config.etherscan_timeout,
                    proxies=proxies,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException as exc:
                if attempt >= config.etherscan_max_retries:
                    raise RuntimeError(
                        f"Etherscan txlist request failed for {wallet} "
                        f"after {attempt} attempts: {repr(exc)}"
                    ) from exc
                sleep_for = config.etherscan_retry_delay_seconds * attempt
                print(
                    f"[transactions] Etherscan request failed for wallet={wallet} "
                    f"page={page} attempt={attempt}: {repr(exc)}. "
                    f"Sleeping {sleep_for:.1f}s."
                )
                time.sleep(sleep_for)

        if payload is None:
            break
        message = str(payload.get("message", ""))
        result = payload.get("result", [])

        if payload.get("status") == "0":
            if message.upper() == "NO TRANSACTIONS FOUND":
                break
            raise RuntimeError(f"Etherscan txlist failed for {wallet}: {payload}")
        if not result:
            break

        all_rows.extend(result)
        if len(result) < config.etherscan_page_size:
            break
        page += 1

    return all_rows


def record_agent_transaction(
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]],
    agent_id: int,
    agent_wallet: Optional[str],
    block_stamp: int,
    tx_hash: Optional[str],
    tx_type: str,
) -> None:
    tx_hash = norm_tx_hash(tx_hash)
    if not tx_hash:
        return

    tx_type = normalize_tx_type(tx_type)
    key = (int(agent_id), tx_hash)
    existing = transactions_by_key.get(key)
    if existing is None:
        transactions_by_key[key] = {
            "agent_id": int(agent_id),
            "agent_wallet": agent_wallet,
            "block_stamp": int(block_stamp),
            "tx_hash": tx_hash,
            "tx_type": tx_type,
        }
        return

    if TX_TYPE_PRIORITY[tx_type] > TX_TYPE_PRIORITY[str(existing["tx_type"])]:
        existing["tx_type"] = tx_type
    existing["block_stamp"] = min(int(existing["block_stamp"]), int(block_stamp))


def scan_agent_wallet_transactions_with_etherscan(
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]],
    identity_records: Sequence[Dict[str, object]],
    signature_map: Dict[str, str],
    config: PipelineConfig,
) -> bool:
    wallet_index = wallet_agent_index(identity_records)
    if not wallet_index:
        print("[transactions] no agent wallets found; skipping Etherscan wallet scan")
        return True
    if not config.use_etherscan or not config.etherscan_api_key:
        return False

    start_by_agent = {
        int(record["agent_id"]): agent_start_block(record, config)
        for record in identity_records
    }
    wallet_by_agent = {
        int(record["agent_id"]): norm_addr(record.get("agent_wallet"))
        for record in identity_records
    }

    print(
        f"[transactions] fetching Etherscan txlist for {len(wallet_index)} agent wallets "
        f"using each agent registration block through {config.observation_block}"
    )

    for wallet, agent_ids in wallet_index.items():
        wallet_start_block = min(start_by_agent[agent_id] for agent_id in agent_ids)
        try:
            rows = fetch_etherscan_wallet_transactions(wallet, wallet_start_block, config)
        except Exception as exc:
            print(f"[transactions] Etherscan failed for wallet={wallet}: {repr(exc)}")
            return False

        print(f"[transactions] Etherscan wallet={wallet} txs={len(rows)} start_block={wallet_start_block}")
        for row in rows:
            tx_hash = norm_tx_hash(row.get("hash"))
            if not tx_hash:
                continue

            row_block = etherscan_row_block(row)
            selector = etherscan_tx_selector(row)
            tx_type = normalize_tx_type(signature_map.get(selector))
            for agent_id in agent_ids:
                if row_block < start_by_agent[agent_id]:
                    continue
                record_agent_transaction(
                    transactions_by_key,
                    agent_id,
                    wallet_by_agent[agent_id],
                    row_block,
                    tx_hash,
                    tx_type,
                )

    return True


def tx_hash_hex(tx) -> Optional[str]:
    tx_hash = tx.get("hash")
    if tx_hash is None:
        return None
    if hasattr(tx_hash, "hex"):
        return norm_tx_hash(tx_hash.hex())
    return norm_tx_hash(str(tx_hash))


def scan_agent_wallet_transactions_with_node(
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]],
    identity_records: Sequence[Dict[str, object]],
    signature_map: Dict[str, str],
    config: PipelineConfig,
) -> None:
    """Scan full blocks from the RPC node and keep txs involving agent wallets.

    This replaces the Etherscan account txlist shortcut. Plain Ethereum RPC has
    no native "list transactions by address" endpoint, so the equivalent node
    implementation is block -> transactions -> address filter.
    """
    wallet_index = wallet_agent_index(identity_records)
    if not wallet_index:
        print("[transactions] no agent wallets found; skipping node wallet scan")
        return

    start_by_agent = {
        int(record["agent_id"]): agent_start_block(record, config)
        for record in identity_records
    }
    wallet_by_agent = {
        int(record["agent_id"]): norm_addr(record.get("agent_wallet"))
        for record in identity_records
    }

    current_block = min(start_by_agent.values()) if start_by_agent else config.start_block
    current_window = config.scan_block_window
    matched_transactions = 0
    scanned_blocks = 0

    print(
        f"[transactions] scanning full node blocks for {len(wallet_index)} agent wallets "
        f"from {current_block} to {config.observation_block}"
    )

    while current_block <= config.observation_block:
        chunk_end = min(current_block + current_window - 1, config.observation_block)
        for block_number in range(current_block, chunk_end + 1):
            block = rpc_call_with_retry(
                lambda block_number=block_number: w3.eth.get_block(
                    block_number,
                    full_transactions=True,
                ),
                config,
                f"get_block(block_number={block_number}, full_transactions=True)",
            )
            scanned_blocks += 1

            for tx in block.get("transactions", []):
                tx_hash = tx_hash_hex(tx)
                if not tx_hash:
                    continue

                involved_wallets = {
                    wallet
                    for wallet in (
                        norm_addr(tx.get("from")),
                        norm_addr(tx.get("to")),
                    )
                    if wallet and wallet in wallet_index
                }
                if not involved_wallets:
                    continue

                selector = transaction_selector(tx)
                tx_type = normalize_tx_type(signature_map.get(selector))
                block_number_for_tx = tx_block_number(tx) or block_number

                for wallet in involved_wallets:
                    for agent_id in wallet_index[wallet]:
                        if block_number_for_tx < start_by_agent[agent_id]:
                            continue
                        record_agent_transaction(
                            transactions_by_key,
                            agent_id,
                            wallet_by_agent[agent_id],
                            block_number_for_tx,
                            tx_hash,
                            tx_type,
                        )
                matched_transactions += 1

        print(
            f"[transactions] node blocks {current_block}-{chunk_end} "
            f"scanned_blocks={scanned_blocks} matched_txs={matched_transactions}"
        )
        if current_window < config.scan_block_window:
            current_window = min(config.scan_block_window, current_window * 2)
        current_block = chunk_end + 1


def scan_agent_wallet_contract_operations(
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]],
    identity_records: Sequence[Dict[str, object]],
    signature_map: Dict[str, str],
    config: PipelineConfig,
) -> None:
    wallet_index = wallet_agent_index(identity_records)
    if not wallet_index:
        print("[transactions] no agent wallets found; skipping wallet scan")
        return

    start_by_agent = {
        int(record["agent_id"]): agent_start_block(record, config)
        for record in identity_records
    }
    wallet_by_agent = {
        int(record["agent_id"]): norm_addr(record.get("agent_wallet"))
        for record in identity_records
    }

    print(
        f"[transactions] scanning ERC-8004 contract logs for {len(wallet_index)} agent wallets "
        f"from {min(start_by_agent.values())} to {config.observation_block}"
    )

    tx_cache: Dict[str, object] = {}
    current_block = min(start_by_agent.values())
    current_window = config.scan_block_window
    contracts = (IDENTITY_REGISTRY, REPUTATION_REGISTRY)

    while current_block <= config.observation_block:
        chunk_end = min(current_block + current_window - 1, config.observation_block)
        try:
            logs = []
            for contract in contracts:
                logs.extend(fetch_contract_logs(contract, current_block, chunk_end))
        except Exception as exc:
            if current_window > 1:
                current_window = max(1, current_window // 2)
                print(
                    f"[transactions] contract log scan failed for {current_block}-{chunk_end}: "
                    f"{repr(exc)}. Retrying with window={current_window}."
                )
                continue
            raise

        for log in logs:
            tx_hash = norm_tx_hash(log["transactionHash"].hex())
            if not tx_hash:
                continue
            if tx_hash not in tx_cache:
                tx_cache[tx_hash] = rpc_call_with_retry(
                    lambda txh=tx_hash: w3.eth.get_transaction(txh),
                    config,
                    f"get_transaction(tx_hash={tx_hash})",
                )

            tx = tx_cache[tx_hash]
            involved_agent_ids = set(wallet_index.get(norm_addr(tx.get("from")), []))
            involved_agent_ids.update(wallet_index.get(norm_addr(tx.get("to")), []))
            if not involved_agent_ids:
                continue

            selector = transaction_selector(tx)
            tx_type = normalize_tx_type(signature_map.get(selector))
            block_number = tx_block_number(tx)
            for agent_id in involved_agent_ids:
                if block_number < start_by_agent[agent_id]:
                    continue
                record_agent_transaction(
                    transactions_by_key,
                    agent_id,
                    wallet_by_agent[agent_id],
                    block_number,
                    tx_hash,
                    tx_type,
                )

        print(f"[transactions] contract logs {current_block}-{chunk_end}")
        if current_window < config.scan_block_window:
            current_window = min(config.scan_block_window, current_window * 2)
        current_block = chunk_end + 1


def scan_agent_nft_transfers(
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]],
    identity_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> None:
    start_by_agent = {
        int(record["agent_id"]): agent_start_block(record, config)
        for record in identity_records
    }
    wallet_by_agent = {
        int(record["agent_id"]): norm_addr(record.get("agent_wallet"))
        for record in identity_records
    }
    current_block = min(start_by_agent.values()) if start_by_agent else config.start_block
    current_window = config.scan_block_window

    print(
        f"[transactions] scanning Agent NFT transfers from {current_block} "
        f"to {config.observation_block}"
    )

    while current_block <= config.observation_block:
        chunk_end = min(current_block + current_window - 1, config.observation_block)
        try:
            logs = fetch_identity_logs(current_block, chunk_end)
        except Exception as exc:
            if current_window > 1:
                current_window = max(1, current_window // 2)
                print(
                    f"[transactions] get_logs failed for {current_block}-{chunk_end}: "
                    f"{repr(exc)}. Retrying with window={current_window}."
                )
                continue
            raise

        for log in logs:
            topics = log.get("topics", [])
            if len(topics) != 4 or topics[0] != TRANSFER_TOPIC:
                continue
            if bytes(topics[1]) == ZERO_TOPIC_BYTES32:
                continue

            agent_id = int.from_bytes(bytes(topics[3]), byteorder="big")
            if agent_id not in start_by_agent:
                continue
            block_number = int(log["blockNumber"])
            if block_number < start_by_agent[agent_id]:
                continue

            record_agent_transaction(
                transactions_by_key,
                agent_id,
                wallet_by_agent[agent_id],
                block_number,
                norm_tx_hash(log["transactionHash"].hex()),
                "ecosystem_operation",
            )

        print(f"[transactions] Agent NFT logs {current_block}-{chunk_end}")
        if current_window < config.scan_block_window:
            current_window = min(config.scan_block_window, current_window * 2)
        current_block = chunk_end + 1


def run_transaction_stage(
    identity_records: Sequence[Dict[str, object]],
    config: PipelineConfig,
) -> List[Dict[str, object]]:
    signature_map = load_signature_map()
    transactions_by_key: Dict[Tuple[int, str], Dict[str, object]] = {}

    scan_agent_wallet_transactions_with_node(
        transactions_by_key,
        identity_records,
        signature_map,
        config,
    )

    scan_agent_nft_transfers(transactions_by_key, identity_records, config)

    rows = sorted(
        transactions_by_key.values(),
        key=lambda row: (int(row["agent_id"]), int(row["block_stamp"]), str(row["tx_hash"])),
    )
    write_csv_table("agent_transaction", rows)
    return rows


# =========================================================
# Pipeline
# =========================================================

def print_stage_stats(stage_name: str, stats: Dict[str, object]) -> None:
    print(
        f"[{stage_name}] success={stats['success']} "
        f"failed={stats['failed']}"
    )


def summarize_reputation_rows(
    reputation_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped: Dict[int, Dict[str, object]] = {}
    feedback_values_by_agent: Dict[int, List[Decimal]] = {}
    clients_by_agent: Dict[int, Set[str]] = {}
    client_types_by_agent: Dict[int, Dict[str, str]] = {}

    for row in reputation_rows:
        agent_id = int(row["agent_id"])
        summary = grouped.setdefault(
            agent_id,
            {
                "agent_id": agent_id,
                "reputation": Decimal(0),
                "feedback_count": 0,
                "client_count": 0,
                "agent_count": 0,
                "contract_count": 0,
                "eoa_count": 0,
            },
        )
        summary["feedback_count"] += 1

        client = norm_addr(row.get("feedback_client"))
        if client:
            clients_by_agent.setdefault(agent_id, set()).add(client)
            client_types_by_agent.setdefault(agent_id, {})[client] = normalize_client_type(
                row.get("feedback_client_type")
            )

        feedback_values_by_agent.setdefault(agent_id, []).append(
            decimal_value(row.get("feedback_value"))
        )

    for agent_id, summary in grouped.items():
        clients = clients_by_agent.get(agent_id, set())
        client_types = client_types_by_agent.get(agent_id, {})
        feedback_values = feedback_values_by_agent.get(agent_id, [])
        if feedback_values:
            summary["reputation"] = sum(feedback_values, Decimal(0)) / Decimal(len(feedback_values))
        summary["client_count"] = len(clients)
        summary["agent_count"] = sum(1 for client in clients if normalize_client_type(client_types.get(client)) == "agent")
        summary["contract_count"] = sum(1 for client in clients if normalize_client_type(client_types.get(client)) == "contract")
        summary["eoa_count"] = sum(1 for client in clients if normalize_client_type(client_types.get(client)) == "eoa")

    return sorted(grouped.values(), key=lambda row: int(row["agent_id"]))


def read_agent_core_records() -> List[Dict[str, object]]:
    rows = read_csv_table("agent_core")
    records = []
    for row in rows:
        if not row.get("agent_id"):
            continue
        records.append(
            {
                "agent_id": int(row["agent_id"]),
                "block_stamp": int(row.get("block_stamp") or 0),
                "client_count": int(row.get("client_count") or 0),
                "owner_wallet": norm_addr(row.get("owner_wallet")),
                "owner_agent_count": int(row.get("owner_agent_count") or 1),
                "agent_wallet": norm_addr(row.get("agent_wallet")) if row.get("agent_wallet") else None,
            }
        )
    return sorted(records, key=lambda row: int(row["agent_id"]))


def load_reputation_records_from_csv() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows = read_csv_table("agent_reputation")
    records = [
        {
            "agent_id": int(row["agent_id"]),
            "feedback_tx": norm_tx_hash(row.get("feedback_tx")) if row.get("feedback_tx") else None,
            "feedback_client": norm_addr(row.get("feedback_client")),
            "feedback_client_type": normalize_client_type(row.get("feedback_client_type")),
            "feedback_type": row.get("feedback_type") or "",
            "feedback_value": decimal_value(row.get("feedback_value")),
        }
        for row in rows
        if row.get("agent_id")
    ]
    return records, summarize_reputation_rows(records)


def load_transaction_records_from_csv() -> List[Dict[str, object]]:
    rows = read_csv_table("agent_transaction")
    records = []
    for row in rows:
        if not row.get("agent_id"):
            continue
        records.append(
            {
                "agent_id": int(row["agent_id"]),
                "agent_wallet": norm_addr(row.get("agent_wallet")),
                "block_stamp": int(row.get("block_stamp") or 0),
                "tx_hash": norm_tx_hash(row.get("tx_hash")),
                "tx_type": normalize_tx_type(row.get("tx_type")),
            }
        )
    return records


def build_agent_statistics(
    identity_records: Sequence[Dict[str, object]],
    reputation_summaries: Sequence[Dict[str, object]],
    transaction_records: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    reputation_by_agent = {
        int(row["agent_id"]): row
        for row in reputation_summaries
    }

    tx_counts_by_agent: Dict[int, Dict[str, int]] = {}
    for row in transaction_records:
        agent_id = int(row["agent_id"])
        tx_type = normalize_tx_type(str(row.get("tx_type") or "other_operation"))
        counts = tx_counts_by_agent.setdefault(
            agent_id,
            {
                "tx_count": 0,
                "identity_operation_count": 0,
                "ecosystem_operation_count": 0,
                "other_operation_count": 0,
            },
        )
        counts["tx_count"] += 1
        counts[f"{tx_type}_count"] += 1

    rows: List[Dict[str, object]] = []
    for record in identity_records:
        agent_id = int(record["agent_id"])
        reputation = reputation_by_agent.get(agent_id, {})
        tx_counts = tx_counts_by_agent.get(
            agent_id,
            {
                "tx_count": 0,
                "identity_operation_count": 0,
                "ecosystem_operation_count": 0,
                "other_operation_count": 0,
            },
        )
        owner_wallet = norm_addr(record.get("owner_wallet")) or ""
        rows.append(
            {
                "agent_id": agent_id,
                "block_stamp": int(record.get("block_stamp") or 0),
                "owner_wallet": owner_wallet,
                "owner_agent_count": int(record.get("owner_agent_count") or 1),
                "agent_wallet": norm_addr(record.get("agent_wallet")) or "",
                "reputation": reputation.get("reputation", Decimal(0)),
                "feedback_count": int(reputation.get("feedback_count") or 0),
                "client_count": int(reputation.get("client_count") or 0),
                "agent_count": int(reputation.get("agent_count") or 0),
                "contract_count": int(reputation.get("contract_count") or 0),
                "eoa_count": int(reputation.get("eoa_count") or 0),
                "tx_count": tx_counts["tx_count"],
                "identity_operation_count": tx_counts["identity_operation_count"],
                "ecosystem_operation_count": tx_counts["ecosystem_operation_count"],
                "other_operation_count": tx_counts["other_operation_count"],
            }
        )
    return sorted(rows, key=lambda row: int(row["agent_id"]))


def run_pipeline(config: PipelineConfig) -> None:
    needs_rpc = config.rerun_all_agent or config.rerun_reputation or config.rerun_transaction
    latest_block: Optional[int] = None
    if needs_rpc:
        if not w3.is_connected():
            raise RuntimeError(f"Ethereum RPC is not reachable: {RPC_URL}")
        latest_block = int(w3.eth.block_number)
        if config.observation_block <= 0:
            config = replace(config, observation_block=latest_block)

    print("Connected:", bool(needs_rpc and latest_block is not None))
    print("Latest block:", latest_block if latest_block is not None else "not required")
    print("Start block:", config.start_block)
    print("Observation block:", config.observation_block)
    print("Output dir:", os.path.abspath(LOCAL_DATA_DIR))
    print(
        "Target agent ids:",
        f"{config.target_agent_id_min}-{config.target_agent_id_max}",
    )
    print("Target candidate count:", config.target_agent_count)
    print("Top agent count:", config.top_agent_count)
    print("Min selected client count:", config.min_selected_client_count)
    print("Rerun All_agent:", config.rerun_all_agent)
    print("Rerun reputation:", config.rerun_reputation)
    print("Rerun transaction:", config.rerun_transaction)

    failed_identity_agents: List[int] = []

    if config.rerun_all_agent:
        discovered_agents = discover_target_agents(config)
        if not discovered_agents:
            print("No agents discovered for the current configuration.")
            return

        identity_records: List[Dict[str, object]] = []
        for batch_number, agent_batch in enumerate(
            chunked(discovered_agents, config.pipeline_batch_size),
            start=1,
        ):
            batch_agent_ids = [int(agent["agent_id"]) for agent in agent_batch]
            print(f"[batch {batch_number}] agent_ids={batch_agent_ids[0]}-{batch_agent_ids[-1]}")
            batch_identity_records, identity_stats = run_identity_stage(agent_batch, config)
            print_stage_stats("identity", identity_stats)
            identity_records.extend(batch_identity_records)
            failed_identity_agents.extend(identity_stats["failed_agent_ids"])

        identity_records = sorted(identity_records, key=lambda row: int(row["agent_id"]))
        pre_records, _, preselection_stats = run_preselection_stage(identity_records, config)
        print_stage_stats("preselection", preselection_stats)
    else:
        pre_records = load_all_agent_records()
        if not pre_records:
            raise RuntimeError(
                "all_agent.csv is missing or empty. "
                "Set RERUN_ALL_AGENT=true to generate it first."
            )
        preselection_stats = {
            "success": len(pre_records),
            "failed": 0,
            "failed_agent_ids": [],
        }
        print(f"[preselection] loaded {len(pre_records)} cached rows from all_agent.csv")

    if config.rerun_reputation or config.rerun_transaction:
        selected_identity_records = select_identity_records_from_pre(pre_records, config)
        print(
            f"[preselection] selected {len(selected_identity_records)} agents "
            f"with client_count >= {config.min_selected_client_count}"
        )
        write_csv_table("agent_core", selected_identity_records)
    else:
        selected_identity_records = read_agent_core_records()
        print(
            f"[preselection] loaded {len(selected_identity_records)} cached rows "
            "from agent_core.csv"
        )

    if config.rerun_reputation:
        reputation_records, reputation_summaries, reputation_stats = run_reputation_stage(
            selected_identity_records,
            config,
        )
        print_stage_stats("reputation", reputation_stats)
    else:
        reputation_records, reputation_summaries = load_reputation_records_from_csv()
        reputation_stats = {
            "success": len({int(row["agent_id"]) for row in reputation_summaries}),
            "failed": 0,
            "failed_agent_ids": [],
        }
        print(
            f"[reputation] loaded {len(reputation_records)} cached feedback rows "
            "from agent_reputation.csv"
        )

    if config.rerun_transaction:
        transaction_records = run_transaction_stage(selected_identity_records, config)
        print(f"[transactions] wrote {len(transaction_records)} rows")
    else:
        transaction_records = load_transaction_records_from_csv()
        print(
            f"[transactions] loaded {len(transaction_records)} cached tx rows "
            "from agent_transaction.csv"
        )

    selected_identity_records = read_agent_core_records()
    reputation_records, reputation_summaries = load_reputation_records_from_csv()
    transaction_records = load_transaction_records_from_csv()
    statistic_records = build_agent_statistics(
        selected_identity_records,
        reputation_summaries,
        transaction_records,
    )
    write_csv_table("agent_statistic", statistic_records)

    final_failed_all = sorted(
        set(
            failed_identity_agents
            + preselection_stats["failed_agent_ids"]
            + reputation_stats["failed_agent_ids"]
        )
    )
    failed_agents_summary = {
        "identity": sorted(set(failed_identity_agents)),
        "preselection": sorted(set(preselection_stats["failed_agent_ids"])),
        "reputation": sorted(set(reputation_stats["failed_agent_ids"])),
        "all_failed": final_failed_all,
    }
    with open(os.path.join(PACKAGE_ROOT, "failed_agents_last_run.json"), "w", encoding="utf-8") as f:
        json.dump(failed_agents_summary, f, ensure_ascii=False, indent=2)

    if config.fail_on_incomplete_snapshot and final_failed_all:
        raise RuntimeError(
            "Pipeline finished with incomplete snapshot. "
            f"failed_agents={len(final_failed_all)}"
        )

    print("Pipeline completed.")
    print(f"Rows: all_agent={len(pre_records)}")
    print(f"Rows: agent_core={len(selected_identity_records)}")
    print(f"Rows: agent_reputation={len(reputation_records)}")
    print(f"Rows: agent_transaction={len(transaction_records)}")
    print(f"Rows: agent_statistic={len(statistic_records)}")


def main() -> None:
    run_pipeline(PipelineConfig())


if __name__ == "__main__":
    main()


