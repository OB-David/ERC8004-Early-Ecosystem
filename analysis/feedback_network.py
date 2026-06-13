#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build and visualize the ERC-8004 trust feedback network.

Inputs:
    ERC8004/data/agent_reputation.csv
    ERC8004/data/agent_core.csv
    ERC8004/data/all_agent.csv

Outputs:
    ERC8004/network/trust_nodes.csv
    ERC8004/network/trust_edges_raw.csv
    ERC8004/network/trust_edges_agg.csv
    ERC8004/network/trust_degree_distribution.csv
    ERC8004/network/trust_network_metrics.csv
    ERC8004/network/trust_network.pdf
    ERC8004/network/feedback_network_stats.pdf
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is not installed. Run:\n"
        "    pip install matplotlib"
    ) from exc

try:
    import networkx as nx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "networkx is not installed. Run:\n"
        "    pip install networkx"
    ) from exc


# =========================
# 0. Paths and parameters
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
NETWORK_DIR = REPO_ROOT / "network"

AGENT_REPUTATION_CSV = DATA_DIR / "agent_reputation.csv"
AGENT_CORE_CSV = DATA_DIR / "agent_core.csv"
ALL_AGENT_CSV = DATA_DIR / "all_agent.csv"

NODE_TYPES = ("agent", "owner", "contract", "eoa")
NODE_COLORS = {
    "agent": "#3B6EA8",
    "owner": "#C97B32",
    "contract": "#8B6BAE",
    "eoa": "#8A8F98",
}
NODE_MARKERS = {
    "agent": "o",
    "owner": "s",
    "contract": "^",
    "eoa": "o",
}
NODE_BASE_SIZES = {
    "agent": 46,
    "owner": 28,
    "contract": 30,
    "eoa": 11,
}
NODE_DEGREE_SCALES = {
    "agent": 8,
    "owner": 4,
    "contract": 4,
    "eoa": 1.5,
}
AGENT_FEEDBACK_EDGE_COLORS = [
    "#3B6EA8",
    "#5E88B8",
    "#7FA3C7",
    "#9BBAD4",
    "#C97B32",
]
AGENT_TO_AGENT_EDGE_COLOR = "#3B6EA8"
OWNER_AGENT_EDGE_COLOR = "#C97B32"
DEFAULT_EDGE_COLOR = "#6F7782"

FEEDBACK_STATS_PDF = NETWORK_DIR / "feedback_network_stats.pdf"

NODE_ORDER = ["eoa", "agent", "owner", "contract"]
NODE_LABELS = {
    "eoa": "EOA",
    "agent": "Agent",
    "owner": "Owner wallet",
    "contract": "Contract",
}
EDGE_ORDER = [
    "eoa -> agent",
    "owner -> agent",
    "agent -> agent",
    "contract -> agent",
]
EDGE_LABELS = {
    "eoa -> agent": "EOA -> Agent",
    "owner -> agent": "Owner -> Agent",
    "agent -> agent": "Agent -> Agent",
    "contract -> agent": "Contract -> Agent",
}

# =========================
# 1. CSV helpers
# =========================

def norm_addr(value: object) -> str:
    return str(value or "").strip().lower()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if math.isfinite(number):
        return number
    return default


def read_csv_rows(path: Path, required_fields: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [field for field in required_fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing fields: {missing}")
        return list(reader)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


# =========================
# 2. Data loading and node classification
# =========================

def agent_node_id(agent_id: int) -> str:
    return f"agent:{int(agent_id)}"


def address_node_id(address: str) -> str:
    return f"addr:{norm_addr(address)}"


def load_agent_core() -> Tuple[Dict[int, Dict[str, object]], Dict[str, List[int]]]:
    rows = read_csv_rows(AGENT_CORE_CSV, ["agent_id", "client_count", "owner_wallet", "agent_wallet"])
    records_by_agent_id: Dict[int, Dict[str, object]] = {}
    agent_ids_by_wallet: Dict[str, List[int]] = {}

    for row in rows:
        agent_id_raw = str(row.get("agent_id") or "").strip()
        if not agent_id_raw:
            continue

        agent_id = int(agent_id_raw)
        agent_wallet = norm_addr(row.get("agent_wallet"))
        owner_wallet = norm_addr(row.get("owner_wallet"))
        record = {
            "agent_id": agent_id,
            "client_count": int(row.get("client_count") or 0),
            "owner_wallet": owner_wallet,
            "agent_wallet": agent_wallet,
        }
        records_by_agent_id[agent_id] = record
        if agent_wallet:
            agent_ids_by_wallet.setdefault(agent_wallet, []).append(agent_id)

    return records_by_agent_id, agent_ids_by_wallet


def load_owner_wallets() -> set[str]:
    rows = read_csv_rows(ALL_AGENT_CSV, ["owner_wallet"])
    return {
        owner_wallet
        for owner_wallet in (norm_addr(row.get("owner_wallet")) for row in rows)
        if owner_wallet
    }


def count_feedback_by_agent(reputation_rows: Sequence[Dict[str, str]]) -> Counter:
    counts: Counter = Counter()
    for row in reputation_rows:
        agent_id_raw = str(row.get("agent_id") or "").strip()
        if agent_id_raw:
            counts[int(agent_id_raw)] += 1
    return counts


def choose_kept_agents(
    agent_records: Dict[int, Dict[str, object]],
    agent_ids_by_wallet: Dict[str, List[int]],
    feedback_count_by_agent: Counter,
) -> Tuple[set[int], Dict[str, int], Dict[int, str]]:
    kept_agent_ids = set(agent_records)
    agent_id_by_wallet: Dict[str, int] = {}
    removed_reason_by_agent: Dict[int, str] = {}

    for wallet, agent_ids in agent_ids_by_wallet.items():
        if len(agent_ids) == 1:
            agent_id_by_wallet[wallet] = agent_ids[0]
            continue

        keep_agent_id = sorted(
            agent_ids,
            key=lambda agent_id: (
                -int(feedback_count_by_agent.get(agent_id, 0)),
                -int(agent_records[agent_id].get("client_count") or 0),
                agent_id,
            ),
        )[0]
        agent_id_by_wallet[wallet] = keep_agent_id
        for agent_id in agent_ids:
            if agent_id == keep_agent_id:
                continue
            kept_agent_ids.discard(agent_id)
            removed_reason_by_agent[agent_id] = (
                f"shared_agent_wallet:{wallet}:kept_agent_id={keep_agent_id}"
            )

    return kept_agent_ids, agent_id_by_wallet, removed_reason_by_agent


def classify_address_node(
    address: str,
    owner_wallets: set[str],
    observed_client_types: Dict[str, str],
) -> str:
    """Classify non-agent wallets with owner wallets taking priority."""
    address = norm_addr(address)
    if address in owner_wallets:
        return "owner"

    observed_type = str(observed_client_types.get(address) or "").strip().lower()
    if observed_type == "contract":
        return "contract"
    return "eoa"


# =========================
# 3. Network construction
# =========================

def build_trust_edges(
    reputation_rows: Sequence[Dict[str, str]],
    agent_records: Dict[int, Dict[str, object]],
    kept_agent_ids: set[int],
    agent_id_by_wallet: Dict[str, int],
) -> Tuple[List[Dict[str, object]], Dict[str, str], Counter]:
    raw_edges: List[Dict[str, object]] = []
    observed_client_types: Dict[str, str] = {}
    skip_reasons: Counter = Counter()

    for row in reputation_rows:
        agent_id_raw = str(row.get("agent_id") or "").strip()
        if not agent_id_raw:
            skip_reasons["missing_agent_id"] += 1
            continue

        target_agent_id = int(agent_id_raw)
        target_record = agent_records.get(target_agent_id)
        if target_record is None:
            skip_reasons["agent_not_in_agent_core"] += 1
            continue
        if target_agent_id not in kept_agent_ids:
            skip_reasons["removed_shared_wallet_agent"] += 1
            continue

        source_address = norm_addr(row.get("feedback_client"))
        if not source_address:
            skip_reasons["missing_feedback_client"] += 1
            continue

        client_type = str(row.get("feedback_client_type") or "").strip().lower()
        if client_type:
            observed_client_types[source_address] = client_type

        source_agent_id = agent_id_by_wallet.get(source_address)
        if source_agent_id == target_agent_id:
            skip_reasons["self_feedback"] += 1
            continue

        if source_agent_id is not None and source_agent_id in kept_agent_ids:
            source = agent_node_id(source_agent_id)
            source_type = "agent"
            source_agent_wallet = source_address
            source_owner_wallet = agent_records[source_agent_id].get("owner_wallet") or ""
        else:
            source = address_node_id(source_address)
            source_type = ""
            source_agent_wallet = ""
            source_owner_wallet = ""

        target = agent_node_id(target_agent_id)
        weight = safe_float(row.get("feedback_value"), default=0.0)
        raw_edges.append(
            {
                "source": source,
                "target": target,
                "source_address": source_address,
                "target_address": target_record.get("agent_wallet") or "",
                "source_agent_id": source_agent_id if source_agent_id is not None else "",
                "target_agent_id": target_agent_id,
                "source_agent_wallet": source_agent_wallet,
                "target_agent_wallet": target_record.get("agent_wallet") or "",
                "source_owner_wallet": source_owner_wallet,
                "target_owner_wallet": target_record.get("owner_wallet") or "",
                "source_type": source_type,
                "target_type": "agent",
                "weight": weight,
                "agent_id": target_agent_id,
                "feedback_tx": norm_addr(row.get("feedback_tx")),
                "feedback_type": row.get("feedback_type") or "",
                "raw_feedback_client_type": client_type,
            }
        )

    return raw_edges, observed_client_types, skip_reasons


def build_node_metadata(
    agent_records: Dict[int, Dict[str, object]],
    kept_agent_ids: set[int],
    raw_edges: Sequence[Dict[str, object]],
    owner_wallets: set[str],
    observed_client_types: Dict[str, str],
) -> Dict[str, Dict[str, object]]:
    nodes: Dict[str, Dict[str, object]] = {}

    for agent_id in sorted(kept_agent_ids):
        record = agent_records[agent_id]
        node = agent_node_id(agent_id)
        nodes[node] = {
            "node_id": node,
            "node_type": "agent",
            "agent_id": agent_id,
            "address": record.get("agent_wallet") or "",
            "agent_wallet": record.get("agent_wallet") or "",
            "owner_wallet": record.get("owner_wallet") or "",
            "client_count": record.get("client_count") or 0,
        }

    for edge in raw_edges:
        source = str(edge["source"])
        if source in nodes:
            continue
        source_address = norm_addr(edge.get("source_address"))
        node_type = classify_address_node(source_address, owner_wallets, observed_client_types)
        nodes[source] = {
            "node_id": source,
            "node_type": node_type,
            "agent_id": "",
            "address": source_address,
            "agent_wallet": "",
            "owner_wallet": source_address if node_type == "owner" else "",
            "client_count": "",
        }

    return nodes


def build_graphs(
    raw_edges: Sequence[Dict[str, object]],
    node_metadata: Dict[str, Dict[str, object]],
) -> Tuple[nx.MultiDiGraph, nx.DiGraph, List[Dict[str, object]]]:
    multi_graph = nx.MultiDiGraph()
    agg_graph = nx.DiGraph()
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}

    for node, metadata in node_metadata.items():
        multi_graph.add_node(node, **metadata)
        agg_graph.add_node(node, **metadata)

    for index, edge in enumerate(raw_edges):
        source = str(edge["source"])
        target = str(edge["target"])
        weight = float(edge["weight"])
        multi_graph.add_edge(
            source,
            target,
            key=index,
            weight=weight,
            target_agent_id=edge["target_agent_id"],
            feedback_tx=edge["feedback_tx"],
            feedback_type=edge["feedback_type"],
        )

        key = (source, target)
        item = grouped.setdefault(
            key,
            {
                "source": source,
                "target": target,
                "source_address": edge.get("source_address") or "",
                "target_address": edge.get("target_address") or "",
                "source_agent_ids": set(),
                "target_agent_ids": set(),
                "weight_sum": 0.0,
                "feedback_count": 0,
                "feedback_types": Counter(),
            },
        )
        item["weight_sum"] = float(item["weight_sum"]) + weight
        item["feedback_count"] = int(item["feedback_count"]) + 1
        if edge.get("source_agent_id") != "":
            item["source_agent_ids"].add(int(edge["source_agent_id"]))
        item["target_agent_ids"].add(int(edge["target_agent_id"]))
        if edge.get("feedback_type"):
            item["feedback_types"][str(edge["feedback_type"])] += 1

    agg_edges: List[Dict[str, object]] = []
    for item in grouped.values():
        feedback_count = int(item["feedback_count"])
        weight_mean = float(item["weight_sum"]) / feedback_count if feedback_count else 0.0
        feedback_types = item["feedback_types"].most_common()
        source = str(item["source"])
        target = str(item["target"])
        source_type = str(node_metadata.get(source, {}).get("node_type") or "eoa")
        target_type = str(node_metadata.get(target, {}).get("node_type") or "eoa")

        agg_graph.add_edge(
            source,
            target,
            weight=weight_mean,
            weight_sum=float(item["weight_sum"]),
            feedback_count=feedback_count,
        )
        agg_edges.append(
            {
                "source": source,
                "target": target,
                "source_type": source_type,
                "target_type": target_type,
                "source_address": item.get("source_address") or "",
                "target_address": item.get("target_address") or "",
                "source_agent_ids": ";".join(str(agent_id) for agent_id in sorted(item["source_agent_ids"])),
                "target_agent_ids": ";".join(str(agent_id) for agent_id in sorted(item["target_agent_ids"])),
                "weight_mean": round(weight_mean, 6),
                "weight_sum": round(float(item["weight_sum"]), 6),
                "feedback_count": feedback_count,
                "top_feedback_types": ";".join(f"{name}:{count}" for name, count in feedback_types[:5]),
            }
        )

    agg_edges.sort(key=lambda row: (-int(row["feedback_count"]), -float(row["weight_mean"]), row["source"], row["target"]))
    return multi_graph, agg_graph, agg_edges


def build_node_rows(
    node_metadata: Dict[str, Dict[str, object]],
    multi_graph: nx.MultiDiGraph,
    agg_graph: nx.DiGraph,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for node in sorted(node_metadata):
        metadata = node_metadata[node]
        rows.append(
            {
                "node_id": node,
                "node_type": metadata.get("node_type") or "",
                "agent_id": metadata.get("agent_id") or "",
                "address": metadata.get("address") or "",
                "agent_wallet": metadata.get("agent_wallet") or "",
                "owner_wallet": metadata.get("owner_wallet") or "",
                "client_count": metadata.get("client_count") or "",
                "raw_in_degree": multi_graph.in_degree(node),
                "raw_out_degree": multi_graph.out_degree(node),
                "raw_total_degree": multi_graph.degree(node),
                "agg_in_degree": agg_graph.in_degree(node),
                "agg_out_degree": agg_graph.out_degree(node),
                "agg_total_degree": agg_graph.degree(node),
            }
        )
    return rows


# =========================
# 4. Metrics
# =========================

def degree_distribution_rows(graph: nx.Graph) -> List[Dict[str, object]]:
    distribution: Counter = Counter(dict(graph.degree()).values())
    return [
        {
            "degree": degree,
            "node_count": count,
        }
        for degree, count in sorted(distribution.items())
    ]


def calculate_metrics(
    multi_graph: nx.MultiDiGraph,
    agg_graph: nx.DiGraph,
    node_types: Dict[str, str],
) -> List[Dict[str, object]]:
    node_type_counts = Counter(node_types.values())
    weak_components = list(nx.weakly_connected_components(agg_graph))
    largest_component_size = max((len(component) for component in weak_components), default=0)

    undirected = agg_graph.to_undirected()
    if undirected.number_of_nodes() > 1:
        avg_clustering_unweighted = nx.average_clustering(undirected)
        avg_clustering_weighted = nx.average_clustering(undirected, weight="weight")
    else:
        avg_clustering_unweighted = 0.0
        avg_clustering_weighted = 0.0

    metrics = [
        ("node_count", agg_graph.number_of_nodes()),
        ("raw_edge_count", multi_graph.number_of_edges()),
        ("aggregated_edge_count", agg_graph.number_of_edges()),
        ("weak_component_count", len(weak_components)),
        ("largest_weak_component_size", largest_component_size),
        ("average_raw_total_degree", mean(dict(multi_graph.degree()).values())),
        ("average_agg_total_degree", mean(dict(agg_graph.degree()).values())),
        ("density_aggregated_directed", nx.density(agg_graph)),
        ("average_clustering_undirected_unweighted", avg_clustering_unweighted),
        ("average_clustering_undirected_weighted", avg_clustering_weighted),
    ]
    metrics.extend((f"node_type_{node_type}_count", node_type_counts.get(node_type, 0)) for node_type in NODE_TYPES)
    return [{"metric": name, "value": value} for name, value in metrics]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


# =========================
# 5. Visualization
# =========================

def choose_visual_nodes(graph: nx.MultiDiGraph, max_nodes: int) -> List[str]:
    if graph.number_of_nodes() <= max_nodes:
        return list(graph.nodes())

    degree_by_node = dict(graph.degree())
    agent_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "agent"]
    other_nodes = [node for node in graph.nodes() if node not in set(agent_nodes)]
    agent_nodes = sorted(agent_nodes, key=lambda node: (-degree_by_node.get(node, 0), str(node)))
    other_nodes = sorted(other_nodes, key=lambda node: (-degree_by_node.get(node, 0), str(node)))

    if len(agent_nodes) >= max_nodes:
        return agent_nodes[:max_nodes]
    return agent_nodes + other_nodes[: max_nodes - len(agent_nodes)]


def community_cluster_layout(graph: nx.DiGraph, node_types: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    undirected = graph.to_undirected()
    if undirected.number_of_nodes() == 0:
        return {}

    if undirected.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight=None))
    else:
        groups: Dict[str, set[str]] = {node_type: set() for node_type in NODE_TYPES}
        for node in undirected.nodes():
            groups.setdefault(node_types.get(node, "eoa"), set()).add(node)
        communities = [nodes for nodes in groups.values() if nodes]

    communities = sorted((set(nodes) for nodes in communities), key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
    community_count = len(communities)
    center_radius = max(1, 0.24 * math.sqrt(max(community_count, 1)))
    positions: Dict[str, Tuple[float, float]] = {}

    for index, nodes in enumerate(communities):
        angle = 2.0 * math.pi * index / max(community_count, 1)
        center_x = center_radius * math.cos(angle)
        center_y = center_radius * math.sin(angle)
        subgraph = undirected.subgraph(nodes).copy()

        if len(nodes) == 1:
            node = next(iter(nodes))
            positions[node] = (center_x, center_y)
            continue

        cluster_scale = 0.20 + 0.062 * math.sqrt(len(nodes))
        local_pos = nx.spring_layout(
            subgraph,
            seed=42 + index,
            k=1.55 / math.sqrt(len(nodes)),
            iterations=260,
            weight=None,
            scale=cluster_scale,
        )
        for node, (x, y) in local_pos.items():
            positions[node] = (center_x + float(x), center_y + float(y))

    return positions


def draw_network_page(
    pdf: PdfPages,
    multi_graph: nx.MultiDiGraph,
    agg_graph: nx.DiGraph,
    node_types: Dict[str, str],
    max_nodes: int,
) -> None:
    visual_nodes = choose_visual_nodes(multi_graph, max_nodes)
    raw_subgraph = multi_graph.subgraph(visual_nodes).copy()
    layout_subgraph = agg_graph.subgraph(visual_nodes).copy()

    fig, ax = plt.subplots(figsize=(10.5, 10.5))
    ax.axis("off")

    if raw_subgraph.number_of_nodes() == 0:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return

    pos = community_cluster_layout(layout_subgraph, node_types)

    pair_seen: Counter = Counter()
    max_parallel_offset = 4
    for source, target, _key in raw_subgraph.edges(keys=True):
        pair_seen[(source, target)] += 1
        offset = min(pair_seen[(source, target)] - 1, max_parallel_offset)
        direction = -1 if offset % 2 else 1
        rad = direction * (0.025 + 0.018 * (offset // 2))
        nx.draw_networkx_edges(
            raw_subgraph,
            pos,
            edgelist=[(source, target)],
            ax=ax,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=6.0,
            width=0.42 if node_types.get(source) != "eoa" else 0.34,
            alpha=0.75 if node_types.get(source) != "eoa" else 0.24,
            edge_color=(
                AGENT_TO_AGENT_EDGE_COLOR
                if node_types.get(source) == "agent" and node_types.get(target) == "agent"
                else OWNER_AGENT_EDGE_COLOR
                if {node_types.get(source), node_types.get(target)} == {"agent", "owner"}
                else DEFAULT_EDGE_COLOR
            ),
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=2.0,
            min_target_margin=2.0,
        )

    degree_by_node = dict(raw_subgraph.degree())
    for node_type in NODE_TYPES:
        nodes = [node for node in raw_subgraph.nodes() if node_types.get(node, "eoa") == node_type]
        if not nodes:
            continue
        sizes = [
            NODE_BASE_SIZES[node_type] + NODE_DEGREE_SCALES[node_type] * math.sqrt(max(degree_by_node.get(node, 0), 1))
            for node in nodes
        ]
        nx.draw_networkx_nodes(
            raw_subgraph,
            pos,
            nodelist=nodes,
            node_size=sizes,
            node_color=NODE_COLORS[node_type],
            node_shape=NODE_MARKERS[node_type],
            edgecolors="white" if node_type != "eoa" else "none",
            linewidths=0.45 if node_type != "eoa" else 0.0,
            alpha=0.9 if node_type != "eoa" else 0.58,
            ax=ax,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=NODE_MARKERS[node_type],
            color="none",
            markerfacecolor=NODE_COLORS[node_type],
            markeredgecolor="white" if node_type != "eoa" else "none",
            markersize=6.5 if node_type != "eoa" else 4,
            label={"agent": "Agent wallet", "owner": "Owner wallet", "contract": "Contract", "eoa": "EOA"}[node_type],
        )
        for node_type in NODE_TYPES
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=AGENT_TO_AGENT_EDGE_COLOR,
            linewidth=1.4,
            label="Agent-to-agent feedback",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=OWNER_AGENT_EDGE_COLOR,
            linewidth=1.4,
            label="Owner wallet-to-agent feedback",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_pdf(
    path: Path,
    multi_graph: nx.MultiDiGraph,
    agg_graph: nx.DiGraph,
    node_types: Dict[str, str],
    max_nodes: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        draw_network_page(pdf, multi_graph, agg_graph, node_types, max_nodes)


def feedback_edge_type_counts(edge_rows: Sequence[Dict[str, object]]) -> Counter:
    counts: Counter = Counter()
    for row in edge_rows:
        source_type = str(row.get("source_type") or "eoa").strip().lower()
        target_type = str(row.get("target_type") or "eoa").strip().lower()
        counts[f"{source_type} -> {target_type}"] += 1
    return counts


def annotate_bars(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def write_feedback_network_stats_figure(
    node_rows: Sequence[Dict[str, object]],
    raw_edge_rows: Sequence[Dict[str, object]],
    agg_edge_rows: Sequence[Dict[str, object]],
) -> None:
    node_counts = Counter(str(row.get("node_type") or "eoa").lower() for row in node_rows)
    raw_edge_counts = feedback_edge_type_counts(raw_edge_rows)
    agg_edge_counts = feedback_edge_type_counts(agg_edge_rows)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "hatch.linewidth": 1.1,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    node_values = [node_counts.get(key, 0) for key in NODE_ORDER]
    node_labels = [NODE_LABELS[key] for key in NODE_ORDER]
    node_colors = [NODE_COLORS[key] for key in NODE_ORDER]
    bars = axes[0].bar(node_labels, node_values, color=node_colors)
    axes[0].set_title("Node Composition")
    axes[0].set_ylabel("Number of nodes")
    axes[0].set_ylim(0, max(node_values) * 1.16 if node_values else 1)
    axes[0].tick_params(axis="x", rotation=20)
    annotate_bars(axes[0], bars)

    edge_labels = [EDGE_LABELS[key] for key in EDGE_ORDER]
    raw_values = [raw_edge_counts.get(key, 0) for key in EDGE_ORDER]
    agg_values = [agg_edge_counts.get(key, 0) for key in EDGE_ORDER]
    x_positions = list(range(len(EDGE_ORDER)))
    width = 0.36
    edge_colors = [NODE_COLORS[key.split(" -> ")[0]] for key in EDGE_ORDER]

    raw_bars = axes[1].bar(
        [x - width / 2 for x in x_positions],
        raw_values,
        width=width,
        color="#F1F3F5",
        edgecolor=edge_colors,
        linewidth=1.2,
        hatch="////",
    )
    agg_bars = axes[1].bar(
        [x + width / 2 for x in x_positions],
        agg_values,
        width=width,
        color=edge_colors,
        alpha=0.9,
        edgecolor=edge_colors,
        linewidth=0.8,
    )
    axes[1].set_title("Feedback Edge Composition")
    axes[1].set_ylabel("Number of edges")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(edge_labels, rotation=25, ha="right")
    axes[1].set_ylim(0, max(raw_values + agg_values) * 1.18 if raw_values or agg_values else 1)
    axes[1].legend(
        handles=[
            Patch(facecolor="#D7DCE2", edgecolor=DEFAULT_EDGE_COLOR, hatch="////", linewidth=1.2, label="Raw edges"),
            Patch(facecolor=DEFAULT_EDGE_COLOR, edgecolor=DEFAULT_EDGE_COLOR, alpha=0.9, label="Aggregated edges"),
        ],
        frameon=False,
    )
    annotate_bars(axes[1], raw_bars)
    annotate_bars(axes[1], agg_bars)

    fig.suptitle("ERC-8004 Feedback Network Statistics", y=1.03, fontsize=13)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FEEDBACK_STATS_PDF, bbox_inches="tight")
    plt.close(fig)

# =========================
# 6. Main flow
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ERC-8004 trust feedback network and render a PDF.")
    parser.add_argument(
        "--max-visual-nodes",
        type=int,
        default=1000,
        help="Maximum nodes shown in the PDF graph. CSV outputs still include the full graph. Default: 1000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)

    agent_records, agent_ids_by_wallet = load_agent_core()
    owner_wallets = load_owner_wallets()
    reputation_rows = read_csv_rows(
        AGENT_REPUTATION_CSV,
        ["agent_id", "feedback_client", "feedback_client_type", "feedback_type", "feedback_value"],
    )
    feedback_count_by_agent = count_feedback_by_agent(reputation_rows)
    kept_agent_ids, agent_id_by_wallet, removed_reason_by_agent = choose_kept_agents(
        agent_records,
        agent_ids_by_wallet,
        feedback_count_by_agent,
    )

    raw_edges, observed_client_types, skip_reasons = build_trust_edges(
        reputation_rows,
        agent_records,
        kept_agent_ids,
        agent_id_by_wallet,
    )
    if removed_reason_by_agent:
        skip_reasons["removed_shared_wallet_agent_nodes"] = len(removed_reason_by_agent)

    node_metadata = build_node_metadata(
        agent_records,
        kept_agent_ids,
        raw_edges,
        owner_wallets,
        observed_client_types,
    )
    node_types = {
        node: str(metadata.get("node_type") or "eoa")
        for node, metadata in node_metadata.items()
    }

    multi_graph, agg_graph, agg_edges = build_graphs(raw_edges, node_metadata)
    node_rows = build_node_rows(node_metadata, multi_graph, agg_graph)
    metrics = calculate_metrics(multi_graph, agg_graph, node_types)
    degree_rows = degree_distribution_rows(agg_graph)

    raw_edge_rows = [
        {
            **edge,
            "source_type": node_types.get(str(edge["source"]), "eoa"),
            "target_type": node_types.get(str(edge["target"]), "eoa"),
        }
        for edge in raw_edges
    ]

    write_csv(
        NETWORK_DIR / "trust_nodes.csv",
        [
            "node_id",
            "node_type",
            "agent_id",
            "address",
            "agent_wallet",
            "owner_wallet",
            "client_count",
            "raw_in_degree",
            "raw_out_degree",
            "raw_total_degree",
            "agg_in_degree",
            "agg_out_degree",
            "agg_total_degree",
        ],
        node_rows,
    )
    write_csv(
        NETWORK_DIR / "trust_edges_raw.csv",
        [
            "source",
            "target",
            "source_type",
            "target_type",
            "source_address",
            "target_address",
            "source_agent_id",
            "target_agent_id",
            "source_agent_wallet",
            "target_agent_wallet",
            "source_owner_wallet",
            "target_owner_wallet",
            "weight",
            "agent_id",
            "feedback_tx",
            "feedback_type",
            "raw_feedback_client_type",
        ],
        raw_edge_rows,
    )
    write_csv(
        NETWORK_DIR / "trust_edges_agg.csv",
        [
            "source",
            "target",
            "source_type",
            "target_type",
            "source_address",
            "target_address",
            "source_agent_ids",
            "target_agent_ids",
            "weight_mean",
            "weight_sum",
            "feedback_count",
            "top_feedback_types",
        ],
        agg_edges,
    )
    write_csv(NETWORK_DIR / "trust_degree_distribution.csv", ["degree", "node_count"], degree_rows)
    write_csv(NETWORK_DIR / "trust_network_metrics.csv", ["metric", "value"], metrics)
    write_pdf(NETWORK_DIR / "trust_network.pdf", multi_graph, agg_graph, node_types, max(1, int(args.max_visual_nodes)))
    write_feedback_network_stats_figure(node_rows, raw_edge_rows, agg_edges)

    print(f"[done] trust network outputs saved to {NETWORK_DIR}")
    print(
        "[nodes] "
        f"agent_core={len(agent_records)} kept_agents={len(kept_agent_ids)} "
        f"removed_shared_wallet_agents={len(removed_reason_by_agent)}"
    )
    if skip_reasons:
        print("[note] skipped reputation rows:", dict(skip_reasons))

if __name__ == "__main__":
    main()


