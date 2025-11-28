"""
Dynamic coalition formation demo tailored to a caching collaboration setting.

Each leader has several subordinate cache nodes. A coalition's value is driven by
how much demand its members can satisfy locally minus coordination cost. Leaders
can merge coalitions or migrate between them if the total system welfare improves.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, TypedDict, cast

MPL_CONFIG_DIR = Path(__file__).resolve().parent / ".mplconfig"
try:
    MPL_CONFIG_DIR.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
except OSError:
    # If the directory cannot be created, fall back to defaults.
    pass

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib may be missing
    plt = None


@dataclass(frozen=True)
class Server:
    name: str
    cache_capacity: float
    op_cost: float = 0.2


@dataclass(frozen=True)
class Vehicle:
    name: str
    demand: float
    cache_capacity: float = 0.0
    coop_cost: float = 0.05


@dataclass(frozen=True)
class Leader:
    name: str
    sub_members: int
    cache_capacity: float
    demand: float
    link_cost: float = 1.0
    servers: Tuple[Server, ...] = field(default_factory=tuple)
    vehicles: Tuple[Vehicle, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Make sure we stay hashable inside sets.
        object.__setattr__(self, "servers", tuple(self.servers))
        object.__setattr__(self, "vehicles", tuple(self.vehicles))


Coalition = Set[Leader]
Partition = List[Coalition]
ValueFn = Callable[[Iterable[Leader]], float]
LeaderRequests = Dict[str, float]
VehicleRequests = Dict[str, Dict[str, float]]
UnitAllocations = Dict[str, Dict[str, Dict[str, float]]]


class Requests(TypedDict, total=False):
    leader: LeaderRequests
    vehicle: VehicleRequests


class SliceResult(TypedDict):
    requests: Requests
    welfare: float
    coalition_sizes: Dict[str, int]
    leader_payoffs: Dict[str, float]


@dataclass
class TEEReport:
    timestamp: float
    payload_hash: str
    signature: str


class TrustedEnclave:
    """
    Lightweight mock TEE: signs payload hashes with a secret key (HMAC).
    In生产中可替换为真TEE的远程证明。
    """

    def __init__(self, key: bytes | None = None) -> None:
        self._key = key or secrets.token_bytes(32)

    def sign(self, payload: dict) -> TEEReport:
        payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        signature = hashlib.sha256(self._key + payload_hash.encode()).hexdigest()
        return TEEReport(timestamp=time.time(), payload_hash=payload_hash, signature=signature)

    def verify(self, payload: dict, report: TEEReport) -> bool:
        expect_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if expect_hash != report.payload_hash:
            return False
        expect_sig = hashlib.sha256(self._key + report.payload_hash.encode()).hexdigest()
        return expect_sig == report.signature


class TrustedLedger:
    """
    Simple append-only ledger; stores signed entries in memory and optionally flushes to disk.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._entries: List[dict] = []
        self._path = path

    def append(self, entry: dict) -> None:
        self._entries.append(entry)
        if self._path:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @property
    def entries(self) -> List[dict]:
        return self._entries


def _leader_request_delta(leader: Leader, requests: Requests) -> float:
    leader_reqs = cast(LeaderRequests, requests.get("leader", {}))
    return leader_reqs.get(leader.name, 0.0)


def _vehicle_request_delta(leader: Leader, vehicle: Vehicle, requests: Requests) -> float:
    vehicle_reqs = cast(VehicleRequests, requests.get("vehicle", {}))
    return vehicle_reqs.get(leader.name, {}).get(vehicle.name, 0.0)


def cache_value_with_requests(coalition: Iterable[Leader], requests: Requests) -> float:
    """
    Value function for a caching alliance with per-period request updates.

    - Benefit: served traffic equals min(total cache capacity, total demand).
    - Cooperation bonus: more subordinate nodes (servers + vehicles) increase locality.
    - Cost: per-leader link cost, per-server op cost, vehicle coop cost, plus light
      coordination overhead on capacity.
    """
    leaders = list(coalition)
    if not leaders:
        return 0.0

    total_leader_demand = sum(l.demand + _leader_request_delta(l, requests) for l in leaders)
    total_vehicle_demand = sum(
        v.demand + _vehicle_request_delta(l, v, requests) for l in leaders for v in l.vehicles
    )
    total_server_capacity = sum(s.cache_capacity for l in leaders for s in l.servers)
    total_vehicle_capacity = sum(v.cache_capacity for l in leaders for v in l.vehicles)
    total_capacity = sum(l.cache_capacity for l in leaders) + total_server_capacity + total_vehicle_capacity
    served = min(total_capacity, total_leader_demand + total_vehicle_demand)

    total_subordinates = sum(l.sub_members + len(l.servers) + len(l.vehicles) for l in leaders)
    cooperation_bonus = 0.12 * served * (1 + total_subordinates / max(1, 8 * len(leaders)))
    link_cost = sum(l.link_cost for l in leaders)
    server_cost = sum(s.op_cost for l in leaders for s in l.servers)
    vehicle_cost = sum(v.coop_cost for l in leaders for v in l.vehicles)
    coordination_cost = 0.05 * total_capacity
    return served + cooperation_bonus - link_cost - server_cost - vehicle_cost - coordination_cost


def cache_value(coalition: Iterable[Leader]) -> float:
    """
    Backwards-compatible value when no per-period requests are specified.
    """
    return cache_value_with_requests(coalition, {"leader": {}, "vehicle": {}})


def welfare(partition: Partition, value_fn: ValueFn) -> float:
    return sum(value_fn(coalition) for coalition in partition)


def evaluate_time_slice(leaders: Sequence[Leader], requests: Requests) -> tuple[Partition, SliceResult, UnitAllocations]:
    """
    Run coalition formation for a single time slice and gather metrics and allocations.

    Returns:
        partition: coalition structure after convergence.
        result: per-slice summary (welfare, coalition sizes, leader payoffs).
        unit_allocations: second-level allocation for each leader -> servers/vehicles.
    """

    def value_fn(coalition: Iterable[Leader]) -> float:
        return cache_value_with_requests(coalition, requests)

    partition = dynamic_coalition_formation(leaders, value_fn=value_fn, epsilon=1e-3)
    leader_payoffs: Dict[str, float] = {}
    coalition_sizes: Dict[str, int] = {}
    unit_allocations: UnitAllocations = {}

    for coalition in partition:
        payoffs = allocate_payoffs(coalition, value_fn)
        for leader in coalition:
            coalition_sizes[leader.name] = len(coalition)
            leader_share = payoffs[leader.name]
            leader_payoffs[leader.name] = leader_share
            unit_allocations[leader.name] = distribute_to_units(leader_share, leader, requests)

    result: SliceResult = {
        "requests": requests,
        "welfare": welfare(partition, value_fn),
        "coalition_sizes": coalition_sizes,
        "leader_payoffs": leader_payoffs,
    }
    return partition, result, unit_allocations


def simulate_time_slices(
    leaders: Sequence[Leader], time_slices: List[Requests]
) -> tuple[List[SliceResult], List[UnitAllocations], List[Partition]]:
    results: List[SliceResult] = []
    unit_allocations_all: List[Dict[str, Dict[str, Dict[str, float]]]] = []
    partitions: List[Partition] = []
    for requests in time_slices:
        partition, summary, unit_allocations = evaluate_time_slice(leaders, requests)
        results.append(summary)
        unit_allocations_all.append(unit_allocations)
        partitions.append(partition)
    return results, unit_allocations_all, partitions


def plot_results(
    results: List[SliceResult],
    filename_prefix: str = "coalition",
) -> List[str]:
    """
    Plot welfare and leader payoffs over time. Returns the list of saved file names.
    """
    if plt is None:  # matplotlib unavailable
        print("matplotlib 未安装，无法绘图。")
        return []

    periods = list(range(1, len(results) + 1))
    welfare_values = [r["welfare"] for r in results]

    # Collect all leader names across slices to keep lines consistent.
    leader_names_set: Set[str] = set()
    for r in results:
        leader_names_set.update(r["leader_payoffs"].keys())
    leader_names: List[str] = sorted(leader_names_set)

    leader_payoff_series = {
        name: [r["leader_payoffs"].get(name, 0.0) for r in results] for name in leader_names
    }
    coalition_size_series = {
        name: [r["coalition_sizes"].get(name, 0) for r in results] for name in leader_names
    }

    saved_files: List[str] = []

    plt.figure(figsize=(6, 4))
    plt.plot(periods, welfare_values, marker="o")
    plt.title("系统总福利随时间变化")
    plt.xlabel("时间段")
    plt.ylabel("福利")
    plt.grid(True, linestyle="--", alpha=0.5)
    welfare_file = f"{filename_prefix}_welfare.png"
    plt.tight_layout()
    plt.savefig(welfare_file, dpi=200)
    saved_files.append(welfare_file)
    plt.close()

    plt.figure(figsize=(7, 4))
    for name in leader_names:
        plt.plot(periods, leader_payoff_series[name], marker="o", label=name)
    plt.title("领导收益随时间变化")
    plt.xlabel("时间段")
    plt.ylabel("收益")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    payoff_file = f"{filename_prefix}_leader_payoffs.png"
    plt.tight_layout()
    plt.savefig(payoff_file, dpi=200)
    saved_files.append(payoff_file)
    plt.close()

    plt.figure(figsize=(7, 4))
    for name in leader_names:
        plt.step(periods, coalition_size_series[name], where="mid", label=name)
    plt.title("每位领导所在联盟规模")
    plt.xlabel("时间段")
    plt.ylabel("联盟成员数量")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    size_file = f"{filename_prefix}_coalition_sizes.png"
    plt.tight_layout()
    plt.savefig(size_file, dpi=200)
    saved_files.append(size_file)
    plt.close()

    return saved_files


def _best_merge(partition: Partition, value_fn: ValueFn, epsilon: float) -> bool:
    best_delta = 0.0
    best_pair: tuple[int, int] | None = None
    for i, j in combinations(range(len(partition)), 2):
        merged_value = value_fn(partition[i] | partition[j])
        delta = merged_value - (value_fn(partition[i]) + value_fn(partition[j]))
        if delta > best_delta:
            best_delta = delta
            best_pair = (i, j)

    if best_pair and best_delta > epsilon:
        i, j = best_pair
        partition[i] |= partition[j]
        partition.pop(j)
        return True
    return False


def _best_move(partition: Partition, value_fn: ValueFn, epsilon: float) -> bool:
    for src_idx, coalition in enumerate(partition):
        for leader in list(coalition):
            for dst_idx, dst in enumerate(partition):
                if src_idx == dst_idx:
                    continue
                new_src = coalition - {leader}
                new_dst = dst | {leader}
                delta = (
                    value_fn(new_src)
                    + value_fn(new_dst)
                    - value_fn(coalition)
                    - value_fn(dst)
                )
                if delta > epsilon:
                    coalition.remove(leader)
                    dst.add(leader)
                    if not coalition:
                        partition.pop(src_idx)
                    return True
    return False


def dynamic_coalition_formation(
    leaders: Sequence[Leader],
    value_fn: ValueFn = cache_value,
    epsilon: float = 1e-3,
    max_iters: int = 100,
) -> Partition:
    """
    Repeatedly applies merge and migrate moves until no coalition can improve
    system welfare by more than epsilon.
    """
    partition: Partition = [{leader} for leader in leaders]

    for _ in range(max_iters):
        if _best_merge(partition, value_fn, epsilon):
            continue
        if _best_move(partition, value_fn, epsilon):
            continue
        break
    return partition


def allocate_payoffs(coalition: Coalition, value_fn: ValueFn) -> dict[str, float]:
    """
    Proportional allocation: each leader's payoff scales with its solo value.
    """
    coalition_value = value_fn(coalition)
    solo_values = {leader: value_fn({leader}) for leader in coalition}
    base = sum(solo_values.values())
    if base == 0:
        share = coalition_value / len(coalition)
        return {leader.name: share for leader in coalition}
    return {leader.name: coalition_value * solo / base for leader, solo in solo_values.items()}


def distribute_to_units(leader_payoff: float, leader: Leader, requests: Requests) -> Dict[str, Dict[str, float]]:
    """
    Second-level split: leader's payoff -> its servers and collaborating vehicles.
    Weighted by caching capacity and current demand to reflect contribution/need.
    """
    weights: List[Tuple[str, str, float]] = []
    for server in leader.servers:
        weight = max(0.0, server.cache_capacity - server.op_cost)
        weights.append(("server", server.name, weight))
    for vehicle in leader.vehicles:
        demand = vehicle.demand + _vehicle_request_delta(leader, vehicle, requests)
        weight = max(0.0, demand + 0.5 * vehicle.cache_capacity - vehicle.coop_cost)
        weights.append(("vehicle", vehicle.name, weight))

    total_weight = sum(w for _, _, w in weights)
    if total_weight <= 0:
        # Nothing to split; give everything back to the leader-level pool.
        return {"servers": {}, "vehicles": {}}

    servers: Dict[str, float] = {}
    vehicles: Dict[str, float] = {}
    for kind, name, weight in weights:
        share = leader_payoff * weight / total_weight
        if kind == "server":
            servers[name] = share
        else:
            vehicles[name] = share
    return {"servers": servers, "vehicles": vehicles}


def record_slice_to_ledger(
    ledger: TrustedLedger,
    enclave: TrustedEnclave,
    slice_idx: int,
    partition: Partition,
    summary: SliceResult,
    unit_allocations: UnitAllocations,
) -> dict:
    """
    Prepare a compact record, sign it via the TEE, and append to the ledger.
    """
    coalition_view = [
        {
            "members": sorted(leader.name for leader in coalition),
            "size": len(coalition),
        }
        for coalition in partition
    ]
    record = {
        "slice": slice_idx,
        "welfare": summary["welfare"],
        "requests": summary["requests"],
        "leader_payoffs": summary["leader_payoffs"],
        "coalition_view": coalition_view,
        "unit_allocations": unit_allocations,
    }
    report = enclave.sign(record)
    entry = {
        "record": record,
        "tee_report": {
            "timestamp": report.timestamp,
            "payload_hash": report.payload_hash,
            "signature": report.signature,
        },
    }
    ledger.append(entry)
    return entry


def demo() -> None:
    leaders = [
        Leader(
            "A",
            sub_members=8,
            cache_capacity=70,
            demand=55,
            link_cost=3,
            servers=(Server("A-s1", 50, 0.3), Server("A-s2", 35, 0.25)),
            vehicles=(Vehicle("A-v1", 12, 5), Vehicle("A-v2", 9, 3)),
        ),
        Leader(
            "B",
            sub_members=6,
            cache_capacity=65,
            demand=50,
            link_cost=2.5,
            servers=(Server("B-s1", 40, 0.2),),
            vehicles=(Vehicle("B-v1", 10, 4), Vehicle("B-v2", 11, 2.5)),
        ),
        Leader(
            "C",
            sub_members=10,
            cache_capacity=85,
            demand=70,
            link_cost=3.5,
            servers=(Server("C-s1", 55, 0.35),),
            vehicles=(Vehicle("C-v1", 14, 6),),
        ),
        Leader(
            "D",
            sub_members=4,
            cache_capacity=55,
            demand=45,
            link_cost=2,
            servers=(Server("D-s1", 30, 0.22),),
            vehicles=(Vehicle("D-v1", 15, 8),),
        ),
    ]

    time_slices: List[Requests] = [
        {"leader": {"A": 10, "C": 5}, "vehicle": {"A": {"A-v1": 6}, "C": {"C-v1": 4}}},
        {"leader": {"B": 15, "D": 10}, "vehicle": {"B": {"B-v2": 8}}},
        {"leader": {"C": 20}, "vehicle": {"A": {"A-v2": 3}, "D": {"D-v1": 10}}},
    ]

    ledger = TrustedLedger(Path("ledger.jl"))
    enclave = TrustedEnclave()

    results, unit_allocations_all, partitions = simulate_time_slices(leaders, time_slices)
    for idx, (requests, partition, units, summary) in enumerate(
        zip(time_slices, partitions, unit_allocations_all, results), start=1
    ):
        def value_fn(coalition: Iterable[Leader], r=requests) -> float:
            return cache_value_with_requests(coalition, r)

        print(f"\n== 时间段 {idx}，请求增量 {requests} ==")
        for coalition in partition:
            names = ", ".join(sorted(l.name for l in coalition))
            value = value_fn(coalition)
            payoffs = allocate_payoffs(coalition, value_fn)
            print(f"联盟({names}) -> 价值 {value:.2f}")
            for leader in sorted(coalition, key=lambda l: l.name):
                leader_share = payoffs[leader.name]
                unit_split = units[leader.name]
                print(
                    f"  领导 {leader.name} 获得 {leader_share:.2f}，"
                    f"服务器分配 {unit_split['servers']}, 车辆分配 {unit_split['vehicles']}"
                )
        print(f"系统总福利: {summary['welfare']:.2f}")
        entry = record_slice_to_ledger(ledger, enclave, idx, partition, summary, units)
        print(f"  已上链记录哈希 {entry['tee_report']['payload_hash'][:16]}...，签名 {entry['tee_report']['signature'][:16]}...")

    saved = plot_results(results, filename_prefix="coalition")
    if saved:
        print("\n生成的图像文件:")
        for name in saved:
            print(f"  {name}")


if __name__ == "__main__":
    demo()
