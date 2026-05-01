"""
多 Agent 驱动的跨境物流配载与费用核验系统 MVP
================================================

这是一个可直接运行的工程化原型，用来展示：
1. 订单解析 Agent
2. 约束识别 Agent
3. 配载候选生成 Agent
4. 费用核验 Agent
5. 全局主问题求解 Agent
6. 诊断与 feedback-regenerate Agent

运行方式：
    python run_demo.py

项目特点：
- 不依赖第三方库，仅使用 Python 标准库。
- 示例数据位于 data/ 目录。
- 输出结果位于 output/ 目录。
- 真实业务中可以将 FeeVerifierAgent 替换为现有 fee_service2.py，
  将 CandidateGenerationAgent / MasterSolverAgent 替换为 ALNS、MILP 或 set-partitioning 求解器。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from collections import Counter, defaultdict
import itertools
import json
import math
import uuid


# ============================================================
# 1. 基础数据结构
# ============================================================

class Country(str, Enum):
    US = "US"
    CA = "CA"
    UK = "UK"
    DE = "DE"


class DeliveryMode(str, Enum):
    FBA = "FBA"
    EXPRESS = "EXPRESS"
    TRUCK = "TRUCK"
    WAREHOUSE = "WAREHOUSE"


class ConstraintLevel(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"


@dataclass(frozen=True)
class Order:
    order_id: str
    order_number: str
    nation_code: Country
    province: str
    city: str
    receiver: str
    loc_outlets_code: str
    destination_warehouse: str
    delivery_mode: DeliveryMode
    product_code: str
    is_fba: bool
    is_dg: bool
    length_cm: float
    width_cm: float
    height_cm: float
    weight_kg: float
    quantity: int = 1

    @property
    def volume_cbm(self) -> float:
        return self.length_cm * self.width_cm * self.height_cm * self.quantity / 1_000_000

    @property
    def total_weight_kg(self) -> float:
        return self.weight_kg * self.quantity


@dataclass(frozen=True)
class Container:
    container_id: str
    lading_number: str
    container_type: str
    shipping_code: str
    shipment_code: str
    destination_code: str
    max_volume_cbm: float
    max_weight_kg: float
    nation_code: Country
    allowed_outlets: Set[str] = field(default_factory=set)
    is_dg_cabinet: Optional[bool] = None
    allowed_fba_warehouses: Set[str] = field(default_factory=set)
    allowed_delivery_modes: Set[DeliveryMode] = field(default_factory=set)
    product_codes: Optional[Set[str]] = None
    max_receivers: Optional[int] = None


@dataclass(frozen=True)
class QuoteRule:
    rule_id: str
    supplier_code: str
    channel_code: str
    nation_code: Country
    shipment_code: str
    destination_code: str
    delivery_mode: Optional[DeliveryMode]
    base_ocean_fee: float
    customs_fee: float
    clearance_fee: float
    pickup_fee: float
    devanning_fee: float
    last_mile_fee_per_cbm: float
    dg_surcharge: float = 0.0
    out_of_area_surcharge_per_cbm: float = 0.0
    max_receivers: Optional[int] = None
    address_range: Optional[Set[str]] = None


@dataclass
class ConstraintViolation:
    level: ConstraintLevel
    code: str
    message: str
    order_ids: List[str] = field(default_factory=list)
    impact_weight: float = 1.0


@dataclass
class FeeBreakdown:
    ocean_fee: float = 0.0
    customs_fee: float = 0.0
    clearance_fee: float = 0.0
    pickup_fee: float = 0.0
    devanning_fee: float = 0.0
    last_mile_fee: float = 0.0
    dg_surcharge: float = 0.0
    out_of_area_surcharge: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.ocean_fee + self.customs_fee + self.clearance_fee +
            self.pickup_fee + self.devanning_fee + self.last_mile_fee +
            self.dg_surcharge + self.out_of_area_surcharge
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "ocean_fee": round(self.ocean_fee, 2),
            "customs_fee": round(self.customs_fee, 2),
            "clearance_fee": round(self.clearance_fee, 2),
            "pickup_fee": round(self.pickup_fee, 2),
            "devanning_fee": round(self.devanning_fee, 2),
            "last_mile_fee": round(self.last_mile_fee, 2),
            "dg_surcharge": round(self.dg_surcharge, 2),
            "out_of_area_surcharge": round(self.out_of_area_surcharge, 2),
            "total": round(self.total, 2),
        }


@dataclass
class CandidateMode:
    mode_id: str
    container_id: str
    order_ids: List[str]
    total_volume_cbm: float
    total_weight_kg: float
    receivers: List[str]
    hard_violations: List[ConstraintViolation] = field(default_factory=list)
    soft_violations: List[ConstraintViolation] = field(default_factory=list)
    fee_breakdown: Optional[FeeBreakdown] = None
    score: Optional[float] = None

    @property
    def feasible(self) -> bool:
        return not self.hard_violations

    @property
    def total_fee(self) -> float:
        return self.fee_breakdown.total if self.fee_breakdown else math.inf


@dataclass
class SolverConfig:
    max_orders_per_mode: int = 6
    min_fill_rate: float = 0.05
    top_k_modes_per_container: int = 8
    uncovered_penalty_per_order: float = 10_000.0
    soft_violation_penalty: float = 500.0
    use_fast_filter: bool = True


@dataclass
class PackingResult:
    selected_modes: List[CandidateMode]
    uncovered_orders: List[str]
    total_cost: float
    diagnostics: Dict[str, Any]
    token_plan: Dict[str, Any]


# ============================================================
# 2. 通用工具
# ============================================================

def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def round2(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def to_jsonable(obj: Any) -> Any:
    """将 dataclass / enum / set 转为可 JSON 序列化对象。"""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, set):
        return sorted(list(obj))
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class AgentBase:
    def __init__(self, name: str):
        self.name = name
        self.logs: List[str] = []
        self.token_counter: int = 0

    def log(self, message: str) -> None:
        self.logs.append(f"[{self.name}] {message}")

    def count_tokens_proxy(self, payload: Any) -> int:
        """近似 token 估算，用于展示 token plan。"""
        text = json.dumps(payload, ensure_ascii=False, default=to_jsonable)
        tokens = max(1, len(text) // 4)
        self.token_counter += tokens
        return tokens


# ============================================================
# 3. 输入加载与解析 Agent
# ============================================================

class OrderParserAgent(AgentBase):
    def __init__(self):
        super().__init__("OrderParserAgent")

    def parse(self, raw_orders: List[Dict[str, Any]]) -> List[Order]:
        orders: List[Order] = []
        for raw in raw_orders:
            self.count_tokens_proxy(raw)
            orders.append(Order(
                order_id=str(raw.get("order_id") or generate_id("order")),
                order_number=str(raw["order_number"]),
                nation_code=Country(raw.get("nation_code", "US")),
                province=str(raw.get("province", "")),
                city=str(raw.get("city", "")),
                receiver=str(raw.get("receiver", "")),
                loc_outlets_code=str(raw.get("loc_outlets_code", "")),
                destination_warehouse=str(raw.get("destination_warehouse", "")),
                delivery_mode=DeliveryMode(raw.get("delivery_mode", "TRUCK")),
                product_code=str(raw.get("product_code", "GENERAL")),
                is_fba=bool(raw.get("is_fba", False)),
                is_dg=bool(raw.get("is_dg", False)),
                length_cm=float(raw.get("length_cm", 0)),
                width_cm=float(raw.get("width_cm", 0)),
                height_cm=float(raw.get("height_cm", 0)),
                weight_kg=float(raw.get("weight_kg", 0)),
                quantity=int(raw.get("quantity", 1)),
            ))
        self.log(f"解析订单 {len(orders)} 单。")
        return orders


# ============================================================
# 4. 约束识别 Agent
# ============================================================

class ConstraintAgent(AgentBase):
    def __init__(self):
        super().__init__("ConstraintAgent")

    def fast_filter(self, container: Container, orders: List[Order]) -> Tuple[bool, List[ConstraintViolation]]:
        violations: List[ConstraintViolation] = []
        total_volume = sum(o.volume_cbm for o in orders)
        total_weight = sum(o.total_weight_kg for o in orders)

        if total_volume > container.max_volume_cbm:
            violations.append(ConstraintViolation(
                ConstraintLevel.HARD, "VOLUME_EXCEEDED",
                f"总体积 {round2(total_volume)} CBM 超过柜子容量 {container.max_volume_cbm} CBM。",
                [o.order_id for o in orders], 5.0,
            ))
        if total_weight > container.max_weight_kg:
            violations.append(ConstraintViolation(
                ConstraintLevel.HARD, "WEIGHT_EXCEEDED",
                f"总重量 {round2(total_weight)} KG 超过柜子承重 {container.max_weight_kg} KG。",
                [o.order_id for o in orders], 5.0,
            ))
        if any(o.nation_code != container.nation_code for o in orders):
            violations.append(ConstraintViolation(
                ConstraintLevel.HARD, "NATION_MISMATCH",
                "订单国家与柜子国家不一致。",
                [o.order_id for o in orders if o.nation_code != container.nation_code], 4.0,
            ))
        return len(violations) == 0, violations

    def validate(
        self,
        container: Container,
        orders: List[Order],
        quote: Optional[QuoteRule] = None,
    ) -> Tuple[List[ConstraintViolation], List[ConstraintViolation]]:
        hard: List[ConstraintViolation] = []
        soft: List[ConstraintViolation] = []
        order_ids = [o.order_id for o in orders]
        total_volume = sum(o.volume_cbm for o in orders)
        total_weight = sum(o.total_weight_kg for o in orders)
        receivers = {o.receiver for o in orders}

        self.count_tokens_proxy({
            "container_id": container.container_id,
            "order_ids": order_ids,
            "quote_id": quote.rule_id if quote else None,
        })

        if total_volume > container.max_volume_cbm:
            hard.append(ConstraintViolation(ConstraintLevel.HARD, "VOLUME_EXCEEDED", "体积超出柜型容量。", order_ids, 5.0))
        if total_weight > container.max_weight_kg:
            hard.append(ConstraintViolation(ConstraintLevel.HARD, "WEIGHT_EXCEEDED", "重量超出柜型承重。", order_ids, 5.0))

        nation_bad = [o.order_id for o in orders if o.nation_code != container.nation_code]
        if nation_bad:
            hard.append(ConstraintViolation(ConstraintLevel.HARD, "NATION_MISMATCH", "订单国家与柜子目的国不一致。", nation_bad, 4.0))

        if container.allowed_outlets:
            bad = [o.order_id for o in orders if o.loc_outlets_code not in container.allowed_outlets]
            if bad:
                hard.append(ConstraintViolation(ConstraintLevel.HARD, "OUTLET_NOT_ALLOWED", "订单所属网点不在柜子允许网点范围内。", bad, 3.0))

        if container.is_dg_cabinet is False:
            dg_bad = [o.order_id for o in orders if o.is_dg]
            if dg_bad:
                hard.append(ConstraintViolation(ConstraintLevel.HARD, "DG_NOT_ALLOWED", "普货柜不能装载危险品订单。", dg_bad, 5.0))

        if container.is_dg_cabinet is True:
            mixed_normal = [o.order_id for o in orders if not o.is_dg]
            if mixed_normal and any(o.is_dg for o in orders):
                soft.append(ConstraintViolation(ConstraintLevel.SOFT, "DG_MIX_WITH_NORMAL", "危险品柜混装普货，需确认业务是否允许。", mixed_normal, 2.0))

        if container.allowed_fba_warehouses:
            fba_bad = [o.order_id for o in orders if o.is_fba and o.destination_warehouse not in container.allowed_fba_warehouses]
            if fba_bad:
                hard.append(ConstraintViolation(ConstraintLevel.HARD, "FBA_WAREHOUSE_NOT_ALLOWED", "FBA 目的仓不在柜子允许范围内。", fba_bad, 4.0))

        if container.allowed_delivery_modes:
            mode_bad = [o.order_id for o in orders if o.delivery_mode not in container.allowed_delivery_modes]
            if mode_bad:
                hard.append(ConstraintViolation(ConstraintLevel.HARD, "DELIVERY_MODE_NOT_ALLOWED", "尾程派送模式不在柜子允许范围内。", mode_bad, 4.0))

        if container.product_codes:
            product_bad = [o.order_id for o in orders if o.product_code not in container.product_codes]
            if product_bad:
                hard.append(ConstraintViolation(ConstraintLevel.HARD, "PRODUCT_NOT_ALLOWED", "产品品类不在柜子允许范围内。", product_bad, 3.0))

        max_receivers = container.max_receivers
        if quote and quote.max_receivers is not None:
            max_receivers = min(max_receivers or quote.max_receivers, quote.max_receivers)
        if max_receivers is not None and len(receivers) > max_receivers:
            soft.append(ConstraintViolation(
                ConstraintLevel.SOFT, "TOO_MANY_RECEIVERS",
                f"收件人数 {len(receivers)} 超过报价/柜子建议上限 {max_receivers}。",
                order_ids, 2.5,
            ))

        if quote and quote.address_range:
            out_area = [o.order_id for o in orders if o.city not in quote.address_range]
            if out_area:
                soft.append(ConstraintViolation(
                    ConstraintLevel.SOFT, "ADDRESS_OUT_OF_RANGE",
                    "部分订单城市超出报价地址范围，将产生额外费用。",
                    out_area, 2.0,
                ))
        return hard, soft


# ============================================================
# 5. 费用核验 Agent
# ============================================================

class FeeVerifierAgent(AgentBase):
    def __init__(self, quote_rules: List[QuoteRule]):
        super().__init__("FeeVerifierAgent")
        self.quote_rules = quote_rules

    def match_quote(self, container: Container, orders: List[Order]) -> Optional[QuoteRule]:
        modes = {o.delivery_mode for o in orders}
        candidates: List[QuoteRule] = []
        for q in self.quote_rules:
            if q.nation_code != container.nation_code:
                continue
            if q.shipment_code != container.shipment_code:
                continue
            if q.destination_code != container.destination_code:
                continue
            if q.delivery_mode is not None and len(modes) == 1 and q.delivery_mode not in modes:
                continue
            candidates.append(q)
        if not candidates:
            return None
        candidates.sort(key=lambda q: (q.delivery_mode is None, q.base_ocean_fee))
        return candidates[0]

    def verify_fee(self, orders: List[Order], quote: QuoteRule) -> FeeBreakdown:
        self.count_tokens_proxy({"quote_id": quote.rule_id, "order_ids": [o.order_id for o in orders]})
        total_volume = sum(o.volume_cbm for o in orders)
        has_dg = any(o.is_dg for o in orders)
        out_area_volume = 0.0
        if quote.address_range:
            out_area_volume = sum(o.volume_cbm for o in orders if o.city not in quote.address_range)
        return FeeBreakdown(
            ocean_fee=quote.base_ocean_fee,
            customs_fee=quote.customs_fee,
            clearance_fee=quote.clearance_fee,
            pickup_fee=quote.pickup_fee,
            devanning_fee=quote.devanning_fee,
            last_mile_fee=quote.last_mile_fee_per_cbm * total_volume,
            dg_surcharge=quote.dg_surcharge if has_dg else 0.0,
            out_of_area_surcharge=quote.out_of_area_surcharge_per_cbm * out_area_volume,
        )


# ============================================================
# 6. 配载候选生成 Agent
# ============================================================

class CandidateGenerationAgent(AgentBase):
    def __init__(self, config: SolverConfig, constraint_agent: ConstraintAgent, fee_agent: FeeVerifierAgent):
        super().__init__("CandidateGenerationAgent")
        self.config = config
        self.constraint_agent = constraint_agent
        self.fee_agent = fee_agent

    def _score_mode(self, mode: CandidateMode, container: Container) -> float:
        fill_rate = mode.total_volume_cbm / max(container.max_volume_cbm, 1e-9)
        soft_penalty = sum(v.impact_weight for v in mode.soft_violations) * self.config.soft_violation_penalty
        return mode.total_fee - fill_rate * 1000 + soft_penalty

    def generate_for_container(self, container: Container, orders: List[Order]) -> List[CandidateMode]:
        modes: List[CandidateMode] = []
        max_size = min(self.config.max_orders_per_mode, len(orders))

        for size in range(max_size, 0, -1):
            for subset_tuple in itertools.combinations(orders, size):
                subset = list(subset_tuple)
                total_volume = sum(o.volume_cbm for o in subset)
                if size > 1 and total_volume / max(container.max_volume_cbm, 1e-9) < self.config.min_fill_rate:
                    continue
                if self.config.use_fast_filter:
                    passed, _ = self.constraint_agent.fast_filter(container, subset)
                    if not passed:
                        continue
                quote = self.fee_agent.match_quote(container, subset)
                if quote is None:
                    continue
                hard, soft = self.constraint_agent.validate(container, subset, quote)
                if hard:
                    continue
                fee = self.fee_agent.verify_fee(subset, quote)
                mode = CandidateMode(
                    mode_id=generate_id("mode"),
                    container_id=container.container_id,
                    order_ids=[o.order_id for o in subset],
                    total_volume_cbm=round2(total_volume),
                    total_weight_kg=round2(sum(o.total_weight_kg for o in subset)),
                    receivers=sorted({o.receiver for o in subset}),
                    hard_violations=hard,
                    soft_violations=soft,
                    fee_breakdown=fee,
                )
                mode.score = self._score_mode(mode, container)
                modes.append(mode)

        best_by_key: Dict[Tuple[str, Tuple[str, ...]], CandidateMode] = {}
        for m in modes:
            key = (m.container_id, tuple(sorted(m.order_ids)))
            if key not in best_by_key or (m.score or math.inf) < (best_by_key[key].score or math.inf):
                best_by_key[key] = m

        final_modes = sorted(best_by_key.values(), key=lambda m: m.score or math.inf)[: self.config.top_k_modes_per_container]
        self.log(f"柜子 {container.container_id} 生成候选模式 {len(final_modes)} 个。")
        return final_modes

    def generate(self, containers: List[Container], orders: List[Order]) -> List[CandidateMode]:
        all_modes: List[CandidateMode] = []
        for c in containers:
            all_modes.extend(self.generate_for_container(c, orders))
        self.log(f"共生成候选模式 {len(all_modes)} 个。")
        return all_modes


# ============================================================
# 7. 全局主问题求解 Agent
# ============================================================

class MasterSolverAgent(AgentBase):
    def __init__(self, config: SolverConfig):
        super().__init__("MasterSolverAgent")
        self.config = config

    def solve(self, modes: List[CandidateMode], all_order_ids: List[str]) -> Tuple[List[CandidateMode], List[str], float]:
        self.count_tokens_proxy({"mode_count": len(modes), "order_count": len(all_order_ids)})
        best_solution: List[CandidateMode] = []
        best_uncovered = list(all_order_ids)
        best_cost = len(all_order_ids) * self.config.uncovered_penalty_per_order
        modes_sorted = sorted(modes, key=lambda m: m.score or math.inf)
        max_pick = min(len({m.container_id for m in modes_sorted}), len(modes_sorted))

        def valid_combo(combo: Iterable[CandidateMode]) -> bool:
            used_orders: Set[str] = set()
            used_containers: Set[str] = set()
            for m in combo:
                if m.container_id in used_containers:
                    return False
                if used_orders.intersection(m.order_ids):
                    return False
                used_containers.add(m.container_id)
                used_orders.update(m.order_ids)
            return True

        for r in range(1, max_pick + 1):
            for combo in itertools.combinations(modes_sorted, r):
                if not valid_combo(combo):
                    continue
                covered = set(itertools.chain.from_iterable(m.order_ids for m in combo))
                uncovered = [oid for oid in all_order_ids if oid not in covered]
                cost = sum(m.total_fee for m in combo) + len(uncovered) * self.config.uncovered_penalty_per_order
                if cost < best_cost:
                    best_solution = list(combo)
                    best_uncovered = uncovered
                    best_cost = cost
        self.log(f"全局求解完成：选择模式 {len(best_solution)} 个，未覆盖订单 {len(best_uncovered)} 单。")
        return best_solution, best_uncovered, best_cost


# ============================================================
# 8. 诊断 Agent
# ============================================================

class DiagnosisAgent(AgentBase):
    def __init__(self, config: SolverConfig):
        super().__init__("DiagnosisAgent")
        self.config = config

    def diagnose(
        self,
        selected_modes: List[CandidateMode],
        all_modes: List[CandidateMode],
        uncovered_orders: List[str],
        orders: List[Order],
    ) -> Dict[str, Any]:
        self.count_tokens_proxy({"selected": [m.mode_id for m in selected_modes], "uncovered": uncovered_orders})
        order_map = {o.order_id: o for o in orders}

        uncovered_detail = []
        for oid in uncovered_orders:
            o = order_map[oid]
            uncovered_detail.append({
                "order_id": oid,
                "order_number": o.order_number,
                "volume_cbm": round2(o.volume_cbm),
                "weight_kg": round2(o.total_weight_kg),
                "is_dg": o.is_dg,
                "delivery_mode": o.delivery_mode.value,
                "destination_warehouse": o.destination_warehouse,
                "impact_weight": self.config.uncovered_penalty_per_order,
            })

        violation_counter = Counter()
        violation_orders: Dict[str, Set[str]] = defaultdict(set)
        for m in all_modes:
            for v in m.hard_violations + m.soft_violations:
                violation_counter[v.code] += 1
                violation_orders[v.code].update(v.order_ids)

        conflict_hotspots = []
        for code, freq in violation_counter.most_common():
            conflict_hotspots.append({
                "conflict_type": code,
                "frequency": freq,
                "affected_order_count": len(violation_orders[code]),
                "severity": self._severity(code, freq),
                "sample_order_ids": sorted(list(violation_orders[code]))[:5],
            })

        high_cost_modes = self._high_cost_modes(selected_modes)
        return {
            "selected_modes": [self._mode_to_dict(m) for m in selected_modes],
            "coverage_gap_report": {
                "uncovered_count": len(uncovered_orders),
                "uncovered_order_ids": uncovered_orders,
                "uncovered_detail": uncovered_detail,
                "estimated_penalty": round2(len(uncovered_orders) * self.config.uncovered_penalty_per_order),
            },
            "conflict_hotspots": conflict_hotspots,
            "high_cost_modes": high_cost_modes,
            "feedback_regenerate": self._build_feedback(uncovered_detail, conflict_hotspots, high_cost_modes),
        }

    def _mode_to_dict(self, mode: CandidateMode) -> Dict[str, Any]:
        return {
            "mode_id": mode.mode_id,
            "container_id": mode.container_id,
            "order_ids": mode.order_ids,
            "total_volume_cbm": mode.total_volume_cbm,
            "total_weight_kg": mode.total_weight_kg,
            "receivers": mode.receivers,
            "fee_breakdown": mode.fee_breakdown.to_dict() if mode.fee_breakdown else None,
            "soft_violations": [asdict(v) for v in mode.soft_violations],
            "score": round2(mode.score or 0.0),
        }

    def _severity(self, code: str, freq: int) -> str:
        high_codes = {"VOLUME_EXCEEDED", "WEIGHT_EXCEEDED", "DG_NOT_ALLOWED", "FBA_WAREHOUSE_NOT_ALLOWED"}
        if code in high_codes or freq >= 5:
            return "HIGH"
        if freq >= 2:
            return "MEDIUM"
        return "LOW"

    def _high_cost_modes(self, selected_modes: List[CandidateMode]) -> List[Dict[str, Any]]:
        if not selected_modes:
            return []
        avg_cost = sum(m.total_fee for m in selected_modes) / len(selected_modes)
        result = []
        for m in selected_modes:
            if m.total_fee >= avg_cost:
                result.append({
                    "mode_id": m.mode_id,
                    "container_id": m.container_id,
                    "total_fee": round2(m.total_fee),
                    "fee_breakdown": m.fee_breakdown.to_dict() if m.fee_breakdown else None,
                    "optimization_potential": self._cost_hints(m),
                })
        return result

    def _cost_hints(self, mode: CandidateMode) -> List[str]:
        if not mode.fee_breakdown:
            return ["缺少费用明细，需重新调用费用核验模块。"]
        fee = mode.fee_breakdown
        hints = []
        if fee.out_of_area_surcharge > 0:
            hints.append("存在地址范围外附加费，可尝试将超区订单拆分至其他报价或柜型。")
        if fee.dg_surcharge > 0:
            hints.append("存在危险品附加费，可检查危险品订单是否单独成柜或匹配 DG 报价。")
        if fee.last_mile_fee > fee.ocean_fee * 0.5:
            hints.append("尾程费用占比较高，可尝试按城市/仓库重新聚类订单。")
        if not hints:
            hints.append("费用结构较均衡，优化重点可转向提升装载率。")
        return hints

    def _build_feedback(self, uncovered_detail, conflict_hotspots, high_cost_modes) -> List[Dict[str, Any]]:
        feedback = []
        if uncovered_detail:
            feedback.append({
                "target": "CandidateGenerationAgent",
                "action": "regenerate_modes_for_uncovered_orders",
                "reason": "存在未覆盖订单，需要优先生成补舱或单独成柜模式。",
                "params": {"order_ids": [x["order_id"] for x in uncovered_detail], "priority": "HIGH"},
            })
        for hotspot in conflict_hotspots[:3]:
            feedback.append({
                "target": "ConstraintAgent",
                "action": "route_by_conflict_type",
                "reason": f"冲突类型 {hotspot['conflict_type']} 出现频率较高。",
                "params": {
                    "conflict_type": hotspot["conflict_type"],
                    "severity": hotspot["severity"],
                    "affected_order_count": hotspot["affected_order_count"],
                },
            })
        for mode in high_cost_modes:
            feedback.append({
                "target": "FeeVerifierAgent",
                "action": "search_lower_cost_quote_or_recluster_orders",
                "reason": "存在高成本模式，需要检查报价匹配或订单聚类方式。",
                "params": {
                    "mode_id": mode["mode_id"],
                    "container_id": mode["container_id"],
                    "total_fee": mode["total_fee"],
                },
            })
        return feedback


# ============================================================
# 9. 编排器
# ============================================================

class LogisticsPackingOrchestrator:
    def __init__(self, config: SolverConfig, quote_rules: List[QuoteRule]):
        self.config = config
        self.parser = OrderParserAgent()
        self.constraint_agent = ConstraintAgent()
        self.fee_agent = FeeVerifierAgent(quote_rules)
        self.generator = CandidateGenerationAgent(config, self.constraint_agent, self.fee_agent)
        self.master_solver = MasterSolverAgent(config)
        self.diagnosis_agent = DiagnosisAgent(config)

    def run(self, raw_orders: List[Dict[str, Any]], containers: List[Container]) -> PackingResult:
        orders = self.parser.parse(raw_orders)
        candidate_modes = self.generator.generate(containers, orders)
        selected_modes, uncovered_orders, total_cost = self.master_solver.solve(
            candidate_modes,
            [o.order_id for o in orders],
        )
        diagnostics = self.diagnosis_agent.diagnose(selected_modes, candidate_modes, uncovered_orders, orders)
        return PackingResult(
            selected_modes=selected_modes,
            uncovered_orders=uncovered_orders,
            total_cost=round2(total_cost),
            diagnostics=diagnostics,
            token_plan=self._build_token_plan(),
        )

    def _build_token_plan(self) -> Dict[str, Any]:
        agents = [self.parser, self.constraint_agent, self.fee_agent, self.generator, self.master_solver, self.diagnosis_agent]
        return {
            "total_token_proxy": sum(a.token_counter for a in agents),
            "by_agent": {a.name: a.token_counter for a in agents},
            "control_strategy": [
                "先用 fast_filter 过滤容量、重量、国家等明显不可行组合。",
                "仅对通过轻量规则的候选模式调用完整约束校验与费用核验。",
                "诊断阶段输出结构化反馈，避免长文本反复推理。",
                "生产环境可加入缓存：相同订单集合、柜型、报价规则的费用结果直接复用。",
            ],
        }

    def collect_logs(self) -> List[str]:
        logs: List[str] = []
        for agent in [self.parser, self.constraint_agent, self.fee_agent, self.generator, self.master_solver, self.diagnosis_agent]:
            logs.extend(agent.logs)
        return logs


# ============================================================
# 10. JSON 数据转换函数
# ============================================================

def load_orders(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_containers(path: str) -> List[Container]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    containers: List[Container] = []
    for raw in rows:
        containers.append(Container(
            container_id=raw["container_id"],
            lading_number=raw["lading_number"],
            container_type=raw["container_type"],
            shipping_code=raw["shipping_code"],
            shipment_code=raw["shipment_code"],
            destination_code=raw["destination_code"],
            max_volume_cbm=float(raw["max_volume_cbm"]),
            max_weight_kg=float(raw["max_weight_kg"]),
            nation_code=Country(raw.get("nation_code", "US")),
            allowed_outlets=set(raw.get("allowed_outlets", [])),
            is_dg_cabinet=raw.get("is_dg_cabinet"),
            allowed_fba_warehouses=set(raw.get("allowed_fba_warehouses", [])),
            allowed_delivery_modes={DeliveryMode(x) for x in raw.get("allowed_delivery_modes", [])},
            product_codes=set(raw["product_codes"]) if raw.get("product_codes") else None,
            max_receivers=raw.get("max_receivers"),
        ))
    return containers


def load_quote_rules(path: str) -> List[QuoteRule]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    quotes: List[QuoteRule] = []
    for raw in rows:
        quotes.append(QuoteRule(
            rule_id=raw["rule_id"],
            supplier_code=raw["supplier_code"],
            channel_code=raw["channel_code"],
            nation_code=Country(raw.get("nation_code", "US")),
            shipment_code=raw["shipment_code"],
            destination_code=raw["destination_code"],
            delivery_mode=DeliveryMode(raw["delivery_mode"]) if raw.get("delivery_mode") else None,
            base_ocean_fee=float(raw["base_ocean_fee"]),
            customs_fee=float(raw["customs_fee"]),
            clearance_fee=float(raw["clearance_fee"]),
            pickup_fee=float(raw["pickup_fee"]),
            devanning_fee=float(raw["devanning_fee"]),
            last_mile_fee_per_cbm=float(raw["last_mile_fee_per_cbm"]),
            dg_surcharge=float(raw.get("dg_surcharge", 0.0)),
            out_of_area_surcharge_per_cbm=float(raw.get("out_of_area_surcharge_per_cbm", 0.0)),
            max_receivers=raw.get("max_receivers"),
            address_range=set(raw["address_range"]) if raw.get("address_range") else None,
        ))
    return quotes


def save_result_json(result: PackingResult, logs: List[str], path: str) -> None:
    payload = {
        "total_cost": result.total_cost,
        "uncovered_orders": result.uncovered_orders,
        "diagnostics": result.diagnostics,
        "token_plan": result.token_plan,
        "agent_logs": logs,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=to_jsonable)


def print_result(result: PackingResult, logs: List[str]) -> None:
    print("\n==================== 最终配载结果 ====================")
    print(f"总成本：{result.total_cost}")
    print(f"未覆盖订单：{result.uncovered_orders}")

    print("\n==================== 选中模式 ====================")
    for m in result.selected_modes:
        print(f"\n模式：{m.mode_id}")
        print(f"柜子：{m.container_id}")
        print(f"订单：{m.order_ids}")
        print(f"体积/重量：{m.total_volume_cbm} CBM / {m.total_weight_kg} KG")
        print(f"费用：{m.fee_breakdown.to_dict() if m.fee_breakdown else None}")
        if m.soft_violations:
            print("软约束提醒：")
            for v in m.soft_violations:
                print(f"  - {v.code}: {v.message}")

    print("\n==================== 诊断报告 ====================")
    print(json.dumps(result.diagnostics, ensure_ascii=False, indent=2, default=to_jsonable))

    print("\n==================== Token Plan ====================")
    print(json.dumps(result.token_plan, ensure_ascii=False, indent=2))

    print("\n==================== Agent Logs ====================")
    for line in logs:
        print(line)
