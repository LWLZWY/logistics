from pathlib import Path

from src.logistics_multi_agent_packing import (
    LogisticsPackingOrchestrator,
    SolverConfig,
    load_orders,
    load_containers,
    load_quote_rules,
    print_result,
    save_result_json,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    raw_orders = load_orders(str(data_dir / "orders.json"))
    containers = load_containers(str(data_dir / "containers.json"))
    quote_rules = load_quote_rules(str(data_dir / "quote_rules.json"))

    config = SolverConfig(
        max_orders_per_mode=4,
        min_fill_rate=0.05,
        top_k_modes_per_container=8,
        uncovered_penalty_per_order=10_000,
        soft_violation_penalty=500,
        use_fast_filter=True,
    )

    orchestrator = LogisticsPackingOrchestrator(config=config, quote_rules=quote_rules)
    result = orchestrator.run(raw_orders=raw_orders, containers=containers)
    logs = orchestrator.collect_logs()

    print_result(result, logs)
    save_result_json(result, logs, str(output_dir / "packing_result.json"))
    print(f"\n结果已导出：{output_dir / 'packing_result.json'}")


if __name__ == "__main__":
    main()
