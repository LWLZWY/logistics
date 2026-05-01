# 多 Agent 驱动的物流配载与费用核验系统 MVP

这是一个面向跨境物流配载场景的 Python 原型项目，用于展示如何用多个 Agent 协同完成订单解析、约束识别、候选配载生成、费用核验、全局选择与诊断反馈。

## 一、项目解决的核心痛点

跨境物流配载不是单纯把订单装进柜子，而是要同时判断：

- 订单体积、重量、数量；
- 柜型容量、承重、起运港、目的港；
- 普货 / 危险品限制；
- FBA 仓库限制；
- 尾程派送模式限制；
- 网点、国家、城市、地址范围；
- 报关、清关、提柜、拆柜、尾程等费用规则；
- 未覆盖订单、冲突模式、高成本模式。

因此，本项目用多 Agent 结构将复杂业务拆解为多个可维护模块。

## 二、项目结构

```text
logistics_agent_packing_package/
├── README.md
├── requirements.txt
├── run_demo.py
├── data/
│   ├── orders.json
│   ├── containers.json
│   └── quote_rules.json
└── src/
    ├── __init__.py
    └── logistics_multi_agent_packing.py
```

## 三、Agent 分工

| Agent | 职责 |
|---|---|
| OrderParserAgent | 解析原始订单，转换为标准 Order 对象 |
| ConstraintAgent | 校验体积、重量、国家、网点、危险品、FBA 仓、派送方式等约束 |
| CandidateGenerationAgent | 生成单柜候选模式，并通过快慢两级评估筛选方案 |
| FeeVerifierAgent | 匹配报价规则，计算海运、报关、清关、提柜、拆柜、尾程等费用 |
| MasterSolverAgent | 在候选模式中选择全局成本最低的组合 |
| DiagnosisAgent | 输出覆盖缺口、冲突热点、高成本模式与 feedback-regenerate 建议 |

## 四、运行方式

进入项目目录后运行：

```bash
python run_demo.py
```

本项目仅使用 Python 标准库，无需安装额外依赖。建议 Python 版本为 3.9 或以上。

## 五、运行输出

程序会在终端输出：

- 最终配载结果；
- 选中模式；
- 费用明细；
- 覆盖缺口报告；
- 冲突热点分布；
- 高成本模式识别；
- feedback-regenerate 建议；
- token plan；
- Agent 日志。

同时会生成：

```text
output/packing_result.json
```

## 六、如何替换为真实业务模块

你可以按下面方式把 MVP 接入已有系统：

1. 将 `FeeVerifierAgent.verify_fee()` 替换为你的 `fee_service2.py` 费用计算逻辑；
2. 将 `ConstraintAgent.validate()` 扩展为完整业务规则库；
3. 将 `CandidateGenerationAgent.generate()` 替换为 ALNS / 回溯搜索 / 动态规划候选生成器；
4. 将 `MasterSolverAgent.solve()` 替换为 MILP / set-partitioning / Gurobi / OR-Tools；
5. 将 `DiagnosisAgent.feedback_regenerate` 输出对接到后续候选模式再生成模块。

## 七、核心创新表达

该项目不是简单聊天机器人，而是将 Agent 嵌入实际物流决策链路：

- 用订单解析 Agent 将业务数据标准化；
- 用约束识别 Agent 降低人工漏判风险；
- 用配载生成 Agent 形成可行单柜模式；
- 用费用核验 Agent 对候选方案进行精确成本计算；
- 用诊断 Agent 把失败原因、高成本原因转化为可操作反馈；
- 通过 feedback-regenerate 机制支持后续自动迭代优化。

## 八、Token Plan 设计

为了避免 Agent 在大量组合中无效推理，系统采用：

1. 快速过滤：先检查容量、重量、国家等基础硬约束；
2. 精确核验：只对通过快速过滤的候选方案进行完整费用计算；
3. 结构化反馈：输出 JSON 格式的冲突与优化建议，减少长文本消耗；
4. 可扩展缓存：生产环境可缓存相同订单集合、柜型和报价规则的费用计算结果。
