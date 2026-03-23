# 从代码骨架到论文级证据：推进路线图

本文档将「可验证的实验声明」映射为仓库内可执行项，与当前实现状态对照。**骨架完成 ≠ 论文可投**：仍需真实数据、统计检验与用户研究。

## 当前仓库状态（诚实评估）

| 维度 | 已有内容 | 论文仍缺 |
|------|----------|----------|
| 接口与目录 | `emokit`/`emosense` 模块、CI、测试 | 真实 DEAP/SEED-V 上的数字 |
| LOSO / 模型 | 实现与单元测试 | 完整 LOSO + 与文献数值对比（±1.5% 追踪） |
| EmoSense | FastAPI + Gradio + 缓冲/推理 | 真实延迟分布（p99）、用户研究、Demo 视频 |

## EmoKit（Open Source Track）

### R1 真实数据管线

- [ ] **R1-1** 使用 `scripts/verify_deap_pipeline.py` 在本地 DEAP（`.bdf`）上跑通；检查形状、频段 DE 排序、标签分布。输出写入 `results/pipeline_sanity.json`。
- [ ] **R1-2** SEED-V：`use_de_features=True` 时跳过 `DEExtractor` 的分支需用真实 `.mat` 或最小 fixture 做回归测试（见 `tests/` 中待补充用例）。

### R2 训练协议与基线

- [ ] **R2-1** 所有实验从 `configs/standard_protocol.yaml` 继承不可变预处理/训练默认值，模型 YAML 仅覆盖自身超参。
- [ ] **R2-2** 运行 `scripts/reproduce_baselines.py`，填写 `PAPER_NUMBERS` 与实测 LOSO 结果，维护 PASS/FAIL（Δ≤1.5%）。
- [ ] **R2-3** DGCNN 邻接可视化：`scripts/visualize_dgcnn_adjacency.py` + `DGCNNModel.get_adjacency_matrix()`（已实现 API）。

### R3 统计与消融

- [ ] **R3-1** LOSO 每被试准确率：对两模型做 `scipy.stats.wilcoxon`（配对、非参数）。
- [ ] **R3-2** 模态消融：对 DGCCA-AM（或等价融合模型）跑 6 组 `modalities`，汇总表格。

### R4 工具对比表

- [ ] **R4-1** 用 `scripts/verify_toolkit_comparison.py` 中的声明逐项核对官方文档/源码，并保存链接与截图索引。

### R5 开源体验

- [ ] **R5-1** 新机器 10 分钟内：`pip install -e .` + 小数据/合成数据 + `python -m emokit.run configs/quick_demo.yaml`（待添加 quick_demo）。
- [ ] **R5-2** Sphinx 文档部署到 GitHub Pages（workflow 待加）。

## EmoSense（Demo Track）

### D1 端到端

- [ ] **D1-1** Replay + WebSocket 在真实或 mock trial 上连续收到 ≥3 条 `inference`（见 `tests/test_replay_pipeline.py` 规划）。
- [ ] **D1-2** 会话中途 `POST /models/active` 不崩溃，后续 `model_name` 切换正确。
- [ ] **D1-3** `scripts/benchmark_latency.py`：真实窗口上报告 mean / p95 / p99，与论文声明一致（勿夸大「<300ms」）。

### D2–D4 可视化验证、用户研究、视频

- 见上文用户提供的脚本思路；需真实标签 trial 与 IRB 批准后执行。

---

**一句话**：把每个论文数字对应到一个脚本、一次统计检验或一次测量，再写进论文。
