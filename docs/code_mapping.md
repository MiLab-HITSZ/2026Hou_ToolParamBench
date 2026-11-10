# 原代码与公开版对应关系

| 原模块 | 整理后的模块 | 处理 |
|---|---|---|
| workspace_entry.py、workspace_structured_entry.py | schema_study/experiment.py | 合并实验构建；structured 为默认入口 |
| rebuilt_entry.py | schema_study/runner.py | 保留共用执行、事件、schema 校验及记录；移除旧 rebuilt 数据入口 |
| lab.py | schema_study/sdk.py | 提取当前所需配置、模型创建和日志；保留 SDK 源码加载支持 |
| workspace_batch.py | schema_study/batch.py | 保留数据集无关的计划、路径和身份逻辑；移除散落的实验常量 |
| 000_run_repeats_20260922.py、000_run_openrouter_20260930.py | run.py、schema_study/cli.py、schema_study/lifecycle.py | 统一模型与实验配置入口，保留完成记录、显式重试和归档行为 |
| scenario_queries.py、ecommerce_queries.py、resource_store.py | schema_study/queries.py | 合并数据读取、共享查询执行；保留场景校验和匹配规则 |
| workspace_queries.py | schema_study/workspace.py | 保留 Workspace 特殊查询语义 |
| workspace_memory.py | schema_study/memory.py | 保留当前 Memory 内容选择与包装 |
| workspace_a2a.py | schema_study/a2a.py | 保留本机 A2A 协议流程 |
| openrouter_model.py | schema_study/openrouter.py | 保留请求间隔、工具调用历史和响应适配 |
| postprocess_20260930/extract_facts.py | schema_study/facts.py | 提取原事实匹配逻辑，去掉历史标签对照入口 |
| paper/build_evidence_v0_17.py 的轨迹关联逻辑 | schema_study/evaluation.py | 去除固定论文输出路径，提供通用轨迹关联与汇总 |

新增 configs 集中保存模型模板、主实验任务和扩展实验组合。新增 run.py / evaluate.py 作为两个面向使用者的入口。

没有迁入：SDK、实验结果、会话日志、历史源码快照、Notebook 输出、密钥配置、论文写作脚本、旧一次性迁移/修复脚本、历史人工标注、旧语义判分流程和 TaskShield 探索代码。原始研究目录不改动。

本次不整理、复制或裁剪 test_data；目录中已有数据保持原样。数据文件内的 mock_backend.py、备份文件等也保留，公开前是否进一步清理由作者决定。

## 评测名称的明确区分

原有部分后处理脚本的 T 是 LLM 评判的业务回答正确性。当前论文 T 表述为下游工具成功执行，因此新入口将 T 明确定义为提交被接受，保留 R/X/S/F 的原始事实提取规则。运行记录内原有 metrics 保留；新评测输出有独立版本号 upload_target_metrics_v1。

整理版不是历史统计结果或人工标签的迁移，不承诺与旧语义判分表逐项相同。使用时以 README 中的指标定义及运行配置为准。
