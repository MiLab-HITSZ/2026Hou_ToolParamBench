# 实验配置与保持项

## 主实验

`main90.json` 引用原 `test_data/main_schema90_v1`，金融、旅游、电商、IT、电网各 18 个任务，共 90 个任务。每个任务使用 Single-Agent、P0、无预置 Memory，对比 absent、optional、required，每个模型每轮 270 次。

Optional 与 Required 的参数名、描述、类型相同，只有 `required` 数组不同。工具使用 `strict_json_schema=False`，执行前仍由 Draft202012Validator 校验，避免 SDK 自动严格化改变可选字段含义。

## 扩展实验

`extended45.json` 包含：

- 原 Workspace、金融、旅游各 001–005，共 15 个任务。
- 主实验五领域各 6 个选定任务，共 30 个任务；任务 ID 在配置文件中逐项列出。

30 个新增任务直接引用主实验数据，不复制数据。每个任务运行以下 6 组：

| 架构 | Memory | Policy | Schema |
|---|---|---|---|
| Single-Agent | None | P0 | Required |
| Single-Agent | Pointer | P0 | Required |
| Single-Agent | Direct | P0 | Required |
| Agent-as-Tool | None | P0 | Required |
| A2A | None | P0 | Required |
| Single-Agent | None | P1 | Required |

每模型每轮 270 次。三类对比共享第一行基线，不能在汇总时把共享基线当作三次独立运行。

## 数据与查询

- 五领域从各自 `resources/*.json` 加载；不回退到 `mock_data.json`，不运行数据目录内 Python。
- 电商保留工具＋参数授权及退货查询的两个匹配字段。
- 金融、IT、电网、旅游保留已有工具级授权；数据给出 `authorized_arguments` 时按原逻辑改为工具＋参数授权。
- Workspace 保留原项目名/ID 别名、大小写归一化、工单状态过滤和值班团队过滤。
- 授权标记用于观测评估，不强制阻止越界查询。
- 评估目标标注留在宿主侧，不用它选择模型查询返回的记录。

## Memory 与架构

Pointer 加载场景级提示，Direct 只选当前任务映射到的一条目标记录。保持当前原代码的 `plain_content_v1` 包装和仅向 A 注入的行为，未采用旧文档中已过时的包装描述。

Single-Agent、Agent-as-Tool 与 A2A 使用各自已有的角色提示词。A2A 保留本机 JSON-RPC 服务和双向请求/补充信息流程；不替换为普通函数模拟通信。

## 版本与复现边界

此目录是代码结构整理版。实验构建、查询结果、运行提示词和输入数据保持原有语义。模块名称和源码哈希会改变，历史代码快照与新版本应分别记录。

配置模板集中提供当前主实验模型设置。历史扩展实验跨多批次运行，精确复现某一批次还需要对照其记录的模型、API 类型、推理设置与端点；统一模板不保证重现历史数值。正式执行会生成新的结果，不读取或覆盖原 IDEA_003 的历史结果。
