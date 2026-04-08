## ADDED Requirements

### Requirement: README 必须覆盖 `inference_artist.py` 的核心用法
README MUST 解释 `inference_artist.py` 的用途（画师风格推理/特征提取），并提供可复制的命令示例，覆盖至少以下信息：
- 必需输入：`--checkpoint`、`--input`
- 分类模式所需：`--class-csv`
- 模式选择：`--mode`（`classify/cluster/both`）
- 输出目录：`--output`
- 常用参数：`--device`、`--batch-size`、`--top-k`、`--threshold`、`--allow-head-reinit`

#### Scenario: 用户可以从 README 直接复制命令进行单图推理
- **WHEN** 用户准备好 checkpoint 与 class CSV，并复制 README 的“单张图片推理（classify）”示例命令
- **THEN** 脚本能输出 top-k 分类结果并在 `--output` 目录下生成对应的 JSON 结果文件

### Requirement: README 必须解释分类/聚类模式对资源的要求差异
README MUST 明确说明：
- `classify` / `both` 模式 MUST 提供 `--class-csv`（与训练导出的映射一致）
- `cluster` 模式不依赖分类头结果，但仍需要 checkpoint 以提取特征
- checkpoint 的 `num_classes` 与 CSV 类别数不匹配时的处理策略（默认终止；可用 `--allow-head-reinit` 或改用 `cluster`）

#### Scenario: 用户理解为什么 classify 必须提供 class CSV
- **WHEN** 用户阅读 README 的“参数说明/常见错误”部分
- **THEN** 用户能知道 `--class-csv` 的来源与作用，并能在缺失时预期脚本会报错

### Requirement: README 必须说明输出产物与格式
README MUST 说明 `inference_artist.py` 在不同输入类型下的输出：
- 单图：`<stem>_result.json`（包含 classification 与/或 features）
- 目录批处理：`batch_results.json`，以及在 `cluster/both` 时可能额外生成 `features.npy` 与 `image_names.txt`

#### Scenario: 用户知道在哪里找到 features.npy
- **WHEN** 用户以目录输入并使用 `--mode cluster` 或 `--mode both` 运行脚本
- **THEN** 用户能在 README 指定的位置找到 `features.npy` 与 `image_names.txt` 并理解其含义
