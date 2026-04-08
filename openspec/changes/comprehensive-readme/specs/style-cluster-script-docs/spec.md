## ADDED Requirements

### Requirement: README 必须以 `style_cluster_script.py` 为唯一核心入口
README MUST 将 `style_cluster_script.py` 作为唯一核心入口脚本进行详细介绍，并避免在 README 中展开介绍其他脚本（其余代码仅作为实现细节/依赖存在）。

#### Scenario: README 中不出现其他脚本的章节化介绍
- **WHEN** 用户通读 README
- **THEN** README 的“脚本/用法”相关内容只围绕 `style_cluster_script.py` 展开，不包含对其他脚本的独立章节或参数说明

### Requirement: README 必须解释脚本的目标与典型使用场景
README MUST 说明脚本用于“基于画风相似度对数据集进行分组/拆分/归档”，并给出典型使用场景（例如按风格归档、辅助构建训练/验证划分、发现风格污染或重复）。

#### Scenario: 用户理解脚本解决的问题
- **WHEN** 用户阅读 README 的“脚本做什么”部分
- **THEN** 用户能明确输入是什么、输出是什么、以及它为何有助于数据集拆分

### Requirement: README 必须描述脚本的整体流程与关键算法点
README MUST 描述流程：扫描图片 → 特征提取 →（可选）PCA → 阈值扫参选择阈值 → 余弦相似度阈值连边 + 并查集聚类 → 输出摘要/可视化/精灵图 →（可选）复制/移动归档。

#### Scenario: 用户能将输出产物对应回流程步骤
- **WHEN** 用户查看输出目录（features/summary/threshold_sweep/scatter/sprites/clusters）
- **THEN** 用户能理解每个产物对应流程中的哪一步与其用途

### Requirement: README 必须说明关键配置项与安全使用建议
README MUST 说明至少以下关键配置项：`INPUT_DIR`、`OUTPUT_DIR`、`MODEL_DIR`、`DEVICE`、`DRY_RUN`、`COPY_MODE`、`COLLISION_POLICY`、`EXPECTED_N_CLUSTERS`、`THRESH_SWEEP_*`、`SIM_CHUNK_SIZE`、`TOP_K_EDGES`，并强调先使用 `DRY_RUN=True` 预览，再执行真实文件操作。

#### Scenario: 首次使用者能避免误移动/误覆盖文件
- **WHEN** 用户首次运行脚本并按 README 建议设置 `DRY_RUN=True`
- **THEN** 用户能先查看计划操作与摘要结果，再决定是否切换到复制/移动归档

### Requirement: README 必须清晰说明输出目录结构与含义
README MUST 明确输出目录的层级结构（`OUTPUT_DIR/tasks/<task_id>/runs/<timestamp>/...`）与各文件含义，并说明在 `DRY_RUN=True` 时不会产生 `clusters/` 的实际归档结果。

#### Scenario: 用户能定位 summary.json 并理解为何没有 clusters
- **WHEN** 用户设置 `DRY_RUN=True` 运行脚本
- **THEN** 用户能在 run 目录找到 `summary.json` 等产物，并理解 `clusters/` 不生成是预期行为

