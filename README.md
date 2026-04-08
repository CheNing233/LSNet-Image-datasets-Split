# lsnet-datasets-split

一个基于 **Kaloscope-2.0** 的“画风相似度”数据集拆分工具：通过提取图片风格特征并聚类/分组，辅助把数据集按画风相近性进行整理与拆分（例如划分训练集/验证集，或按风格归档）。

本仓库的**核心入口**是 `style_cluster_script.py`（离线脚本流程，Python），不包含 GUI / ComfyUI 节点实现。

## Quickstart

1) 安装依赖（Windows / PowerShell）

```powershell
pip install -r requirements.txt
```

2) 配置 `style_cluster_script.py`

打开 `style_cluster_script.py`，在文件顶部“配置区”至少设置：

- `INPUT_DIR`：你的图片数据集目录
- `OUTPUT_DIR`：输出目录
- `MODEL_DIR`：模型目录（默认 `.\Kaloscope-2.0`）
- `DEVICE`：`cuda` 或 `cpu`

建议第一次先设：

- `DRY_RUN = True`（只预览将要归档的结果，不实际复制/移动文件）

3) 运行

```powershell
python style_cluster_script.py
```

## 核心脚本：style_cluster_script.py（做什么、怎么做）

`style_cluster_script.py` 用于对一批图片做“画风相似度分组”，输出可视化与分组结果，并可选把文件按簇归档到不同目录。典型用途：

- 按画风相近性把数据集拆分成若干簇，便于后续挑选/构建训练集与验证集
- 对混杂来源的数据做风格归档，快速发现风格重复/污染

### 工作流程（概览）

- **扫描图片**：支持递归扫描与常见图片后缀
- **提取风格特征**：对每张图片提取特征向量（batch 处理）
- **可选降维**：对特征做 PCA 降维（更快/更稳，可能损失信息）
- **阈值自动选择**：在一个阈值范围内扫参，选择最接近期望簇数的阈值
- **聚类分组**：用“余弦相似度阈值 + 并查集”把相似图片连通成簇
- **输出结果**：保存特征/摘要/可视化/精灵图
- **可选归档**：按簇复制或移动图片（以及同名 `.txt` sidecar）

### 资源放置约定（模型目录）

默认 `MODEL_DIR = .\Kaloscope-2.0`，该目录需要包含：

- `best_checkpoint.pth`
- `class_mapping.csv`
- （可选）`config.json`（形如 `{"model": "lsnet_xl_artist"}`，用于选择模型变体）

也可用环境变量覆盖少量关键配置：

- `LSNET_STYLE_INPUT_DIR`
- `LSNET_STYLE_OUTPUT_DIR`
- `LSNET_STYLE_MODEL_DIR`

### 输出说明（非常重要）

脚本会在 `OUTPUT_DIR/tasks/<task_id>/runs/<timestamp>/` 下生成：

- `features.npy` / `image_names.txt`：特征矩阵与文件名列表
- `summary.json`：分组结果与统计信息（阈值选择、簇大小、每张图的簇 ID 等）
- `threshold_sweep.png`：阈值扫参与簇数关系图
- `scatter.png`：二维可视化（若启用）
- `sprites/cluster_XXX*.png`：每个簇的精灵图（若启用，图片多会分页）
- `clusters/cluster_XXX/*`：按簇归档后的文件（仅当 `DRY_RUN=False`）

### 关键配置项（建议先了解）

- **安全模式**：`DRY_RUN=True` 只打印计划操作；确认无误后再设为 `False`
- **文件操作**：
  - `COPY_MODE=True` 复制；`False` 移动（更危险）
  - `COLLISION_POLICY="rename"` 避免同名覆盖
- **阈值与簇数**：
  - `THRESH_SWEEP_MIN/MAX/POINTS` 控制扫参范围与密度
  - `EXPECTED_N_CLUSTERS` 用于选择“最接近期望簇数”的阈值
- **性能/内存**：
  - `SIM_CHUNK_SIZE` 控制相似度分块计算的块大小（越大越快但更吃内存）
  - `TOP_K_EDGES` 可限制连边数量，降低“链式效应”与 \(O(N^2)\) 压力


## 模型下载（Kaloscope）

本仓库相关脚本使用的画风模型为 **Kaloscope** 系列（通常你会获得一个 checkpoint 文件，以及训练导出的类别映射 CSV）。

### v2（Kaloscope2.0）

- HuggingFace：https://huggingface.co/heathcliff01/Kaloscope2.0
- ModelScope：https://www.modelscope.cn/models/Heathcliff02/Kaloscope-2.0/summary

### v1（Kaloscope）

- HuggingFace：https://huggingface.co/heathcliff01/Kaloscope/tree/main
- ModelScope：https://www.modelscope.cn/models/Heathcliff02/Kaloscope/files

### “最新模型”参考

最新模型与对齐信息可参考 spawner 的仓库（但这里并不常更新）：https://github.com/spawner1145/comfyui-lsnet

## FAQ / 排错

### 1) CUDA 不可用 / 只能用 CPU

- 现象：`torch.cuda.is_available()` 为 false 或报 CUDA 相关错误
- 建议：先用 `--device cpu` 跑通流程，再排查 CUDA/PyTorch 安装与驱动环境

### 2) 分类模式提示必须提供 --class-csv
如果你在使用 `style_cluster_script.py` 时缺少 `class_mapping.csv`，请检查 `MODEL_DIR` 目录是否包含该文件。该 CSV 用于与训练配置保持一致，并作为模型资源的一部分被脚本要求存在。

### 3) 输出目录为空 / 没有 clusters 目录

- 如果 `DRY_RUN=True`：脚本不会实际复制/移动文件，因此不会生成 `clusters/` 的归档结果（这是预期行为）。
- 请查看 `OUTPUT_DIR/tasks/<task_id>/runs/<timestamp>/summary.json` 是否已生成，以确认聚类流程是否完成。

---

如果 README 中的示例命令与你本地环境不一致，请以 `--help` 输出为准，并欢迎提 issue/PR 补充可复现步骤。

