## Context

本仓库（`lsnet-datasets-split`）以 Python 为主，包含：

- `lsnet_model/`：注册/实现 LSNet 相关模型（看起来通过 `timm.create_model` 使用）。
- `inference_artist.py`：面向“画师风格/风格分类与特征提取”的推理脚本，支持 `classify/cluster/both` 三种模式。
- `style_cluster_script.py`：独立可运行的批量聚类脚本，基于特征提取 + 阈值聚类 + 可视化/精灵图输出（并包含 Windows 友好的路径/字体处理）。
- `requirements.txt`：依赖声明（包含 `torch`、`timm`、`triton-windows` 等）。

当前缺少根目录 `README.md`，导致使用者无法快速回答以下关键问题：

- 项目定位是什么？与参考模型仓库 `comfyui-lsnet` 的关系与差异是什么？
- 如何在 Windows 上准备 Python 环境、安装依赖？
- 需要哪些外部资源（checkpoint、`class_mapping.csv`、数据目录），从哪里获得？
- 如何运行推理/聚类脚本、输出会生成在哪里、常见报错如何排查？

约束与假设：

- README 内容以中文为主（与脚本内文档语言一致），并提供可复制命令。
- 仓库不包含权重/数据集等大文件，README 只能通过“下载链接/放置路径约定”来引导。
- README 需要明确本仓库所用画风模型（`Kaloscope-2.0`）的下载渠道（v2/v1，HuggingFace/ModelScope）以及“最新模型参考”来源说明。
- 对齐 `comfyui-lsnet` 时，优先描述概念/命名映射与可复现路径，避免声称本仓库具备 ComfyUI 节点能力（除非代码中确实存在）。

## Goals / Non-Goals

**Goals:**

- 产出一份可直接上手的根目录 `README.md`，覆盖：
  - 项目简介（用途、适用场景、与 `comfyui-lsnet` 的关系）
  - 目录结构（关键文件/脚本入口）
  - 环境安装（Windows/conda 或 venv + pip）
  - 快速开始（最小推理示例、批处理/聚类示例）
  - 资源准备（checkpoint、类映射 CSV、可选 config 等的放置约定）
  - 常见问题与排错（CUDA/CPU 回退、分类头不匹配、路径与编码问题）
- README 的命令与参数示例与仓库当前脚本一致（不编造不存在的参数）。
- 明确脚本输出产物与目录（例如 `inference_artist.py` 的 `--output`，`style_cluster_script.py` 的 `OUTPUT_DIR/tasks/<task_id>/runs/<timestamp>/...`）。

**Non-Goals:**

- 不实现/新增任何模型训练流程或 ComfyUI 插件/节点（README 只做说明与对齐，不扩展功能）。
- 不将外部数据集或模型权重纳入仓库版本控制（README 仅提供下载与放置指引）。
- 不对现有脚本做大规模重构（若需轻微调整以匹配文档，将在任务阶段单独列出并最小化改动）。

## Decisions

- **README 信息架构采用“从快到深”**：先给出 Quickstart（最小可跑命令）→ 再展开目录/脚本/高级用法 → 最后 FAQ。  
  - *Alternative*: 先讲原理/模型结构再讲使用。  
  - *Why*: 文档的首要价值是让人“跑起来”，原理放后更符合使用者路径。

- **以仓库真实入口脚本为权威来源**（`inference_artist.py`、`style_cluster_script.py`、`requirements.txt`），README 只描述已存在能力。  
  - *Alternative*: 参照 `comfyui-lsnet` README 直接迁移章节。  
  - *Why*: 迁移容易引入不匹配内容（例如 ComfyUI 节点/工作流），会误导用户。

- **“与 `comfyui-lsnet` 的关系”采取对齐映射而非功能等价声明**：  
  - 描述可能的共同点：同名模型/权重来源、输入尺寸约定、checkpoint 格式。  
  - 同时明确本仓库关注点：数据/推理/聚类脚本、离线流程。  
  - *Why*: 避免用户期待“装上就能在 ComfyUI 里用”的体验。

- **示例路径策略**：README 命令示例同时给出 Windows（PowerShell）与跨平台写法（相对路径/引号），但不强制支持 Linux。  
  - *Why*: 用户环境信息显示为 Windows；同时尽量不妨碍其他平台用户阅读。

## Risks / Trade-offs

- **[风险] 对齐 `comfyui-lsnet` 时信息不全/版本差异** → **缓解**：只写“可核对项/映射思路”，并在 README 中指向参考仓库作为权威来源。
- **[风险] README 示例命令与脚本参数漂移**（未来脚本改动）→ **缓解**：README 明确“以 `--help` 输出为准”，并在关键脚本章节提示查看帮助。
- **[风险] Windows CUDA 环境复杂** → **缓解**：提供 CPU 回退说明、常见错误（CUDA 不可用、`triton-windows` 安装问题）与排查步骤。
