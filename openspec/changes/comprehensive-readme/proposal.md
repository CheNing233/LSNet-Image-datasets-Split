## Why

当前仓库缺少一份可直接上手的 README，导致新使用者难以快速理解：项目是什么、如何安装依赖、如何准备数据/权重、如何运行推理脚本、以及与参考实现 `comfyui-lsnet` 的关系与差异。补齐 README 可以显著降低使用门槛，并减少重复沟通成本。

## What Changes

- 新增一份**全面的**根目录 `README.md`，覆盖项目定位、目录结构、环境安装、最小可运行示例、常见问题与排错。
- README 将包含面向 Windows/Python 的可复制命令（如 `pip install -r requirements.txt`、脚本调用示例）。
- README 将聚焦说明 `style_cluster_script.py` 的作用、配置方法、输出结构与常见排错，并将其它代码视为该入口的实现细节（不在 README 中展开）。
- README 将补充 `Kaloscope-2.0`（本仓库使用的画风模型）权重下载渠道与版本差异（v2/v1，HuggingFace/ModelScope）及放置约定。
- README 将加入“与 `comfyui-lsnet` 的关系”章节：说明本仓库用途（如数据集拆分/推理/训练辅助）、如何对齐模型命名与权重来源、以及不包含/不覆盖的功能边界。
- 不引入破坏性变更；仅新增文档与可能的轻量示例资源链接（不提交大文件）。

## Capabilities

### New Capabilities

- `project-readme`: 为本仓库提供一份可操作的 README（安装、快速开始、目录结构、脚本用法、FAQ/排错）。
- `style-cluster-script-docs`: 针对 `style_cluster_script.py` 的画风相似度分组/拆分流程提供清晰说明（配置项、输出结构、阈值扫参与归档策略、FAQ/排错）。
- `comfyui-lsnet-alignment`: 解释与参考仓库 `comfyui-lsnet`（LSNet 模型）之间的对应关系（模型名、输入尺寸、checkpoint/类映射文件等）与使用边界。

### Modified Capabilities

- （无）

## Impact

- 受影响文件/目录：新增根目录 `README.md`；可能新增 `docs/` 或 `assets/` 用于存放小体积示例（如目录树截图/小示例文件说明），但不引入大型数据与权重。
- 对外接口/API：无。
- 依赖：无新增强制依赖（README 可能建议可选依赖/工具，如 `conda`、`uv`，但不作为硬要求）。
