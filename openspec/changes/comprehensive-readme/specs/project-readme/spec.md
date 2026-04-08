## ADDED Requirements

### Requirement: 根目录必须提供可用 README
仓库根目录 MUST 提供 `README.md`，用于让首次接触本项目的用户在不阅读源码的前提下完成“安装 → 准备资源 → 运行脚本 → 理解输出”的闭环。

#### Scenario: 首次阅读 README 的用户能完成快速开始
- **WHEN** 用户按照 README 的“快速开始（Quickstart）”步骤执行命令
- **THEN** 用户能明确需要准备的文件/目录（如 checkpoint、CSV 映射文件、输入图片目录），并能运行至少一个脚本得到输出结果

### Requirement: README 必须描述项目定位与边界
README MUST 明确说明本仓库的用途（如数据处理/推理/聚类脚本与模型包装），并明确不提供的能力边界（例如不承诺提供 ComfyUI 节点/工作流，除非代码中存在）。

#### Scenario: 用户能区分本仓库与参考仓库的功能边界
- **WHEN** 用户阅读“项目简介/范围（Scope）”章节
- **THEN** 用户能理解本仓库提供的脚本入口与预期用途，并知道需要到哪里获取/使用 ComfyUI 相关能力（若需要）

### Requirement: README 必须提供目录结构与关键入口说明
README MUST 以 `style_cluster_script.py` 为核心入口进行说明，并避免展开介绍其他脚本；允许仅在目录结构中将其余代码/模块描述为“实现细节/依赖”而不点名脚本文件。

#### Scenario: 用户能找到推理与聚类的入口文件
- **WHEN** 用户查看 README 的“目录结构”章节
- **THEN** 用户能定位到 `style_cluster_script.py` 作为入口，并知道它解决的问题与使用方式

### Requirement: README 必须包含安装与环境说明
README MUST 提供 Windows 下可复现的安装步骤（Python 版本建议、venv/conda 任选其一、`pip install -r requirements.txt`），并提醒 GPU/CPU 差异与常见安装风险点。

#### Scenario: 用户能在 Windows 创建环境并安装依赖
- **WHEN** 用户在 Windows（PowerShell）执行 README 给出的环境创建与依赖安装命令
- **THEN** 用户能完成依赖安装，并知道遇到 CUDA 不可用时如何回退到 CPU 或如何排查

### Requirement: README 必须提供模型下载与版本说明
README MUST 提供本仓库使用的画风模型（`Kaloscope-2.0`）下载渠道与版本差异说明（至少包含 v2/v1），并说明权重/映射文件应如何放置以供脚本使用。

#### Scenario: 用户能找到并下载正确版本的模型
- **WHEN** 用户阅读 README 的“模型下载/资源准备”章节
- **THEN** 用户能定位到 v2/v1 的下载入口，并理解各版本的来源与推荐使用顺序

### Requirement: README 必须包含 FAQ/排错章节
README MUST 包含常见问题与排错指引，至少覆盖：CUDA 不可用、checkpoint/类别数不匹配、路径与编码问题、输出目录找不到/为空等。

#### Scenario: 用户遇到常见报错时能在 README 中找到解决思路
- **WHEN** 用户在运行推理/聚类脚本时遇到常见错误
- **THEN** 用户能在 FAQ/排错章节中找到对应条目与处理建议
