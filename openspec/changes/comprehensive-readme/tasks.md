## 1. 盘点与信息收集

- [x] 1.1 盘点仓库可公开说明的入口脚本与依赖（`requirements.txt`、`inference_artist.py`、`style_cluster_script.py`、`lsnet_model/`）
- [x] 1.2 梳理需要用户自行准备的外部资源清单（checkpoint、`class_mapping.csv`、可选 `config.json`、输入图片目录）

## 2. 设计 README 信息架构

- [x] 2.1 确定 README 章节结构（简介/Quickstart/安装/目录结构/脚本用法/资源准备/FAQ）
- [x] 2.2 明确与 `comfyui-lsnet` 的对齐与边界表述（避免不准确功能声明）

## 3. 编写根目录 README.md（主体）

- [x] 3.1 新增 `README.md`：项目简介 + 目录结构说明
- [x] 3.2 新增 `README.md`：环境安装（Windows/PowerShell，venv 或 conda）与依赖安装说明
- [x] 3.3 新增 `README.md`：快速开始（最小可运行命令），包含 `inference_artist.py` 的 classify 示例
- [x] 3.4 新增 `README.md`：`inference_artist.py` 参数与模式说明（classify/cluster/both），以及输出产物位置说明
- [x] 3.5 新增 `README.md`：`style_cluster_script.py` 用法说明（配置区、环境变量覆盖、输出目录结构、DRY_RUN/COPY_MODE 等关键点）
- [x] 3.6 新增 `README.md`：与 `comfyui-lsnet` 的关系/对齐章节（链接、映射思路、权威来源提示）

## 4. FAQ / 排错与可维护性

- [x] 4.1 补充常见问题与排错（CUDA 不可用回退 CPU、`--class-csv` 缺失、分类头不匹配、路径/编码问题）
- [x] 4.2 校对 README 中的命令/参数与脚本保持一致（必要时引用 `--help` 作为权威）
