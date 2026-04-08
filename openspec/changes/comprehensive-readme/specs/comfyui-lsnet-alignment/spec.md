## ADDED Requirements

### Requirement: README 必须描述与 `comfyui-lsnet` 的关系与对齐方式
README MUST 包含“与 `comfyui-lsnet` 的关系”章节，内容至少包括：
- `comfyui-lsnet` 的定位（参考实现/模型来源/ComfyUI 生态相关）与本仓库定位（脚本化推理/聚类/数据处理）
- 共同点与可核对项：模型变体命名、输入尺寸、checkpoint 组织方式（在本仓库脚本中对应的参数/配置）
- 约束：本仓库不承诺提供 ComfyUI 节点或工作流（除非实际存在），并建议以参考仓库为权威来源

#### Scenario: 用户知道该去哪里获取权重与参考说明
- **WHEN** 用户阅读该章节
- **THEN** 用户能明确参考仓库用于“模型/权重/ComfyUI 使用”，而本仓库用于“离线脚本流程”，并能找到参考仓库链接

### Requirement: README 必须避免不准确的功能声明
README MUST 只描述本仓库当前存在的文件/脚本能力，不得暗示已集成 ComfyUI 或提供图形化工作流（除非仓库内有对应实现与入口）。

#### Scenario: README 的功能描述可被仓库内容验证
- **WHEN** 读者根据 README 中提到的脚本/文件名在仓库内检索
- **THEN** 这些脚本/文件实际存在，且参数/用法与 README 示例一致或可通过 `--help` 验证
