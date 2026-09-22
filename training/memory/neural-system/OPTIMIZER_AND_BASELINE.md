# CG-003：固定优化循环与完整回复基线

目前已经实现有界训练引擎、检查点写入/加载校验，并完成固定24例的无记忆 C 基线。
**没有对真实0.5B记忆模型进行新的参数更新，没有新的候选模型或记忆准确率结果。**

## 1. 优化循环

实现：`python/joint_optimizer.py`。公开训练入口仍首先调用无条件拒绝执行的 `require_joint_launch`；
不能仅改 JSON 的 approval 字段就运行，命令行也没有放行训练。
内部引擎已实现，不再是只写方案；但 GPU 正式启动、授权清单与端到端训练验收还未完成。

- 固定 seed 1013 初始化两组相同权重，各50步，不恢复旧检查点，不自动延长。
- 每步4对/8次记录使用，先对每对取两条样本损失均值，再加成对损失，最后对4对取均值。
- 每对完整前向后计算梯度，再逐对累积，保留成对因果图，同时避免常驻4组大词表计算图。
- AdamW：学习率1e-4、weight decay 0.01，梯度裁剪1.0，FP32，无 AMP/TF32。
- 只优化 `requires_grad` 参数，冻结输出头和未用私有分类器；不更新骨干，不训练记忆数据。
- 每对验证应连接梯度及有限性，整个批次成功后才更新。中途失败会清空累积梯度，不把半个批次当一步。
- 每步检查权重/AdamW 状态有限，冻结参数不变。某分支本步未连接时梯度为 None，不沿用上一步梯度。

完整特征包与已冻结采样表的只读绑定核验通过：50步、200对、400次记录使用、268条独立记录。
采样表摘要：`46dff2f29d5a37a9d5cbeb255ce8dc1b5cb8124856d921ebfc0f90b33d5cc7ce`。
每对的 split、world、language、relation、scenario 与原数据一致；完整包样本接口仍拒绝 dev 优化器输入，test 不入包。
这只是固定抽样的可追溯性，不是跑完训练或跑完数据全量 epoch。

## 2. 检查点

研究格式 `cg003-joint-research-checkpoint-v1`，只允许第0步和第50步保存，不能作为服务模型发布。
包含策略、初始/当前参数摘要、精确参数名、优化器状态、随机状态、骨干/模板身份、
完整特征/采样表/实验/源码/启动证据绑定。优化器状态仅供审计，没有 resume 入口。

先写临时文件并刷盘，再以不覆盖目标的方式发布；已有检查点不能被重跑覆盖。
加载首先核对文件 SHA，使用 `weights_only=True`；检查策略、绑定、参数名、shape/dtype、有限性、
状态摘要和冻结参数，全部成功后才修改目标模型。目标必须是相同的全新初始化。
错误模型、损坏文件、来源不匹配、改动冻结参数及伪装第0步均拒绝。
本轮只在临时目录保存人工小模型检查点，未产生0.5B候选检查点。

## 3. 固定24例无记忆基线

面板来自预先冻结的 `checks/CG-003-data.json`，没有根据生成效果筛选：
dev世界 `jb-008` × 两语言 × 两关系 × 六种条件，24例；test保持封存。
greedy、64-token上限、128-token容量，保持登记的 GGUF ChatML 模板。

`python/eval_joint_baseline.py` 从自然问题开始，每个实际生成位置调用 C fresh context。
没有读取标签、参考回复或离线前缀特征包；无记忆来源进入生成提示。
同批次仅共享只读骨干权重，每个样本/位置都重建上下文，不复用历史 KV；单次 prefill 内仍使用 K/V 工作缓冲。

| 可验证项目 | 结果 |
| --- | ---: |
| 固定用例 | 24/24 |
| 正常 stop token 结束 | 24/24 |
| UTF-8 完整 | 24/24 |
| 截断 | 0/24 |
| 真实 C 生成位置 | 252 |
| C batch 调用 | 12 |
| 跨请求/位置缓存复用 | 0 |

基线 SHA：`4389fc1c29a28f81353dac9ecc52bec8bb024fd54c658ceecc8485db5874433d`。
原始结果 `build/neural-memory-cg003-baseline/`；报告按原始字节保留在 `reviews/CG-003/baseline.json`。
`python/audit_joint_baseline.py` 已逐个核对252个位置的输入 bytes、自然初始前缀、实际生成后继、
原始输出 SHA、fresh-context trace、argmax token、终止和文本解码；没有使用新的生成替换失败。
审计见 `checks/CG-003-baseline-audit.json`。

无记忆基线对于同一问题的正确来源、换值、角色交换及空记忆条件输出相同。
例如英文居住问题回复 `Seline is currently living in the United States.`，中文居住问题回复 `Seline目前居住的城市是东京。`。
这不能当作 recall：它根本没有读取记忆，也没有随值变化。
正式独立语义评审还未执行，因此这里不列“准确率”，也不拿基线行为选模型或调整训练预算。

已经生成去身份盲包 `reviews/CG-003/baseline-blind.json`，只包含自然上下文、来源和实际回复，
没有候选名称、标准答案或训练损失。后续候选仍须在同一面板预测路由生成，并由两位独立评审核验。

## 4. 软件验证和剩余工作

新增11项测试；Python **245/245**，现有 CTest **21/21** 通过。
单元测试在 **8维隐藏层/13词表的人工张量** 上实际执行优化器，验证两种策略的完整50步、
累积梯度等价性、冻结参数、错误批次不更新、非活动分支梯度清空、最终检查点往返与原始前缀篡改拒绝。
这些小模型更新仅测试软件行为，不是使用真实训练数据的新候选，不计作记忆能力证据。

下一项：独立语义评审/评分接口、完整训练启动证据绑定与GPU执行路径资格检查。
还需验证完整训练包同步、启动后硬步数限制、候选检查点及固定面板端到端衔接。
两组各50步/总100步的正式预算仍未批准；保持 `training_approved=false`、`prerequisites_complete=false`、`launch_command=null`。
不把已通过的初始化 CPU/CUDA parity 当作新优化器循环或训练后模型的 GPU 数值证书。

## 复现基线和审计

```sh
python3 python/eval_joint_baseline.py \
  --corpus training/memory/neural-system/data/JB-001 \
  --audit training/memory/neural-system/checks/CG-003-data.json \
  --audit-sha256 f191a6aa27c5f6420d8fb8e0f318b562986ec9d1298eda766a1092ff51b8601f \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --prefix-probe build/memory_prefix_probe --tok-probe build/tok_probe \
  --output build/neural-memory-cg003-baseline-new

python3 python/audit_joint_baseline.py \
  --baseline build/neural-memory-cg003-baseline/baseline.json \
  --baseline-sha256 4389fc1c29a28f81353dac9ecc52bec8bb024fd54c658ceecc8485db5874433d \
  --panel-audit training/memory/neural-system/checks/CG-003-data.json \
  --corpus training/memory/neural-system/data/JB-001 \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --tok-probe build/tok_probe \
  --output build/neural-memory-cg003-baseline-audit-new
```

输出路径须不存在。新生成报告如摘要变化，应审查具体差异，不覆盖冻结基线或偷偷替换期望 SHA。
