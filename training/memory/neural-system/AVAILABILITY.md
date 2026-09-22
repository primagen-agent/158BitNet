# 神经可用性/任务分支

状态：按 DESIGN 第 4 节补齐原型；不是已训练的拒答器，也不是完整记忆系统。
DG-006 已通过本轮登记的数值与 v2 后端诊断，语义能力仍未训练/验收。
数值与 v2 生成诊断登记于 `experiments/DG-006.json`，先登记、再实现和执行，不放行 CG-001。

## 固定边界与假设

旧内容分支在空库时为常量零，冻结 backbone 下无法从生成损失学习证据不足的回应。
新增一个与内容读出分开的神经分支，输入仅为当前 C hidden、神经内容读出、
两者交互、NULL 注意力质量和是否有来源的结构事实，不接收 gold 标签、场景 ID 或答案。
本原型仍只支持单个给定事件；它不是已完成的多事件 reader，更不是自动 writer。

神经 softmax 预测三个状态：无需记忆、证据可用、证据不足。
冲突在此属于证据不足，尚未单独实现或验证冲突归因。
空库从结构上没有可用记忆内容，屏蔽“证据可用”状态；不是用规则识别问题语义。
普通聊天和私有信息请求都仍由神经模块区分，不按问号、姓名或关键词选择分支。

```text
Δhidden = p(证据可用) × Δcontent + p(证据不足) × U(state_hidden)
logits = C_base_logits + frozen_GGUF_output_weight × Δhidden / logit_scale
```

所有投影全秩；不使用 LoRA、SVD、NAS、RAG 原文拼接或固定拒答文本。
“无需记忆”是零增量选项，不能把每个问题都强制导向不确定回答。
可用性分支通过完整生成损失学习回复，状态分类标签仅用于后置辅助 loss，不能传入 forward。

## 初始化、验收和停止条件

- 新分支的 U 输出矩阵零初始化，没有另加零 gain，避免两个零因子造成永久死梯度。
  初始 logits 保持基础模型；第一步生成损失可到达 U，状态辅助 loss 可训练分类网络。
  不能声称零初始化时生成损失已经传到上游每个参数。
- 功能关闭必须精确保留基础 logits；空库内容增量必须为零，可用性增量允许非零。
- 增加普通聊天与证据不足的对照，报告它们的原始预测分布；随机初始化不能当作正确分类。
- 数值验证同时检查输出容差和真实 C 损失方向，不只检查 grad 非零。
- v2 自由生成保持全前缀重算，停止/截断原样记录，不复制参考答案，不做训练或参数选择。
- 后续真实训练必须同时含无需记忆、缺证据与可用证据，且验证普通聊天不退化。
  当前三状态原型并不自动证明能识别无关、矛盾、否定、错误主体或时间。

DG-006 通过至多证明空库可学习路径存在及本轮数值/后端行为；不能解除数据冻结、
独立评分校准、可学习性和泛化验收等前置项。原空库 bypass 数值证据保留为旧内容分支对照。

## 实现与复现

- `python/memory_availability.py`：三状态 softmax 与独立可用性生成增量。
- `python/native_continuous_generation.py`：新增开启但空库/有来源两个诊断条件，记录每个生成位置的状态概率。
- `python/diagnose_memory_availability.py`：真实 C 投影、梯度方向、参数恢复及 v2 生成检查。
- `tests/test_memory_availability.py`：9 项初始化、空库梯度、后置分类 loss、关闭、通路、身份和重载检查。

```sh
python3 tests/test_memory_availability.py
python3 python/diagnose_memory_availability.py \
  --experiment training/memory/neural-system/experiments/DG-006.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --probe build/memory_continuous_probe \
  --reference-probe build/memory_gradient_reference --encoder-probe build/memory_feature_probe \
  --tok-probe build/tok_probe --lib build/libggwshim.so --output build/neural-memory-dg006-new
```

输出目录拒绝覆盖。保留随机初始化权重以便复现，文件明确为 `untrained-module.pt`，
不是训练产物；临时梯度扰动结束即恢复。backbone、词表输出矩阵均未训练。
神经融合/可用性模块仍在 Python，C 只执行冻结 backbone 和连续输出投影，未接入完整服务。

## 实测结果与准确性边界

登记 `checks/DG-006-result.json`；原始产物 `build/neural-memory-dg006/`。
三个登记样例（空库私有问题、空库普通算术、单个中文来源）全部通过本轮数值门槛：

- 关闭和默认零初始化在三个输入上均与普通 C logits 逐位一致。
- 空库内容增量仍为零，生成损失到可用性输出矩阵的初始梯度范数分别为 2.5961、3.0093，
  不再是旧内容 bypass 的无训练通路。
- 21 组真实 C/Torch 前向对照通过登记容差，合成 logits 最大绝对误差 `9.54e-7`。
- 3/3 个登记有限差分与自动求导一致；6/6 组矩阵扰动使 C NLL 在下降方向降低、反方向升高。
- v2 共生成 18 个 token（含停止 token），每个位置从零重算完整前缀，无跨 token KV 复用。
  两个英文回复达到预登记的 6 token 上限，中文回复遇 EOS；均保留原文和截断标记。

| 场景 | 随机模块原始输出 | 能力结论 |
| --- | --- | --- |
| 空库询问 Nora 住处 | `Nora lives in the small`（截断） | 无证据仍开始猜测，不是通过 |
| 空库二加二 | `Two plus two is equal to`（截断） | 未完成答案，不报告正确 |
| 来源是“小安现在住在苏州” | `小安住在上海。`（正常停止） | 与来源不符，recall 未成功 |

随机状态分类也不可靠：算术问题“证据不足”概率约 0.6901，不能把 softmax 数字当作校准置信度。
数值目标 token 只用于方向诊断，按单独文本编码得到；不是完整回答的训练 NLL 或准确率。
本轮 Python **120/120**、C **21/21**；优化器步骤为 0，未提交代码或修改部署模型。

## 下一工作项

数值前置检查已有证据，下一步转向三状态监督与生成监督的独立数据/训练接口，而非继续要求随机权重答对：

1. 冻结新语义世界、三种状态、换值/删目标/无关/冲突/普通聊天等配对条件及比例，标签和输入物理分离。
2. 状态辅助监督以当前用户前缀为依据，不让 teacher-forced 答案前缀泄漏目标状态；
   本原型生成时每个位置重新计算状态，尚未完成 P2C 的固定读决策集成。
3. 完整回复训练目标必须通过同一个 C tokenizer 对完整上下文及回复联合编码，核验前缀边界；
   不能直接沿用诊断中孤立目标文本的 dummy-space 处理作为训练标签对齐方案。
4. 完成既定的独立评分校准、冻结训练配置/停止条件和来源清单后，才启动 CG-001。
   语义正确性在受控训练后验收，不把“未训练随机分类器还不准确”变成无限延迟训练的前置条件。

CG-001 的 P1/数据/校准前置项仍未全部满足；未启动正式训练，不进行 LoCoMo 选模。

后续 P1-supervision-12 已完成首版新世界语料、分离 loss 接口、真实 C 完整回复边界审计，
并经用户授权完成小规模隔离模型盲审；见 `TRAINING_DATA.md`。实际特征抽取/训练器衔接尚待实施，
不把 teacher-forced 数据准备或盲审夹具正确率当作训练/记忆成果。
