# P1 独立评分与 C/Torch 特征核验

状态：工具与小样本数值诊断；不是记忆准确率报告。阶段状态见 `PLAN.md`。

## 1. 自由生成的离线独立评分

入口：`python/review_neural_memory.py`，只读取本地文件，不调用在线裁判或发送数据。

预测 JSONL 每条包含 `id`、原始 `text`、`input_sha256`、`producer_id`、`trace`。
`input_sha256` 是 `prepare_neural_memory_protocol.digest(runtime_input)`；trace 必须完整满足
`InferenceTrace` 契约，明确组件/oracle/端到端口径，不能混入 KV 复用或整句替换。
同一次评分禁止混合不同被测模型或不同评测层级。

两步流程：

```sh
# 必须已有模型实际生成的 predictions.jsonl；不能用标准答案填充。
python3 python/review_neural_memory.py pack \
  --config training/memory/neural-system/data/P1-smoke.json \
  --corpus build/neural-memory-p1-data-02 \
  --predictions build/predictions.jsonl --output build/blind-review.json

# 将两个独立评审者填写的记录汇入 reviews.jsonl 后执行。
python3 python/review_neural_memory.py score \
  --config training/memory/neural-system/data/P1-smoke.json \
  --corpus build/neural-memory-p1-data-02 \
  --predictions build/predictions.jsonl --reviews build/reviews.jsonl \
  --reviewer reviewer-a --reviewer reviewer-b --output build/review-score.json
```

盲审包只有当前上下文、来源、回复、匿名化 ID 和包哈希。
原始 case ID 含场景名字，也被替换成摘要；不包含标准答案、目标事件、场景标签、模型名称或分数。
评审者只根据上下文和来源，提取回复的主体/关系/值/时间/事实状态，并判断自然性、无依据确定回答及操作旁白。
代词必须结合语境消解；角色方向、历史时间、否定、假设和引用不可被改成正向当前事实。

评审 JSONL 每条为 `id`（盲审包的匿名 ID）、`packet_sha256`、`adjudication`；后者字段为：
`response_sha256`、`reviewer_id`、`claims`、`acknowledges_missing_evidence`、`natural_reply`、
`unsolicited_memory_narration`。claims 每条必须有 subject/relation/value/time/status。

校验规则：

- 两个登记的独立评审者必须一致，才能判定自由回复通过/失败。相同分数但事实提取不同仍是分歧，不用多数票自动通过。
- 缺输出、缺评审或评审分歧记 `needs_review`，保留在总分母中；空回复直接失败。
- 拒绝重复投票、未知评审者、被测模型自己评审及不匹配的上下文/回复哈希。
- 汇总同时保留场景、语言和语义世界。704 行不当作 704 个独立样本；未全部审完不报告完整准确率。
- 工具输出 `promotion_eligible=false`：完成评分流程本身不代表阶段能力达标。

审阅者 ID 不是独立性的自动证明，登记和校准由可信操作者负责。
正式校准需覆盖正确答案及错主体、角色、时间、否定/假设、无目标、自然性等反例，
保存双方原始意见和分歧解决依据；再用未参与规则修改的样例复核。
不得把两份由同一被测模型产生的评审当成两位独立评审者。
本轮只完成软件工作流和人工编写的单元样例，**尚未完成实际独立人工校准，也未评分真实新模型回复**。

## 2. 数值对照范围与复现

探针：`tools/memory_feature_probe.c`；诊断脚本：`python/diagnose_neural_memory_features.py`。
只对固定 SHA-256 的 BitCPM 0.5B 做 CPU FP32 对照，未加载记忆模型。
固定输入见 `data/P1-feature-probes.json`：英文、中文、角色方向、ChatML、多字段消息框及重复英文。
共 6 次输入、5 个不同文本、24 个层位置，不是语义测试样本量。

```sh
cmake -S . -B build
cmake --build build --target memory_feature_probe tok_probe -j 8
arch -arm64 cc -O2 -fPIC -shared python/ggwshim.c -I src -o build/libggwshim.so
python3 python/diagnose_neural_memory_features.py \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --config training/memory/neural-system/data/P1-feature-probes.json \
  --output build/neural-memory-feature-check-new \
  --probe build/memory_feature_probe --tok-probe build/tok_probe --lib build/libggwshim.so
```

输出目录必须不存在。探针保存实际 C token ID、原始 embedding、output-normalized hidden、
每层 RMS 归一化残差；脚本保存 Torch 对应张量、逐项误差、源码/二进制/原始文件哈希。
这里只对 `hidden` 进行测量，没有运行生成、激活或记忆准确率测试。

所有输入使用同一 tokenizer。探针保留 BOS 行；同时另报去掉 BOS 后的 L2-normalized contextual/lexical
拼接特征，使旧 reader 的 BOS mask 与新草稿去 BOS 的差别可见，不默认为同一输入协议。
每个 C 输入创建全新 context，零初始位置、单次全前缀 prefill，输入之间无 K/V 复用。
C attention 内部仍使用本次 prefill 的 K/V 缓冲；报告显式记为 true，不谎称未分配 K/V。
Torch 不使用跨步/跨输入缓存；这里没有跨会话记忆验证，正式服务的缓存约束仍需单独验证。

## 3. 实测结果与修复边界

结果登记：`checks/P1-feature-parity.json`，引用各原始报告和张量哈希。
事前容差保持 `atol=1e-5, rtol=1e-4`，未因失败而放宽。

| 诊断条件 | output hidden 相对 L2 误差范围 | 严格特征对齐 |
| --- | --- | --- |
| 原始 Torch 参考 | 3.380%–5.012% | 未通过 |
| 仅纠正 FP32 RoPE 表精度 | 3.377%–4.998% | 未通过 |
| 仅模拟 C NEON 按 token int8 激活量化 | 3.724%–4.952% | 未通过 |
| 两项组合 | 3.491%–5.251% | 未通过 |

各组 token ID 一致，原始 embedding 逐元素相同，C 重复输入的全部观测张量完全一致。
误差在第一层已存在；并非仅最终 output norm 的差异。
int8 对照只是 Torch 反量化后 dense matmul 的诊断公式，不复刻 C 整数累加/舍入顺序，
不能据它未消除误差就排除量化路径的影响，也不能把局部下降说成整体对齐。

已修复一个确定问题：`TorchBackbone.forward` 原来总用 BF16 构造 RoPE 表，之后转 FP32 无法恢复精度。
现在遵从请求的 dtype，BF16 训练默认未改变；FP32 修复经先失败、后通过的单元测试与再次实测确认。
修复后实测与上述“仅纠正 RoPE”对照一致，**不代表已经消除特征域差异**。
同时移除了 Torch 文件中“与 C 完全一致”的不准确说明。

上述首次对照引出了第一层算子回放，最新定位与修复见下一节。
这些数值不能直接换算成记忆错误率，也没有证明原模型失败的唯一原因。

## 4. 第一层定位、ARM SiLU 修复与剩余问题

结果登记：`checks/P1-operator-parity.json`。新增工具：
`tools/memory_operator_probe.c`、`python/diagnose_neural_memory_operators.py`。
观测器直接编译实际 `bitnet.c`，普通库里的观测宏为空操作，没有服务端开关或常驻 callback。
同一输入下，带观测与正常探针的输出逐字节相同，已在修复前后验证，不用另一份 C 推理重写作为基准。

```sh
cmake --build build --target memory_operator_probe memory_feature_probe -j 8
python3 python/diagnose_neural_memory_operators.py \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --config training/memory/neural-system/data/P1-feature-probes.json \
  --lib build/libggwshim.so --probe build/memory_operator_probe \
  --reference-probe build/memory_feature_probe \
  --output build/neural-memory-operator-check-new
```

两种口径分开：

- **单算子回放**：每步喂入真实 C 中间量，检查该算子的局部误差，不让上游误差干扰定位。
- **连续前向**：自己计算上一算子的输出并向下传递；诊断链的结果必须与真实 Torch 第一层前向完全相同。

发现并修复的缺陷位于 `src/ops.c` 的 ARM SiLU 指数近似：
旧公式 `trunc(t - 0.5) + 1` 不是正确的 floor/nearest，会把多项式输入带出约定区间；
三阶近似精度也不足。极端负输入还会因指数钳位得到明显非零的 SiLU。
现在使用 nearest 范围缩减、分段 ln(2) 常量、六阶指数多项式及负尾屏蔽。
没有把 Torch 改成复刻错误近似，也没有修改 x86 算子。

修复前，同输入 SiLU×up 的相对 L2 误差为 **0.7238%–0.7430%**；
修复后为 **3.95e-8–4.45e-8**。
使用与 C 一致的激活量化公式后，6 次输入 × 17 项单算子对照 **102/102** 满足原容差。
这是第一层局部数值结果，不是 102 项记忆问答。

旧 `test_ops` 用同一 SiLU 实现计算 fused SiLU 的参考值，可能两个入口一起错但仍通过。
新增独立 double `exp` 参考，覆盖 SIMD/尾部长度、4,099 点网格及极端值；旧实现先失败，修复后通过。

**完整 backbone 仍不通过**：修复后的 dense Torch 对照最终 hidden 相对 L2 为 3.738%–5.410%；
诊断性的 int8 激活量化对照为 **2.859%–4.365%**，仍超出原容差。
局部算子合格不保证连续量化前向合格：观察到 ChatML 在第一层 FFN 入口有 1/16,384 个量化码不同，
下游 activation 已变为 11/65,536 个；消息框输入分别观察到 attention 的 1/20,480、FFN 入口的 9/20,480，
及 activation 的 29/81,920 个。小浮点差异跨过量化取整边界，会产生离散变化并继续传播。
这些记录证明该机制确实出现，不代表已证明后续 24 层所有误差都只有这一来源。

下一步需制定并验证统一的训练/部署量化前向契约，不能把 dequantized dense Torch 与 C 整数算子默认视为同一特征编码器。
严格门槛未放宽；EP-001 继续阻止启动。
本轮 Python 回归 **64/64**、C **21/21**，默认配置/API 两个 HTTP 回归通过。
没有运行新记忆准确率或性能对比，不宣称更快，也未完成 Android/x86 真机复验。

注意：SiLU 修复改变了 ARM backbone 特征域。不要把旧编码器生成的持久记忆状态与新特征混用于公平评估；
后续应以明确的编码器版本绑定并从原始来源重新编码。本轮没有删除旧状态、重启既有服务或替换记忆模型产物，
也没有将旧模型效果数字当作修复后的结果。
