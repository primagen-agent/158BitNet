# 连续神经记忆增量分支：独立架构对照

状态：DG-003 已通过本轮数值诊断；尚未验证记忆学习/泛化，原层内注入方案保留。
不是发布模型或自动记忆系统。
登记：`experiments/DG-003.json`。本分支经用户确认作为独立对照推进，不作为原全层设计已完成的替代说法。

## 1. 为什么做此对照

DG-001 的近似反向未通过真实 C 方向检查。DG-002 又表明，最终整数码不变时，
NLL 可能仅因 logits 整体缩放而下降，目标排名不变。
本分支检验：让神经记忆增量避开后续整数取整，能否获得可信梯度并实际改变候选排序。
这不预先保证记忆学习或泛化成功，也不推翻原层内方案的所有可能实现。

## 2. 结构与边界

固定骨干与来源编码器都使用同一 0.5B GGUF。当前前缀先由 C 正常计算：

```text
当前前缀 → 冻结 C backbone → base logits、最终 hidden
                                  ↓
激活的事件内容 → 神经交叉注意力/门控 → Δhidden
                                  ↓
                冻结 GGUF 输出权重的 FP32 投影
                                  ↓
             logits = base logits + Δlogits
```

- 当前原型复用 `GatedMemoryFusion` 的全秩投影与门控；直接返回 residual，避免先加回 hidden 再相减造成消减误差。
- 本实验只给定一个来源事件，用于隔离数值问题；没有完成多事件激活或自动 writer，不可据此宣称 recall 成功。
- 记忆增量不再次量化为 int8；不训练 GGUF 输出矩阵，不加入 LoRA、SVD 或 NAS。
- C 基础分布保持原有实现；关闭/零增量必须与普通 C 库逐位相同。
- Torch 使用普通自动求导。零增量的 bit-preserving 加法保留数学上的加法梯度，使零初始化 gate 可以打开；不是恒等 STE 或替代骨干后缀梯度。
- 来源只通过独立神经特征进入记忆分支，不拼入 prompt、不查找答案、不直接输出固定文本。最终仍由 token 分布生成。
- 训练的是记忆模块。冻结的基础 logits/hidden 可以作为固定输入，但没有冻结记忆模块自身的梯度。

这是多因素架构变化：查询表示移到最终 hidden，注入点移到输出分布，且增量路径使用连续 FP32。
即使结果改善，也不能归因于某一个因素，不能称原全层注入已通过。

## 3. 实际输出头与实现范围

该 0.5B GGUF 没有独立 `output.weight`，使用 F16 `token_embd.weight` 作为绑定输出头，`logit_scale=4`。
C 基础推理会为该头建立量化缓存；连续增量使用 GGUF 中原始绑定权重转 FP32 计算，不改写基础路径。
不另存或训练一个词表权重文件，模型仍必须绑定此 GGUF。

初次初始化因为工具假定存在独立输出头而在评测前失败，保留在 `build/neural-memory-dg003/`；
补齐实际绑定头读取后重新运行，未改变样例、随机种子、扰动幅度或门槛。

代码边界：

- `python/continuous_memory.py`：连续投影和保留零值 bit 的加法。
- `tools/memory_continuous_probe.c`：普通 C 前向及冻结权重的连续增量投影；仅隔离研究工具。
- `python/diagnose_continuous_memory.py`：Torch 神经分支产生增量，真实 C 检验前向、扰动损失与排名。

**尚未在 C 中实现整个神经分支**，探针接收 Python 模块算出的增量。
因此本轮验证范围是新生成接口/输出投影/梯度，不是完整 C 记忆推理或 HTTP 链路。
正式部署前仍须移植并验收神经激活、门控、持久化、普通/流式生成和算子一致性。

## 4. 验收口径

使用 DG-001 的三个中英文样例，不重新筛选困难样例。
前向容差保持 `atol=1e-5, rtol=1e-4`；gain 有限差分门槛在 DG-003 中事前单独登记。
除 gain 标量检查，还沿融合输出矩阵的单位 Frobenius 梯度方向做正/反扰动，结束即恢复。

必须同时满足：

1. 关闭/零增量与普通 C 基础输出逐位一致。
2. 每个实际 C 增量与合成 logits 都在 Torch 对照容差内。
3. gain 的每个登记步幅都通过导数误差与符号检查。
4. 每个样例至少 3/4 个输出矩阵扰动具有实际 C 损失下降/反向上升。
5. 每个样例至少一次扰动同时降低损失并提高目标 token 排名，不能仅凭 NLL 改善通过。

本轮没有优化器步骤、没有持久化扰动后的参数、没有学习曲线。
即使上述门槛全部通过，也只允许进入后续前置审查与受控可学习性验证，不能叫记忆准确率。

## 5. 复现

```sh
cmake -S . -B build
cmake --build build --target memory_continuous_probe memory_gradient_reference memory_feature_probe tok_probe -j 8
python3 python/diagnose_continuous_memory.py \
  --experiment training/memory/neural-system/experiments/DG-003.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_continuous_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --lib build/libggwshim.so \
  --tok-probe build/tok_probe --output build/neural-memory-dg003-new
python3 tests/test_continuous_memory.py
```

输出目录必须不存在，完整保留每个正/反扰动和所有未达标项。
阶段计划、旧失败报告和“同一 backbone、不靠 KV、不使用 RAG”的约束保持有效。

## 6. 实测与推进边界

结果登记：`checks/DG-003-result.json`，原始记录位于 `build/neural-memory-dg003-tied-head/`。
三个样例均满足全部事前数值门槛，没有改动阈值：

- 零/关闭分支与普通 C 输出逐位一致。
- 51 组 C/Torch 对照，每组同时检查增量和合成 logits，全部通过原容差；
  最大增量绝对误差 `5.96e-8`，最大合成 logits 绝对误差 `1.91e-6`。
- 12/12 个 gain 有限差分检查与普通自动求导匹配。
- 12/12 组输出矩阵扰动，预测下降方向使原生 C NLL 降低，反方向使其上升。
- 12/12 个下降方向扰动同时提高目标 token 排名；不是仅调整分布温度。

下面仅展示事前登记的最大矩阵扰动幅度 0.3（Frobenius 范数）；完整记录包含全部步幅：

| 样例 | 初始分支目标排名 | 扰动后排名 | 原生 NLL 变化 |
| --- | --- | --- | --- |
| 英文居住地 | 5,394 | 5,204 | 22.9602 → 22.8586 |
| 中文居住地 | 29,270 | 28,831 | 26.9023 → 26.8109 |
| 英文饮料偏好 | 54,231 | 53,880 | 29.1654 → 29.0703 |

**top-1 未变，目标 token 仍远未成为正确输出。** 这只是局部梯度/可改变排序的证据，
不是训练后的准确率，也不证明模型会泛化或自动记忆。临时参数扰动全部恢复，没有优化器步骤或新模型产物。
本轮软件回归 Python **87/87**、C **21/21**；尚未做性能或跨平台实测。

下一步不再靠增加数值诊断样例宣称能力：先补齐剩余 P1 数据/评分/来源契约，
审查并登记连续分支的小规模可学习性实验，之后按 P2B 的明确 oracle 口径检查自由生成。
实验必须包含未见语义世界/主体、同问题换记忆、目标移除/无关记忆、错误主体/时间与无记忆基线，
分开报告 NLL、目标排名、完整回答及因果一致性。DG-003 的三个已知样例不能用作独立验收集。
原全层方案、DG-001/DG-002 的失败记录与阶段门槛不变；EP-001/正式训练尚未放行。

上述下一步中的生成输入隔离和成对验收工具已完成，见 `CAUSAL_GENERATION.md`；
`CG-001` 已登记为未启动。尚需真实自由生成后端认证、新世界数据及独立评分校准，
不能把 transport 的单元测试当成实际后端或记忆能力通过。
