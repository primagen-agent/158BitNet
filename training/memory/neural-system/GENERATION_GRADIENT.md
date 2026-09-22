# DG-001：真实 C 前向与近似反向的最小对照

结论：**前向通路通过，两个近似反向候选均未通过预先登记的方向检查，不放行训练。**
登记见 `experiments/DG-001.json`；完整结果摘要见 `checks/DG-001-result.json`。
这是 P1 数值诊断，不是记忆准确率、自由生成质量或 LoCoMo 测试。

## 范围与实现

- 同一 0.5B、最后一层 attention 后、前缀最后一个位置；其余 backbone 保持冻结。
- 原始自然问题进入 C，来源文本只经独立 C 编码器提供给未训练的融合模块，不拼进问题。
- 一个 next-token NLL 作为诊断目标。目标 token 只进入损失，不能出现在 C 推理输入里。
- C 工具只在隔离诊断构建中注入残差；普通库的宏为空操作，无 HTTP 接入或持久状态写入。
- 每次原生前向都新建 context、完整计算前缀，不复用跨调用 K/V；内部 attention 缓冲仍存在。
- 原生 logits 直接作为 autograd 前向返回值，使用自定义 Function 避免 `native + proxy - detach(proxy)` 的消减误差。
- backward 明确标为 surrogate：dense 后缀反向或 int8 前向、恒等 STE 反向。不声称它是离散 C 程序的精确导数。
- 没有优化器、没有训练步、没有新模型文件。扰动后立即恢复参数，属于方向检查。

两个近似都穿过末层 FFN、最终 RMSNorm、输出投影；并非把 loss 直接连到伪造答案。
检查注入关闭/零残差时，诊断与普通 `libbitnet` 的 hidden、logits 必须逐位相同。
非零记忆残差必须改变实际 C logits，同时损失要能传到融合参数。

## 预先固定的方向门槛

随机种子、三个中英文样例、初始 gain=0.1、两个反向候选、四个 gain 扰动（0.001/0.01/0.05/0.1）在运行前登记。
每个样例每个候选至少 3/4 个扰动需满足：沿预测下降方向，真实 C NLL 下降超过 1e-6；
反方向真实 C NLL 高于原点。持平和不明确不从分母删去。
这个门槛是本次候选筛查约束，不是证明所有随机梯度法必须每步下降的普遍定理。

## 结果

| 样例 | dense 后缀反向 | int8 STE 后缀反向 |
| --- | --- | --- |
| 英文居住地 | 3/4 | 3/4 |
| 中文居住地 | 1/4 | 1/4 |
| 英文饮料偏好 | 1/4 | 1/4 |

两种候选都有有限非零梯度，但没有达到“每个样例至少 3/4”的登记条件。
这两个候选对本次 gain 的梯度符号相同，所以选出的正/反扰动点相同；
两列的相同损失不是两个独立成功证据，也不等于两个完整梯度向量相同。

已通过：

- 三个样例的零残差/关闭路径，与普通 C 库逐位一致。
- 使用非零残差时，三个样例的实际 C logits 都发生改变。
- autograd 前向与 C logits 逐位一致；融合参数获得有效梯度，backbone 参数没有更新。

失败具有可观测后果。例如中文样例在 gain 扰动 0.001 时，
原 NLL=26.8362283，近似梯度建议方向变成 26.8362497（更差），反方向为 26.8361950（更好）。
因此不能把“梯度非零”作为训练方向可信的充分证据。
NLL 大小本身不作为本轮记忆能力判定：融合权重未训练，且这里只测试一个 token。

## 限制与后续

只测最后一层一个位置的 gain，不能推断所有层、全部参数、长期训练或所有 STE 方法均无效。
当前证据足以拒绝把这两个未经校准的反向近似直接接入全量训练，但不证明神经记忆设计整体失败。
`native_generation_bridge.py` 仅由诊断与测试使用；现有训练入口仍被拦截。

该后续检查已由 DG-002 完成，见 `QUANTIZATION_CELL.md`：局部导数未全部过关，
并实测到同区间 NLL 下降却不改变目标排名。DG-001 的失败判定保持不变。
如要尝试连续记忆增量支路或改变注入位置，必须先说明架构变化并另行登记，不能把它默认为全层方案已完成。
不更改 DG-001 扰动幅度或门槛来覆盖本次失败记录，不以扩大训练量处理该问题。

## 复现与软件回归

```sh
cmake -S . -B build
cmake --build build --target memory_gradient_probe memory_gradient_reference memory_feature_probe tok_probe -j 8
python3 python/diagnose_native_generation_gradient.py \
  --experiment training/memory/neural-system/experiments/DG-001.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_gradient_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --lib build/libggwshim.so \
  --tok-probe build/tok_probe --output build/neural-memory-dg001-new
python3 tests/test_native_generation_bridge.py
```

输出目录必须不存在。保留每次扰动的输入、原始 C hidden/logits、日志和哈希。
本轮 Python 回归 **76/76**、C **21/21**；软件正确性通过不抵消梯度候选筛查失败。
