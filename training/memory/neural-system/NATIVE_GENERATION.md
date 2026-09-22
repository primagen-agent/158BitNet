# 原生 C 自由生成后端与基础模板检查

实验：`DG-004`（后端）和 `DG-005`（无记忆模板单因素对照）。
只使用绑定的本地 0.5B GGUF，不训练、不替换正式模型或 HTTP 服务。

## 结构与可核查边界

`native_continuous_generation.py` 将 P1 的自由生成 transport 接到实际 C：

1. 当前完整前缀交给独立 C 进程，新建 context，从位置 0 开始 prefill。
2. 获取 C 的最终 hidden 与基础 logits；已编码的单个来源事件通过 Python 神经融合模块得到增量。
3. 新 C 进程重新计算相同前缀并应用连续输出投影。两次基础 hidden/logits 必须逐位一致。
4. 贪心选择 token，下一步只能追加这个实际预测 token；不可追加监督答案。

本轮逐步另用链接普通 `libbitnet` 的 reference 探针验证基础 logits。
每个连续探针日志记录初始位置 0、最终位置等于完整前缀长度、单次 eval，并核验 ARM NEON。
每次推理后进程和 context 结束；没有 token 间或请求间 KV 复用。
**一次完整 prefill 内仍有 attention K/V 工作缓冲**，不声称没有分配任何 K/V。

来源编码器保持既有 C FP32 编码域与身份校验。来源只允许单个给定事件，不允许生成途中换记忆；
这不是学习到的多事件激活，也没有自动 writer。神经融合仍在 Python 中，C 只负责 backbone 和输出投影。
每步重载模型是隔离诊断实现，不是已优化的生产 serving，不作延迟/吞吐结论。

## 对照与停止语义

DG-004 固定中英文两个问题，每个有五个条件：普通 C、功能关闭、空记忆、零 gate、随机非零分支。
关闭条件向 C 故意传非零增量但关闭开关，以验证不是“恰好增量为零”。
前三种记忆关闭/空/零条件要求每步 logits、完整 token 序列、正文及结束原因与普通 C 相同。
随机分支只检查增量非零和实际参与计算，不要求或宣称它会回答记忆内容。

停止 ID 来自真实 C EOS 元数据和 `<|im_end|>` 的 token 字节，不硬编码词表数字。
该 GGUF 的 SentencePiece 在单独编码结束标记时先添加一个空格；空格不能被当作结束 token。
首次 DG-004 初始化把编码结果误当作只有一个 token，在生成前停止；修正了适配器假设，未修改 tokenizer。

真实生成记录停止 token，但不将其写入正文。达到长度上限显式标记截断；未完成 UTF-8 的原始字节保留。
另用真实 tokenizer 的字节、脚本指定 logits 检验 EOS/部分 UTF-8 边界；该回放明确不是模型自然输出。

## 基础模板风险与独立对照

DG-004 的普通 C 基线出现 `<tool>` 标记，而没有自然回答。
读取 GGUF 的 `tokenizer.chat_template` 后发现：元数据只以 assistant header 结尾，
研究 prompt 额外添加了 `<think>\n\n</think>\n`。这只能先作为故障假设，不能直接认定唯一原因。
DG-005 保持 GGUF、tokenizer、system text、问题、贪心生成与无记忆条件不变，只比较这段额外后缀。
两个原问题加一个普通算术问题，完整记录两个变体，不按结果筛选样例。

元数据含真实换行；最初逐行解析会截断 Jinja 字符串，DG-005 在生成前拒绝启动。
工具已改成完整多行解析并严格匹配实际声明，不改变实验问题或阈值。
原 DG-001—DG-004 与 P1 输入审计仍保留原模板身份；不得改写旧证据。
如果今后变更研究模板，需显式版本化并重跑受影响的输入和数值检查。

## 复现

```sh
cmake --build build --target memory_continuous_probe memory_gradient_reference memory_feature_probe tok_probe gguf_inspect -j 8
python3 python/diagnose_native_free_generation.py \
  --experiment training/memory/neural-system/experiments/DG-004.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --probe build/memory_continuous_probe \
  --reference-probe build/memory_gradient_reference --encoder-probe build/memory_feature_probe \
  --tok-probe build/tok_probe --output build/neural-memory-dg004-new
python3 python/diagnose_native_prompt.py \
  --experiment training/memory/neural-system/experiments/DG-005.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --reference-probe build/memory_gradient_reference \
  --tok-probe build/tok_probe --inspect build/gguf_inspect --output build/neural-memory-dg005-new
```

两项均拒绝覆盖输出目录，保存原始生成文本、token、逐步前缀、C 二进制、模型和源码哈希。
失败的初始化目录保留，不把初始化失败计入记忆准确率；本轮无优化器步骤。

## 结果与修正

DG-004：`checks/DG-004-result.json`，原始记录 `build/neural-memory-dg004-stop-token/`。
两个问题各五种条件，共 60 个生成 token（含停止 token），156 次生成 C 前向、4 次来源编码。
后端门槛全部通过，10 次生成都在真实 EOS 正常结束；未训练随机分支不改变这些样例的贪心输出。
关闭/空/零条件都输出与普通 C 相同的 `<tool>\n\n`，故不能把后端通过当成正常回答或记忆成功。

DG-005：`checks/DG-005-result.json`，原始记录 `build/neural-memory-dg005-multiline/`。

| 问题 | 旧研究后缀 | GGUF 声明后缀 |
| --- | --- | --- |
| Nora 住在哪里 | `<tool>` | `Nora lives in the small town of Nora.` |
| 小安现在住在哪里 | `<tool>` | `小安住在上海。` |
| 二加二 | `<answer>` 后为 `2+2=4` | `Two plus two is equal to four.` |

六次均正常停止，原始回复保留空白。前两个新模板回复**没有来源支持**，不能记为正确回答。
对这三个固定问题，唯一改变是后缀，足以说明它确实影响输出；不足以证明此前所有失败都由模板导致。

修正：新研究生成默认 `gguf-chatml-v2`，不再添加元数据未声明的空 think 段。
DG-004 和 DG-005 的旧条件显式指定 `research-no-think-v1`，不会因默认值变化而悄悄改实验。
v2 重新通过 704 输入/416 成对 C tokenizer 审计，最大 prompt 47 token；
原 v1 审计及数值结果未覆盖。v2 连续融合的完整数值/生成认证仍待补做。
当前修正仅作用于新研究生成输入，不修改旧部署模型、旧训练入口或正式 HTTP 服务。

另一个已确认的缺口：`GatedMemoryFusion.prepare(empty)` 返回 None，residual 为常量零；
冻结 backbone 时空库输出没有可训练参数通路。设计要求的神经可用性/任务控制尚未实现，
无法单靠给当前内容分支更多空库样例来训练拒绝无依据断言。
必须保留“关闭功能精确 bypass”与“功能开启但证据不足的可学习响应”两个独立语义，
不能改成关键词拒答、固定句式或把所有普通聊天都拒绝。

本轮最终 Python **111/111**、C **21/21**；没有新训练模型或记忆准确率。
下一工作项：按既有设计补齐神经可用性/任务分支，并在 v2 上重验连续生成；CG-001 暂不启动。

后续 DG-006 已补齐可训练通路并通过登记的 v2 数值和短序列生成对照，见 `AVAILABILITY.md`。
这不是正确 recall 或完整回复验收；下一步转向新世界数据和训练监督接口，旧诊断记录不覆盖。
