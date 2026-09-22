# 生成输入隔离与成对因果验收

P1-causal-09：只补齐生成接口与评分协议，不启动训练，不修改正式服务。
DG-003 的数值通过不等于本协议通过。当前 16 世界协议集已经公开检查，只作软件 smoke。

## 事前固定的工作范围

1. 生成 prompt 只能来自当前 conversation context 和固定 ChatML 模板。
   来源事件保持独立 JSON role/speaker/text 编码，不拼入 prompt。
   样例 ID、场景/世界、标签、目标事件索引均不进入生成调用。
2. 自由生成只追加刚预测出的 token，不允许教师强制目标作为续写输入。
   每步提交完整前缀；后端必须从零重算，禁止 decode/跨请求 KV 复用。
   此接口本身不能证明外部后端无缓存，真实后端认证仍是前置条件。
3. 一个问题的两个记忆条件必须同时正确才记成对通过。
   换值/更新要求答案跟随新事实；移除/无关/错关系等要求承认缺失证据；
   加干扰、调序、同值不同人要求保持正确归属。不能只检查文本是否改变。
4. 评分沿用独立双评审，不增加词法匹配裁判。缺回复、缺评审、分歧保留分母；
   任一端已确认失败则成对失败，任一端仍待评审也另外标记未完成。
5. 不修改原 704 条数据、种子、标签或登记清单。不把 416 个相关对照当成
   416 个独立世界；分开报告 train/dev、语言、对照类型和世界。

同一问题的来源变化是受控干预，但这些对照通常同时改变多个事件/属性，
不能据此单独归因于某一个内部算子。错主体与删目标在此 smoke 内有重复条件，
其分数不能被当成独立重复证据。

## 接口限制

固定简短回答 system 提示保留。DG-005 之后新研究 prompt 默认使用 GGUF 声明的
`gguf-chatml-v2`（无额外 no-think 段）；旧实验显式使用 `research-no-think-v1` 以便复现。
支持普通 user/assistant 上下文。带不同 speaker 身份的消息、tool/system 上下文
与保留模板标记暂时拒绝，不能静默丢弃说话者/角色语义。
这是显式支持范围，不是完整 P1.1/P1.2 已完成。

空记忆是证据缺失条件；功能关闭则是原始 backbone 基线，两者不得混为同一评分目标。
停止 token 不作为正文，截断必须标记。所有生成 token（包括停止 token）保留在记录中。
不使用整句替换、答案前缀提示或手工选择候选 token。

## 下一实验的登记边界

`experiments/CG-001.json` 只登记小规模连续分支可学习性研究，仍未放行。
先完成来源、模板/自由生成后端与评分校准，以及新独立世界数据冻结；再定训练命令与哈希。
P2B 的正确激活只能用于明确的 oracle 条件，不能称为神经 reader 或自动记忆成功。
正式 P2B/P2C 门槛、EP-001 启动锁和原层内对照保持不变。

## 工具与复现

- `python/neural_memory_generation.py`：隔离 prompt/来源和逐步自由生成 transport；调用方不能传标签。
- `python/neural_memory_causality.py`：成对协议和独立双评审后的联合评分；不自行理解或裁判自由文本。
- `python/audit_neural_memory_generation.py`：校验原冻结数据，使用同一 GGUF 的真实 C tokenizer 审计所有前缀。
- `tests/test_neural_memory_generation.py`：14 项手工软件反例，不是人工校准或记忆问答。

```sh
python3 tests/test_neural_memory_generation.py
python3 python/audit_neural_memory_generation.py \
  --config training/memory/neural-system/data/P1-smoke.json \
  --corpus build/neural-memory-p1-data-02 \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --tok-probe build/tok_probe \
  --output build/neural-memory-p1-causal-new

# 有实际生成的回复及独立评审后才运行；只统计原冻结 corpus 的 dev。
# 盲审材料仍用 review_neural_memory.py pack 生成。
python3 python/neural_memory_causality.py \
  --config training/memory/neural-system/data/P1-smoke.json \
  --corpus build/neural-memory-p1-data-02 \
  --predictions build/predictions.jsonl --reviews build/reviews.jsonl \
  --reviewer reviewer-a --reviewer reviewer-b --output build/causal-score.json
```

输出拒绝覆盖。审计保留完整 token 序列、每个来源文本摘要、每个成对输入哈希和源码/二进制身份。
摘要通过只证明本轮适配器遵守输入边界；不能认证尚未实现的真实生成后端、训练器或服务。

## 本轮结果

结果登记 `checks/P1-causal-generation.json`，原始产物 `build/neural-memory-p1-causal-09-final/`。
704 条输入、16 个世界、13 类对照，共 416 对（train/dev 各 208）；
每对上下文及 C token 前缀完全相同，来源事件不同，未改动原数据及登记哈希。
最大 prompt 56 token，另预留 32 个生成 token，均在当前 128 token 诊断容量内。
新增测试 14/14；包含原数值/数据/评分回归的 Python 合计 101/101。

模型推理调用和优化器步骤均为 0，没有新增模型、生成回复或记忆准确率。
真实 C 后端的全前缀无 KV 重算、功能关闭一致性、生成停止与 UTF-8 尚待下一工作项认证。
原 smoke 已被查看，不作为 `CG-001` 独立验收集；新世界数据与真实独立评审校准仍未完成。

后续更新：上述旧模板后端对照已由 DG-004 完成；DG-005 暴露模板与证据缺失问题，见
`NATIVE_GENERATION.md`。v2 同样完成 704 输入/416 成对审计，最大 prompt 47 token，原始数据未改动。
结果另存 `checks/P1-causal-generation-v2.json`，不覆盖 v1；v2 连续融合尚需独立重验证。
