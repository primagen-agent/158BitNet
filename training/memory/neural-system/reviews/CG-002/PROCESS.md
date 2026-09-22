# CG-002：固定面板 C 自由生成与双评审

结论：**本面板拒答和普通聊天保真改善，但有依据 recall 仍失败。不得发布或继续增加训练步数。**
这是两个开发世界、中英双语、单个给定事件的组件实验，不是 LoCoMo、自动写入或完整 C/HTTP 系统。

## 执行与数值证据

先登记 `experiments/CG-002-step50-generation.json`，再执行原定 16 条样例；
基线、顺序、greedy、32-token 上限和 GGUF ChatML 模板不变。
检查点 SHA-256：`eb6b9a551c616d08a2aa0e8d183a63233c87a3c10e296f290a7b48b99dea716d`。
基线仍是 `reviews/CG-001/baseline.json`，SHA-256 固定在实验登记中。

`native_role_generation.py` 只在初始用户 prefill 后调用一次神经决策，之后绑定同一预测路线；
没有使用 CG-001 的旧混合分支，没有 gold 状态或答案路由。来源由 C 单独编码，未拼入 prompt。
神经模块仍在 Python；C 执行同一冻结 0.5B backbone 与输出投影。

两个独立 case worker 各有 tokenizer 和 C 子进程；只共享只读神经参数，结果按原面板顺序写入。
16/16 条均正常停止，未截断，无不完整 UTF-8；共 155 个生成位置、465 次真实 C 前向。
每个位置都重建 context，普通 C 参考与基础 hidden/logits 逐位一致。
不复用跨 token/request KV；单次 prefill 内仍有 attention K/V 工作缓冲。

独立重放检查覆盖 **155/155** 个位置：

- 原始文件 SHA-256、预测前缀连续性、实际预测 token 与报告一致。
- 第一个前缀重新计算神经决策，后续不重判；16 条路线均与实际记录一致。
- 训练残差逐位重放；C/Torch correction/logits 满足原定 `atol=1e-5, rtol=1e-4` 联合容差，top-1 一致。
- 最大 logit 绝对误差约 `1.04904e-5`；容差是逐元素的绝对加相对容差，不是仅用最大绝对误差判定。
- 两条预测为普通路线的完整回复与基线 token/字节完全相同。

本轮预测路线：内容 10 条（含两条误激活的英语算术），证据不足 4 条，普通 2 条。
在实际前缀上，内容增量改变了 10 个下一词选择，不确定增量改变了 14 个，普通路线改变 0 个。
改变下一词不代表读取了正确记忆；这只是说明增量确实参与了生成。

## 盲审方式

两个全新、互不共享上下文的实例 `/root/cg002_judge_a` 与 `/root/cg002_judge_b`
只读取混合去身份后的 `blind.json` 和相同语义规范，不读取代码、标签、记忆摘要或对方结果。
评审只标注回复实际声称的事实、缺证据声明、自然表达及主动记忆操作旁白，不自行给准确率。
仍是同源模型的隔离上下文，不能宣称真人或跨模型家族验证。

32 个基线/候选输入-回复记录有 6 个完全相同，去重为 **26 个唯一盲包**；两位各标注 26 个，
26/26 语义签名一致，无缺失或分歧。每种条件仍按完整 16 条计分，没有缩减分母。
“无法提供实时更新的信息”被视为缺证据声明；裸 Ok 是自然表达但没有算术结论；
英国按国家 United Kingdom 提取，不把国家错误地当作城市。没有修改原始标注来提高分数。

## 结果

| 固定面板 | 无记忆基线 | CG-001 第 50 步 | CG-002 第 50 步 |
| --- | ---: | ---: | ---: |
| 总通过 | 3/16 | 1/16 | 6/16 |
| 有记忆问答（原值 + 换值） | 0/8 | 0/8 | 0/8 |
| 空记忆承认未知 | 1/4 | 1/4 | 4/4 |
| 普通算术正确 | 2/4 | 0/4 | 2/4 |
| 换值成对通过 | 0/4 | 0/4 | 0/4 |
| 移除记忆成对通过 | 0/4 | 0/4 | 0/4 |

CG-001 列引用其已保存的独立评分；本轮重新盲审的是相同基线与 CG-002。
**不能把 6/16 写成“记忆准确率 37.5%”。** 它由 4 条拒答与 2 条基础算术组成，没有一次正确的有依据 recall。
空库一端全部正确，也不能让“移除记忆”成对控制通过，因为有证据一端仍全错。

典型回复（完整原文及 token 在 generation.json）：

- Celia 来源从 Turin 换到 Graz，英文仍都答 `Celia currently lives in New York City.`；中文仍都答住在上海。
- Adela 来源从 Rennes 换到 Turin，英文仍都答住在 United States；中文仍都答住在英国。
- 空库时，英文能说明不知道/没有足够信息；中文输出“我还不知道Celia目前住在哪座城市。”或“我还不知道。”。
- 中文算术恢复基线的 34 和 32；英语仍是基线的 Ok，不能算答对。

## 判断与下一项

职责分离在此面板改善了缺证据回应，并避免此前中文普通聊天退化；
但实体/值绑定和精确内容读出没有通过。训练诊断中错主体状态识别仍是 0/16，
本轮没有把该诊断混入 16 条自由生成分母，也没有用基线保真掩盖误激活。

已有证据不支持继续堆训练步数，也不足以证明只有一个结构原因或神经记忆整体不可行。
现有两种训练候选的筛选预算已用完。下一项回到零训练步的内容通路定位：
对同问题的换主体/换值对，分别观察来源表示、神经读出和目标 token 分布，
区分“未保留信息”“没有正确绑定”“读出未映射到生成”与“概率变化未跨过 top-1”。
在完成该定位并重新登记方案前，不新增第三次训练；不使用 LoCoMo 选模，不用原文检索或整句复制绕过神经激活。

本轮 Python 170/170、C 21/21 通过，未提交代码、未替换部署模型、未增加训练步数。

## 复现

```sh
python3 python/eval_role_checkpoint.py \
  --checkpoint build/neural-memory-cg002-step-000050.pt \
  --baseline build/neural-memory-cg001-baseline/baseline.json \
  --config training/memory/neural-system/data/CG-001-pilot.json \
  --corpus build/neural-memory-cg001-data \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_continuous_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --tok-probe build/tok_probe \
  --output build/neural-memory-cg002-step50-generation-new
```

随后使用 `audit_role_generation.py` 审计全部位置；
`review_availability_generation.py --candidate-kind cg002` 生成盲包及汇总标注。
输出目录/文件拒绝覆盖；审计、评分、两份原始标注与原始回复完整保留在本目录。
