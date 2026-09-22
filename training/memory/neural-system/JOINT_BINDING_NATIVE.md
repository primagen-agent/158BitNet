# DG-012：联合绑定反例与真实 C 生成适配

范围：零训练步的组合接口诊断。新 reader/因子头未训练，连续内容与拒答分支来自 CG-002。
这不是新候选的能力验收，也不是完整 C/HTTP 服务。
登记：`experiments/DG-012.json`。

## 1. 为什么需要同词、不同归属的反例

同一个问题询问 Estelle 的现居城市；来源为一条包含两句话的事件：

| 条件 | 来源事实 | 主体出现 | 居住关系出现 | Estelle—居住关系联合成立 |
| --- | --- | --- | --- | --- |
| 正确归属 | Estelle 住在 Graz；Fabian 喜欢 Graz 的建筑 | 是 | 是 | 是 |
| 交叉归属 | Fabian 住在 Graz；Estelle 喜欢 Graz 的建筑 | 是 | 是 | 否 |

中英文各一对，问题完全相同，每对来源的字符计数相同，仅改变角色归属。
两种独立 presence 标签在两侧都是真；其乘积无法作为联合绑定的充分条件。
正确状态必须来自对同一事实内部主体—关系归属的判断，而非看到某个人名、某个关系就接受。

合成事实标注只在 runner 中用于解释对照，不传入 `NativeTokenBackend`、输入编码或模型前向。
此处的 `joint_annotation` 只是这些固定反例的后置标注器，不是通用冲突/时序判断器，更不是运行时规则路由。
人名和值沿用已有训练世界，不能宣称未见实体或未见值泛化。

## 2. 实际生成链路

`python/native_token_generation.py` 接收原始自然输入，分别 fresh C 编码查询和来源。
固定初始神经路线后，每个输出位置执行：

1. 独立普通 C reference 与连续分支 C probe，均从位置 0 完整 prefill。
2. 基础 hidden/logits 与普通 C 路径逐位核对。
3. 按预测路线使用基础输出、C 投影的拒答增量，或 C 投影的内容增量加 Python token 概率混合。
4. 与同一冻结输出头的 Python 组合实现比较数值及 top-1。
5. 只允许下一次前缀接上本次实际 greedy token；首个前缀必须等于自然问题的完整 prompt。

不读取 prefix bank，不使用参考答案前缀或 gold 路由。每次 C 调用是新进程/新 context，不跨步骤复用 KV。
完整 prefill 内部仍有 K/V 计算缓冲；不是声称 attention 不需要 K/V。
reader、因子头和最终 copy 概率混合仍在 Python，不能称为神经模块全部用 C 实现。

四条 active 生成和两条保留正确来源但关闭记忆的生成，固定 greedy、最多 24 个新 token、128 token 上下文。
达到长度上限或 UTF-8 不完整都保留在报告里，不重试挑选成功回复。
disabled 对照仍编码来源，但基础输出必须每一步逐位不变，不能因移除来源而弱化这个检查。

## 3. 分支数值检查不能冒充实际调用

每条 active 样例的首个自然前缀，另外检查 supported/content/copy 和 uncertainty 两条分支。
这个检查不改变预测路线，也不把某条分支的结果作为实际生成输出。
必须单独记录实际 copy 使用位置数；若所有样例都走拒答，不能宣称 copy 在自由生成中已经成功工作。
数值容差为 `atol=1e-5, rtol=1e-4`，并额外要求 top-1 一致；不能只比较绝对误差。

## 4. 已完成结果

完整结果 `checks/DG-012-result.json`，独立复核汇总 `checks/DG-012-audit.json`；
原始输出、输入、C 日志和特征在 `build/neural-memory-dg012/`。

| 模式 | 语言/来源 | 初始预测路线 | 实际输出观察 | 结束 |
| --- | --- | --- | --- | --- |
| active | 英文正确归属 | insufficient | I do not have enough information to say where Estelle lives now. | EOS，15 token |
| active | 英文交叉归属 | insufficient | 与正确来源相同的拒答 | EOS，15 token |
| active | 中文正确归属 | insufficient | 重复“现有信息不足以确定” | 上限截断，24 token |
| active | 中文交叉归属 | insufficient | 相同的重复拒答 | 上限截断，24 token |
| disabled | 英文正确归属 | 不执行预测路线 | 基础模型的无实时信息说明，未说完 | 上限截断，24 token |
| disabled | 中文正确归属 | 不执行预测路线 | 基础模型建议搜索引擎/社交媒体，未说完 | 上限截断，24 token |

六条原始回答全部保留，没有删掉或重试截断样本。6/6 字节序列 UTF-8 完整，但 4/6 未在上限内结束。
disabled 路线仍可观察到初始化分类结果，但其结果不控制输出，不能把它计为实际拒答 specialist 调用。

两条正确来源被接纳为 supported：**0/2**；两条交叉来源被判 insufficient：**2/2**。
这只是此固定面板的初始路线观察，没有独立语义评审，不报告综合“记忆准确率”。
四条 active 全部拒答，不能把负例拒绝率当作绑定能力提升。
中文启用分支的短语重复并未在对应 disabled 的 24-token 观察窗口出现；
这提供了拒答分支退化的局部对照，但不是所有问题上的普遍因果结论。

执行审计结果：

- **126/126** 个实际解码位置，通过普通 C 基础结果核对、组合数值及 top-1 核对。
- 共 **338 次** C decoder/reference/projection 调用（不含输入 encoder）；每一步均完整重新 prefill。
- 初始自然 prompt、所有实际预测前缀延续及固定路由/概率均重新核验；没有离线前缀库或 gold 回答回退。
- **1,038 个**原始文件哈希复核通过。两条 disabled 生成的全部 **48 个**位置与基础输出逐位相同。
- 最大 C/Torch 组合绝对误差 `1.33515e-5`，在预登记 `atol=1e-5, rtol=1e-4` 的组合容差内通过，不能称为绝对误差均小于 `1e-5`。
- 四次首位置独立分支检查均通过；实际自由生成使用 copy 的位置为 **0**。
  因此只能说支持分支数值接线已验证，不能说它已在自由回复中成功取值。
- 新增四项反例/前缀/数值保护测试，Python **205/205**；现有 build CTest **21/21**。
- 参数未更新，原 CG-002 checkpoint 哈希未变，没有新训练产物或部署替换。

## 5. 推进边界

真实 C 前缀/投影通过只证明组合链路可执行，不能抵消语义失败。
新的训练必须包含同词角色交换的联合监督，并同时检查正确来源的接纳和错误来源的拒绝。
不能仅用后者掩盖“所有问题都拒答”。联合目标、门控是否需要校准/消融、预算和早停条件需先登记。
不得直接延长 CG-002，也不得以接口原型名义绕过第三候选训练预算。
自动 writer、多事件激活、状态持久化、作用域隔离和完整 HTTP 验收仍未完成。

下一项应定版联合绑定训练数据与新的有限训练方案，而不是继续放大现有输出或将所有问题拒答当作安全成功。
至少包含同词角色交换、同主体换值、错关系、正确来源保留，以及完整自由回复的重复/EOS 检查。
保留独立 presence 辅助监督与乘法路由门控的消融，避免把概率相乘当成联合事实证明。
训练预算须重新登记并确认；首段后用真实预测路线复测，正确来源仍不能被接纳或出现重复时必须停止。
本轮并未重新测量 CG-002 原固定面板的有依据 recall，其旧 0/8 结论保持原范围，不与本轮路由比例混合。

## 复现

```sh
python3 python/diagnose_joint_native_generation.py \
  --experiment training/memory/neural-system/experiments/DG-012.json \
  --features build/neural-memory-cg001-features \
  --checkpoint build/neural-memory-cg002-step-000050.pt \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_continuous_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --tok-probe build/tok_probe \
  --output build/neural-memory-dg012-new
```

features 参数只读取经过哈希验证的输出头及 manifest，不实例化或访问离线前缀库。
输出目录必须不存在；不覆盖既有报告或保存新的训练 checkpoint。
