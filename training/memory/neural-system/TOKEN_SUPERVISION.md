# DG-011：后置监督、复制覆盖和生成分支组合

本轮完成可执行的监督与组合接口，没有训练新候选，也没有完成端到端 C 自由生成。
登记 `experiments/DG-011.json`；完整结果 `checks/DG-011-result.json`。
真实记忆能力结论不变，不能把以下 token 覆盖或梯度检查报告为 recall 准确率。

## 1. 后置监督，而非 gold 前向

`python/token_memory_supervision.py` 接收完成前向后的 `ComposedOutputs` 和独立的 `TokenSupervision`：

- 两个辅助因子监督为“来源中存在请求主体”和“来源中存在请求关系”。
  标签由合成数据的事实字段生成，绝不在运行时通过名称/字符串匹配决定来源。
  普通聊天因子为 unknown，不强行教它匹配一个无关实体。
- 支持事实的完整回复保留全词表 CE，含 EOS；初始状态保留三分类 CE。
- 对已标注的目标值位置，指针监督对正确来源值区间内的所有安全、同 ID 位置求概率和。
  重复位置不是只选第一个；其它回复位置不添加 gold 指针目标。
- 若目标值 token 在安全来源中不能表达，目标为 NULL，但仍保留完整词表 CE。
- 普通聊天和不足证据保留复制质量惩罚，不给正向复制标签；普通聊天对候选生成分支保留基础分布 KL。

gold 主体/关系/值及其字节区间只用于这些后置目标。
结构性的 `copy_allowed` 仍只来自文本边界与控制 token 检查，没有变成 gold 选择器。
当前损失系数在诊断中均为 1，不代表已批准的训练配方；没有搜索系数或选择 checkpoint。

## 2. 四条通路的组合及梯度

实现：`python/token_memory_composition.py`。

| 初始预测路线 | 生成通路 |
| --- | --- |
| normal | 原始基础 logits，逐位保留 |
| insufficient | 只调用不依赖来源值的 uncertainty 分支，不调用 copy/content |
| supported | 先计算连续内容分支，再将其词表分布与神经 pointer 的 payload 分布混合 |
| disabled | 原始基础 logits，逐位保留 |

训练可计算所有分支并在前向后施加损失；实际 `read` 不接受 gold 路由。
初始决策保持回复内固定，状态绑定包括新 reader、因子头和原有 specialist 的参数版本。

修正的关键连接点：DG-010 pointer 接口只允许冻结 base。
若把可训练 content logits detach 后送入它，会切断连续内容分支梯度。
现在 pointer 只提供神经位置分布及复制质量，组合层独立混合可微 content 分布，**没有 detach 内容输出**。
NULL 质量留给词汇生成；重复 token 的质量累加。无合格 payload 时仍保留连续内容分支，不能误退成裸骨干。
概率端点使用 FP64 epsilon 防止饱和 FP32 gate 产生 log(0) 梯度；不是阈值搜索或语义选择。

## 3. 必须保留的设计限制

两个因子目前是必要条件的辅助监督，**不是联合绑定已完成**：
“A 喜欢建筑，同时 B 住在某城”可以同时包含主体 A 和居住关系，却没有 A 的居住事实。
此时仍须由联合状态/事件绑定判断；不能将两个 presence 标签当成正确答案的充分条件。
否定、引用、历史和冲突也可以两个因子均为真，但状态必须为 insufficient。

本诊断用两个神经 sigmoid 的乘积削减支持质量，并把被削减的质量转给 insufficient；normal 质量保持不变。
这是待消融的软门控，**不是经校准的联合概率，也没有证明两因子统计独立**。
随机初始化两因子均约 0.5 时，支持质量会额外下降到约四分之一，存在过度拒答倾向。
因此不能看到错主体被拒绝就认为匹配结构成功；需要同时检查正确来源的接纳及后续训练校准。

原 CG-002 的 content/uncertainty 参数只用于零步组合检查；旧私有状态分类器不参与新路由，
其无梯度是预期结果。尚未批准将这种混合初始化用于下一次训练。

## 4. 训练集完整覆盖审计

审计冻结合成训练集全部 **512 条**，无开发集选择或 LoCoMo 使用：

| 指标 | 结果 |
| --- | ---: |
| 保留的完整回复 token（含 EOS） | 5,352 |
| 有支持来源的记录 | 128 |
| 答案值 token | 280 |
| 可由安全来源 token 表达 | 253 |
| 无法由安全来源 token 表达 | 27 |

27 个缺口全部属于 16 条英文 paraphrase 记录，原因都是 **token ID/语境分词边界不同**，不是结构 mask 排除。
例如源句首 `Graz` 被分为 `Gr`、`az`，答案中的 ` Graz` 被分为 ` Gra`、`z`。
来源字节完整不意味着存在这两个答案 token ID。

所有缺口和记录均保留在完整回复损失及评测分母中。
253/280 只说明当前模板内的指针可表达覆盖，不是复制正确率、更不是 90.4% 记忆准确率。
本轮没有按问题重分词来源、搜索原文或用答案 token 反向构造前向 payload。
后续若增加上下文边界适配/字节通道，需要单独验证；当前依赖连续内容和词汇生成处理不可复制 token。

## 5. 固定组合检查

复用并核验 DG-010 的 C 查询/来源特征原始文件哈希；解码前缀来自已验证的冻结 C prefix bank。
不是本轮新执行的 C decoder，也不是完整 C 实现。
第一训练世界、中英两种语言，各 original/swapped_values/wrong_subject，共 **6 条**；新模块 seed 1011。

- 完整参考回复下的归一化与禁用恒等检查 **6/6** 通过，最大分布归一化误差约 `8.71e-8`。
- 支持样例的连续内容、复制位置、复制门和两个因子头都有有限非零梯度。
- 错主体样例的拒答分支、复制关闭目标和因子监督有有限梯度；不向内容输出矩阵传播回答 CE，符合分工。
- **预测路线 6/6 都是 insufficient**，包括 4 条有正确来源的记录。
  两条错主体被拒绝不能算改进，因为同时误拒绝了正确记忆。
- 参数逐位不变，无累积 `.grad`、无优化器步数、无新模型文件、无新自由回答。
- 新增 10 项回归，全套 Python **201/201**；现有 build 的 CTest **21/21**。

最终原始报告 `build/neural-memory-dg011-verified/report.json`。
早期报告仍留在 `build/neural-memory-dg011/`；最终版补充不可复制 token 的 ID/字节归因并重跑，未调损失或改分母。

## 6. 下一项

1. 在固定对照中补上联合主体—关系绑定与过度拒答边界，不能把独立 presence 监督包装成完成绑定。
2. 接上真实 C full-prefix 逐步生成适配，验证预测路线、连续内容与 pointer 混合、拒答隔离、token 边界和禁用恒等；
   仍无历史 KV 复用，不允许离线 prefix bank 回退或 gold 回答前缀冒充自由生成。
3. 数值/生命周期门槛通过后再登记新的有限训练预算、消融与早停。不得直接延长 CG-002 或静默启动第三个候选。

尚未解决自动 writer、多事件唤醒、作用域/持久化和完整 HTTP 验收；原 CG-002 有依据自由 recall **0/8** 结论不变。

## 复现

```sh
python3 python/diagnose_token_supervision.py \
  --experiment training/memory/neural-system/experiments/DG-011.json \
  --features build/neural-memory-cg001-features \
  --config training/memory/neural-system/data/CG-001-pilot.json \
  --corpus build/neural-memory-cg001-data \
  --checkpoint build/neural-memory-cg002-step-000050.pt \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --tok-probe build/tok_probe \
  --native-report build/neural-memory-dg010 \
  --output build/neural-memory-dg011-new
```
