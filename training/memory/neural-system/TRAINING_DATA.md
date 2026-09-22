# 三状态组件训练数据与监督接口

P1-supervision-12：准备 CG-001 的可复现组件试验数据，不启动训练。
冻结前的范围：24 个新语义世界，16 train / 8 dev；每世界 16 场景 × 中英文，共 768 条。
不是正式 P2 验收集或独立最终测试。世界/主体隔离，城市词表在 train/dev 共享；
不同 split 使用不同问法，但不声称广泛语言、关系或精确值泛化。

## 数据约束

- 每条只含零或一个来源 utterance，保留边界；不按 gold 目标索引过滤来源。
- supported：原值、换值、来源改写、明确更新，共 4 类。
- insufficient：空库、错主体、错关系、未确认的冲突报告、假设、否定、引用、仅历史事实，共 8 类。
- no_memory_needed：算术问题分别搭配空库、有效事实、错主体事实、冲突报告，共 4 类。
- 配对保持问题完全一致，只改变来源；普通聊天的来源有无/内容不能决定是否需要记忆。
- 原始每状态比例为 1:1:2（normal:supported:insufficient），预登记每样例权重 2:2:1，
  加权贡献相等；权重不等于独立样本数，必须分别保存原分母。
- 输入、标签/参考回复、索引/规范事实、成对关系分文件；ID 不含场景名且不进入神经 forward。
- 训练参考回复是离线监督样例，不是推理时拼接/固定回复；开发评分不能只做参考回复字符串匹配。
- 只用合成世界，不用 LoCoMo 或旧 DG 问题；完整数据重生成及源码/产物哈希必须一致。

## 监督边界

生成请求只含当前 prefix token 与独立来源文本。状态辅助 loss 只用当前用户前缀位置，
不把 teacher-forced 答案前缀输入状态监督。生成 loss 包含完整回复和结束 token，
对 prompt 位置不计 loss。教师强制只用于训练损失，不能作为自由生成准确率。

回复目标由同一 C tokenizer 对 `prompt + reply + im_end` 联合编码得到，
必须核验 prompt token 完整前缀不变、回复字节可逆、最后 token 确为结束标记；
不允许把孤立回复的 dummy-space token 拼到训练前缀上，也不静默截断超限样例。
本轮不缓存 backbone KV，不训练任何参数。

## 限制与启动条件

这里只覆盖居住地事实与普通算术、单来源、模板化双语示例；关系多样性、多事件绑定、
自然聊天写入和持久化不在该 pilot 的通过口径内。
先验证生成器、监督隔离和真实 tokenizer，再冻结 manifest。独立评分校准、真实 C 特征/前缀抽取、
训练配置及启动审查仍须完成，不能仅把 JSON 的 prerequisites_complete 改成 true 来开训。

## 实现与复现

- `prepare_availability_curriculum.py`：世界划分、三状态监督、标准事实与成对关系，验证主体与原子事实不跨集合重复。
- `availability_supervision.py`：完整文本联合分词、显式训练用 teacher forcing、回复 loss 与仅首位置状态 loss。
- `audit_availability_supervision.py`：真实 C tokenizer 全量前缀/字节/容量审计，不运行模型生成或优化器。
- `calibrate_availability_review.py`：编写的正反例盲审与离线联合评分，不伪装成模型预测。

```sh
python3 python/prepare_availability_curriculum.py \
  --config training/memory/neural-system/data/CG-001-pilot.json --output build/neural-memory-cg001-data-new
python3 python/prepare_availability_curriculum.py \
  --config training/memory/neural-system/data/CG-001-pilot.json --output build/neural-memory-cg001-data-new --verify
python3 python/audit_availability_supervision.py \
  --config training/memory/neural-system/data/CG-001-pilot.json --corpus build/neural-memory-cg001-data-new \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --tok-probe build/tok_probe --output build/neural-memory-cg001-supervision-new
```

冻结登记：`data/CG-001-pilot.manifest.json`；输入/标签/索引/成对数据可由保留的配置和生成器逐字节重建。
`--verify` 拒绝内容改动、额外文件或缺失文件。修改生成器必须显式生成新清单，不覆盖旧实验依据。
真实 tokenizer 审计登记：`checks/P1-supervision-audit.json`，原始 token 记录在 `build/neural-memory-cg001-supervision/`。

## 本轮结果

- 768 条：训练 512、开发 256；24 世界（16/8），672 成对控制（训练 448、开发 224）。
- 训练原始状态数：normal 128、supported 128、insufficient 256；权重后各 256。
  开发原始分母 64/64/128 保留，不用加权数目或成对数替代独立世界数。
- 全部联合编码前缀一致、回复字节精确可逆、EOS 正确；最大 prompt 38、completion 16、联合 50 token。
- **768/768 条孤立回复编码与联合编码得到的回复序列不同**。新的监督接口避免将这些边界不同的
  token 序列直接拼到真实 prompt 上；这不能证明之前所有训练失败都由分词边界导致。
- 数据重生成/冻结校验通过；Python 137/137 通过；没有模型推理、梯度更新或新训练权重。

## 独立模型评审与适用范围

用户授权由助手指定两位评审后，使用 reviewer-a / reviewer-b 两个隔离上下文模型实例完成 16 条盲审。
两位均未读取 gold/预期判定或对方输出，16/16 原始标注一致，16/16 评分符合事先编写的正反例。
两位为同源模型，不能称为跨模型独立验证，也不是人工验收；自然性与“承认知识不足”的边界歧义
保留在 `calibration/AV-001/PROCESS.md`。16 条小样本不能证明任意自由回复的评分可靠性。
该目录保存实际盲审包、未提供给评审者的标签、两份原始标注和含哈希的评分结果。

## 固定训练配置与剩余项

`experiments/CG-001.json` 已登记 AdamW、FP32、batch 4、lr 1e-4、梯度裁剪 1、状态 loss 系数 0.2；
均匀抽样训练行，权重 2/2/1，不再次按状态重采样。批次使用语料平均权重 1.5 归一化，
避免小批次随机状态比例改变目标权重。最大 1,000 步，检查点 0/50/200/500/1,000。
完整开发集只报告 teacher-forced 与状态指标，不能叫自由生成准确率；另有预先固定的 16 行自由生成面板。
若到 200 步没有成对语义改善，或普通聊天退化，不扩大训练；评审未完成也不能假定通过。
这些是拟实施配置，不是已运行的训练日志或已实现训练器。

下一工作项：实现并核验真实 C 的来源/完整前缀特征抽取、与上述分离监督接口衔接的训练器，
固定源码与产物身份后再同步到 `test` tmux GPU 服务器运行。骨干、输出头冻结，训练记忆模块参数；
缓存的离线特征是训练输入，不是要部署的用户记忆，也不是跨请求 KV。
当前 `source_manifest`、合格特征产物、训练命令及相应启动实现仍缺失，CG-001 未放行。
原 EP-001 草稿训练器的拦截不变；正式阶段的大样本、人类复核和更广场景门槛未被这个 pilot 取代。
