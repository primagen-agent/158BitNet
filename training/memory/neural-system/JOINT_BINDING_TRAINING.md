# 联合绑定训练：数据已定版，训练尚未放行

本轮完成数据、采样计划和新实验登记。**没有执行训练，没有新模型或准确率。**
实验配置 `experiments/CG-003.json`；数据审计、全部样本记录、逐步采样表和固定面板 ID 在 `checks/CG-003-data.json`。
原 CG-001/CG-002 的两候选预算已经用完；CG-003 是新预算提案，不能视为旧授权的自动延长。

## 1. 数据目标与保留材料

让神经模型根据“问题中的主体/关系”和“同一来源事实的主体/关系归属”决定是否支持回复，
并在换值时改变输出，而不是记住一份姓名—城市映射或检测几个关键词是否出现。
数据为自建合成小型课程，不含 LoCoMo；还不覆盖通用自动 writer 或所有自然记忆类型。

- 生成器：`python/prepare_joint_binding_curriculum.py`。
- 配置：`data/JB-001.json`。
- 完整数据、独立标签、元数据、成对关系及 SHA manifest：`data/JB-001/`。
- C 分词审计：`python/audit_joint_binding_data.py`。
- 固定采样器：`python/joint_binding_schedule.py`。

| 划分 | 世界 | 记录 | supported | insufficient | 普通聊天 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 12 | 576 | 192 | 288 | 96 |
| dev | 4 | 192 | 64 | 96 | 32 |
| sealed test | 4 | 192 | 64 | 96 | 32 |

共 20 个世界、960 条记录、800 对对照，中英两种语言、居住地和工作地两种关系。
主体和城市值均跨三种划分隔离，也与 CG-001 指定的主体/城市集合分开；这不是声称 backbone 预训练从未见过这些词。
test 仅生成、保存并作机械一致性检查，没有进行 C 分词、神经前向、选择或训练。

## 2. 防止新数据形成新捷径

每个世界对同一主体分别询问居住地和工作地，并在世界之间交替选择目标人物，
不能让“某个人只被问居住、另一个人只被问工作”成为关系标签的替代品。
单元测试还验证：完全相同的来源事件，在问题关系改变后，可以分别为支持和不支持。

每组含 12 个场景：正确归属、同词角色交换、换值、句序交换、改写、错主体、错关系、
否定、历史、空记忆、有来源的算术、无来源的算术。

- **角色交换**：问题与来源字符计数保持不变，两个 presence 标签都为真，联合事实真假相反。
- **换值**：主体和关系不变，正确回答必须改变；不能只优化拒答或语言流畅性。
- **句序交换/改写**：正确来源不能因目标事实不在第一句，或城市位于句首而失效。
- **否定/历史**：主体和关系出现不等于现在有有效事实。
- **普通聊天**：不需要记忆时保持基础模型行为，不通过猜测某个记忆答案提高分数。

标签和事实字段都在 sidecar；运行时输入仍严格只有 `id/context/episodes`。
这些模式还不是开放域记忆数据：没有声称已经覆盖引用、别名、复杂冲突、多事件合并、更新与遗忘。

## 3. C token 与可复制覆盖

对全部 train/dev 共 **768 条**做同一 0.5B C tokenizer 检查，验证完整回复（含 EOS）、
query/source 字节、原 prompt 边界、128-token decoder 容量及 512-token encoder 容量。

| 指标 | train | dev |
| --- | ---: | ---: |
| 完整回复 token | 5,944 | 2,000 |
| 目标值 token | 444 | 168 |
| 不能直接复制的目标值 token | 36 | 12 |
| 最长参考回复完整前缀 | 48 | 50 |

48 个不可复制 token 全部留在损失与分母中，不用 gold 重分词构造前向 payload。
这是分词/覆盖审计，不是完整 C 特征包，更不是记忆准确率。
实际 contextual/lexical features 和逐回复位置 C hidden/logits 尚需另行提取与绑定；不可复用旧语料前缀伪装为新数据。

初稿中的主体—关系关联捷径在训练前修正。初稿可恢复于 `build/neural-memory-jb001-draft-corpus/`，
其旧审计在 `build/neural-memory-jb001-audit/`；最终审计为 `build/neural-memory-jb001-verified/report.json`。
没有用模型效果来挑选这次数据修改。

## 4. 两组小规模实验提案

拟登记 **两个候选，每组 50 步，总共 100 个 optimizer step**，不追加 seed、不自动延长。
两组使用同一份初始权重、同一采样表、同一优化器和监督，唯一预定差异是路由：

| 组 | 因子头参与训练 | 因子头额外改变路由 |
| --- | --- | --- |
| joint_aux | 主体/关系 presence BCE | 否，仅三分类神经状态 |
| joint_product | 相同 BCE | 是，保留 DG-011 乘法支持门控 |

这样可以检查乘法门控是否帮助联合识别或只增加误拒绝。
独立 presence 不是联合绑定证明；两组都必须接受 role-swap 的联合状态与成对损失。
相对 CG-002 同时改动了数据和模型，不能把跨实验进步归为某一个单因素。

计划从全新记忆参数开始，不继承 CG-001/CG-002 的失败 checkpoint 或重复拒答权重。
连续输出矩阵零初始化、内容 gain 非零；骨干固定同一 0.5B，不使用 LoRA/SVD/NAS。
旧私有分类器不参与新路由，须排除在优化器之外；这不是冻结正在学习的记忆模块。

## 5. 采样和损失边界

首段每步 4 对，共 200 对、400 次记录使用，实际 **268 条不同训练记录**。
十类 pair 各 20 次；两组共用这份已经固定的表，不可各自抽到更有利的样本。
400 次使用中，supported 240、insufficient 120、普通聊天 40；英文 196、中文 204。
**均衡的是 pair 种类，不是状态类别；这也不是完整训练集的一轮训练。**

保留 DG-011 的状态 CE、presence BCE、完整词表回复 CE/EOS、normal KL、值位置 marginal/NULL 和 copy-off。
role-swap 与 value-swap 另有系数 1 的成对差异项；两端 CE 保留，不能只要求差异方向为正。
所有系数冻结为 1；两样本取均值后加 pair 项，再对四对取均值。具体配置见实验 JSON。
成对损失和新数据编译器尚需实现/验证，不能因已有单样本 loss 就宣称训练器已完成。

## 6. 首段验收和停止条件

开发全量 192 条只用于声明范围内的诊断。C 自由回复固定为开发世界 `jb-008` 的 **24 个 ID**：
中英两种语言、两种关系、正确/角色交换/换值/空记忆/普通有来源/普通无来源六种条件。
ID 已保存在审计结果中，不按未来效果重新挑选。

先记录 24 条 disabled/base 参考，再对每个 50-step 候选各跑 24 条，最多 72 条回复；
greedy，最多 64 新 token，128 context，全前缀 C 重算，无历史 KV 或答案前缀回退。
这部分 C 测试可能比小规模 GPU 训练耗时更长，不把 optimizer 步数当作总运行成本。

拟定同时满足的门槛：

- 有依据完整回复至少 **6/8** 正确，换值两端都正确至少 **3/4**。
- 角色交换 **4/4** 不接受错误归属；空记忆 **4/4** 不编造。
- 普通聊天路线与回复 **8/8** 精确保留基线。
- 候选回复不截断、UTF-8 完整；同一至少 4-token 块不能连续重复三次。
- 两位独立盲审检查完整自由回答，保留分歧、重复包和模型家族限制；未解决分歧不放行。

这些是拟注册的小样本推进门槛，不是统计充分的生产标准。
任一项失败就停止该候选，不凭 NLL 好看追加步数。
两组皆失败则停；皆通过时按完整换值对、完整有依据回复数依次选择，完全相同优先 aux-only。
最终 test 192 条在候选冻结后另行登记评估，不用于本轮选择或回填训练。

## 7. 当前就绪状态和下一项

已完成：确定性数据、分割与反例检查、C token 审计、采样表、固定面板、方案登记。
本轮新增 8 项回归；Python **213/213**，现有 build CTest **21/21**。

后续已经实现两种路由、成对 forward/loss 核心、特征打包接口并通过 smoke C 包的 CPU 零步预检，
具体证据见 `JOINT_TRAINING_IMPLEMENTATION.md`；不能把该预检当作完整训练器或全量特征已经就绪。
完整新 C 特征包和80组扩展 CPU 零步预检现已完成，见 `FULL_FEATURE_PREFLIGHT.md`。
尚未完成：正式 optimizer/检查点循环、CUDA 预检查、
新策略 C 生成认证、新基线自由回复/评审、完整源码同步、启动身份校验，以及新预算确认。
实验保持 `training_approved=false`、`prerequisites_complete=false`、`launch_command=null`。
下一项认证新策略 C 生成与 CPU/CUDA 一致性，前置检查通过并确认新预算后才能到 tmux GPU 服务器启动。

## 复现

```sh
python3 python/prepare_joint_binding_curriculum.py \
  --config training/memory/neural-system/data/JB-001.json \
  --output training/memory/neural-system/data/JB-001 --verify

python3 python/audit_joint_binding_data.py \
  --config training/memory/neural-system/data/JB-001.json \
  --corpus training/memory/neural-system/data/JB-001 \
  --previous-config training/memory/neural-system/data/CG-001-pilot.json \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf --tok-probe build/tok_probe \
  --output build/neural-memory-jb001-new-audit
```
