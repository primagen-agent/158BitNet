# CG-003：成对训练核心和 C 特征接口预检

状态：训练准备推进，**optimizer step 仍为 0**。
本轮没有生成新的记忆准确率、CUDA 结果、自由回答或部署模型。

## 1. 已实现的代码

| 文件 | 本轮职责 |
| --- | --- |
| `python/joint_memory_model.py` | 独立实现 aux/product 两路对照、回复内路由绑定、相同随机初始化 |
| `python/joint_training_data.py` | train/dev 编译、分离前向特征与监督、身份绑定包读取 |
| `python/prepare_joint_training_features.py` | 同一 C encoder、prefix extractor、GGUF 输出头的 smoke/full 打包 |
| `python/train_joint_memory.py` | 成对前向、后置损失和 CPU 预检；正式训练入口仍硬阻止 |
| `tests/test_joint_training_core.py` | 新增 9 项公式、隔离、身份和启动保护测试 |

没有修改 DG-011 的 `token_memory_composition.py` 或 DG-012 的原始 backend，旧证据仍对应旧源码。
新 `JointState` 包装绑定策略；参数、输入或策略变化后禁止沿用旧回复状态。
**新模型尚未接入/重新认证 NativeTokenBackend**，不能直接继承 DG-012 的新策略生成证书。

两组的所有初始 state_dict 张量逐位相同。aux 只使用 reader 的三分类路由，factor loss 仍参与训练；
product 使用已有乘法门控。相同输入下，supported/uncertainty/位置概率/复制质量和 factor 输出保持一致，
只有状态路线公式不同。冻结的四个旧私有分类器张量并不参与新决策，其余有效记忆参数仍可求梯度。

## 2. 成对监督与安全边界

前向仅接受 query/source C 特征、冻结解码 hidden/base 和输出头，不接受答案、场景标签或 gold 路由。
前向之后才计算两个单样本损失的均值，以及预定 role-swap/value-swap 差异项。
value-swap 强制检查同一初始 prompt、首个差异是值位置、该位置的 C hidden/base 逐位一致。
普通/否定/历史等其它 pair 不偷偷添加额外对比损失。

封存 test 在编译入口即拒绝。优化器样本接口拒绝 smoke 包和 dev 记录；
预检可以读取它们计算诊断梯度，但没有 optimizer。
`--mode train` 目前无条件报错，改 JSON 中的 approval 标志也不能启动；
实际 50-step optimizer 循环和完整授权校验尚未实现，不能把 loss/preflight 核心称作完整训练器。

## 3. 已完成的真实 C 小包

固定第一训练世界、home_city、中英两种语言的 correct/role_swap，共 4 条记录：

- 独立 fresh C 查询/来源编码 **6 份文本**。
- 完整教师强制前缀 **44 个**，包括完整回复的 EOS 监督。
- 乱序及重复的 4 个提取器控制，与独立 C reference 的 hidden/logits **4/4 逐位一致**。
- 输出头、分词器、encoder、prefix probe、语料和记录均保存身份绑定。
- 包路径 `build/neural-memory-cg003-smoke-features/`。
- 包摘要 `83b805f58132130affea6b431e9a48ef44593c60fba8cbd8c51ab590dbc2fab8`，元数据副本见 `checks/CG-003-smoke-features.json`。

这些前缀是离线训练特征，不是 serving 记忆，不允许作为自由生成回退。
提取器复用模型权重，但每条前缀重建 C context；没有跨前缀 KV 复用，内部 prefill K/V 缓冲仍存在。

## 4. CPU 零步预检

证据：`checks/CG-003-cpu-preflight.json`，原始记录 `build/neural-memory-cg003-cpu-preflight.json`。
两种策略 × 中英两种语言，共 **4 组成对前向/损失/梯度检查**通过。

- 有效训练参数的梯度均存在且有限；内容输出、拒答输出、复制位置、复制门与 factor 头有非零梯度。
- 零初始化的输出矩阵会使部分上游梯度暂时为零，这是预期现象；不宣称所有张量都有非零梯度。
- 初始参数前后逐位相同，无 `.grad` 累积，无 optimizer 更新。
- disabled 路线逐位保留基础输出；全新两组不继承旧的重复拒答权重。
- 初始 aux 在四条样例上均选 normal，product 均选 insufficient。
  **两者都没有表现出正确 recall，不能据此选优或宣称乘法门控更安全。**

这个真实 C 小包只覆盖 role-swap。value-swap 的损失方程及同前缀保护已过单元测试，
但它的完整真实 C 数据/梯度验收，以及空记忆/普通聊天的真实 C 新策略检查仍待全包或额外预检。
本轮 Python **222/222**；现有 build CTest **21/21**。

## 5. 全量清单不等于全量特征已就绪

`--scope full --plan-only` 已编译全部 768 条 train/dev，没有执行完整神经特征提取：

| 清单项 | 数量 |
| --- | ---: |
| 独立完整前缀 | 1,764 |
| 独立查询/来源文本 | 608 |
| 回复 token | 7,944 |
| 值 token | 612 |
| 不可直接复制但保留的值 token | 48 |

每条记录的 token 数及不可复制位置与此前独立 C 分词审计一致；无 test 记录。
清单在 `build/neural-memory-cg003-full-feature-plan/`，摘要见 `checks/CG-003-feature-inventory.json`。
目录没有合格特征包 manifest，不能被 loader 当成训练包使用。

后续完整特征已经在单独目录 `build/neural-memory-cg003-full-features/` 实际提取完成，
80组扩展CPU预检也已通过；完整证据见 `FULL_FEATURE_PREFLIGHT.md`。
上面的 plan-only 目录仍然只是清单，不因新包完成而变成可训练特征包。

## 6. 后续顺序

1. 完整 C 特征及全包完整性审计已完成，见 `FULL_FEATURE_PREFLIGHT.md`。
2. 两语言/两关系/十类 pair 的80组 CPU 检查已完成；覆盖一个固定训练世界，不是全量语义准确率。
3. 新策略真实 C 生成适配和 CPU/CUDA 数值预检已完成，见 `PLATFORM_PREFLIGHT.md`；新固定面板基线及盲包现已完成，独立语义评审仍待完成。
4. 固定50步优化引擎与检查点验证已实现，见 `OPTIMIZER_AND_BASELINE.md`；授权绑定、完整训练同步到 tmux GPU 主机及新优化器GPU路径资格检查仍待完成。
5. 新两组预算确认、前置检查通过之后才能启动。当前配置继续 blocked，不能按旧两候选授权执行。

## 可复现命令

只重新编译全量清单，不提取或训练（新目录）：

```sh
python3 python/prepare_joint_training_features.py \
  --config training/memory/neural-system/data/JB-001.json \
  --corpus training/memory/neural-system/data/JB-001 \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --prefix-probe build/memory_prefix_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --tok-probe build/tok_probe \
  --lib build/libggwshim.so --scope full --plan-only \
  --output build/neural-memory-cg003-plan-new
```

CPU 预检（只读现有 smoke 包，输出文件须不存在）：

```sh
python3 python/train_joint_memory.py --mode preflight \
  --features build/neural-memory-cg003-smoke-features \
  --package-digest 83b805f58132130affea6b431e9a48ef44593c60fba8cbd8c51ab590dbc2fab8 \
  --output build/neural-memory-cg003-preflight-new.json
```
