# CG-003：完整 C 特征与扩展 CPU 预检

结论：**完整特征包已经生成，80/80 组零训练步预检通过；尚未训练或验证新的记忆准确率。**
本轮没有更新参数、没有新自由生成回复，也没有替换部署模型。

## 1. 特征包已实际提取，不再只是清单

路径：`build/neural-memory-cg003-full-features/`，约 1.1 GiB，包含训练准备所需的原始核对文件。
包摘要：`d299ad9f59c5a8500d751ee71b4f9139232cec9f80a2607e14953442e883b6fb`。
元数据副本：`checks/CG-003-full-features.json`。

| 项目 | 已验证数量 |
| --- | ---: |
| train/dev 记录 | 576 + 192 = 768 |
| 独立完整 C 前缀 | 1,764 |
| 独立查询/来源文本 | 608 |
| 被记录覆盖的回复位置 | 7,944 |
| 与 smoke 包逐字节相同的前缀 | 44 |
| 与 smoke 包逐字节相同的查询/来源编码 | 6 |
| dev 被优化器样本接口拒绝 | 192/192 |

前缀提取器的乱序/重复控制与独立 C reference 的 hidden/logits **4/4 逐位一致**。
完整记录和清单的哈希与提前保存的 `CG-003-feature-inventory.json` 一致；
每个参考回复位置都能找到对应 C 前缀，48 个不可复制值 token 的监督没有被丢掉。
逐字节比较包含 dtype、shape 和原始 bytes，不用浮点相等掩盖正负零等差异。

封存 test 未进入特征包；没有用于本轮神经前向或选择。
提取器可以复用模型权重，但每个输入使用新的 C context；没有跨前缀/请求 KV 复用。
单次 prefill 内部仍使用 K/V 工作缓冲。这些特征仅用于离线训练，不能用作自由生成的答案前缀回退。

## 2. 扩展预检范围和结果

先登记 `experiments/CG-003-full-preflight.json`，再运行 `python/preflight_joint_full.py`。
固定第一个训练世界，两种语言 × 两种关系 × 十类 pair × 两种策略，共 **80 组成对检查**。
每组的原始结果都保留，包括失败字段，不按通过情况缩小分母。
检查的是初始权重上的局部计算，不是80个独立记忆成功样本；同一正例会在多个对照中重复使用。

最终证据 `checks/CG-003-full-preflight.json`，原始报告 `build/neural-memory-cg003-full-preflight-verified/report.json`。
结果：**80/80 通过**，覆盖 role-swap、value-swap、句序、改写、错主体、错关系、否定、历史、空记忆和普通聊天。

通过的内容包括：

- 成对 query 一致；换值损失使用相同 C 因果前缀且首个差异确实在值位置。
- 有限损失和应连接分支的梯度；内容/复制没有被意外 detach。
- 只含 supported 的 pair 可以没有 uncertainty 梯度；aux 普通 pair 可以没有 factor-head 梯度。
  这些不是断链；相反，应连接的矩阵没有梯度仍会导致失败。
- 零初始化输出矩阵后面的上游零梯度是预期，不误报为学习成功或算子损坏。
- 空记忆禁止 supported 和复制，disabled 逐位保留 base。
- 预测 normal 的输出等于 base，预测 insufficient 的输出等于仅 query 驱动的 uncertainty 通路。
- 使用 copy 时检查 `exp(log_distribution)` 的和，而不是再 softmax 一次来掩盖归一化错误。
- 两组初始参数相同，前后参数逐位不变，无 `.grad` 累积，无 optimizer 更新。

第一次扩展运行保留在 `build/neural-memory-cg003-full-preflight/`。
最终版强化了 full/smoke 的字节级比较后重跑，没有改变数据、权重或损失，也没有重新挑选样本。

## 3. 不能据此宣称具备记忆能力

这些是教师强制参考位置上的计算检查，没有完整自由回复或独立语义评分。
初始路线观察也没有表现出可靠 recall：

- aux 的 80 次样本使用均选择 normal。
- product 的 80 次样本使用中，72 次选择 insufficient、8 次选择 normal。
- 两组都没有在本面板选择 supported。

这些计数包含重复使用的样本，且是随机初始化结果，不用于选优或修改预算。
“80/80 预检通过”绝不等于“记忆准确率100%”。CG-002 已有小面板的有依据 recall 0/8 仍是其原范围内的结论。

## 4. 软件回归和剩余门槛

新增7项预检保护测试：完整分母、两语言/两关系覆盖、活动与非活动分支梯度、非有限值拒绝、严格字节一致性及留存 manifest 摘要。
manifest 副本按原始字节保留，避免 JSON 重序列化把 `4.0` 改成 `4` 后改变身份摘要；原始特征包没有修改。
Python **229/229**，现有 build CTest **21/21**。

后续已完成新策略模型的真实 C 生成适配与 CPU/CUDA 预检，见 `PLATFORM_PREFLIGHT.md`。
旧 DG-012 只认证旧组合类型，新适配器单独验证 `JointTokenMemory`，没有挪用旧证书。
预检源码与小型特征夹具已完整同步至 GPU；这不等于完整训练包已经同步或启动授权已完成。
固定 optimizer/检查点循环、新固定面板基线、训练启动身份验证及新两组预算确认仍未完成。
因此 `training_approved=false`、`launch_command=null` 和硬阻止训练的入口保持不变；GPU 只运行零步核对，没有开训。

## 复现预检

```sh
python3 python/preflight_joint_full.py \
  --experiment training/memory/neural-system/experiments/CG-003-full-preflight.json \
  --features build/neural-memory-cg003-full-features \
  --package-digest d299ad9f59c5a8500d751ee71b4f9139232cec9f80a2607e14953442e883b6fb \
  --smoke-features build/neural-memory-cg003-smoke-features \
  --smoke-digest 83b805f58132130affea6b431e9a48ef44593c60fba8cbd8c51ab590dbc2fab8 \
  --output build/neural-memory-cg003-full-preflight-new
```

输出目录必须不存在。此命令没有 optimizer，不会训练或写出模型 checkpoint。
