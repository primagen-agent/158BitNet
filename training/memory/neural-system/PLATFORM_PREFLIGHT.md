# CG-003：真实 C 生成适配与 CPU/CUDA 零步核对

结论：新路由模型的研究适配器与固定 CPU/CUDA 计算检查通过。**没有训练，也没有新的记忆准确率结果。**
本报告只覆盖数值、路线和实际前缀链路；神经模块仍在 Python，骨干及输出投影在 C，不是完整 C/HTTP 服务。

## 1. 预先登记与执行范围

先登记 `experiments/CG-003-platform-preflight.json`，再执行本地和 GPU 预检。
两种策略 `joint_aux` / `joint_product` 使用 seed 1013 的相同初始参数；不继承 CG-002 权重。
冻结同一 0.5B GGUF，禁止 LoRA、SVD、NAS、跨请求/位置 KV 复用、原文拼入生成提示和答案前缀回退。
单次 fresh prefill 仍有内部 K/V 工作缓冲，不能描述成完全没有 K/V 运算。
输入是指定的单条复合事件，不包括自动判断写入、长程多事件管理或 LoCoMo。

## 2. 本地实际 C 链路

`python/native_joint_generation.py` 只接收新 `JointTokenMemory`，不改动 DG-012 的旧适配器及证据。
每步从自然问题与实际已生成 token 重新进行 C prefill，核对普通 C reference、连续投影和 Python 组合结果；首轮神经路线固定用于后续位置。
来源单独编码为记忆输入，不放进生成提示。支持分支的 token 概率来自神经位置激活，不做文本查询或整句答案旁路。

| 范围 | 用例 | 实际位置 | C decoder/reference/projection 调用 | 实际 copy 位置 | 最大数值绝对误差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始初始化的短前缀链路 | 4 | 16 | 48 | 0 | 0 |
| 单独的非零数值夹具 | 8 | 8 | 32 | 2 | 1.90735e-6 |

调用数不包含输入编码器。所有比较要求 `atol=1e-5, rtol=1e-4` 且 top-1 相同；normal/disabled 必须逐位等于 base。
原始初始化覆盖两种策略和中英两语言，每例仅生成4个位置；aux 全部走 normal，product 全部走 insufficient。
这不是完整回复，也不是新的语义基线。

另设8个独立夹具，为两种策略分别覆盖 normal、supported、insufficient、disabled。
夹具在单独模型副本上固定 seed 4013，将内容/U 输出矩阵设为非零小量，并固定神经分类头的偏置来实现所需路线。
**这是人为构造的路径覆盖，不能把 supported 或 copy 激活解释为模型学会了记忆。**
每个实际首位置还单独核对支持/U 分支；该检查不会替代实际预测路线。
夹具初始化之后参数保持不变，不写出训练检查点，不用于模型选择。

原始记录：`build/neural-memory-cg003-native/report.json`、`case-00..11/backend.json` 及其带 SHA 的 C 输出。
报告副本：`checks/CG-003-native-preflight.json`；其中相对 backend 路径以原始 build 目录为根。
已逐个复核原始输出 SHA、实际预测前缀和固定路线概率。

## 3. GPU 零训练步核对

在登录的 tmux `test` 会话运行，主机 `pcm-6cb311adc5a8`、RTX 5070，Torch `2.13.0+cu130`。
使用独立目录 `build/cg003-platform-a067290e/`，没有覆盖旧训练目录。
完整源码快照451个文件和12条固定训练记录的 C 特征经过归档、传输和逐文件 SHA 核验。
GPU 复用的 GGUF 与冻结输出头均只读，并与本机身份完全一致：

- GGUF：`44cb4e0db8374d4247bba391b3d3b7c0b3ee815c36cf2e20d0d1a3570677bd95`。
- 输出头：`8a44471dd9502adcfaf8a04eceb4716e1fbfd228fbbb6832f06520a699b93cbd`。
- 源码/夹具归档：`a067290e49411c44e9b98130b1af0525570dede509bfa3e0935c9f2e40846ad6`。
- 夹具摘要：`4912e5c8041f1901858b2b22c6ac33ce58176f282a7fe3f7870c88935cf43ae8`。

`python/joint_cuda_preflight.py` 在服务器上比较 CPU 与 CUDA，FP32、AMP/TF32 关闭。
第一个训练世界的 home-city 关系，两语言 × role-swap/value-swap/empty/ordinary-empty 四类 pair × 两策略，共 **16/16 组成对检查通过**。
12条记录会重复配对，不能当作16个独立记忆样本；也未覆盖全部训练世界或训练后参数。
CPU/CUDA 使用完全相同的 CPU 初始化权重，不分别随机初始化。

| 数值项目 | 比较数量 | 最大绝对误差 | 预设容差 atol / rtol |
| --- | ---: | ---: | --- |
| 输出、位置分布、copy mass、因子和状态概率 | 192 | 1.90735e-6 | 1e-5 / 1e-4 |
| 成对总损失 | 16 | 1.78719e-5 | 1e-4 / 1e-3 |
| 有连接的参数梯度张量 | 460 | 1.37091e-6 | 1e-4 / 1e-3 |

支持/U/状态 top-1 一致；应连接梯度有限、未连接梯度模式一致。零初始化导致的上游零梯度不算能力证据。
没有 optimizer，没有参数更新，没有 `.grad` 累积，没有保存模型检查点。
本轮 CUDA 对照不包含人为非零夹具；后续训练后检查点仍需重新核对 native 数值与自由生成。

报告已下载并验证 SHA `469213c04cf2053983606891e4e3a4d949ac01257303fbf62c68b3f31a82325a`。
留存 `checks/CG-003-cuda-preflight.json`、`CG-003-cuda-fixtures.json`、`CG-003-platform-archive.json` 和 `CG-003-platform-source.json`。
源代码快照代表实际运行时点；本报告和计划更新发生在核对之后，不声称后续文档字节仍与快照相同。

## 4. 软件回归与下一门槛

Python **234/234**，现有 build CTest **21/21** 通过。
新增测试覆盖旧模型拒绝、新策略非零路径夹具、实际前缀约束、非有限/超容差拒绝、监督与前向输入隔离。

后续已实现固定 optimizer/检查点引擎，并完成24例无记忆 C 基线及盲包，见 `OPTIMIZER_AND_BASELINE.md`。
正式启动身份验证、独立语义评审和新优化器GPU路径资格检查仍待完成。
新两组各50步/总100步预算仍未获批准，`training_approved=false`、`launch_command=null`，训练入口继续硬阻止。
不能将本报告当作基线语义评审、训练批准、记忆准确率通过或部署批准。
原有 CG-002 固定面板有依据 recall **0/8** 的失败结论没有被本轮软件预检改变。

## 复现入口

```sh
python3 python/qualify_joint_native.py \
  --experiment training/memory/neural-system/experiments/CG-003-platform-preflight.json \
  --features build/neural-memory-cg003-full-features \
  --package-digest d299ad9f59c5a8500d751ee71b4f9139232cec9f80a2607e14953442e883b6fb \
  --corpus training/memory/neural-system/data/JB-001 \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_continuous_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --tok-probe build/tok_probe \
  --output build/neural-memory-cg003-native-new

python3 python/joint_cuda_preflight.py export \
  --features build/neural-memory-cg003-full-features \
  --package-digest d299ad9f59c5a8500d751ee71b4f9139232cec9f80a2607e14953442e883b6fb \
  --output build/neural-memory-cg003-cuda-fixtures-new
python3 scripts/package_joint_preflight.py \
  --fixtures build/neural-memory-cg003-cuda-fixtures-new \
  --output build/neural-memory-cg003-platform-new.tar.gz
```

CUDA 主机运行归档中的 `source/python/joint_cuda_preflight.py run`，传入 `--experiment`、`--fixtures`、
`--fixture-digest`、`--head`、`--gguf`、`--snapshot`、`--output`。使用新归档实际输出的摘要，不沿用旧身份。
这些入口只做零步核对；所有输出目录/文件须为新路径，避免覆盖证据。
