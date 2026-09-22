# P1 数值与编码器身份契约

状态：P1-native-05 固定 reader 输入桥接通过小型数值验证；生成对齐未通过，P1 未整体完成。
不放行 EP-001，不替换部署模型。

## 决策与范围

现有证据表明，dense Torch / 模拟 int8 Torch 与 C 的连续骨干结果不能视为同一编码器。
继续复制浮点算子公式不能保证相同量化码；本轮不降低容差，也不靠训练抵消未标明的特征域变化。

对于不需向骨干回传梯度的 reader 输入，选择验证 **同一 C 编码器的前向结果直接用于离线训练和运行输入**。
GPU 仍训练神经记忆模块，冻结的是原本就要求冻结的 backbone，不是冻结记忆模型。
离线特征文件只是训练输入，不是用户的记忆状态、不是记忆模型权重，也不是 K/V cache。
不按问题检索事件、不拼接来源到 prompt、不把标签变成特征。

此决策不覆盖生成融合：融合分支仍须保留穿过冻结 backbone 的梯度；不能用静态 C 特征替代随融合变化的生成前向。
P2B/P2C 的可微且与部署一致的生成数值路径仍是未完成项，禁止将 reader 输入桥接成功当作全系统对齐。
统一/共享算子或带经过验证的 surrogate backward 的真实部署前向需要另行立项，不在本轮悄悄加入 STE。

## 首版 reader 输入格式

- 只接受既定自然消息框（role/speaker/text），监督侧文件不进入编码器。
- 固定同一 0.5B GGUF；每个文本独立新建 C context，单次全前缀计算；禁止跨输入 K/V 复用。
- C 内部本次 attention 的 K/V 工作缓冲如实记录，不能声称未分配 K/V。
- 保留实际 C token ID；BOS 参与骨干，但从 reader 特征行移除。
- C 内部对每个 output-normalized hidden 和原始 lexical embedding 分别作 FP32 L2 归一化，拼接成 `[T-1, 2H]`。
- Python/GPU 按 float32 原样加载，不再重算骨干、归一化或降成 FP16/BF16；保留事件边界。
- 同一输入重复编码、分批/换序、磁盘保存/加载应逐字节一致。大于当前探针长度限制时显式失败，不静默截断。

## 身份与产物约束

编码器 ID 至少绑定：GGUF SHA、实际 C 二进制 SHA（包含 tokenizer/算子）、消息框协议、输出格式、
归一化/BOS 规则、固定线程与加速环境、平台架构及实际 dispatch 名称。
更换二进制、SiLU/tokenizer、协议、CPU 实现或精度时默认拒绝混用；不能只检查 GGUF。
此首版是保守的同平台约定，不意味着 macOS/Android/x86 已互相数值兼容。

特征产物需要有输入/编码器/文件哈希、实际 token ID、形状和 dtype 校验；未完成、损坏、旧身份或缺少来源的文件不允许悄悄回退。
读取接口只返回向量，不返回 gold 字段或现成答案。训练标签变更不能影响这些向量。
对未知新文本仍应实际编码，不能把预制特征库当作运行时记忆检索器。

## 本轮验收与停止条件

1. 同一程序的 raw 探针与 reader-format 探针均正常运行，既有诊断格式不被覆盖。
2. 固定双语/角色/ChatML/消息框样例，比较新进程重算、换序、批边界和产物重载；要求特征与 token ID 精确一致。
3. 身份错、特征篡改、输入改动、额外 gold 字段、非法几何被拒绝。
4. 接到实际 reader 的同一前向边界，确认标签不参与输入、重载不改变 logits，且 reader 参数仍可获得梯度。
5. 软件与探针通过只能完成此 reader 输入子项。此前 C/Torch 全骨干连续对照的失败记录和门槛保持不变。

## 实现与实测

代码：`python/native_memory_encoder.py`、`python/diagnose_native_memory_encoder.py`；
现有 `memory_feature_probe` 增加显式 `reader-v1` 模式，默认 raw 诊断格式保持不变。
`ModelBinding` 映射已实现，用于状态/读取视图拒绝不同编码器身份；这不是已发布模型格式。
首版直接绑定探针二进制，是保守的研究身份。正式服务须使用相同共享编码实现，或经过明确认证的身份映射；
不能把探针的通过结果直接当成不同 `openai_server`/跨平台二进制已通过认证。

复现命令（输出目录须不存在）：

```sh
cmake --build build --target memory_feature_probe -j 8
python3 python/diagnose_native_memory_encoder.py \
  --config training/memory/neural-system/data/P1-smoke.json \
  --corpus build/neural-memory-p1-data-02 \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_feature_probe \
  --output build/neural-memory-native-check-new
python3 tests/test_native_memory_encoder.py
```

实测登记见 `checks/P1-native-encoder.json`：

- 事前选择首个开发世界的全部 22 类场景 × 中英文，共 44 条记录、44 个不同编码文本；只有 **1 个独立世界**。
- 独立进程重算、输入顺序反转、批大小 16 改 7：token ID 和 float32 特征逐值一致。
- NPZ 保存/重载不改变特征，实际未训练 reader 前向的 logits **44/44** 完全一致。
- reader 权重可以获得非零有限梯度，但没有做优化器步骤，不是训练或能力结果。
- 额外的新文本可以直接编码，不要求先存在于离线特征文件中。
- 首批 16 条另经 raw 模式核对，并对照 float64 L2 数学参考，归一化最大绝对误差约 `5.35e-7`，通过原容差。
- 6 项新增单元检查覆盖身份、损坏、输入篡改、非法 dtype/形状、标签字段拒绝和读快照错版本；总 Python 回归 **70/70**，C **21/21**。

生成目录保留实际 C 输入、输出、日志、特征及哈希。GPU 可以读取这些 float32 输入训练记忆模块，
但需要从已登记模型/实验配置传入预期编码器 ID 和 manifest hash，不能从待加载文件自己决定“可信身份”。
未知输入在离线 bank 上显式失败；运行时应调用编码器处理新文本，不可降级成答案查找。

限制：未切换草稿训练器到该新格式，未准备正式全量训练特征，未验证 GPU 或其他 CPU 平台。
reader 的结构、训练数据和自由生成能力均未因这项基础设施验证而达标。
后续 DG-001 已验证真实 C logits 前向可以保留，但 dense/STE 两个近似反向候选均未通过真实 C 方向检查。
见 `GENERATION_GRADIENT.md`。它未进行优化器更新，不放行训练；不得将该局部前向精确性当成全层对齐或记忆准确率。

DG-002 又验证了量化码固定时 NLL 改善可能仅来自输出整体缩放、不能改变候选排序。
详情见 `QUANTIZATION_CELL.md`；其后用户确认的连续分支 DG-003 已实现并通过局部数值门槛，
见 `CONTINUOUS_MEMORY.md`。这不是原量化 backbone 全层对齐通过，也不放行正式训练。
