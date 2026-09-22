# C 特征抽取与 CG-001 训练运行链路

目标：只训练神经记忆/可用性参数，冻结同一 0.5B GGUF 的骨干与输出头。
训练依赖真实 C 数值域，不以另一套 Torch backbone 代替 C；不使用 LoRA、SVD、NAS 或 KV 复用。

## 离线输入与前向隔离

`memory_prefix_probe` 只复用只读模型权重。每个完整前缀均创建新 context，
验证起始位置为 0、结束位置为前缀长度；输出完整 hidden/logits 和位置记录。
单次 prefill 内仍有 attention K/V 工作缓冲，但不复用前一前缀或请求的 K/V。

完整回复监督仍使用真实 C 联合分词，教师强制前缀只含之前的答案 token，不含当前/未来 token。
相同冻结前缀可以去重计算，但这些产物只作为训练输入，不是部署记忆数据或回答缓存。
自由生成不从教师强制前缀库读取答案，仍根据实际生成 token 重新执行 C。

训练 forward 接口仅接收 `ForwardFeatures(hidden, base_logits, source)`；目标 token、
状态标签、世界/场景 ID 和参考文本不进入该接口。生成损失覆盖回复及 EOS，
状态辅助损失只取初始用户前缀的位置，不在答案前缀上监督状态。

## 产物与身份

`prepare_availability_features.py` 产生：

- `prefixes/`：带源 C binary/编码器身份的只读前缀库。
- `sources/`：同一 C 编码器的来源特征，文本键只用于精确读取已提供的训练来源，不做问题检索。
- `output-head.npy`：从绑定 GGUF 读取的冻结词表投影，不优化、不作为独立记忆模型发布。
- `records.json`：训练用记录与分离目标；`manifest.json` 绑定上述文件、数据集、模板及程序身份。

完整包包含 768 条记录、1,473 个唯一前缀、528 段唯一来源。
抽取前对起始/中间/末尾前缀、调序与重复项做独立进程比对，4/4 hidden/logits 逐位一致。
Python 3.10 的 SHA-256 兼容修复只替换文件读取辅助函数，不改变特征数值或哈希算法。
抽取原始源清单与训练代码清单分别保留，不能将兼容修改伪装成之前执行的代码版本。

## 训练器与启动限制

`train_availability_memory.py --preflight-only` 校验全部产物、三个状态的非零分支前向、
CPU/目标设备容差、梯度有限性以及冻结头无梯度，不调用优化器。
实际训练同时要求注册启动命令、源码身份、特征包身份、完整零步自由生成基线和 CUDA 预检查通过。
改一个 `prerequisites_complete` 标志不能单独放行。

训练只构造 `AvailabilityMemoryFusion` 参数优化器；骨干不实例化为可训练 Torch 模型，
词表头、C hidden/logits 与来源特征均不参与参数更新。
首段固定 50 步，保存 0/50 步检查点、完整采样行 ID、loss/梯度范数和全部 256 条开发集的教师强制指标。
50 步后必须停止，等待固定面板的真实 C 自由生成与独立评审，不自动向 200/1,000 步推进。
教师强制 NLL/状态分类与记忆正确率分开报告；不能仅凭 loss 降低选为发布模型。

## 服务器同步

`package_availability_training.py` 生成完整代码快照和所需数据/基线，排除 GGUF。
GPU 服务器已有 GGUF 的 SHA-256 已核对为同一绑定模型；新的工作目录与既有非 Git 项目副本分离。
快照带逐文件哈希，不向旧目录零散覆盖几份代码后声称“最新验证”。

直连路径不可用时，`transfer_training_archive.py` 仅通过已有认证的 `test` SSH pane 传输：
确认主机名，独占新文件名，临时关闭输入回显，每块 ACK/SHA-256，最后核验全文件哈希；
成功或超时恢复 TTY。不中转至公共存储、不新开网络监听、不读取 SSH 凭据。
此过程临时占用 `test` 输入，传输期间不能在该 pane 中输入其它命令。

当前预检查、源码快照与首段运行的实际状态，以 `checks/` 和 `experiments/CG-001.json` 的登记为准。

## 首段实测与停止结论

GPU：`pcm-6cb311adc5a8`，RTX 5070，Torch 2.13.0+cu130 / Python 3.10。
数据包摘要 `8f5b2d2ae685a7995d45422e661a8f59811357186d0cff370211eb596d5f36a6`；
完整传输包 637,460,507 字节，SHA-256 为 `4f0a5ccdedfc57cb4865051437673683f8462eca073c3cbb4a9379d0913eab8d`。
新目录内 325 个源码文件全部通过逐文件校验；之后只应用了单独登记的三份启动元数据。
首次 macOS tar 附带 AppleDouble 条目，被精确文件白名单拒绝，未启动优化器；重新使用干净归档后正常放行，未放宽检查。

三个状态的 CPU/CUDA 非零分支预检查全部通过，最大 logit 误差 `9.54e-7`，
梯度有限、冻结词表头无梯度。预检查见 `checks/CG-001-cuda-preflight.json`。
固定 16 条零步 C 基线均正常结束，记录中无 KV 复用；基线包含无依据居住地猜测和只答“Ok.”的英语算术回复，
不能要求记忆分支以生成监督补偿任意基础语言行为而忽略其路由职责。

首段 **50 个优化器步骤已完成并停止**，原始记录 `build/neural-memory-cg001-run-report.json`，
登记见 `checks/CG-001-step50.json`。256 条开发样例的结果：

| 指标 | 第 0 步 | 第 50 步 |
| --- | --- | --- |
| 教师强制平均 NLL | 2.6278 | 1.7181 |
| 状态总正确数 | 90/256 | 173/256 |
| 无需记忆状态 | 50/64 | 8/64 |
| 证据可用状态 | 0/64 | 61/64 |
| 证据不足状态 | 40/128 | 104/128 |
| 错主体场景的状态判断 | 7/16 | 2/16 |

**不是记忆准确率。** 证据可用分类改善，但无需记忆/错主体判断明显退化；
不能用总分或 NLL 改善掩盖风险，也不能推断完整回答已经正确。
普通聊天分类退化不自动等同于最终答案退化，需通过预先固定的 C 自由生成面板核实；
错主体可增加单独诊断，不改变原 16 项面板或分母。

GPU 原始检查点保留于：

- `/home/ubuntu/158bitnet-locomo/build/cg001-stage-4f0a5ccd/run-first50/step-000000.pt`
- `/home/ubuntu/158bitnet-locomo/build/cg001-stage-4f0a5ccd/run-first50/step-000050.pt`

后者 138,546,037 字节，SHA-256 `a8976841239f18520842e75551dfbc8681c27f6d25e0784fed03c1efcb5585d9`，
包含记忆模块及优化器状态，不是 `.bnmem` 发布产物；GGUF、词表投影未训练。
第 50 步权重已完整下载到 `build/neural-memory-cg001-step-000050.pt`，全文件 SHA-256 与服务器一致。
下载使用 `scripts/download_training_artifact.py` 经现有认证 tmux 输出通道完成，未新建网络服务；
原始传输保留为 `.transport`。首次实时解析未处理 shell 的 bracketed-paste 控制前缀，
随后修正解析并从完整原始流恢复，通过全部 2,819 块顺序及全文件哈希校验；未重训或更改权重。
检查点是研究组件而非本地部署模型，`models/memory/resident-0.5b/` 未替换。

下一步运行固定面板的真实 C 自由生成和独立评审，重点检查普通聊天干扰及换入/移除记忆后的变化。
再根据证据检验生成损失与“无需记忆”目标是否冲突、实体绑定是否不足；这是待检验原因，不是本轮已证实的唯一解释。
不自动训练到 200/1,000 步，不替换部署产物，不宣称新系统通过。
本轮 Python **143/143**、C **21/21**；未提交代码。

## 第 50 步自由生成复现

协议冻结在 `experiments/CG-001-step50-generation.json`。来源经独立 C 编码器进入神经分支，
每个预测位置执行普通 C 参考、C 基础前向和 C 连续投影，逐位核对基础 hidden/logits。
神经融合仍是 Python，不称为完整 C/HTTP 记忆系统；不读取教师强制前缀库生成回答。

```sh
python3 python/eval_availability_checkpoint.py \
  --checkpoint build/neural-memory-cg001-step-000050.pt \
  --baseline build/neural-memory-cg001-baseline/baseline.json \
  --config training/memory/neural-system/data/CG-001-pilot.json \
  --corpus build/neural-memory-cg001-data \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --probe build/memory_continuous_probe --reference-probe build/memory_gradient_reference \
  --encoder-probe build/memory_feature_probe --tok-probe build/tok_probe \
  --output build/neural-memory-cg001-step50-generation-new
```

`review_availability_generation.py` 把基线和候选转成去身份盲包，再用两个独立评审的
语义标注评分；分歧、缺失、截断保留分母，成对条件要求两端均通过。
`audit_availability_generation.py` 校验原始产物哈希、初始位置的训练残差精确重放，
以及真实 C/Torch 词表投影在原有容差内一致；分支范数和状态变化仅作解释性诊断。
`diagnose_availability_objectives.py` 仅在相同 16 条面板上计算后置教师强制 loss 对状态头的梯度，
不更新参数、不生成新答案、不测记忆准确率，不能把局部梯度冲突直接宣称为唯一失败原因。

固定面板已经完成：基线 3/16，第 50 步 1/16，有依据 recall 0/8，普通聊天从 2/4 降至 0/4。
两位独立上下文评审对 29 个混合盲包全部一致。所有 16 条生成正常停止，168 个位置无 KV 复用；
16 个初始位置的实际 C/Torch 投影对照通过，最大 logit 误差 `4.2915e-6`。
已触发停止条件，不继续该候选。完整回复、评分、数值与梯度证据见 `reviews/CG-001/`，
下一诊断计划见 `CG001_FOLLOWUP.md`；本轮 Python 149/149、C 21/21，未提交代码。
