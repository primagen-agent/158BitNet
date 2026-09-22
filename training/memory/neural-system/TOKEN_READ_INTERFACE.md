# DG-010：神经匹配与 token 读出接口

状态：**零训练步的研究接口通过结构检查，尚未学会主体绑定或 recall。**
没有替换 CG-002、resident 模型或 HTTP 服务；没有导出可部署权重。
登记在 `experiments/DG-010.json`，完整结果在 `checks/DG-010-result.json`。

## 1. 设计修正及实际实现

DG-009 表明成对损失提供了来源差异梯度，但不能据此保证现有共同输出通路能实现精确读取。
本原型将“是否有关”和“具体输出哪个 token”分成可独立监督的接口。
这是待对照的新假设，不是已证明优于原结构的结论。

| 部件 | 输入及作用 | 本轮边界 |
| --- | --- | --- |
| C 查询编码 | 只编码自然对话的逐 token contextual/lexical 特征 | 与来源分开；不含答案或来源原文 |
| C 来源编码 | 单条供给事件的逐 token 特征和原始 token ID | 事件尚非自动 writer 产生或自动从多事件中激活 |
| 两路匹配表示 | 可学习 query-token pooling，各自与来源位置交叉注意力，并有 NULL 位置 | 拟用于主体/关系，但没有训练或监督证明槽位已有这种语义 |
| 初始状态分类 | 两路 query、证据及交互特征预测 normal/supported/insufficient | 每次用户 prefill 形成回复内状态，不随输出前缀重判路线 |
| 逐位置读出 | 当前冻结 C hidden 加初始 query 表示，产生来源位置分布和 NULL | 只有神经网络选择位置，不用实体字符串匹配、gold mask 或答案索引 |
| token payload | 位置权重累加到原始 token ID，并与基础词表分布混合 | 重复 token 累加概率；不输出整个预制答案 |

`python/memory_token_read.py` 不接收问题字符串、source 原文、标注或参考答案。
原始 token ID 不参与匹配和位置注意力，只参与最终概率分配。
替换 payload ID 而保持特征不变的单元测试用于检查这个依赖关系，
**不是允许生产中把 ID 与实际编码特征错配，也不是未见值泛化证据**。

冻结骨干的初始 decode hidden 仍来自原 C prompt，不用 JSON 查询编码的末行冒充生成 hidden。
读取输入可使用逐 token 查询特征，并不意味着将来源拼入生成 prompt。

## 2. 概率与路线

设神经位置 softmax 为 `a`（第 0 位为 NULL），复制门为 `g`，原模型分布为 `p_base`：

`p(t) = [(1-g) + g*a(NULL)] * p_base(t) + g * sum(a(i), payload_id(i)=t)`。

采用稳定的 log-domain 混合，并检查完整词表的归一化与梯度。
normal、insufficient、禁用、空来源和无合格 payload 的情况，复制通路不改变基础 logits。
其中 **insufficient 返回原 logits 只是此组件的旁路行为，不是完整系统的拒答实现**；
与已有拒答 specialist/连续内容分支的联合训练和接入仍未完成，不能直接拿这个类替换服务。

`proposal` 是不依赖 gold 路由的可微分支输出，便于训练时在前向后选择监督项；
实际 `read` 必须用初始神经分类的预测路线，不能拿训练目标替代它。
测试里强制分类器偏置只验证路线接线，不能算错主体识别成功。

回复内状态绑定模块、输入/中间张量身份和版本；参数、来源、payload、mask、分类或表示被修改后拒绝读取。
这不是持久化格式，也未实现作用域、跨事件撤销或模型热更新。

## 3. 必须保守处理 token/字节边界

`python/native_token_payload.py` 的结构校验不识别城市、姓名、关系或答案。
只允许 **整个 token 字节都位于 JSON text 内容内部**的原始 token：

- JSON role/speaker/括号/引号等包装不允许复制。
- 跨内容边界的 token 被屏蔽；不能切开一个 token 后仍声称保留原 ID。
- BOS/EOS 和控制 token 被排除。
- 内容含 JSON 转义（例如换行、引号、反斜杠）时，本版关闭整条来源的复制资格，保留神经匹配和普通词汇生成。
  这是明确的覆盖损失，不是已经解决转义文本精确复制。
- 重复值不做字符串查找或去重；多位置的概率按 token ID 求和。

来源 token 与输出语境的空格/分词边界可能不同，跨 token 字节、UTF-8 完整性、连续多 token 值及
来源不含目标 token 的情况，还需要生成协议与后续验收。
只验证来源字节正确不等于输出拼接就正确，不能报告“任意内容都能精确复制”。
本轮不引入自动重分词、按问题搜原文或正则答案提取作为兜底。

## 4. 本轮固定检查结果

固定 seed 1010，第一训练世界 `av-002`、中英两种语言，各 original/swapped_values/wrong_subject，共 6 条记录。
fresh C 编码 2 份查询、6 份来源；验证 GGUF、tokenizer、encoder 身份。无扫描种子/阈值/权重。

- **8/8** 份编码的 C token ID、还原字节核对通过。
- 同语言换来源时，查询完全不变，验证未将来源夹带进查询。
- **6/6** 条输出概率归一化、禁用恒等检查通过；14 个参数张量均有有限、非零的诊断梯度。
  梯度来自基础 top-1 的数值探针和状态平方项，不是训练损失、标准答案或能力证明。
- 中英两条空记忆检查均禁止 supported/copy，原始 logits 逐位不变。
- 真实 C 输入上的未训练路线：**6/6 supported，包括 2/2 wrong_subject**。
  实际复制概率质量约 0.0130—0.0172；这是随机初始化的接口输出，不是正确复制率。
- 9 项新增单元测试覆盖重复 token、结构边界、极端 gate、梯度、状态生命周期和旁路。
  Python 回归 **191/191**，现有 build 的 CTest **21/21**。

运行证据：`build/neural-memory-dg010/`，含 fresh C 输入、二进制特征和日志及其哈希。
生成 hidden/base logits 使用已验证的冻结 C prefix bank，**不是本轮新跑的 decoder 或端到端 C 自由生成**。
C 编码每份输入新建 context 并完整 prefill；不复用之前输入的 KV。prefill 内部仍使用 K/V 计算缓冲。
优化器步数 0、新自由回答 0，参数不变。原 CG-002 自由 recall **0/8** 结论未改变。

## 5. 下一阶段及停止条件

下一项是 **零训练步的监督契约和生成适配检查**，不是直接跑第三次训练：

1. 为两路匹配定义后置、可检验的主体/关系监督和成对对照；不能仅给两个槽位起名字就声称实现分解。
   保留 normal/insufficient 和多种事实状态，不能只训练 supported 的 copy。
2. 分清目标 token 可由当前安全 payload 输出、不可复制和语境边界不同三种情况；
   不可复制样本必须留在分母中，使用基础生成/连续分支损失，不能筛掉来抬高结果。
3. 明确 copy、连续内容、拒答和普通生成四者如何组合及梯度流向。
   先检查完整词表 CE、错误来源反事实与关闭分支保真，再做真实 C full-prefix 生成适配，禁用 KV 历史复用。
4. 新的受控训练须重新登记预算、对照和停止条件，并保留仅改监督与增加通道的消融。
   当前接口并没有解除两候选预算限制，也没有满足部署/LoCoMo 门槛。

未覆盖的自动 writer、多事件激活、时间/否定/更新/遗忘、持久化与完整 HTTP 链路继续保留在总计划中。

## 复现

```sh
python3 python/diagnose_token_read_interface.py \
  --experiment training/memory/neural-system/experiments/DG-010.json \
  --features build/neural-memory-cg001-features \
  --config training/memory/neural-system/data/CG-001-pilot.json \
  --corpus build/neural-memory-cg001-data \
  --gguf models/bitcpm4-0.5b-tq2_0.gguf \
  --tok-probe build/tok_probe --encoder-probe build/memory_feature_probe \
  --output build/neural-memory-dg010-new
```

输出目录必须不存在。参考命令只做未训练接口检查，不启动训练或保存候选权重。
