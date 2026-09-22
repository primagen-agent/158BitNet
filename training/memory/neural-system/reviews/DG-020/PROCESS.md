# DG-020：自主模式控制器契约

结论：**控制器接口与状态转换的软件检查通过；12条真实C首决策均停在NEEDS_UNCERTAINTY且没有发出token。**
这是未训练模型的真实输出，不是记忆准确率、完整拒答或自然回复成功。
不确定回复生成器尚未实现，V2整体未完成。

注册：`../../experiments/DG-020.json`。结果：`native-decisions.json`，
SHA-256 `f259c416bc25edaf852b085648c9331d720f0675ed35de3114890805a89a665a`。
原始C输入/输出保留在`build/neural-memory-dg020-controller/`。

## 1. 控制器只消费运行时预测

`python/autonomous_value_controller.py`接收FineSpanReader、自然输入C特征、字节映射、只读snapshot、请求绑定和实际前缀。
每一步接受明确类型的LiveFrame，包含确切prefix、冻结FP32隐藏态/logits和来源标识。
它不接收外部route覆盖、标准答案、teacher轨迹或gold句柄。
预测START时才从同一神经前向得到字节句柄；NULL/gold/不一致句柄不能进入引用。

Reply初始化时检查完整模型参数摘要与编码域、用户作用域、来源和上下文；
之后检查参数/输入tensor版本、codec绑定、真实前缀及pending动作。
来源标识和Python类型用于防误用，不是对任意伪造Python调用的安全认证；C适配器仍必须证明frame出处。

## 2. 状态行为

| 情况 | 行为 |
| --- | --- |
| normal | 原样选base argmax，普通路径不强制copy |
| supported、未预测START | 选base argmax，等待真实模式预测 |
| supported、预测START | 编译预测值，首token进入游标；禁止oracle句柄 |
| CONTINUE | 不重新选事实，按绑定游标输出下一token |
| END | 值完成后的首步选base argmax，清除游标、返回idle |
| insufficient | NEEDS_UNCERTAINTY，无token、无固定文本，进入等待状态 |
| EOS | commit后关闭reply，不能继续生成 |
| 不一致/过期/越界/超预算 | FAILED，不静默返回base，也不报告成功 |

首次神经路由在reply内固定；其后路由意外改变会失败。引用中的source/model变更同样失败。
propose和commit分开：commit只接受该实例刚发出的同一个Decision对象，并核对实际输出token。
复制的action、另一请求的action、重复propose、错误token或确认前被改动的C frame均拒绝。
START先为END交接预留一个token和context位置；引用开始之前就检查值预算/安全性。
当前pilot最多引用一个值，再次预测START会明确预算失败，不通过偷偷压制重复来伪造成功。
这不是多事实、多值完整回复的最终策略。

空来源允许绑定空snapshot，但不会伪造空FactPayload或让supported的NULL句柄走copy。
普通生成留下不完整UTF-8前缀时，若此时试图START，当前版本严格失败，不丢字节修补；这仍是需要后续覆盖的运行限制。

## 3. 25项人工状态测试

测试覆盖normal/EOS、supported idle、START→CONTINUE→END、UTF-8、空来源、insufficient无token、
teacher/oracle拒绝、错prefix/action/ack、来源/模型/codec/frame变化、跨请求动作、值/回复预算和重复激活。

为了覆盖未训练模型当前不会走到的分支，单测使用明确标记的synthetic frame和stub预测，
构造器必须显式允许synthetic；native检查没有启用，也没有修改预测或权重。
这些软件fixture不计入真实记忆能力或12条native结果。

## 4. 固定12条真实C首决策

沿用jb-000、home_city、英/中、6类自然输入的固定范围，只读inputs/index选取，不读labels文件或teacher轨迹归档。
旧DG-019报告只用于固定权重/源码身份，监督字段不进入当前决策。
query/source使用身份验证一致的C编码，实际初始生成前缀通过新的fresh C进程执行。

- 12个首前缀各有普通C reference和零增量continuous探针，共24次C forward。
- 12/12隐藏态与logits逐位一致，增量为零、fresh-context检查通过。
- 全部12条模型预测insufficient；控制器全部返回NEEDS_UNCERTAINTY，selected_token为null。
- 全部保留0个输出token，0条完整回复。没有模板拒答、base回退或teacher内容补齐。
- 原始文件摘要、prefix wire、逐位数值、动作和无输出状态在运行结束后再次重读核验。

因此不能把12/12接口通过写成12/12正确拒答：范围中还包含真正有记忆和普通问题，当前未训练路由并不正确。
同样，native没有实际覆盖START/CONTINUE/END；这些分支本轮只有人工软件fixture验证。
不复用跨请求/跨位置KV，单次prefill内部仍有工作缓冲。
模型参数摘要与DG-019相同且运行前后不变，原训练源码/产物身份不变。

检查入口首次因缺少一个导入在C执行前退出；修复后原12条清单完整重跑，没有删除样例或改门槛。

## 5. 下一项

下一步先定义并验证**仅依赖当前问题/实际前缀的uncertainty生成分支**，接收NEEDS_UNCERTAINTY，
不得访问来源payload、copy或teacher答案；normal旁路保持不变。
可以复用现有query-only数值接口的验证经验，但不默认旧候选权重已合格，也不把固定拒答模板作为模型能力。
需要先验证C/Python数值、梯度、参数与prefix绑定、终止/预算，再制定训练和完整回复评分。
状态头预测insufficient和生成一句自然、真实的拒答是两个不同门槛，不能再合并统计。

还需source语义绑定、模式学习、多值/更新/时间反例和完整自由回复评审；未批准新的训练预算。
不接HTTP、不发布模型、不启动LoCoMo，不把用户的记忆目标改成查询式RAG。

最终新增状态测试25/25，完整相关Python **365/365**、现有CTest **21/21** 通过。
未更新权重、未启动训练、未提交代码，也未改README中的最终模型事实。

复现（输出路径须不存在）：

```sh
python3 -m unittest discover -s tests -p test_autonomous_value_controller.py
python3 python/check_autonomous_value_controller.py \
  --raw build/dg020-replay --output build/dg020-replay.json
```
