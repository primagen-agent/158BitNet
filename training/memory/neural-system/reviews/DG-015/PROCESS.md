# DG-015：V1 事实句柄与值传递软件原型

当前状态：**原型和边界测试已实现，但V1原定native验收尚未全过：10/11。**
未实现神经事实/跨度预测器，没有训练、没有自由回复生成，也没有新的记忆准确率。
本轮不进入V2训练准备，不接入HTTP，不修改现有CG-003模型或旧生成路径。

## 已实现

- `python/value_transport.py`：不可变payload快照、reply-local事实句柄、连续UTF-8值跨度、预算、编译后游标。
- `python/native_value_codec.py`：使用同一0.5B GGUF和C tokenizer；编译前后核对文件身份。
- `python/check_value_transport.py`：11条预登记native人工夹具、来源标注、完整分母、失败原样输出。
- `tests/test_value_transport.py`：人工字节codec上的软件契约回归，不是模型测试。

句柄绑定骨干/tokenizer/协议/编码器、记忆模型身份、用户作用域、记忆实例、版本、完整payload摘要、请求和上下文摘要。
即使版本号相同，payload字节改变也会拒绝；进入值模式后不能换事实或换请求。
快照全量摘要是正确性优先的原型检查，不是已经优化的长期记忆热路径。

`compile_value`只根据已选择的跨度取值，没有问题匹配、实体词典、来源检索或gold答案解析。
真实C tokenizer对“已有输出前缀+所选值”编码；只有旧token前缀保持不变、后缀字节完全等于值时才创建游标。
这不是把整个来源写入backbone提示；本轮甚至没有backbone decode调用。

游标每次给出下一个已编译token，`commit`核对实际前缀和实际发出的token后才返回新游标。
值内不能跳步、提前EOS或被base token覆盖；重复token按位置计数，结束后恢复原base token选择。
NULL不调用tokenizer，普通路径不改base选择。所有失败均为显式错误，没有截掉值、补词或部分成功fallback。
`committed_text`只暴露已经完整的UTF-8字符，暂存跨token半个字符；这仍不是SSE实现。

所有本轮人工句柄均标记`oracle_fixture`，默认拒绝运行，只有测试显式启用。
接口中的`neural_prediction`只是未来调用方的来源声明，**不是可信来源认证或已经存在的神经预测器**。
选择错误事实或过短跨度时，本算子也会忠实传递错误内容，专门有回归证明它不会靠语义规则偷偷纠正。

## Native结果：保留未通过项

预登记：`../../experiments/DG-015.json`，没有改写fixture预期。
最终证据：`native-fixtures-certified.json`；较早运行保留在`native-fixtures.json`，未覆盖。
后一次增加了训练证书/特征包固定摘要检查和前缀token字节证据，没有改变传递算法或fixture。

| 类别 | 例数 | 实际结果 |
| --- | ---: | --- |
| Riga、Quimper、中文、emoji、重复片段、含前导空格的值 | 6 | 完整传递、顺序一致、有限结束 |
| 前缀空格合并、前缀字母合并 | 2 | 按登记要求在发出任何值token前拒绝 |
| 控制token、超token预算 | 2 | 按登记要求在发出任何值token前拒绝 |
| 带引号值 | 1 | 原期望可传递，实际拒绝：**未通过** |

失败例：前缀`Value:`、值`"Riga"`。
原C token字节为`[" Value", ":"]`，合并编码为`[" Value", ":\"", "R", "iga", "\""]`。
第二个token从冒号变为冒号加引号，违反“不改写已输出token”契约，因此安全拒绝。
这不是原权重错误或模型生成错误；是首版“全串canonical分词+稳定前缀”适配器的覆盖限制。
不能把安全拒绝重新标为该正例通过，也不能去掉引号例来宣布V1完成。

**10/11是软件fixture符合预期的数量，不是记忆正确率。**
其中6条实际传递值、4条按预期拒绝、1条非预期拒绝；没有神经激活或正确答案选择测量。
全部127项已绑定的旧训练源文件/证书/检查点在运行前后摘要不变。
0次optimizer更新、0次backbone生成；C只做tokenizer操作，不存在本轮KV记忆测量。

## 下一步

V1维持未完成。下一项先设计并预登记**冻结实际token前缀下的边界安全续接**，再处理引号/空格/字母合并。
任何新方法必须使用C运行时token字节验证，不可悄悄回写已输出前缀、丢空格、插入原文提示或退回模型猜词。
如采用非canonical但字节等价的分词，必须明确变更当前契约并验证新actual-prefix下的C行为；不能把它当成当前版本已支持。
同时保留预算、控制token、UTF-8和旧token前缀完全不变的硬门槛。

V1通过后才实现神经联合激活/跨度/模式监督接口的零更新验证。CG-003的100步训练预算已用尽，不自动续训。
现在仍缺神经选择、实际decoder接线、自然回复质量、自动写入、生命周期与HTTP端到端证据。

最终新增软件单测23/23通过，完整相关Python回归296/296、现有CTest21/21通过。
这些软件回归不覆盖掉native门槛10/11的未通过结论；模型参数未改、没有提交代码或更新README中的最终模型事实。

复现（native检查因保留的quoted-value失败返回非零是当前预期状态）：

```sh
python3 -m unittest discover -s tests -p test_value_transport.py
python3 python/check_value_transport.py --output build/dg015-check.json
```
