# DG-019：模式切换教师轨迹与实际C特征

结论：**12条固定合成训练样例的完整教师轨迹、C特征、监督和零更新梯度检查通过。**
这不是模型自己生成的答案，不是12/12记忆准确率；模型没有训练，也没有新增可部署产物。
V2整体仍未完成。

注册：`../../experiments/DG-019.json`。结果：`zero-update.json`，
SHA-256 `5da3ffa91a74d5d5bfbefb0a7571adb7e4366b46d8e5e24083814466e697b533`。
独立原始文件核对：`raw-audit.json`。数据/原始C文件：`build/neural-memory-dg019-trajectories/`。

## 1. 固定范围

JB-001训练世界jb-000、home_city、英/中两种语言，各保留correct、value_swap、role_swap、empty、ordinary_source、ordinary_empty。
共12条：supported 4、insufficient 4、normal 4。没有接触dev选优、sealed-test或LoCoMo。
4条supported使用来源中的精确值字节，不加空格、不trim；所有参考回复字节和EOS完整保留。
只有一个训练世界，不能外推泛化能力。

| 轨迹阶段 | 位置数 | 监督/控制含义 |
| --- | ---: | --- |
| GENERATE | 104 | 普通生成位置，idle目标0 |
| START | 4 | 每条supported的第一个值token，idle目标1 |
| CONTINUE | 6 | 值内剩余token，由游标决定，不计算idle CE |
| END | 4 | 值后第一个token，游标已结束，idle目标0 |
| 合计 | **118** | 含12条完整回复的EOS |

因此有112个idle监督位置，而不是118个；END不等于整段回复EOS，CONTINUE/END也不是新训练的独立分类头。
每条正例恰好一个START和END，负例/普通问题不能进入值模式。

## 2. 教师轨迹与自主推理分开

`value_teacher_trajectory.py`是训练专用编译器：参考回复划为前文、精确值、后文，分别用相同C tokenizer的边界隔离续接编码。
正例值使用DG-016的真实游标，但句柄明确标记`oracle_fixture`；整个轨迹标记`teacher_forcing_oracle`。
不能将oracle句柄改名为神经预测。编译/读取接口拒绝inference用途及dev/test记录。
这些用途检查是本流程的防误用约束，不是防止任意调用方伪造数据来源的安全认证。

后续神经前向只接收自然问题/来源的冻结C特征以及**过去教师token构成的实际前缀隐藏态**。
不会接收未来token、标签、正确边界或oracle游标状态。标签在前向完成后参与损失。
这里允许标准teacher forcing的“过去参考token”，但不能把这种条件下的检查当作自主生成或在服务中复用轨迹。
运行时是否开始引用、选哪条事实和值，仍须由真实预测决定。

## 3. C特征提取与核对

118次位置使用对应64个不同的token前缀。仅按完整token ID序列去重，同一文本不同分词不能共用特征。
用`memory_prefix_probe`提取全部64行，每行一个fresh context；共享只读GGUF权重，不共享会话/跨前缀KV。
这些是离线训练特征，不是推理检索缓存。单次prefill内部K/V工作缓冲仍存在。

直接ordinary C reference的选点预先固定：每条首/中/末位置，加全部START/CONTINUE/END，去重后31条。
31/31隐藏态和logits逐位一致；全部64行检查几何、有限性、输入token字节、fresh-context trace和文件摘要。
**不是64/64都跑了第二份reference**；总C forward次数为64+31=95，进程调用数不是95。

独立审计器`audit_value_teacher_trajectories.py`再次核对：

- 12条固定清单、完整回复字节、EOS、来源精确值及状态/阶段标签；
- 118个实际前缀使用与64个去重前缀一一对应；
- C batch输入wire、输出行、CPU dispatch及fresh-context记录；
- 31个直接reference的原始输入/输出及FP32逐位一致；
- 所有阶段位置计数及训练专用标识。

没有把旧canonical前缀特征替代新续接后的token序列。来源/问题编码仍复用身份一致的旧C编码，因为自然输入没有改变。

## 4. 零更新梯度检查

使用DG-018相同随机种子和几何；初始参数摘要与DG-018逐位一致，没有新增模型候选或换初始化选优。
每条记录先算一次路线/事实/值/字节边界静态loss，再对其112个idle位置中的所属位置取模式CE均值。
每条记录独立清空梯度检查，不根据句子长短重复累计静态loss；CONTINUE不误计为“不要开始引用”的负监督。
本轮没有优化普通措辞token CE，不能声称已教会冻结骨干自然回复或拒答。

12/12记录梯度有限、非零；4条正例的粗跨度、细字节头连通，所有记录的模式头连通。
模型参数和原训练源码/检查点运行前后保持相同摘要，未调用optimizer，未保存新checkpoint。
当前随机模型在12条初始输入均预测insufficient；112个idle位置的组合策略都没有START。
这是包含路线门控的结果，不是模式head单独的准确率；不把教师输出的正确内容归给该模型。

## 5. 下一项与未解决问题

先补齐独立的**自主模式控制器接口**：只用真实神经路由、边界、START预测与实际前缀，不能读取教师轨迹、标签或后续目标token。
要定义并测试normal旁路、NULL/insufficient信号、START句柄、CONTINUE游标、END返回、状态失效/预算失败。
insufficient如何成为自然且真实的回复目前未接通；不能把它当normal直接放行然后宣称已正确拒答，也不能用固定答案模板伪装学习成功。
回到base后继续扩写实体或添加错误内容的风险，也必须由完整自由回复评测覆盖。

仍需更多合成世界、时间/否定/多值等反例、自由生成接口、独立评审与新训练预算。
本轮只验证了控制器相关监督和特征，不代表设计已经具备端到端高精度记忆。
不自动延长CG-003的100步预算，不接HTTP，不发布模型，不修改README中的最终模型事实。

新增8项软件回归；完整相关Python **340/340**、现有CTest **21/21** 通过。
没有提交代码。

复现（输出目录/文件须不存在）：

```sh
python3 -m unittest discover -s tests -p test_value_teacher_trajectory.py
python3 python/check_value_teacher_trajectories.py \
  --raw build/dg019-replay --output build/dg019-replay.json
python3 python/audit_value_teacher_trajectories.py \
  --report build/dg019-replay.json --output build/dg019-replay-audit.json
```
