# DG-021：query-only uncertainty分支与诊断交接

结论：**不确定回复的神经算子与控制器交接接口已经实现，C数值和零更新梯度门槛通过。**
没有训练；零增量分支仍与base完全相同，所以不能声称学会拒答。本轮每例只确认首token，完整回复为0条。
V2整体未完成，诊断包装器默认禁止作为正式服务使用。

注册：`../../experiments/DG-021.json`。结果：`zero-update.json`，
SHA-256 `ca508eaa66b3eaff19987b9cae07674c61078017cd5782f8d0f4b9f304a1c9a1`。
原始C文件：`build/neural-memory-dg021-uncertainty/`。

## 1. 分支不读取记忆内容

`query_only_uncertainty.py`只接收绑定的实际prefix、冻结C hidden/base logits、冻结GGUF输出投影和scale。
没有source特征、payload、值句柄、teacher答案或回应模板参数。查询hidden的来源仍必须由普通C前缀证明，
只靠Python类型不能证明调用方没有把来源文本塞进prompt；本次使用已审计的自然问题前缀，没有这样做。

残差为 `Wout·tanh(Win·LayerNorm(h))`，两层均为1024维稠密投影。
`Wout`从精确零初始化，其他参数正常初始化、可求梯度；没有LoRA、SVD、NAS或骨干权重更新。
残差经过绑定0.5B GGUF的冻结输出头与logit_scale，加到普通base logits。
使用已有经过验证的连续输出数学接口，但没有装载或复用旧候选uncertainty权重。

disabled开关直接返回原base，不调用残差网络。enabled且零残差也保持base的逐位表示，包括signed zero。
这只是正确的初始化/旁路性质，不是有意义的不确定性语言行为。

## 2. 控制器如何交接

`uncertainty_reply_controller.py`包装未修改的DG-020控制器。
只有包装器自己调用core得到的NEEDS_UNCERTAINTY才启动分支，不接受外部route或伪造交接Decision。
正常/引用动作仍是原core动作，不调用uncertainty分支；新分支仅使用当前实际prefix的frame。

uncertainty token有单独动作类型/分支标识，仍需propose/commit确认实际token；
错前缀、旧动作、frame/head/参数变化、EOS后续写和预算耗尽都失败，不静默fallback或插入固定话术。
reader和uncertainty参数分别绑定并固定；这还是研究期组件身份管理，不是最终自包含模型文件格式。

构造器必须显式指定`diagnostic_only=True`；不能把零初始化的base相同输出冒充已合格的拒答模型部署。
DG-020独立控制器原来的WAIT行为和证据不变。

## 3. Native检查结果

沿用DG-020全部12条自然输入和其已审计的初始C frame，重新运行未训练reader的实际决定。
所有决定仍是insufficient，因此进入query-only诊断分支；没有改route或调权重来触发交接。
每例对零残差新跑一次C投影，重新核对fresh前缀和旧ordinary reference：

- **12/12**零残差、C/Python/base逐位一致；每例仅确认1个首token。
- 首token与base相同是零初始化的必然结果，不作为正确拒答或记忆成绩。
- 没有生成完整回复，没有独立语义评分。

另外，在预登记的英文/中文两例上建立独立人工非零实例：`Wout=0.001I`。
这不是训练更新，没有覆盖零初始化实例，更不是新的训练候选。

| 非零测试实例 | C/Python logits最大误差 | C有限差分 | autograd方向导数 |
| --- | ---: | ---: | ---: |
| 英文 | 9.536743e-7 | 0.182131397 | 0.182131246 |
| 中文 | 9.536743e-7 | 0.387815322 | 0.387811899 |

logits门槛为atol1e-5、rtol1e-4并要求同一top-1；有限差分使用固定epsilon0.01、atol0.0005、rtol0.02。
目标是base argmax的CE，**仅为求导夹具，不是正确答案**；没有搜索epsilon、种子或权重。
两例还验证了带非零delta的disabled C路径仍等于base。
总计新C调用 **20次**：12次零路径，2×4次非零/disabled/正负残差扰动；所有原始输入、输出和fresh trace再次重读核验。
没有跨位置/请求KV复用，保留单次prefill内部工作缓冲。

## 4. 零初始化的梯度含义

只对4条真值为insufficient的合成训练记录，在前向后才读取其首个目标token做CE；
没有因为未训练reader全判insufficient就把另外8条也当拒答训练样例。
4/4输出矩阵Wout梯度有限、非零；Win及其bias梯度为零。
这是Wout=0时链式法则的预期结果，**不是冻结了Win**；未执行第一步优化，不能报告Win已经开始学习。
独立非零实例中两层都得到有限非零梯度，另有C有限差分验证。

reader、零分支、人工非零实例的参数摘要前后一致，未调用optimizer、未保存新checkpoint，原源码/产物身份保持不变。
冻结输出头仍通过原始GGUF/feature manifest SHA绑定；不能替换其他backbone或tokenizer。

## 5. 下一项

先扩展到**完整不确定回复的逐token监督和轨迹级零更新检查**，而不是只验证首token。
复用合格的训练侧实际token轨迹时仍需精确prefix绑定；仅在post-forward loss中使用标签，不能把teacher答案作为服务输入。
normal旁路、supported copy和uncertainty分支应分别核验；保留错误路由产生的真实失败，不用固定模板掩盖。
随后需要完整自由回复协议和独立评审、更多语义世界与反例，以及明确的新训练预算。
CG-003的100步预算不自动延长。本轮不接HTTP、不发布模型、不启动LoCoMo。

新增14项测试；完整相关Python **379/379**、现有CTest **21/21** 通过。
没有提交代码，没有修改README的最终模型事实。

复现（输出目录/文件须不存在）：

```sh
python3 -m unittest discover -s tests -p test_query_only_uncertainty.py
python3 python/check_query_only_uncertainty.py \
  --raw build/dg021-replay --output build/dg021-replay.json
```
