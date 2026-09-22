# DG-023：完整自由回复协议与独立双评审

结论：**协议已认证：12/12条真实逐token自由回复全部通过C核验与零更新门槛；未训练语义基线如实记录为 4/12。**
normal算术4/4；supported 0/4（编造"New York City"）；uncertainty 0/4（英文编造事实、中文跑偏到"请稍等查询"并截断乱码）。
没有训练、没有权重更新；通过的是协议与证据链，不是记忆能力。

注册：`../../experiments/DG-023.json`。生成与核验：`free-replies.json`。
原始C文件：`build/neural-memory-dg023-freereply2/`（280次冻结C调用，全哈希登记）。

## 1. 生成与C核验

12条记录（DG-020面板全部）在DG-023身份下新建控制器；每个位置对"自然提示+模型自身已生成token"
做冻结`memory_continuous_probe`全新前缀前向（零残差），无跨位置KV复用，披露单次prefill内部工作缓冲。
硬门槛全部通过：每位置零残差与base逐位一致（12/12）；wrapper决策token==C logits argmax（全位置）；
8条normal/supported记录的branch回复与纯base贪心回复**逐token一致**（零刔分支≡base的协议级验证）；
参数摘要前后逐位一致。回复预算24 token：8条EOS停止、4条触及预算（截断如实保留，含`<reserved_NN>`乱码）。
重跑一次生成完全确定（回复/停止/长度与packet id全部一致）。

## 2. 双评审（两轮，全部保留）

盲包只含 id/语言/问题/回复/截断标记，无gold、标签、路由或模型名。
四位**全新**隔离同族模型实例（a/b、a2/b2），互不可见；评分模式预登记于注册。

**v1发现两个协议缺陷（本轮的实质产出）：**

1. **端标记伪影**：packet把`<|im_end|>`字面量带进回复文本，评审按预登记rubric判natural_reply=false，
   导致答案其实全对的normal算术0/4。这是测量伪影，不是模型失败。v2包剥离尾随端标记（=服务端呈现方式），
   乱码与截断原样保留；两位全新评审重审。
2. **route-2门槛欠约束**：预登记的uncertainty判据（acknowledges_missing_evidence且无对所问事实的无据断言）
   未含natural_reply——v1中两条含乱码的截断中文回复因此"通过"。已作为限制披露，
   下次注册必须加入natural_reply约束；本轮未回溯改门槛。

## 3. 结果（v2，两位评审12/12逐项一致）

| 路由 | 分母 | a2通过 | b2通过 | 事实 |
| --- | ---: | ---: | ---: | --- |
| normal（算术） | 4 | **4** | **4** | "3 + 7 = 10" / "3加7等于10。" |
| supported | 4 | 0 | 0 | 编造"Brenna lives in New York City"（gold为Riga等） |
| uncertainty | 4 | 0 | 0 | 英文编造New York City；中文"请稍等，我马上为您查询"+保留标记乱码，24 token截断 |

未训练分支输出=base贪心延续（协议级验证），因此该4/12不是拒答能力：normal通过项是base骨干自身的算术能力。
v1结果（normal 0/4、uncertainty 2/4）与全部四份评审文件、两版盲包一并保留。
评审为同族隔离模型实例，不是人工或跨族校准。

## 4. 下一项

协议已可用于任何后续候选。按计划顺序，下一项是**扩充语义世界与反例的数据/监督预登记**
（更多世界、错时间/否定/多事实干扰、状态监督广度），训练仍需明确批准的新预算；
CG-003的100步预算保持耗尽，不自动延长。不接HTTP、不发布、不启动LoCoMo。

新增8项单测（`tests/test_free_reply_protocol.py`）；未训练权重、参数零更新、未提交代码。

复现（输出路径须不存在）：

```sh
python3 -m unittest discover -s tests -p test_free_reply_protocol.py
python3 python/check_free_reply_protocol.py \
  --raw build/dg023-freereply-repro --output build/dg023-freereply-repro.json
```
