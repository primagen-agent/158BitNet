# EV-001：V3-001（50步监督补全候选）自由回复评估与双盲评审

结论：**候选未通过预登记门槛。** 训练后的reader将全部24条路由到normal/base——
uncertainty分支与值通路从未激活，所有回复均为base延续。不选优、不部署、不自动追加训练。
注册：`../../experiments/EV-001.json`。生成核验：`free-replies.json`；评审：`score.json`。

## 1. 协议层（通过）

24/24条真实逐token自由生成（每位置冻结reference取base帧+训练残差注入continuous探针，
重跑确定性验证、packet id一致）；由于路由坍缩，残差注入路径实际从未触发，C奇偶校验平凡通过。
截断10/24、EOS 14/24，如实保留。

## 2. 语义门槛（两位全新隔离评审，24/24逐项一致）

| 门槛 | 结果 | 判定 |
| --- | ---: | --- |
| normal保持 | **4/4** | ✅ |
| uncertainty承认缺证且不编造 | **3/10**（≥4） | ❌ |
| grounded主体+值正确 | **0/10**（≥3） | ❌ |
| 截断 | 10/24（≤2） | ❌ |

3/10的uncertainty通过项来自base自身的AI口吻回避（"I'm sorry, but as an AI language model…"），
不是训练得到的拒答；grounded全败（base编造New York City或跑偏）。

## 3. 失败归因（已核实的事实）

- **路由坍缩到normal**：所有24条首决策均为base（DG-020未训练时全判insufficient；50步后翻转为全判normal）。
  训练分布为r0=96/r1=316/r2=528——坍缩到**少数类**normal，说明不是简单多数偏置，
  route头的50步/800样本/lr1e-4更新未建立判别，翻向了另一端的退化解。
- 分支CE与跨度/模式监督确实在训（损失5.3→~1.7），但推理路径从不经过它们——
  **监督补全的下游行为被路由头单点卡死**。
- 运行前如实修正：提案门槛"normal 8/8"与实际面板（4条normal）不符，已在生成前改为4/4并登记。
- 评审标注 schema 放宽：允许空value（对"Tanja在2020年住在。"这类无值断言的诚实标注），
  两位评审独立一致；基础设施修正，非门槛变更。

## 4. 下一项

**路由头归因诊断（零更新）**：在缓存特征上检查训练后route logits分布/边际与逐场景混淆，
区分（a）欠训练（route头未收敛）与（b）监督目标冲突（static loss内route项被其他头淹没）；
据此决定是延长预算申请还是先改损失加权。任何新预算均需用户批准。

复现：

```sh
python3 python/check_v3_eval.py --raw build/ev001-repro --output build/ev001-repro.json
python3 build/ev001-aggregate.py
```
