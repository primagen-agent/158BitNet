# EV-005：V3-009（三组件候选）评估与双盲评审

结论：**两项门槛通过（uncertainty 9/10、normal 4/4），总门槛未过（grounded 0/10、截断6/24）。**
首次在路由判别（82%）、mode校准（268/268）、uncertainty分支三者齐备下评估——三通路首次全部实际行使。
注册：`../../experiments/EV-005.json`；生成核验：`free-replies.json`（parity 0失败）。

| 门槛 | EV-004 | EV-005 | 判定 |
| --- | ---: | ---: | --- |
| uncertainty | 10/10 | **9/10** | ✅（≥4） |
| normal | 2/4 | **4/4** | ✅ |
| grounded | 0/10 | 0/10 | ❌（≥3） |
| 截断 | 0/24 | 6/24 | ❌（≤2） |

事实：normal算术中英全对（路由到base通路）；uncertainty仅wrong_time/zh一条编造"上海"；
supported项仍走base延续或截断——**值通路（START→cursor复制）虽激活条件齐备，但START决策在
自由生成中未触发**（回复无一次经复制路径），且jb2-009英文项落入"as an AI"或"To provide"截断轨迹。

## 下一项
**DG-033零更新归因**：在EV-005轨迹上逐位置检查supported项的路由/mode决策序列（START为何未在
任何位置触发）；区分mode在自由前缀上的行为与教师前缀上的268/268差异。grounded是最后一个未通的门槛。
