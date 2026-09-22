# DG-033：START未触发的逐位置归因（零更新）

结论：**mode头被LBFGS阶段饱和（logit幅度±40~71），且教师前缀268/268与自由前缀0触发并存——
mode学到"教师token身份记忆"而非"值应开始"的通用信号。**

## 三个事实
1. 路由正确且恒定：route argmax=1（supported）全程——路由头输入不含prefix（设计如此）。
2. mode饱和：margin~140、argmax恒0——LBFGS在2180:268不平衡上收敛到极端置信。
3. 教师/自由前缀分歧：pos5起base写"New York"而教师值"Riga"——DG-032的268/268实为教师轨迹
   前缀记忆（DG-029预警的位置规律风险兑现）。

## 修法方向（需登记+预算）
mode训练改用自然前缀对照（pre-value由base生成替代仍标START），或START解耦prefix身份。
