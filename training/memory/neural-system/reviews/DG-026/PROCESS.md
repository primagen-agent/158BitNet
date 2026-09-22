# DG-026：JB-002 零更新监督核验扩展

结论：**时间限定/假设/转述拒答与时间值/多事实轨迹的零更新监督门槛全部通过。**
零训练、零参数更新；这是数值与监督接口证据，不是记忆能力。注册：`../../experiments/DG-026.json`。
结果：`zero-update.json`，SHA-256 `699d9fbe36e184c67b05a9fe3c82608eb4fe78cb87b620331feba639170dcede`。
原始C文件：`build/neural-memory-dg026-supervision/`（201前缀+72参考，全哈希登记）。

## 0. 运行前修复（先红后绿）

DG-026的wrong_time编译暴露`compile_teacher_time`委托条件过窄（时间限定+insufficient两条路径均被拒）。
按TDD修复：无跨度需求（state!=supported）一律委托原编译器；回归测试后JB-002全部384条train记录可编译，
current路径委托逐位不变。

## 1. 面板（jb2-009，20条，位置数全pin）

- **不确定臂12条**：wrong_time（时间限定拒答）/hypothetical（假设）/quoted（转述）×en/zh×home/work。
  每条教师轨迹经委托编译；166个位置全部C前缀批量提取+每条首/中/末参考探针逐位一致。
  query-only分支每位置零残差与base逐位相同、disabled恒等、逐token CE等权聚合反传符合
  零初始化链式法则（Wout非零、encoder精确为零）。均值损失30.79。
- **支持臂8条**：historical_stated（2020值跨度，DG-025时间分支编译）与multi_fact（多事实干扰）。
  reader静态fine_loss+idle模式CE在真实C前缀上反传：梯度有限非零、route-1正例
  （offset/boundaries/fact_value/value/value_query/mode头）全部连通，参数摘要前后逐位一致。

特征由冻结编码器探针对28条面板文本新提取（`build/neural-memory-jb002-features/`，
bank摘要`e6bf0382…`，encoder_id与CG-003身份绑定；冻结特征包只读未动）。

## 2. 分母披露

hypothetical与quoted在query-only监督臂上**信号完全重合**（问题与拒答模板相同，来源不同但分支不读来源侧）：
12条记录实际为8组不同监督轨迹（wrong_time×4 + hypothetical≡quoted×4），与DG-022的role_swap/empty重合同类。
支持臂8条互不相同（来源不同、值不同）。

## 3. 下一项

监督与协议基础设施至此覆盖JB-001+JB-002全部场景（当前值、换值、角色、时间限定、多事实、假设、转述）。
下一步进入 **V3训练方案预登记**：固定数据/预算/早停/对照后**向用户申请明确批准**；
未获批准前不启动任何优化器步。

复现（输出路径须不存在）：

```sh
python3 -m unittest discover -s tests -p test_compile_time_scoped_teacher.py
python3 python/check_jb002_supervision.py \
  --raw build/dg026-repro --output build/dg026-repro.json
```
