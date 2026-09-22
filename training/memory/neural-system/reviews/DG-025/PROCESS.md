# DG-025：时间限定跨度编译接口

结论：**historical_stated 现已可教师编译；被 pin 的 compile_teacher/positive_bytes 链逐字节未动，current 时间路径通过委托保持完全一致。**
零模型前向、零优化器步。注册：`../../experiments/DG-025.json`；检查：`checks/DG-025-compile.json`。

## 1. 方案

`python/compile_time_scoped_teacher.py`：
- **委托**：`query_time` 为 current 或缺省（JB-001 全部、JB-002 correct/multi_fact）时直接调用原
  `compile_teacher`——数据类全等断言（JB-001 jb-000 六条 + JB-002 96 条 current-time supported）零偏差。
- **时间分支**：`positive_bytes_time` 按 `meta['query_time']` 选择 2020 年 actual 事实，
  用同一 `time_sentence` 过去式定位子句/值字节；复制路径复用原 cursor/append 机器与
  `TeacherTrajectory` 格式（oracle 标识、显式 EOS、容量检查、逐字节回复保持全部沿用）。

## 2. 核查（train-only，镜像被 pin 编译器的训练侧策略）

| 项 | 结果 |
| --- | ---: |
| JB-002 train supported 编译 | 144/144 |
| 其中 historical_stated（route-1，2020值跨度） | 48/48 |
| 其中 current-time 委托 | 96/96 |
| 委托一致性偏差 | 0/6 |
| 重名/缺失2020事实拒绝 | 通过（测试覆盖） |
| dev/test | 保持未编译 |

每条 historical_stated：恰一次 START/END、EOS 结尾、值跨度字节=2020子句值、
编译两次确定、同人 current 事实不被选中。dev/test 按被 pin 编译器语义保持拒绝。

## 3. 下一项

在 JB-002 上扩展 **DG-022 式零更新监督核验**（时间限定与多事实干扰轨迹的梯度/数值门槛），
协议沿用 DG-022；训练仍需单独批准的新预算。

新增5项单测（`tests/test_compile_time_scoped_teacher.py`）；未提交代码。

复现：

```sh
python3 -m unittest discover -s tests -p test_compile_time_scoped_teacher.py
```
