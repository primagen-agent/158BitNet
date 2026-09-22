# DG-024：扩充语义世界与反例的数据/监督预登记（JB-002）

结论：**24个新语义世界、768条双语记录、480组成对对照已冻结；重生成逐字节一致；
与JB-001人名/城市/世界完全不相交；零训练、零模型前向。**

注册：`../../experiments/DG-024.json`。语料：`data/JB-002/`（manifest SHA-256
`0e5e1a06e7dff6ecc5e30fa62e15487430e121fe7c81146e3ae37a379f435c22`）。
检查：`checks/DG-024-data.json`。生成器：`python/prepare_expanded_memory_worlds.py`。

## 1. 场景与状态表（8场景 × 每组）

| 场景 | 状态 | 来源构成 |
| --- | --- | --- |
| correct | supported | 目标当前事实 + 他人异关系干扰 |
| role_swap | insufficient | 角色互换（JB-001 对照） |
| empty | insufficient | 无存储 |
| wrong_time | insufficient | 仅当前事实；问2020年（时间限定拒答） |
| historical_stated | supported | wrong_time来源 + 恰好一条2020年事实 |
| multi_fact | supported | 目标 + 他人干扰 + 同人异关系干扰 |
| hypothetical | insufficient | 假设句替代事实（获奖将搬） |
| quoted | insufficient | 他人转述替代事实 |

分组：12 train / 6 dev / 6 test 世界（世界级划分），每世界2人2城，48+48全新人名/城市，
与JB-001池断言不相交。计数与注册完全一致：train 384（144 supported/240 insufficient）、
dev 192、test 192；test保持封存（未分词、无模型前向、不用于选择）。

## 2. 成对对照（每语言×关系组验证，96组）

- **time_pair（新）**：wrong_time 与 historical_stated 共用**同一时间限定问题**，
  来源恰好相差一条2020年子句——状态翻转 insufficient/supported 的最小对照。
- hypothetical/quoted/correct 共用当前问题，来源只在事实状态上不同。
- role_swap/empty 与 correct 成对（沿用JB-001口径）。
- 监督三文件物理分离；输入仅为用户消息；标签/索引不进入运行输入（digest绑定+测试）。

## 3. 管线兼容与登记的接口边界

- `compile_teacher` 在全部96条current-time supported记录（correct+multi_fact）上编译通过
  （route=1、显式EOS、冻结tokenizer实测）。
- **historical_stated 按登记被拒绝**（`positive_bytes` 只认current风格子句）——时间限定值跨度
  需要时间感知的跨度编译器，已登记为下一接口项，不做静默近似。

## 4. 下一项

**预登记时间限定跨度编译接口**（positive_bytes/compile_teacher 的时间感知扩展，
使 historical_stated 可教师编译），随后在 JB-002 上扩展 DG-022 式零更新监督核验。
任何训练仍需单独批准的新预算；CG-003 的100步保持耗尽。

新增9项单测（`tests/test_expanded_memory_worlds.py`）；未训练、未提交代码。
复现（输出目录须不存在）：

```sh
python3 -m unittest discover -s tests -p test_expanded_memory_worlds.py
python3 python/prepare_expanded_memory_worlds.py \
  --config training/memory/neural-system/data/JB-002.json \
  --output build/jb002-repro
python3 python/prepare_expanded_memory_worlds.py \
  --config training/memory/neural-system/data/JB-002.json \
  --output training/memory/neural-system/data/JB-002 --verify
```
