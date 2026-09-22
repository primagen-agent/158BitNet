# CG-003 固定基线：独立语义评审

这是关闭记忆的24例 C 自由回复基线，不是训练候选或最终系统成绩。
基线 SHA：`4389fc1c29a28f81353dac9ecc52bec8bb024fd54c658ceecc8485db5874433d`。
24例全部正常停止，UTF-8完整；252个真实生成位置的原始 C 输入、输出、预测前缀和 fresh-context 已核验。

## 评审方法

两位全新、隔离上下文的评审实例 `cg003_baseline_judge_a` 与 `cg003_baseline_judge_b`
只读取 `baseline-blind.json` 与公共标注结构，不读取标签、候选权重、先前成绩或对方结果。
分别提取回复实际声称的主体、关系、值、时间和事实状态，并标注缺证据声明、自然性和主动记忆旁白。
不是用字符串匹配自动给模型判正确；实际正确性由保存的独立语义标注与固定 rubric 比较。

两位评审各提交24条，24/24语义签名一致，无缺失或待裁决项。
“工作的城市是瑞士”按 country claim 记录，不能把 Switzerland 当成正确城市；
美国同样是 country claim，不因来源问的是城市就强行改成 city。
这是同一模型家族的两个独立上下文，不是人工或跨模型家族验证。
原始标注保留在 `reviewer-a.jsonl`、`reviewer-b.jsonl`，没有改写评审提高分数。

## 基线结果

| 固定分母 | 通过 |
| --- | ---: |
| 有依据回答：正确来源与换值 | 0/8 |
| 角色交换反例 | 0/4 |
| 空记忆不虚构 | 0/4 |
| 普通算术 | 8/8 |
| 换值两端均正确 | 0/4 |

总通过8/24全部来自算术，**不是记忆准确率33.3%**。
例如同一居住问题在换值、角色交换、空来源下仍输出相同城市/国家，符合未读记忆的基线行为，
但不能作为任何记忆能力证据。后续候选必须在相同面板上比它改善内容读取，同时保留普通回复。

评分程序 `python/review_joint_generation.py` 保留24例完整分母，按语言与关系分别形成4组换值对，
不会把两个关系混为一对；机械截断、无效UTF-8、重复和normal路线偏离基线均不能被语义通过掩盖。
评审分歧保持 `needs_review`，不会因已有机械失败而擅自宣布语义评审完成。
通过小面板也不直接放行部署。

复现：

```sh
python3 python/review_joint_generation.py score \
  --baseline training/memory/neural-system/reviews/CG-003/baseline.json \
  --panel-audit training/memory/neural-system/checks/CG-003-data.json \
  --corpus training/memory/neural-system/data/JB-001 \
  --reviews training/memory/neural-system/reviews/CG-003/reviewer-a.jsonl \
  --reviews training/memory/neural-system/reviews/CG-003/reviewer-b.jsonl \
  --reviewer cg003-blind-a --reviewer cg003-blind-b \
  --output build/neural-memory-cg003-baseline-score-new.json
```
