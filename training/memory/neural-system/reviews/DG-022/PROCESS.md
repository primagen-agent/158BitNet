# DG-022：完整不确定回复的逐token监督与轨迹级零更新验证

结论：**四条真值insufficient记录的完整不确定回复监督与轨迹级零更新检查全部通过；C数值、梯度结构和三路分离证据齐备。**
没有训练；零初始化分支在每个位置仍与base逐位相同，不能声称学会拒答。本轮完整自由回复为0条、提交token为0。

注册：`../../experiments/DG-022.json`。结果：`zero-update.json`，
SHA-256 `907d750d0df14ab568e1e9f37a338acd3d0c6995ecf8cad4eca27aa63823c511`。
原始C文件：`build/neural-memory-dg022-supervision/`（40MB，全部哈希登记于结果内）。

## 0. 环境恢复（先于本轮，已登记）

17:03本地`build/`因未提交的Q8K期C改动整体重建，CG-003冻结的五个二进制身份全部漂移；
`libbitnet.a`之外的四个可执行文件已从哈希核验过的CG-003平台源码快照
（`build/neural-memory-cg003-platform.tar.gz`，sha256 `a067290e…`）在独立目录**逐字节重建成功**
（tok_probe/prefix/reference/encoder四个探针均与冻结pin一致；`libbitnet.a`未复现，不作为身份）。
`memory_continuous_probe`此前无pin，本轮起登记其冻结哈希。本轮全部C调用使用该冻结目录
`build/neural-memory-cg003-frozen-build/`，未改动用户在用的工作区二进制，manifest一律未改。
DG-019/020/021的报告与源码哈希在运行前后核验不变。

## 1. 监督范围与标签纪律

沿用DG-019 oracle编译器，把四条route-2记录（jb-000 world，role_swap/empty × en/zh）
编译为全GENERATE教师轨迹：13/12/13/12个位置共50个，显式EOS结尾，回复字节完整，无截断。
每个位置的分支前向只接收冻结C frame、冻结输出投影和scale；目标token在该前向之后才进入CE。
等权平均损失33.967；逐token损失全部保留在报告中。

**分母披露（登记于结果JSON）**：面板刻意让role_swap与empty共用同一问句文本（普通对亦然），
两者只在已存episode上不同；query-only分支不读来源侧，因此4条记录只含**2组不同监督轨迹**
（en/zh各一组，各出现两次）。这是接口属性，不是样本量加倍；报告按4条记录等权、按2组去重披露。

## 2. 原生核验

31个唯一前缀全部经冻结`memory_prefix_probe`批量提取（每前缀全新context，披露单次prefill内部K/V工作缓冲，
无跨前缀复用）；16个位置（每条不确定轨迹的首/中/末 + 每条normal记录首位置）与冻结
`memory_gradient_reference`逐位一致。零残差分支在全部50个位置与base逐位相同；
两条非fixture记录在位置0另经冻结`memory_continuous_probe`零增量核验，逐位等于base。

控制器交接在DG-022身份下重新锚定：每条route-2记录用位置0真实C frame驱动未训练reader，
四次均产生真实NEEDS_UNCERTAINTY交接；未提交任何token，教师位置1..n-1为模块级前向
（控制器commit只确认真实发射，教师token不作服务输入）。

## 3. 梯度（零更新）

- 零初始化聚合反传（50位置等权）：Wout梯度有限非零，encoder权重/偏置梯度**精确为零**——
  这是Wout=0时链式法则的预期结果，不是冻结，也不代表Win开始学习。参数摘要前后逐位一致。
- 两个人工非零实例（Wout=0.001I，独立副本，en/zh轨迹各一）在各自三个参考位置上：
  两层全部连通；C/Python logits最大误差9.5e-7（atol1e-5/rtol1e-4，top-1一致）；
  轨迹级部分损失有限差分——按位置残差梯度取组合单位方向、ε=0.01——
  实测导数0.858366对autograd 0.858382（en）、0.797695对0.797723（zh），在登记容差内；
  disabled路径在该非零delta下仍逐位等于base。
- 目标为部分教师目标CE，是监督目标上的求导夹具，不是回答准确率。

## 4. 三路分离

- **normal旁路**：四条普通记录的reader首决策全部仍为route 2（错误路由**原样保留**为真实失败，
  未遮蔽、未改标签）；其全部参考帧上disabled与零残差enabled均与base逐位一致；不进入监督。
- **supported copy**：四条引用记录编译结构不变（各恰一次START/END）；按构造与显式断言排除在
  不确定监督之外；DG-019证据链核验未变。
- **uncertainty分支**：即上文50位置监督。监督ID集合恰为4条route-2记录，与另两路零交集。

## 5. 下一项

按计划顺序，接下来是**完整自由回复协议与独立双评审的预登记**（含不确定分支与正常旁路的
真实逐token自由生成、截断/UTF-8/预算记录、两位隔离评审与完整分母），以及更多语义世界与反例；
训练仍需明确批准的新预算，CG-003的100步预算不自动延长。本轮不接HTTP、不发布、不启动LoCoMo。

新增10项单测；直接链路五文件 **68/68**（teacher 8、controller 25、query-only 14、
supervision 10、fine_span 11）、现有CTest **22/22** 通过
（当前工作区ctest登记22项，含Q8K期新增的test_quant_q8k；前几轮"21/21"为当时登记数）。
没有提交代码，没有修改README的最终模型事实；未训练权重、参数零更新。

复现（输出路径须不存在）：

```sh
python3 -m unittest discover -s tests -p test_uncertainty_reply_supervision.py
python3 python/check_uncertainty_reply_supervision.py \
  --raw build/dg022-supervision --output build/dg022-supervision.json
```
