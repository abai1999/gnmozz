# VLA2 项目代码库架构诊断与重设计建议

## 审查范围与总体判断

我基于公开仓库的主分支，静态审查了 `README.md`、`scripts/evaluate_rlbench.py`、`scripts/evaluate_rlbench_modes.py`、`prismatic/robot/coarse2contact/supervisor.py`、`prismatic/robot/contact_refiner.py`、`prismatic/robot/depth_force_contact_controller.py`、`prismatic/robot/residual_safety.py`、`prismatic/robot/residual_transforms.py`、`prismatic/robot/stage_aware_refiner.py` 等文件。仓库 README 仍把 VLA2 主线定义为 **planner-only baseline**，并明确把 alignment/student/residual chains 排除在主线之外；但主评测脚本同时导入了 `Coarse2ContactSupervisor`、`ContactRefiner`、`StageAwareRefiner` 以及大量 alignment/student 相关模型，并在运行时同时构建 `refiner` 与 `coarse2contact` 后再用互斥条件阻止二者共用。这说明公开仓库目前仍处在**“旧评测大脚本 + 新 Coarse2Contact scaffold 并存”**的过渡状态，而不是一个已经收敛到单一路径的清爽主线。citeturn23view0turn42view0turn43view1turn51view0

我的结论很明确：**你的大方向是对的，但当前代码并没有把“冻结 VLA 负责 coarse approach，depth/force 负责高精度接触修正”真正落成一个有效的 two-stage system。**现在真正被执行的 Coarse2Contact 路径，本质上是一个以腕部深度 proxy 为基础的视觉微调器、一个无记忆的一步式力反射器，以及它们对 planner chunk 的直接相加；而现有文献里真正对高精度插入有效的方案，通常依赖显式的局部位姿伺服、视觉与力的**相位拥有权切换**、以及带恢复原语的 visual-force / impedance 框架，而不是“proxy residual + threshold reflex”的拼接。RLBench 本身也正是一个包含多阶段操作任务的 benchmark；在近年的 contact-rich VLA 与视觉-力插入工作中，纯视觉大策略在接触与遮挡阶段的不足，已经是被反复验证的问题。citeturn55search0turn25view3turn28view0turn25view6turn29view0turn56view0turn56view1turn56view2

与附件报告里的高层判断一致，我这次看完公开代码后的更强结论是：**当前失败不是因为“外挂高精度修正器”这个研究方向不成立，而是因为你现在的 depth/force 分支在估计错误的量、在错误的时机介入、并且通过错误的融合方式污染了 planner。**如果按现状继续做补丁式修修补补，论文很难成立；但如果把它重写成一个真正的 coarse-to-contact 层级控制系统，这条线是完全有论文价值的。citeturn25view3turn28view0turn29view1turn56view0turn56view1turn56view2

## 根因表

| 严重性 | 根因 | 精确代码位置 | 为什么它会导致失败 |
|---|---|---|---|
| 致命 | depth 分支估计的是“最近前景 proxy”，不是配合几何的相对位姿 | `supervisor.py` 中 `DepthVisualAligner` 明确自称 *“Conservative wrist-depth proxy for pre-contact correction”*；`estimate()` 先取 valid depth 的 5% 分位 `prox`，再以 15% 分位和 `prox+0.03` 构造近端 mask，随后用 mask 的质心 `(u,v)` 与 PCA 主轴角直接生成 `correction[0], correction[1], correction[2], correction[5]`；核心链路在约 L1437–1617、L1531–1588。citeturn25view3turn28view0 | 这会把“离相机最近、最显著、最容易被遮挡的 blob”误当成“插入几何中心与相对 yaw”；它甚至直接改写 z 修正，因此不是局部对准器，而是 proxy 驱动的 residual 注入器。citeturn28view0 |
| 致命 | depth 对 planner 的污染发生在动作层，而不是输入层隔离之后的 owner handoff | `supervisor.py` 的 `step()` 中，`use_visual` 对 `depth_shadow/depth_apply/depth_force` 都成立；visual delta 与 force delta 被直接相加成 `correction = visual_delta + force_delta`，再加到 `local_out` 上。真正决定“只记录不执行”的不是 `mode`，而是另一个独立的 `shadow_only` 标志；相关逻辑在约 L2134–2170。citeturn25view1turn29view0 | 这意味着 depth 分支不是在“接管局部子空间”，而是在与 planner 共享同一动作子空间并直接相加；同时 `depth_shadow`/`depth_apply` 的语义并不由 `mode` 本身保证，门控一旦配错，就可能出现“名义 shadow，实际 apply”或反过来的情况。citeturn25view1turn29view0 |
| 致命 | force_reflex 没有形成有状态的退让-重试闭环，只是一步式反射 | `RecoveryPrimitiveBank.action()` 在 `JAM` 只输出一次 `backoff_m + lateral_m`，在 `EDGE` 只输出一次侧向/偏航 relief；`ForceReflexController.correction()` 只是在 `force_stop` 时返回 `-local_base`，或调用一次 primitive，或在 `TOUCH` 时给一个单步 `touch_slowdown`；相关逻辑在约 L1824–1901。citeturn27view4turn25view6turn25view7 | 这不是“恢复控制器”，而只是“当下这一步怎么躲一下”。没有 `RETRACT → UNLOAD → SEARCH → RE-APPROACH` 的持久状态；所以系统只会出现“撞一下、退一点、继续撞”的循环。citeturn25view6turn25view7 |
| 极高 | `EDGE` 状态被 `PARTIAL` 逻辑提前短路，导致真正需要 lateral relief 的时刻很可能根本进不了 `EDGE` | `ContactStateEstimator.update()` 先判 `SEATED`，再判 `PARTIAL`：只要 `contact` 且 `depth_gap > seated_depth_threshold * 2.0` 就进入 `PARTIAL`；`EDGE` 的 `visual_xy_error > 0.004` 检查排在后面。默认 `seated_depth_threshold=0.012`，因此 `depth_gap > 0.024` 的接触都会先被打成 `PARTIAL`。相关逻辑在约 L1663–1799。citeturn28view0 | 早期偏心接触本来最需要 lateral relief，但由于先被 `PARTIAL` 吃掉，`EDGE` 的专用解卡逻辑就可能长期不触发。citeturn28view0 |
| 极高 | 一旦进入接触相位，视觉局部修正被关掉；真正需要视觉-力协同时，系统实际上只剩力 heuristics | 在 `step()` 中，visual delta 仅在 `phase in (VISUAL_ALIGN, PROBE_CONTACT)` 时生效；force delta 则在 `PROBE_CONTACT / CONTACT_INSERT / RECOVER` 时生效。与此同时，`_update_phase()` 会把 `TOUCH/EDGE/PARTIAL` 统一映射到 `CONTACT_INSERT`。相关逻辑在约 L2137–2143 与 L2258–2279。citeturn25view2turn29view1 | 这意味着**一碰上就关视觉**，而接触后的力控制又不是稳定的相容控制，只是一步式 reflex。所谓“depth_force 协同”因此在最关键的 handoff 区间并没有发生。citeturn25view2turn29view1 |
| 极高 | public 仓库里的“learned depth-force controller”并不在正向执行链路上；真正执行的 `depth_force` 只是两个 heuristic 的相加 | `depth_force_contact_controller.py` 的模块注释明确写着：它是 *“Shadow-only wrapper … It does not apply actions by default; it only scores local candidate actions and returns diagnostics needed for shadow evaluation.”*；`shadow_step()` 只返回 `DepthForceContactDecision`，不执行动作。相反，`supervisor.py` 中的 `depth_force` 仅仅是 `use_visual` 与 `use_force` 同时成立，然后做 `visual_delta + force_delta`。citeturn34view4turn35view0turn35view2turn29view0 | 这直接解释了为什么 `depth_force` 没起协同作用：你以为在跑 learned multimodal local policy，实际上跑的是“proxy depth servo + heuristic force reflex”的求和版。citeturn34view4turn29view0 |
| 高 | contact/force 统计口径被拆成多套：planner 看 `force_hist`，local branch 多看 `raw_force`；`DepthForceContactController` 还要求 `depth_prox` 有效才开门 | 主评测循环里 `process_obs()` 同时返回 `depth_tensor`、`force_hist`、`depth_tensor_96`、`raw_force`；planner 用 `depth_tensor` 和 `force_hist`，refiner/coarse2contact 则用 `depth_proximity` 与 `raw_force`。`ContactRefiner.compute_depth_proximity()` 和 `DepthForceContactController.compute_depth_proximity()` 都是 valid depth 的 5% 分位；`DepthForceContactController.shadow_step()` 的 `gate_open` 还要求 `np.isfinite(depth_prox)`。citeturn45view0turn30view1turn35view0turn35view2 | 这会造成典型的“planner 在 history 分布上工作，局部分支在 raw/proxy 分布上工作”的失配；而一旦接触或遮挡导致 depth 不稳，learned depth-force controller 的 gate 反而关闭。citeturn35view0turn35view2 |
| 高 | `InvalidActionError` 路径没有并入 supervisor 的主恢复回路，仓库里只有一个孤立的一步式 helper | `ResidualSafety.compute_invalid_action_recovery()` 只会在有 force reflex 时输出一次 backoff，否则就是“保持一帧零动作并等待 planner replan”；但 `Coarse2ContactSupervisor.step()` 的正向路径只调用 `check_force_stop()`、primitive bank 和 `clip_final_action()`，没有把 invalid action recovery 纳入相位机。citeturn32view0turn25view6turn29view0 | 这意味着 IK/执行异常并不会把系统显式推进到受控恢复态，而只是外层异常处理问题；闭环因此断掉。citeturn32view0turn29view0 |
| 中高 | 你无法从现有 trace 判断“修正是否真的执行了”，因为 trace 存的是 local 提案，不是 post-clip/post-exec 世界动作 | `supervisor._last_trace` 记录了 `visual_correction_local_6d`、`force_correction_local_6d`、`local_correction_local_6d`、`planner_chunk_local_6d`、`final_action_local_6d`；但 `clip_final_action()` 是在 world-frame `out[:6]` 上调用的，trace 里没有 post-clip world action，也没有 executed action 与 pre/post clip attenuation。citeturn29view0turn32view0 | 这会让你误以为“depth/force 明明有输出”，却看不到它们是否被 clip、被 IK 否决、或在环境层被拒绝。citeturn29view0turn32view0 |
| 中 | 主评测脚本仍然是旧架构大熔炉，不适合作为 Coarse2Contact 论文主线的实验入口 | `evaluate_rlbench.py` 顶部仍导入大量 alignment/student/teacher 相关模块，并同时支持 `ContactRefiner`、`StageAwareRefiner` 和 `Coarse2ContactSupervisor`；`evaluate_rlbench_modes.py` 也仍暴露出巨量 legacy 参数面。虽然脚本里有“Coarse2Contact runtime is an independent first-stage scaffold”的 guard，但这不是干净架构，只是互斥保护。citeturn42view0turn43view1turn51view0 | 这会让实验解释变得困难，任何结果都很难让审稿人相信“你真的只在评 Coarse2Contact，而不是某个 legacy runtime side effect”。citeturn23view0turn42view0turn43view1 |

上表里最关键的三点，其实已经足够解释你提出的三个主要症状：**depth 分支污染了 planner**，因为它不是孔位/配合位姿估计，而是 proxy residual；**force_reflex 不稳定**，因为它只有一步式躲避没有闭环恢复；**depth_force 没协同**，因为真正执行的不是 learned multimodal controller，而是两个 heuristic 的加和。citeturn28view0turn25view6turn29view0turn34view4turn35view2

## 信号链剖析

主评测循环已经把最关键的信号链暴露得很清楚了：`process_obs()` 会产出 `depth_tensor`、`force_hist`、`depth_tensor_96` 和 `raw_force`；planner 预测动作时消费 `depth_tensor` 与 `force_hist`，而 local branch 则更多消费由 `depth_tensor_96` 进一步压缩得到的 `depth_proximity`，以及未经时序建模的 `raw_force`。同一帧观测因此被拆成了两套统计口径。更糟的是，`ContactRefiner` 和 `DepthForceContactController` 都重复使用了“valid depth 的 5% 分位”作为 `compute_depth_proximity()`，而 `Coarse2ContactSupervisor` 又完全绕过 `ContactTrigger`，自己定义了另一套 `ContactStateEstimator`。这不是一个统一的 contact stack，而是多套各自阈值化、各自尺度化的小栈并存。citeturn45view0turn30view1turn30view4turn35view0turn24view0turn28view0

下表是我认为当前系统里最重要的“污染信号”。这些量可以做**诊断**，但不应该直接做**控制主变量**。

| 信号 | 生成位置 | 为什么它会污染控制 |
|---|---|---|
| `z_gap / depth_proximity` | `DepthVisualAligner.depth_proximity()` 与 `estimate()` 都把 valid depth 的低分位当作接近度；同时 `ContactRefiner` 和 `DepthForceContactController` 也沿用同样的 5% 分位标量。citeturn26view4turn30view1turn35view0 | 它表达的是“最近可见表面多近”，不是“配合几何的相对位姿误差”；拿它去驱动接触态判断、`SEATED/PARTIAL` 标签乃至 z 推进，都会把遮挡与最近前景误当成对准进度。citeturn28view0 |
| `centroid_u/v` 与 `xy_error` | `estimate()` 对近端 mask 做质心，随后按 `pixel_to_meter` 线性映射到 xy correction。citeturn28view0 | 这只是在把“近端前景 blob 是否偏在图像中心”当成插入横向误差；在 peg、自遮挡、夹爪边缘占主导时，方向会系统性错。citeturn28view0 |
| `yaw_error` | `estimate()` 用近端 mask 的 PCA 主轴角，再折叠到 90° 对称区间。citeturn28view0 | 它对方孔/方 peg 这类接触几何并不等价于真实相对 yaw；如果主轴来自遮挡边而不是配合边界，偏航修正会被错误放大。citeturn28view0 |
| `visual_confidence` | `confidence = 0.15 + 0.70 * conf_area * conf_depth`，只依赖 mask 面积和 prox 是否进入阈值。citeturn26view1 | 这是可观测性 proxy，不是几何可信度；它完全不知道你看到的是目标孔、peg 侧壁还是夹爪遮挡。citeturn26view1turn28view0 |
| `force_norm / delta_force / fz / torque_xy` | `ContactStateEstimator.update()` 直接从 raw force slice 计算，没有显式 bias 校正、低通或历史特征；`ResidualSafety.compute_reflex_override()` 也直接用 raw `force_reading[:6]`。citeturn28view0turn32view0 | 这些量适合作安全阈值，不适合作连续接触控制主变量；它们既无法稳定位姿解歧，也极易受瞬时碰撞尖峰影响。citeturn28view0turn32view0turn56view2 |

让 depth/force 长时间不触发的门控条件，也已经能从代码里精确定位出来：

| 门控条件 | 位置 | 造成的后果 |
|---|---|---|
| visual 只有在 `phase in (VISUAL_ALIGN, PROBE_CONTACT)` 且 `estimate.valid` 且 `confidence >= 0.20` 才生效 | `supervisor.step()` 约 L2137–2139。citeturn25view2 | 一旦进入 `CONTACT_INSERT`，视觉修正立即停用；而接触初期恰恰是最需要“视觉确认 + 力约束”的 handoff 区间。citeturn25view2turn29view1 |
| `TOUCH/EDGE/PARTIAL` 都会把 phase 推到 `CONTACT_INSERT` | `_update_phase()` 约 L2258–2279。citeturn29view1 | 这相当于“只要接触就把视觉 owner 撤走”，导致 depth 和 force 不是协作，而是强制接力。citeturn29view1 |
| learned `DepthForceContactController` 的 `gate_open` 需要 `not force_stop`、`np.isfinite(depth_prox)`、`switch_prob >= threshold` | `shadow_step()` 约 L800–810。citeturn35view0turn35view2 | 接触与遮挡阶段 depth 最容易失效，此时 learned multimodal branch 反而关门；即使 force history 有用，它也进不了执行态。citeturn35view2 |
| `PARTIAL` 判定排在 `EDGE` 之前 | `ContactStateEstimator.update()` 约 L1778–1793。citeturn28view0 | 早期偏心接触很可能被 `PARTIAL` 提前吃掉，从而根本触发不到 `EDGE` 的 lateral relief。citeturn28view0 |

还有一个对调试极其致命的问题：你现在**看不到真正执行了什么**。`_last_trace` 记录的是 local proposal 和 `final_action_local_6d`，但 `clip_final_action()` 是在 world frame 上做的，而且 trace 里没有 `executed_action_world_7d`、没有 pre/post clip 衰减比例、也没有 InvalidAction 后的恢复动作。对于这种接触丰富任务，这会让实验陷入“分支明明一直在出动作，但为什么 success rate 不动”的假象。citeturn29view0turn32view0

## 为什么当前 depth 和 force 没有形成正向作用

depth 分支之所以不但没提升，反而污染 planner，核心不是它“太弱”，而是它在当前代码里担任了**错误职责**。`DepthVisualAligner` 不是一个“配合几何局部定位器”，而是一个“最近深度前景 proxy 转 residual”的模块；它把近端 mask 的质心和主轴方向转成 xy/yaw 修正，又把 `prox` 直接转成 z 修正。也就是说，它既控制横向、也控制接近深度，而且控制源还是最容易被 peg、夹爪和遮挡污染的区域。这种模块一旦和 planner action 直接相加，就不是“细化 coarse approach”，而是在用高噪声的局部 proxy 去持续拖拽一个原本还能工作的 planner。citeturn25view3turn28view0turn29view0

force_reflex 之所以没有形成稳定闭环，是因为它不是控制器，而是**反射器**。在当前实现里，`force_stop` 直接输出 `-local_base`，`JAM` 只给一次 `backoff_lateral_nudge`，`EDGE` 只做一次侧向 relief，`TOUCH` 只做一次 `touch_slowdown`。这里没有 dwell time，没有 unload 判据，没有 retry primitive，也没有“恢复 succeeded 之后再重新接近”的显式状态返回。你确实写了 `compute_invalid_action_recovery()`，但它本身也只是一帧 backoff 或一帧 hold，而且我没有在 `Coarse2ContactSupervisor.step()` 的正向路径里看到它被纳入相位机。结果就是：异常、碰撞、卡滞都没有被真正吸收到控制状态里。citeturn25view6turn25view7turn32view0turn29view1

`depth_force` 没协同，原因更直接：**运行时并没有真的用上仓库里的 learned depth-force contact policy。**`depth_force_contact_controller.py` 自己已经写明这是 shadow-only wrapper，不默认执行动作；而 `supervisor.py` 里的 `depth_force` 只是“visual delta 和 force delta 一起加”。因此你当前测到的 `depth_force` 结果，并不能证明“多模态 local policy 没用”，它只能证明“proxy depth servo 和 heuristic force reflex 的求和没用”。这两件事完全不是一回事。citeturn34view4turn35view0turn35view2turn29view0

从研究角度看，这一点非常重要。近年的高精度插入与 VLA contact-rich 研究之所以有效，不是因为“把更多模态塞进网络”这件事本身，而是因为它们把**谁在什么时候拥有哪几个自由度的控制权**设计清楚了。Insert-One 用的是 6-DoF 视觉跟踪迭代控制与阻抗控制的混合框架；ForceVLA 强调的是 force-aware、phase-aware 的 action generation；TaF-VLA 则强调接触信号具有时间历史依赖，静态单帧并不够。你的当前代码几乎在这些点上都反着来：它没有真正的局部位姿目标，没有 phase ownership，也没有稳定的时序接触状态。citeturn56view0turn56view1turn56view2

## 一个真正可用的重设计框架

我建议你不要继续在现有 `evaluate_rlbench.py` 这类大脚本上堆条件分支，而是**单独切一条 Coarse2Contact 主线**。README 已经把 VLA2 主线定义成 planner-only baseline，并明确排除了 alignment/student/residual chain；所以最合理的做法，不是再让 `Coarse2ContactSupervisor` 依赖旧 refiner，而是新建一个只服务 C2C 的评测入口，例如 `scripts/evaluate_c2c_rlbench.py`，只保留 planner、raw observation packaging、Coarse2Contact orchestrator、日志与 ablation。旧的 `StageAwareRefiner` 和 `ContactRefiner` 可以保留作历史 baseline，但不要继续作为主线正向依赖。citeturn23view0turn42view0turn43view1turn36view0

新的系统建议采用下面这个职责划分：

```text
obs_raw
 ├─ planner_obs --------------------------> Frozen VLA --------------------> a_plan
 ├─ wrist_depth_raw + valid_mask --------> Depth localizer ---------------> delta_align, depth_conf
 └─ raw_wrench + wrench_filt + hist -----> Phase / jam estimator ---------> phase

phase owner:
  COARSE        -> planner owns xyz+yaw+gripper
  PRECONTACT    -> depth owns {x, y, yaw}; z only allowed as guarded approach
  FIRST_TOUCH   -> force governor owns z compliance; depth may still veto if visible
  SLIDE/INSERT  -> force-guided tangential relief + guarded insertion
  JAM/INVALID   -> recovery primitive owns action
  VIEW_RECOVERY -> perception recovery primitive owns action
```

这个框架的关键不在“多模态”，而在**owner-by-phase**。也就是说，不允许再出现 `visual_delta + force_delta + planner_delta` 这种同自由度直接求和。进入 `PRECONTACT` 之后，planner 不再持续拥有 `{x, y, yaw}`；它只保留 coarse 语义和 gripper 意图，横向与偏航改由 depth localizer 接管。进入 `FIRST_TOUCH/SLIDE/INSERT` 后，depth 不再用 proxy 去硬拽动作，而只作为**可视时的 veto/consistency check**；真正的 z 推进与切向 relief 改由经过滤波和滞回的 force controller 管理。这样 depth 不会污染 planner，force 也不会和 depth 抢控制权。citeturn29view0turn25view2turn56view0turn56view1

depth 局部修正器的第一版，我强烈建议你做成**解析几何 localizer**，而不是 learned residual policy。原因不是“学习没用”，而是你现在的第一优先级是证明**架构正确**，不是给错误架构再加一层网络。对于 `insert_onto_square_peg` 这类强几何结构任务，一个更可信的 depth 模块应该输出的是 `delta_align = [dx, dy, dyaw, standoff_z]` 与 `depth_conf`，其意义是“当前配合几何在末端插入坐标系里的相对偏差”，而不是“最近前景在图像里偏了多少”。具体做法可以是：基于 planner 给出的接近方向，在 wrist depth 里裁一个 TCP-centered ROI；把 raw metric depth 回投到 wrist frame 点云；对局部面和空洞/边界做拟合，估计配合几何中心与主方向；若观测质量不足，则不输出控制 residual，而是进入 `VIEW_RECOVERY`，执行小退让、小抬升、小侧移，使配合区域重新进入可观测范围。只有在 `depth_conf` 过门之后，depth servo 才拥有 `{x, y, yaw}`。这与高精度视觉-力插入方法里“视觉跟踪式迭代控制 + 接触后阻抗/力引导”的思路是一致的。citeturn56view0turn28view0

如果你后续确实想把 depth 模块学起来，最合理的下一步也不是学习 full action residual，而是学习一个小型 `DepthAlignNet: depth ROI -> [dx, dy, dyaw, conf]`，它仍然服从同样的控制接口。这样你的论文会更干净：第一阶段先证明 **controller structure** 有效；第二阶段再证明 learned localizer 相比 analytic localizer 能否进一步提升。这样既避开了“residual learner 是不是只是在补错架构的洞”的质疑，也符合你“不把 residual learner 放第一优先级”的约束。citeturn56view0turn56view1

force 模块则应该从“阈值反射器”重写为**带恢复原语的接触状态机**。至少要有 `FREE / PRECONTACT / FIRST_TOUCH / SLIDE / INSERT / JAM / VIEW_RECOVERY / DONE / FAIL` 这些状态，并维护 bias-corrected、低通后的 TCP wrench、轴向推进进度、横向负载峰值和持续时间。控制逻辑也不应该再是“哪一维超阈就往反方向怼一下”，而应该是：`FIRST_TOUCH` 阶段冻结或限速 z，直到横向负载回到低区；`SLIDE` 阶段只允许小尺度切向 relief；`INSERT` 阶段只有在横向力、扭矩和推进一致性都满足时，才恢复小步 z 推进；`JAM` 必须触发一个多步 primitive：`RETRACT → UNLOAD → MICRO-SEARCH → RE-APPROACH`。`InvalidActionError` 不应再被当成“评测计数事件”，而应直接把系统推进到 `JAM` 或 `VIEW_RECOVERY`。这才是能在论文里自洽地称为“force reflex / recovery”的设计。citeturn25view6turn25view7turn32view0turn56view1turn56view2

最后，日志系统也必须重写。当前 trace 只记录 local proposal，不记录 post-clip/post-exec 结果。论文主实验至少要记录：`owner`、`phase`、`phase_reason`、`depth_conf`、`depth_obs_quality`、`raw_wrench`、`filtered_wrench`、`planner_action_local/world`、`depth_action_local`、`force_action_local`、`recovery_action_local`、`pre_clip_action_world`、`post_clip_action_world`、`executed_action_world`、`retry_id`、`invalid_action_flag`。如果这些字段没有，审稿人很容易质疑：你到底是在比较 controller，还是在比较一堆互相吞动作的 logs。citeturn29view0turn32view0

## 立即执行的最小实验

下面这组实验，我建议你按顺序做。它们不是“补丁验证”，而是直接对应上面识别出的四类核心故障：proxy depth、门控错位、恢复闭环缺失、以及执行不可见。citeturn28view0turn29view0turn35view2turn32view0

| 实验 | 实施方式 | 成功判据 |
|---|---|---|
| 离线 depth 诊断实验 | 固定 planner 轨迹，不与环境交互；对每一步 wrist depth 同时计算当前 `DepthVisualAligner` 的 `(u,v,yaw,prox)` 与新 depth localizer 的 `[dx,dy,dyaw,conf]`；比较二者对“最终是否进入插入 basin”的单调性、符号一致率和校准误差 | 如果当前 proxy 指标与成功 basin 的相关性弱，而新 localizer 的相关性显著更强，就证明根因在“估计错误的量” |
| 相位占用与开门率审计 | 在现有 `depth_apply / force_reflex / depth_force` 上记录每个 episode 的 `phase` 占比、`visual gate open` 占比、`force gate open` 占比、`EDGE` 命中率、`JAM` 命中率 | 你大概率会看到 `EDGE` 极少命中、visual 在 contact 后快速归零、learned depth-force gate 在 occlusion 下长期关闭 |
| 变换与裁剪单元测试 | 针对 `world_delta_to_local` / `local_delta_to_world` 做 round-trip synthetic test；同时记录每步 `pre_clip_action_world` 与 `post_clip_action_world` 的差异 | round-trip 应接近零误差；任何系统性符号翻转或 post-clip attenuation 异常都必须先修 |
| 新 depth owner-only 闭环实验 | 去掉 force，保留 frozen planner；进入 `PRECONTACT` 后仅由 depth owner 接管 `{x,y,yaw}`，并把 z 变成 guarded approach；对比 planner-only 与当前 `depth_apply` | 即使没有 force，新版本也应该明显优于当前 `depth_apply`；否则先别做 force 融合 |
| jam 恢复原语实验 | 人为注入初始横向偏差或 yaw 偏差，使系统高概率 first-touch 偏心；比较“当前单步 reflex”与“RETRACT→UNLOAD→SEARCH→RE-APPROACH”多步 primitive | 新恢复器应显著降低连续 jam 次数，提高 episodes 内重试后的最终成功率 |
| 主论文 ablation | 固定 planner checkpoint，比较 `planner_only`、`planner + depth owner`、`planner + force recovery`、`full C2C` 四组；指标不仅看 success rate，还看 first-touch lateral load、平均重试次数、进入 `JAM` 的频率、执行动作衰减比例 | 只有当 `depth owner` 单独有效、`force recovery` 单独有效、`full C2C` 再进一步增益时，论文故事才真正站得住 |

如果你的目标最终是上实机，那么路线仍然应该是**先在 RLBench 把 owner/gate/recovery/logging 全部跑通，再迁移到 RealMan**。因为你现在最缺的不是数据，而是“控制职责是否真的正确”这个系统性证据；而这种证据，先在可重放、可日志化的仿真里做出来，成本最低、说服力也最高。插入类文献里，真正能迁移到真实装配的方案，往往正是因为它们在局部视觉-力控制结构上足够显式，而不是把所有能力都埋在端到端黑箱里。citeturn55search0turn56view0turn56view1

## 结论

当前 depth/force 之所以没有形成正向作用，原因不是“VLA 做 coarse approach、外挂高精度修正器”这条研究路线不可行，而是**这套代码还没有真正实现一个高精度修正器**。当前 depth 分支是 proxy-based visual nudging，不是配合几何 localizer；当前 force_reflex 是单步反应，不是 recovery controller；当前 `depth_force` 也不是 learned multimodal local policy 的执行版，而只是 visual heuristic 与 force heuristic 的加和。再加上 visual 在 contact 后被关掉、`EDGE` 被 `PARTIAL` 短路、InvalidAction 恢复未并入主相位机，最终结果自然不会正向。citeturn28view0turn25view6turn29view1turn34view4turn35view2

最值得先改的三个地方，我会这样排：第一，**把 Coarse2Contact 从 `evaluate_rlbench.py` 这种 legacy monolith 里独立出来**，做一条干净的 `evaluate_c2c_rlbench.py` 主线；第二，**彻底替换 `DepthVisualAligner`**，把它从“最近深度 proxy”改成“输出 `[dx,dy,dyaw,conf]` 的局部几何定位器”；第三，**把 `ForceReflexController + RecoveryPrimitiveBank` 改成显式的恢复状态机**，并把 `InvalidActionError` 正式纳入 `JAM/RECOVER` 正向路径。只要这三件事不做，后面的阈值微调、平滑、甚至再训练一个 residual learner，都只是在放大架构噪声。citeturn23view0turn25view3turn28view0turn25view6turn32view0turn42view0

如果只允许我先改一版，我会做一个非常克制、但论文上最有说服力的 **Coarse2Contact v1**：保留 frozen VLA planner 完全不动；删掉所有同自由度 residual 求和；实现一个 analytic wrist-depth localizer 只接管 `PRECONTACT` 的 `{x,y,yaw}`；实现一个 filtered-wrench jam FSM 只负责 `FIRST_TOUCH/SLIDE/INSERT/JAM`；新增 `VIEW_RECOVERY`；重写 trace，把 pre-clip、post-clip、executed action 全部记录下来。**第一版不训练新的动作网络。**只有当这一版已经在 RLBench `insert_onto_square_peg` 上稳定超过 planner-only、并且每个 ablation 都可解释时，才值得把 learned depth localizer 或 learned multimodal contact policy 接回系统里。按现在的公开代码状态，这才是最短、也最稳的论文路径。citeturn55search0turn56view0turn56view1turn56view2turn29view0turn34view4