# Coarse2Contact 代码库架构诊断与重设计建议

## 审查边界与总判断

先说明审查边界：我在当前环境里无法直接拉取你给出的 GitHub 仓库，因此做不到“逐行行号级”的源码核验。下面的“代码位置”会精确到**文件—责任段—搜索关键词/调用点**，并结合你给出的完整信号链、模块命名和症状，给出一次**高置信度的架构诊断**与**可直接落地的一版重设计**。这不是“调阈值式补丁”，而是围绕输入、状态、门控、控制、恢复、记录、验证的系统级重构。  

总体结论非常明确：**“冻结的 VLA 只做 coarse approach，depth/force 在 near-contact 接管局部修正”这个思路是可行的，而且与近两年的主流研究方向一致。**近年的多篇 VLA 与接触丰富操作论文都明确指出，纯视觉 VLA 在接触、遮挡、力学不确定性明显的任务上存在弱点，加入力/触觉或局部几何修正后，插入、插拔、接触式装配这类任务的成功率会显著提升。与此同时，RLBench 上的高精度操作也越来越依赖 coarse-to-fine、3D 几何或局部配准，而不是让一个单体策略直接“从像素一步到位”输出最终毫米级动作。citeturn13academia1turn13academia0turn18academia3turn18academia0turn16academia1turn16academia2

但这里有一个决定成败的前提：**外挂模块必须是“局部相对位姿/接触状态控制器”，而不是“把几个 noisy depth/force 指标塞进主动作通道里做 residual nudging”。**文献里真正有效的方法，几乎都在做以下几件事之一：用腕部视觉做局部 visual servo / relative pose alignment；用力/触觉做接触位姿推断与接触相切方向修正；再配一个显式的阶段机或双策略结构来区分非接触、接触、卡滞、恢复。相反，如果 depth/force 分支只是依据阈值对 planner 动作做一点加减，往往既不能提供真正的局部几何约束，也不能形成稳定的退让-重试闭环。citeturn25academia0turn25academia1turn17academia0turn17academia3turn26academia4turn20academia0

所以，我对你这个项目的判断不是“方向错了”，而是：**当前 depth/force 没起正向作用，核心原因大概率不是模块不够强，而是它们被放在了错误的架构位置，估计了错误的量，并在错误的时机介入。**

## 失效机理判断

### depth 分支为什么不但没提升，反而污染了 planner

从你给的症状看，当前 depth 分支大概率犯了三个连锁错误。

第一，它很可能**没有在估计“孔相对 peg/末端的局部位姿误差”**，而是在估计一个更易得但控制意义很差的 proxy，例如 `wrist_depth` 的最小值、前景质心、最近点、某种阈值化的前景面积中心，或者“图像里离相机最近的东西”。在插入任务里，这类量往往更容易锁到 peg 自身、夹爪边缘、遮挡边、桌面边缘，或者孔口前的局部高反差区域，而不是锁到**孔位中心 + 孔轴方向**。一旦把这种 proxy 直接映射成 `x/y/yaw` 修正，它就不是“局部几何校正”，而是在把一个**诊断信号**误当成**控制信号**。已有视觉伺服、局部配准与触觉插入工作都强调，真正稳定的近接触修正必须围绕**目标相对位姿**展开，而不是围绕“离什么东西更近/更显著”展开。citeturn25academia1turn25academia0turn25academia3turn17academia0

第二，depth 分支很可能**介入得过早、过宽，和 planner 在同一自由度上抢控制权**。一旦 `depth_apply` 在 coarse approach 期间就持续输出横向微调，那么 planner 还在朝“语义目标区”收敛时，depth 分支就会拿一个局部、噪声大、视野受遮挡的信号去拉扯同一组 `x/y/yaw` 维度。这样一来，planner 不再沿它熟悉的数据分布工作，depth 也没有足够稳定的局部观测，两者就会互相破坏。RLBench 上的高精度工作之所以常采用 coarse-to-fine，是因为**粗定位与精对准的可观测性、控制带宽和目标函数本来就不同**。citeturn16academia1turn16academia2turn25academia0turn25academia1

第三，depth 分支很可能在实现上**污染了 frozen planner 的输入分布**。这通常发生在 `process_obs` 里对共享 observation dict 原地修改，或者把 depth 衍生量、阶段标志、门控结果写回 planner 会再次读取的字段，导致同一个 frozen VLA 在 baseline 模式与 depth 模式下看到的是两种不同分布的输入。对于冻结的大模型 policy，这种“看似无害的观测改写”常常比动作 residual 更伤。近年的 VLA 适配工作普遍选择把额外模态做成旁路条件或独立控制器，而不是直接篡改原始视觉-动作主干输入，原因就在这里。citeturn13academia1turn13academia0turn18academia3turn18academia0

### force_reflex 为什么没有形成稳定的退让-重试闭环

从你描述的链路看，`force_reflex` 大概率只是一个**事件反射**，不是一个**有状态的恢复控制器**。

也就是说，它可能做的是：当 `|F| > τ` 时临时减一点速度、给一个瞬时 backoff、或者把某个维度置零；但它**没有把系统显式推进到“恢复态”**，也没有在接下来的若干步里持续执行“退让 → 解除侧向载荷 → 重新对齐 → 重新接近”的完整 primitive。这样一来，所谓 reflex 就只是一帧或几帧的局部反应，之后系统又回到原来的 approach 动作，于是碰撞—退一点—继续撞—再退一点，永远形不成闭环。接触丰富插入文献里，真正稳定的方法几乎都区分了接触、对准、插入、卡滞与恢复阶段；FoAR 这样的工作也强调力信号的价值在于**阶段化地调节控制**，不是孤立地做阈值报警。citeturn20academia0turn17academia0turn17academia3turn26academia4

另一个高概率问题是：`raw_force` 与 `force_hist` 的使用不一致。很多实现里，门控时看的是瞬时 `raw_force`，控制时用的是平滑后的 `force_hist`，或者反过来。结果就是**触发条件与执行依据不在同一个统计分布上**：要么触发太晚，要么一触发就抖振，要么触发后很快又被“恢复为正常态”。尤其在仿真里，力值尺度、接触峰值、偏置漂移与真实机器人可能都不同，如果阈值不是按分布校准，而是手写几个常数，稳定闭环很难出现。PolyFit 和力-扭矩动力学工作都强调，低维 F/T 信号本身就有歧义，只有把它放到**历史、接触多样性与阶段上下文**里才可能稳定用于修正。citeturn17academia3turn26academia4turn26academia1

最后，`InvalidActionError` 很可能只是被记录或直接导致 episode 失败，并**没有真正接到 recovery primitive**。如果异常路径只是“打印日志/记一次失败”，那么系统就在最需要恢复的时候彻底失去控制主导权了。

### depth_force 组合为什么也没有协同作用

这类组合失败，通常不是因为两种模态都没信息，而是因为**它们被以错误的方式组合**。

最常见的问题是**把 depth 与 force 做“同一步的线性相加”**。但实际上，这两种模态在插入任务中的最佳工作区间并不一样：depth 在**接触前、孔口尚可见时**最有价值；force 在**初始接触后、视觉被遮挡或几何退化时**最有价值。FoAR 明确利用阶段差异来动态调整力信息使用方式；视觉伺服与触觉/力插入工作也几乎都在做先对准、后接触、再插入，而不是同时把两个误差信号往同一个动作向量里生加。citeturn20academia0turn25academia0turn25academia1turn17academia0turn25academia3

另一种常见问题是**门控交集过窄**：如果 `depth_force` 只有在 “depth_valid 且 force_valid 且 phase==contact” 时才启用，那么这个条件窗口往往非常短，甚至几乎不存在。因为一旦真正接触，腕部深度图常常恶化；而在刚好还能看清孔口的时候，力信号又可能还没达到接触态。结果就是“组合模式”在逻辑上听起来更强，实际上几乎从不工作。

还有一个很危险的实现错误是**坐标系不一致**。如果 depth residual 在相机/末端局部系中定义，而 force reflex 在 TCP 或 world 系中定义，最后又在 `supervisor` 里直接相加，你会得到“每个分支单看方向都对，合起来就失真放大”的现象。尤其在插入任务的最后几毫米，frame 方向错一点都足以把修正放大成破坏。

## 根因表与代码定位

下面这张表按严重性排序。由于仓库本体当前不可拉取，我给的是**文件 + 责任段 + 搜索关键词/调用点**，这是我认为最应该先做定点核查的地方。

| 严重性 | 根因 | 精确代码位置 | 为什么它是根因 |
|---|---|---|---|
| 致命 | **局部修正与 planner 没有做相位隔离，双方在同一动作子空间抢控制权** | `prismatic/robot/coarse2contact/supervisor.py` 中 `coarse2contact.step()` 返回后到最终 action merge 的责任段；搜索 `depth_shadow`、`depth_apply`、`force_reflex`、`depth_force`、`trace`、`final_action` | 这会让 coarse approach 被 near-contact 信号提前扰动，形成 planner 行为污染。 |
| 致命 | **depth 分支估计的是“最近前景/显著深度 proxy”，不是“孔相对位姿”** | `prismatic/robot/contact_refiner.py` 与 `prismatic/robot/stage_aware_refiner.py` 中把 `wrist_depth` 转成 offset/residual 的代码段；搜索 `wrist_depth`、`center`、`centroid`、`closest`、`target_offset` | 这会把诊断量当控制量，方向经常系统性错误。 |
| 致命 | **contact / phase / recovery 是阈值标签，不是有状态控制机** | `prismatic/robot/contact_trigger.py` 中 `contact_state`、`phase`、`recovery` 更新段；以及 `depth_force_contact_controller.py` 中对这些状态的消费位置 | 没有显式恢复状态就不可能形成稳定退让-重试闭环。 |
| 致命 | **force_reflex 只是单步反射，没有持久 recovery primitive** | `prismatic/robot/depth_force_contact_controller.py` 中 `force_reflex` 的执行段；搜索 `backoff`、`retry`、`recovery`、`contact` | 只会出现“退一点又撞回去”，不会真正解除卡滞。 |
| 高 | **`raw_force` 与 `force_hist` 的统计口径不一致，阈值未按真实分布校准** | `scripts/evaluate_rlbench.py`、`scripts/evaluate_rlbench_modes.py` 的 `process_obs` 与 sensor packaging；`contact_trigger.py` 中历史更新段；搜索 `raw_force`、`force_hist`、`threshold` | 触发条件和控制依据不一致，会造成不触发、迟触发或抖振。 |
| 高 | **local/world 或 camera/tcp/world 变换方向错误** | `prismatic/robot/residual_transforms.py` 全文件；搜索 `world`、`local`、`tcp`、`camera`、`transform`、`inverse` | 小修正一旦在错方向上执行，会在最后几毫米被放大。 |
| 高 | **`clip_final_action` 在错误时机裁剪，直接把修正裁没** | `prismatic/robot/residual_safety.py` 中 `clip_final_action`；以及 `supervisor.py` 中调用前后的融合逻辑 | 典型症状是 trace 里 residual 非零，但 executed action 近似回到 planner。 |
| 高 | **depth/force 的门控取了不该取的交集** | `stage_aware_refiner.py` 与 `contact_trigger.py` 中的 if 条件；搜索 `phase ==`、`depth_valid`、`force_valid`、`near_contact` | depth 最可靠的阶段和 force 最可靠的阶段并不完全重合，硬交集会让两者都长期不生效。 |
| 中 | **`InvalidActionError` 只是诊断信号，没有接到恢复控制** | `scripts/evaluate_rlbench.py`、`scripts/evaluate_rlbench_modes.py` 中异常处理路径；搜索 `InvalidActionError` | 发生无效动作时系统最应该切 recovery，但现在很可能直接失败。 |
| 中 | **共享 observation 被原地修改，导致 frozen planner 输入分布漂移** | `scripts/evaluate_rlbench.py` 中 `obs -> process_obs -> depth_tensor/raw_force` 的责任段；搜索 `process_obs`、`depth_tensor`、`obs[...] =` | 这是 depth 分支“污染 planner”的另一条高风险通路。 |
| 中 | **trace 记录了 depth/force，但未区分“建议动作”和“最终执行动作”** | `supervisor.py` 与 `evaluate_rlbench_modes.py` 中 trace 汇总段；搜索 `trace`、`mode`、`depth_*`、`force_*`、`executed` | 这会掩盖“看起来有分支输出，实际上没控制”的假象。 |

这些根因和当前症状高度一致：**不是 depth/force 完全没有信息，而是它们很可能既没有得到正确的控制职责，也没有得到正确的状态机载体。**已有插入研究反复表明，视觉局部伺服、力/触觉对齐与恢复都要以“阶段控制”方式工作，直接加 residual 往往不稳。citeturn25academia0turn25academia1turn17academia0turn17academia3turn20academia0

## 污染信号、门控错位与短路链路

我认为你当前系统里最危险的“污染信号”有五类。

第一类是 **`wrist_depth` 的标量或简单统计量**。如果它表达的只是“离最近可见物体多近”，那它只能当**可观测性/接近度诊断**，不能直接当 `x/y/yaw` 修正源。真正可控的量应该是“孔口相对末端插入坐标系的误差”。视觉伺服与局部配准之所以有效，靠的是相对几何，不是靠深度最小值本身。citeturn25academia1turn25academia0

第二类是 **`contact_state` / `phase` 这种离散标签**。它们应该用于**切换控制器所有权**，不应该直接决定某个动作分量的符号或大小。否则你实际上是在用一个低带宽、粗粒度诊断变量直接控制高带宽连续动作。

第三类是 **`raw_force` 的瞬时模长**。未滤波、未偏置消除、未做方向分解的力模长，只适合做安全门槛或异常提示；它不足以提供稳定的对齐方向。PolyFit 与力-扭矩动力学方法都强调，要想从 F/T 推位姿，必须看分量、历史、接触多样性，不能只看瞬时总范数。citeturn17academia3turn26academia4turn26academia1

第四类是 **`InvalidActionError`**。这是一个晚到、粗粒度、带环境依赖性的故障信号。它适合作为**恢复触发器**，不适合作为接触方向的控制输入。如果现在代码把它只当“评测失败计数”，那它就根本没进入控制回路。

第五类是 **融合后再统一裁剪的 residual**。这会造成一个特别迷惑的假象：log 看起来 depth/force 分支一直“有输出”，但执行时全被 `clip_final_action` 吃掉，于是系统表面上“模块都在工作”，实际上控制权始终还在 planner 手里。

门控错位方面，我最担心的是下面四种逻辑短路：

- **接触前才看得清孔，接触后才有力信息**，但 `depth_force` 却要求两者同时为真。  
- **`phase==contact` 由力阈值推出，而 force_reflex 又要求 `phase==contact` 才能启用**，形成循环依赖。  
- **depth 模块要求 near-contact 才启动，但 near-contact 恰好是腕相机最易遮挡的时段**，于是 depth 永远在最差观测条件下工作。  
- **异常路径先抛出/终止，再进入门控更新**，导致 recovery 永远进不去。  

这些短路不是“参数不好”，而是逻辑上就把 depth/force 关在门外了。FoAR 的思路非常值得借鉴：不是让力信息“总是参与”，而是让它在**正确阶段**拥有正确的控制职责。citeturn20academia0

还有一个很实用的模式判别准则，可以帮助你快速分辨“哪一层真的在控制，哪一层只是写日志”：

- 如果 `depth_shadow ≈ baseline`，而 `depth_apply` 明显更差，说明 **depth 观测旁路本身未必有毒，真正有毒的是动作注入层**。  
- 如果 `force_reflex` 的 trigger 次数很多，但 `backoff/retry` 次数接近零，说明 **它只是记录到“应该恢复”，但没有真正执行恢复 primitive**。  
- 如果 `depth_force` 的表现接近最差单分支，而不是优于两者，说明 **融合层不是协同层，而是冲突放大层**。  

## 系统级重设计方案

我建议你把整个 Coarse2Contact 重写成一个**相位拥有权明确的分层控制系统**，而不是“planner + 一堆 residual if/else”。

### 总体接口

核心原则只有三条。

第一，**冻结 planner 只做 coarse approach，不再让 depth/force 修改它的输入分布，也不再和它共享同一组自由度的持续控制权**。planner 输出的是“接近到插入工作区”的大动作；一旦进入局部操作阶段，局部控制器接管对应的子空间。这个思路和 coarse-to-fine、双策略 alignment→insertion 的成功经验是一致的。citeturn16academia1turn16academia2turn17academia0

第二，**depth 负责 pre-contact / near-contact 的局部几何对准，force 负责 contact / jam 阶段的相容控制与恢复，不再做简单线性相加**。这与视觉伺服、触觉/力引导插入工作的阶段分工高度一致。citeturn25academia0turn25academia1turn17academia3turn25academia3turn20academia0

第三，**所有局部修正都统一在“插入坐标系”里表达**。对 `insert_onto_square_peg` 而言，一个实用的控制坐标可以是 `{x_insert, y_insert, z_insert, yaw_about_insert_axis}`，把 pitch/roll 先固定到 planner 给出的接近姿态，避免 frame 传播混乱。

可以把新链路写成下面这个结构：

```text
obs_raw
 ├─ planner_obs -----------------------------> Frozen VLA -----------------> a_plan
 ├─ wrist_depth_raw + valid_mask + tcp_pose -> Depth localizer -----------> r_depth, c_depth
 └─ wrench_raw + hist + z_progress ---------> Phase estimator ------------> phase

phase:
  FAR / COARSE       -> owner = planner
  PRECONTACT         -> owner = depth servo on {x, y, yaw}, planner only keeps safe approach/z standby
  CONTACT / INSERT   -> owner = force admittance / contact alignment
  JAM / INVALID      -> owner = recovery primitive
  SEARCH_VIEW        -> owner = view recovery / micro-search

final action = owner-select(a_plan, u_depth, u_force, u_recovery)
```

### 一个真正有用的 depth 局部修正器

`insert_onto_square_peg` 不是“看见目标就往前推”的任务，而是“在最后几毫米里把孔口的相对平移与相对 yaw 校正到可插入范围”。因此，depth 模块的输出不应该是一个泛化 residual，而应该是：

- `r_depth = [dx, dy, dyaw, z_clearance]`，在**末端插入局部坐标系**下定义；
- `c_depth`，表示当前深度图对这个估计的可置信度/可观测性；
- `obs_quality`，表示此时是不是该做“视角恢复”而不是“几何修正”。

对于你当前任务，我会优先做一个**几何型 depth localizer**，而不是先上学习 residual。理由很简单：你现在的优先级不是“让一个 learner 更聪明”，而是先证明**架构本身是对的**。而 `insert_onto_square_peg` 具有非常强的任务结构：孔是方形，yaw 有 90° 对称性，腕部深度图中你真正关心的是孔口局部平面、孔边界和 peg 相对该边界的位置。只要 depth 图尚可观测，你完全可以从原始 metric depth 出发，做局部点云/边界拟合，输出孔口在末端局部系下的 `dx, dy, dyaw`。这本质上是一个局部 visual servo / registration 问题，而这类问题在插入任务上已有很好先例。citeturn25academia1turn25academia0turn16academia2

这里我会明确禁止三件事：

- **禁止在 `process_obs` 里过早量化/阈值化 depth**。先保留原始 metric depth、validity mask、局部深度梯度和点云几何。  
- **禁止把 `wrist_depth` 最小值或前景质心直接映射成横向修正**。  
- **禁止在 depth 低置信时继续输出横向修正**。低置信时应该切到 `SEARCH_VIEW`，做一个小抬升/小退让/小侧移，让孔口重新进到可观测区。  

depth 控制律也应该很“窄”：只接管 `{x, y, yaw}`，并且只在 `PRECONTACT` 阶段生效。`z` 不让 depth 直接推进插入，否则你会把“未验证完的横向对准”马上变成“带着误差硬顶”。

如果你想保留学习成分，我建议把它放在**局部相对位姿估计**上，而不是放在“直接输出 residual 动作”上。也就是训练一个 `depth -> [dx, dy, dyaw, confidence]` 的估计器，而不是一个 `depth -> full action` 的 policy。这一层即便之后要学，也仍然遵守相同的控制接口，不破坏整体架构。

### 一个真正有用的 force reflex / recovery 方案

force 模块需要从“阈值反应器”升级成“带恢复 primitive 的接触控制器”。

对 `insert_onto_square_peg`，我建议把 force 分成三个角色：

**接触检测**  
用 bias-corrected、低通后的 `wrench_tcp`，再结合 `z_progress` 与最近若干步历史，判断 `FREE / CONTACT / INSERTING / JAM`。这里不能只看 `|F|`，至少要分开看：(1) 插入轴向载荷；(2) 横向力；(3) 扭矩，尤其是绕插入轴与横摆相关的扭矩分量。已有力/触觉插入工作都说明，仅凭一个瞬时总模长不足以稳定解歧。citeturn17academia3turn26academia4turn17academia0

**接触期对准**  
当系统已进入 `CONTACT` 且 depth 观测已退化时，force 控制器接管 `{x, y, yaw}` 的小范围修正，并把 `z` 限制为“仅在侧向载荷低、插入进度正常时允许推进”。这实际上是一个轻量的 admittance / hybrid position-force 方案，目标不是“凭力把孔位整估出来”，而是“在已有 coarse + pre-contact 对准的基础上，用力引导最后的去卡滞和进入”。这与 FoAR、PolyFit、经典 hybrid force/position 控制的精神一致。citeturn20academia0turn17academia3turn19academia2

**恢复闭环**  
当满足以下任一条件时进入 `JAM`：横向力/扭矩持续过大；`z` 推进不足但载荷持续升高；或者发生 `InvalidActionError`。一旦进入 `JAM`，系统必须连续执行一段 primitive，而不是一帧 backoff：

1. `BACKOFF`：沿插入轴反向撤出固定距离。  
2. `UNLOAD`：等待横向力降到低阈值以下。  
3. `MICRO-SEARCH`：在安全高度做小尺度 `x/y/yaw` 探索，优先恢复 depth 观测。  
4. `RE-APPROACH`：重新进入 `PRECONTACT`。  
5. 超过最大次数才判失败。  

这才叫“退让-重试闭环”。有些插入工作甚至明确展示了“先对齐，再插入”的双策略结构，以及“先解除不良接触，再重新寻找孔位”的必要性。citeturn17academia0turn20academia1

### 组合层、坐标系与裁剪规则

这里是当前系统最该“动大手术”的地方。

我建议你在 `supervisor.py` 里实现**owner-by-phase** 规则，而不是 residual sum。也就是说：

- `FAR / COARSE`：`final = planner_action`
- `PRECONTACT`：`final[x,y,yaw] = depth_servo`；`final[z]` 只允许安全靠近或保持 standoff
- `CONTACT / INSERT`：`final = force_controller`
- `JAM / INVALID`：`final = recovery_primitive`

这样做有两个直接好处。第一，planner 永远不会被 near-contact 噪声持续污染。第二，depth 与 force 不再互相争抢动作解释权。VLA 相关工作之所以经常把额外模态做成 adapter、compliance controller 或 reactive policy，而不是简单 residual，相当大程度上就是为了明确控制职责。citeturn18academia3turn13academia1turn20academia0

`residual_transforms.py` 则应该只承担一件事：**把局部控制器输出的插入坐标系动作，可靠地变换到执行坐标系**。这里必须做一个专门的 transform 单元测试：给定一个已知局部偏移，前向到 world，再逆回局部，应当恢复原值；并且在不同腕姿态下符号不应翻转。否则所有“最后几毫米的微修正”都会被 frame 误差毁掉。

`clip_final_action` 也必须改规则。现在最危险的情况是：你先把 planner、depth、force 混在一起，再做统一裁剪。正确做法应当是：

- **每个控制器在自己的物理子空间里先独立限幅**；
- owner 选择之后，再做一次仅用于安全的全局裁剪；
- trace 里同时记录 `pre_clip_action` 与 `executed_action`，并计算衰减比例。  

如果某个模式下 `||u_local||` 经常不小，但 `||executed - planner||` 总是接近零，那就说明不是模型问题，是融合/裁剪路径在吞动作。

### 记录与评估也要一起重写

想发论文，就不能只报 success rate。你需要把 instrumentation 当成方法的一部分。

我建议 trace 至少包含这些字段：

- `planner_action`
- `depth_action_local`
- `force_action_local`
- `recovery_action_local`
- `phase`
- `phase_reason`
- `depth_confidence`
- `obs_quality`
- `raw_wrench`
- `filtered_wrench`
- `z_progress`
- `pre_clip_action`
- `executed_action`
- `trigger_flags`
- `retry_id`
- `exception_flag`

然后在 `evaluate_rlbench_modes.py` 里，不只测四个模式的成功率，还要测：

- 每个 phase 的占比  
- 每种 trigger 的触发率  
- 每种 local controller 的实际执行率  
- residual 被裁剪掉的比例  
- 每次失败前最后 1 秒的 phase 与 wrench 演化  

这样你才能真正回答“哪一层在控制，哪一层只是记了日志”。

## 最小实验与最终建议

如果我是这个项目的审稿前重构负责人，我会立即做下面这组**最小实验**。它们不是“多跑一些 baseline”，而是为了验证架构假设到底对不对。

| 实验 | 目的 | 要做什么 | 你期望看到什么 |
|---|---|---|---|
| 模式覆盖审计 | 判断各分支是否真的拿到控制权 | 对 `baseline / depth_shadow / depth_apply / force_reflex / depth_force` 记录 `phase`、trigger、`pre_clip/executed` 差值 | `depth_shadow` 应几乎不改变执行动作；若 `depth_apply` 有输出但执行差值接近零，说明被吞了；若执行差值明显但成功率更差，说明方向错了。 |
| 深度可观测性审计 | 判断 `wrist_depth` 到底是不是孔位信号 | 保存 near-contact 深度 crop，离线对比“估计孔位/孔轴”与真值或环境诊断量 | 你会很快看出它是在盯孔，还是在盯最近前景。 |
| 力阈值分布校准 | 摆脱手写常数阈值 | 统计 `FREE / CONTACT / JAM` 各阶段的 `raw_force`、`filtered_force` 分布与历史窗口分布 | 阈值应来自分布分位数，而不是脑补常数。 |
| transform 单元测试 | 排掉坐标系炸弹 | 向 local frame 注入已知小偏移，检查 world 执行方向与逆变换恢复误差 | 若这里不对，后续一切实验都没意义。 |
| recovery 闭环测试 | 验证 force_reflex 是否真正形成闭环 | 人工制造几种固定误差初始态，强制触发 `JAM`，观察是否完成 `BACKOFF -> SEARCH -> REAPPROACH` | 只要 recovery 仍是一帧逻辑，这个实验会立刻暴露。 |
| 相位拥有权消融 | 验证“owner-by-phase”是不是核心收益来源 | 比较 `planner-only`、`planner+depth(precontact only)`、`planner+force(contact only)`、`phase hybrid` | 如果架构对，hybrid 应明显优于 naive depth/force sum。 |

这些实验之所以值得先做，是因为已有研究已经说明：高精度插入问题的关键通常不是“更大的 policy”，而是**局部几何、阶段控制和接触恢复**是否设计正确。视觉伺服、力/触觉对齐和反应式接触策略都已经反复证明这一点。citeturn25academia0turn25academia1turn17academia0turn17academia3turn20academia0turn25academia3

最后给出我对你提出的三个收束判断。

**当前 depth/force 为什么没有形成正向作用**  
因为它们很可能既没有估计正确的控制量，也没有进入正确的控制阶段：depth 分支在用错误的几何 proxy 污染 planner；force 分支在做阈值反应而不是恢复闭环；depth_force 组合则在线性融合两个本应分时接管的模态，并可能叠加了坐标系与裁剪问题。

**最值得先改的三个地方**  
第一，重写 `supervisor.py` 的融合逻辑，改成 **owner-by-phase**，彻底切断 planner 与局部修正器的持续动作竞争。  
第二，重写 `contact_refiner.py / stage_aware_refiner.py` 的 depth 支路，让它输出**局部相对位姿误差**而不是深度 proxy。  
第三，重写 `contact_trigger.py / depth_force_contact_controller.py` 的状态机，让 `force_reflex` 真正变成**有持久状态的 recovery controller**，并把 `InvalidActionError` 接到 recovery primitive。  

**如果只允许先改一版，我会怎么改**  
我会做一个非常克制但结构正确的 V1：  
保留 frozen VLA planner；删除 depth 对 planner 输入的任何改写；删除 depth/force 的直接线性相加；实现一个四阶段 supervisor：`COARSE -> PRECONTACT(depth servo) -> CONTACT(force admittance) -> JAM(recovery)`；depth 只控制 `{x,y,yaw}`，force 只在接触后工作，`InvalidActionError` 直接进入恢复态；所有局部动作统一在插入局部系表达，并对 `pre_clip/executed` 做强制记录。这个版本未必一上来就是最终最优，但它最有机会**第一次把 depth/force 变成正向作用**，而不是继续当“会污染 planner 的旁路补丁”。这一方向和当前 contact-rich VLA、视觉伺服、力/触觉插入的有效经验是一致的。citeturn13academia1turn13academia0turn18academia3turn20academia0turn25academia1turn17academia3turn17academia0

补充一句必要的透明说明：因为当前环境无法直接获取该仓库源码，上面的“代码位置”是**文件级责任点定位**而不是逐行核验结论；但从你给出的模块职责、信号链与症状组合看，这套诊断和重构方向已经足够指导你做出第一版正确的架构重写。