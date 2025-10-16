# 8个拦截弹对抗2个来袭导弹的改造清单

下列条目按照“为什么需要调整 → 需要修改的内容 → 修改后的作用”给出，请在实施前逐项确认。

## 1. 场景基础参数
- **拦截弹数量保持 8 枚**
  - *原因*：当前环境与初始化脚本均以 8 枚拦截弹构建状态矩阵和拦截弹对象，若误调默认值会造成动作空间与库存错配。
  - *修改*：复核 `ManeuverEnv` 构造函数与 `init_env` 入口的 `InterceptorNum`/`interceptor_num` 默认值保持为 8，并确保所有建环境入口都显式传入该值以避免后续改动时被覆盖。【F:Environment/env.py†L31-L70】【F:Environment/init_env.py†L8-L65】
  - *作用*：保证仿真与训练阶段的拦截弹数量与 8v2 目标一致。
- **来袭导弹数量改成 2 枚**
  - *原因*：环境构造、随机初始化与重置流程仍默认生成 3 枚导弹，会让状态维度、奖励缩放和训练脚本停留在 8v3 设定。
  - *修改*：将 `ManeuverEnv`、`init_env` 与 `reset_para` 的 `missilesNum/num_missiles` 默认值统一调整为 2，并在 `ManeuverEnv.reset()` 中同步实例属性；调用这些接口的脚本也要传入 2。【F:Environment/env.py†L31-L90】【F:Environment/init_env.py†L8-L73】【F:Environment/reset_env.py†L7-L73】
  - *作用*：确保环境每回合实际只生成 2 枚威胁弹，后续状态与奖励逻辑才能按照 8v2 尺度运行。

## 2. 状态与动作空间
- **观测矩阵扩缩到 11×6**
  - *原因*：状态拼接依赖 `self.missileNum` 与 `self.interceptorNum` 计算形状，若外部工具继续假设 12 行（1+3+8）会在回放与可视化时错位。
  - *修改*：将 `utils/predict.py` 中的 `DEFAULT_NUM_MISSILES` 及相关派生变量改为 2，所有根据常量截取状态的函数（如 `predictResult`、`ComparepredictResult`、`aniPlot` 等）需以新的常量或环境返回的 `obs_rows` 计算维度；同时检查其它脚本是否写死 3 枚导弹的长度假设。【F:utils/predict.py†L24-L195】【F:Environment/env.py†L66-L75】【F:Environment/env.py†L849-L882】
  - *作用*：保证渲染、回放与模型输入统一使用 11×6（飞机 1 + 导弹 2 + 拦截弹 8）状态矩阵。
- **动作库目标索引校验**
  - *原因*：动作库在分配目标时依赖 `missile_num` 的取值范围，若外部生成器仍传入 3 会导致索引越界或错打。
  - *修改*：在调用 `getNewActionDepository` 或自定义动作生成逻辑处传入一致的导弹数量，并可在 `LockConstraint` 或动作解析处增加断言以捕获目标上限超出 2 的情况。【F:Environment/ActionDepository.py†L34-L53】【F:Environment/env.py†L339-L360】
  - *作用*：避免因为导弹索引错配导致的训练崩溃或策略失效。

## 3. 拦截弹发射与锁定策略
- **Lock 限额重新标定**
  - *原因*：`LockConstraint` 依据 `self.interceptorNum / self.missileNum` 估算单目标可集中的拦截弹数量，在导弹仅剩 2 枚时默认允许同一目标最多 4~5 枚拦截弹，需要评估是否会导致火力过度集中。
  - *修改*：根据战术预期调整 `base_limit` 与 `focus_bonus` 的策略，或在危险阶段额外约束单目标锁定数；同时确认 `LanchGap=70` 的发射节奏在 8 枚拦截弹面对 2 枚威胁时是否需要缩短以提升响应速度。【F:Environment/env.py†L19-L21】【F:Environment/env.py†L339-L360】
  - *作用*：保证拦截弹分配在 8v2 场景下既不过度堆砌、也能在关键时刻迅速响应。
- **剩余拦截弹奖励系数调优**
  - *原因*：`commonReward` 中剩余弹量惩罚按导弹数量归一化，导弹数减至 2 后单位罚分会翻倍，可能促使模型过早倾泻所有拦截弹。
  - *修改*：基于新的导弹数量重新设置危险时的 `focus_penalty` 缩放方式（例如按最大导弹数 2 归一化或引入上限），以免在威胁阶段过度惩罚未发射的拦截弹。【F:Environment/env.py†L508-L517】
  - *作用*：在保持威胁压力的同时，避免激励结构鼓励无差别倾泻火力。

## 4. 奖励函数权重与阈值
- **危险距离与加权复核**
  - *原因*：2 枚导弹同时逼近时的风险密度低于 3 枚版本，沿用原 `DANGER_DISTANCE` 与 `DANGERSCALE` 可能导致危险标志过早触发、奖励剧烈波动。
  - *修改*：结合 8v2 轨迹重新评估危险距离，必要时降低 `DANGERSCALE` 或增加与导弹数量相关的缓冲系数；检查距离奖励、剩余弹惩罚等在危险模式下的梯度是否过强。【F:Environment/env.py†L16-L22】【F:Environment/env.py†L430-L517】
  - *作用*：让奖励在 2 枚导弹场景中保持稳定梯度，避免训练振荡。
- **导弹威胁度与锁定奖励缩放**
  - *原因*：`rl`、`ri` 段落的奖励/惩罚目前以绝对数量计算，导弹数量减少后每次锁定的权重应适当缩放，防止一次错误锁定导致巨额负分。
  - *修改*：将 `engaged_on_threat`、`wrong_lock` 的系数按导弹数量或剩余威胁归一化，必要时调低 `ERRACTIONSCALE` 或 `1.8/0.6` 等常数；同理调整 `ri / lock_num` 的缩放以匹配新的威胁峰值范围。【F:Environment/env.py†L562-L600】
  - *作用*：保持奖励函数的相对幅值与 8v3 时相近，让网络能延续既有架构稳定收敛。
- **稀疏奖励惩罚校准**
  - *原因*：`SparseReward` 中对剩余导弹数量的惩罚会因威胁数量减半而自动减半，需确认是否仍能提供足够驱动力。
  - *修改*：视训练需求调节 `SPARSE_REWARD_SCALE` 或在危险模式下引入最小惩罚，以维持拦截失败时的明确负反馈。【F:Environment/env.py†L18-L22】【F:Environment/env.py†L618-L633】
  - *作用*：在 8v2 场景中保持稀疏奖励对“拖延时间”与“漏拦截”的敏感度。

## 5. 训练、预测与验证脚本
- **入口脚本默认参数同步为 2 枚导弹**
  - *原因*：`train.py`、`predict.py`、`validate.py` 等入口目前仍以 3 枚导弹初始化环境，导致训练与评估数据维度不符。
  - *修改*：将 `num_missiles`、`DEFAULT_NUM_MISSILES` 以及 `EvaluationConfig.num_missiles` 改成 2，并在命令行参数解析的默认值中同步更新；保存新模型时使用独立目录以避免旧权重形状不匹配。【F:utils/train.py†L112-L167】【F:utils/predict.py†L35-L178】【F:utils/validate.py†L19-L162】
  - *作用*：确保训练、验证与推理流程全程使用 8v2 环境参数，避免加载/保存模型时出现维度冲突。

## 6. 数据记录与可视化
- **图例与统计字段检查**
  - *原因*：可视化脚本的图例与颜色列表基于“来袭导弹/拦截弹群/飞行器”三类对象，但部分统计可能隐含 3 枚导弹的长度假设。
  - *修改*：在绘图与 CSV 导出时依据 `DEFAULT_NUM_MISSILES` 或环境实际值动态生成导弹曲线与统计字段，必要时更新图例说明以强调 2 枚来袭弹的设定。【F:utils/predict.py†L169-L226】
  - *作用*：保证展示与数据分析准确反映 8v2 场景，避免因长度错配导致的图像或表格异常。

如确认无遗漏，可按以上顺序逐项实施和回归测试。
