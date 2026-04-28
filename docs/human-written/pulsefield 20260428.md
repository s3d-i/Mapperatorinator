---
date: 20260428
status: valid
commit: ba7c1f5a4e767db6730e0009ff615c03c861b937
---
## project state
a. we have trained stage 1 models using self-attention + audiomel&oracle dense timing encoder on small datasets. these models have several problems:
1. it handle cross-window long notes very bad because of our training design. 
2. due to autoregressive self-attention bad tokens will affect follow-up tokens. and the model lacks the ability to globally plan the beatmap

b. we have selected out useful features in train/stage1_oracle/features/control_v3.py.

c. we want to start a fresh stage_2 control model, decoder mapper model train.
## issues
### 1
a)
open_hold_mask 适合作为 legality state
但不适合作为完整的 LN continuity conditioning

new LN carry contract:
`@dataclass(frozen=True)`
`class CarryLNState:`
    `open_mask: int  # 0..15`
    `open_age_ms_by_lane: tuple[int, int, int, int]`
```
open_mask:
  legality state

open_age_ms_by_lane[lane]:
  closed -> 0
  open   -> write_start_ms - hold_start_ms
```

需注意, contract不等于模型输入feature方式. 未来可以依据config做一些transform
b)
decoder HEAD output logits缺乏对于何时在哪里关闭ln的倾向, 如果只有在窗口开始的时候喂进CarryLNState
因此, 我们需要维护一个per step dynamic state, 并且对于HEAD加一个adapter, 产生bias偏好(后续我们也许可以做gated carryLNstate body injection, 但是现在我们先做HEAD adapter)
显式维护：
dynamic_open_mask_t
dynamic_open_age_ms_by_lane_t

logits = base_logits
logits = logits + dynamic_state_logits_adapter(dynamic_state_t)
logits = logits + hard_grammar_mask(dynamic_open_mask_t)
含义:
```
base_logits:
  model 根据音乐、control、prefix 认为下一步应该出什么

dynamic_state_logits_adapter:
  根据当前 LN 状态调整“合理性”

hard_grammar_mask:
  把非法 token 直接打成 -inf
```

这个dynamic_state_logits_adapter长什么样?
```
dynamic_state

  -> DynamicStateEncoder

  -> low-rank bottleneck

  -> lane_action_bias[4 lanes, 4 actions]

  -> project to flattened EV vocab

  -> add to logits

  -> apply hard grammar mask (invalid -inf)

```
性质期望:
```
低秩：
  不要过强，不要变成另一个 decoder

action-aware：
  不要无结构地改所有 token，而是按 lane/action 改 EV token
```
伪代码：
```
state_emb = DynamicHoldStateEncoder(
    dynamic_open_mask,
    dynamic_open_age_ms_by_lane,
)  # [B, T, D]

z = down_proj(state_emb)        # [B, T, r]
z = gelu(z)
lane_action_bias = up_proj(z)   # [B, T, 4 * 4]
lane_action_bias = lane_action_bias.view(B, T, 4, 4)

r = 8, 16, or 32
对于 flattened EV token：EV_NONE_HOLD_END_NONE_TAP, 它的bias是:
bias =
  lane_action_bias[lane0, NONE]
+ lane_action_bias[lane1, HOLD_END]
+ lane_action_bias[lane2, NONE]
+ lane_action_bias[lane3, TAP]
```

### 2
在目前的model里面difficulty作为一个简单的bucket prefix, 有如下缺点:
1. bucket不显示表示difficulty的连续顺序
2. 信号只在序列开头, 容易被自回归history稀释
3. 不能被每个layer, 每个token感知到. 它应该是一个global condition

解决方案:
### control model 侧
difficulty contract
```
@dataclass(frozen=True)
class DifficultyCondition:
    raw_stars: float          # original requested stars, e.g. 4.37
    norm_scalar: float        # [-1, 1]
```
difficulty_embed = scalar_mlp(norm_scalar)
h = FiLM(h, difficulty_embed)
transformer block后film或者每几层之后film. 它更像先理解音乐，再按难度调节 pressure。
### decoder 侧
```
forced prefix:
  BOS

side condition:
  CarryLNState

cross-attention:
  audio/timing memory
  control memory
```
