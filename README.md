# Interaction-Aware CF-JEPA: Learning When Uncontrollable Objects Matter

**상호작용 인지 CF-JEPA: 조종할 수 없는 물체가 중요해지는 순간 학습하기**

Research proposal · Artificial Intelligence, Ajou University (Fall 2026)

## Abstract

**World model**은 "이 action을 하면 다음 장면이 어떻게 될까"를 예측하는 모델이다. Robot은 이 예측을 이용해 실제로 움직이기 전에 미리 planning한다.

최신 world model인 CF-JEPA는 장면 정보를 두 부분으로 나눈다.
- **controllable** 부분 `z^c`: agent가 action으로 바꿀 수 있는 정보. 예를 들어 agent의 위치다.
- **uncontrollable** 부분 `z^u`: action과 무관하게 움직이는 정보. 예를 들어 화면 속 distractor다.

이렇게 나눈 덕분에 CF-JEPA는 관계없는 distractor가 많아도 무너지지 않는다. 대신 CF-JEPA는 **"`z^u`는 `z^c`에 절대 영향을 주지 않는다"고 가정**한다. 그래서 움직이는 object가 agent와 **collision**하는 상황은 예측할 수 없다.

본 연구는 CF-JEPA의 예측에 **interaction gate**를 하나 추가한다. Interaction gate는 평소에는 꺼져 있다가, 부딪히는 순간에만 켜지는 스위치다. 목표는 평소의 robustness는 유지하면서, collision이 있는 상황의 prediction과 planning을 개선하는 것이다.

## Key Concepts

| Term | 설명 |
|---|---|
| **World model** | 현재 상태와 action으로 다음 상태를 예측하는 모델 |
| **JEPA** | 다음 *이미지*를 그리는 대신, 다음 *latent*를 예측하는 방식 (Joint-Embedding Predictive Architecture) |
| **Latent** | Encoder가 이미지를 압축해 만든 짧은 숫자 vector |
| **Planning (CEM)** | 여러 action 후보를 world model로 미리 굴려 보고, goal에 가장 가까워지는 action을 고르는 방법 |
| **Distractor** | Task와 무관하게 화면에서 움직이는 object |
| **Controllable / Uncontrollable** | Agent의 action으로 바꿀 수 있는 정보 / 바꿀 수 없는 정보 |
| **Inverse dynamics loss** | 전후 상태만 보고 어떤 action을 했는지 맞히게 하는 보조 loss |
| **Gradient reversal** | 학습 신호의 부호를 뒤집는 장치. 한쪽 network는 action을 맞히려 하고, encoder는 못 맞히게 만든다 |
| **Representation collapse** | 모든 입력을 같은 latent로 보내 loss를 낮추는 실패 현상. SIGReg가 이를 막는다 |
| **Ablation** | 제안한 구성 요소를 하나씩 빼 보며, 그 요소가 정말 필요한지 확인하는 실험 |
| **AUROC** | 점수가 정답을 얼마나 잘 구분하는지 나타내는 지표. 1.0은 완벽한 구분, 0.5는 무작위 수준이다 |

## Research Questions

- **RQ1.** Interaction gate를 추가하면, collision이 있는 환경에서 prediction과 planning 성능이 좋아지는가?
- **RQ2.** 모델은 "지금 부딪혔다"를 스스로 알아차리는가? 즉, gate value `α_t`가 실제 contact 순간에 커지는가?

## Background

**LeWM** ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312))
- 이미지(pixel)에서 바로 학습하는 작은 JEPA world model이다 (~15M parameters).
- 우리 연구의 기반 모델이다.

**CF-JEPA** ([arXiv:2610.00727](https://arxiv.org/abs/2610.00727))
- LeWM의 latent(192-dim)를 `z^c`(128-dim)와 `z^u`(64-dim)로 나눈다.
- 각 부분을 따로 예측한다:
  - `z^c`는 `g(z^c_t, a_t)`로 예측하고, action을 입력으로 받는다.
  - `z^u`는 `h(z^u_t)`로 예측하고, action을 입력으로 받지 않는다.
- 두 부분을 나누는 방법은 다음과 같다:
  - **Inverse dynamics loss**: `z^c`의 변화로 action을 맞히게 한다. 그러면 action 관련 정보가 `z^c`에 모인다.
  - **Adversarial loss (gradient reversal)**: `z^u`로는 action을 못 맞히게 한다. 그러면 `z^u`에서 action 정보가 빠진다.
- Planning은 `z^c`만 보고 한다. 그래서 `z^u`에 들어간 distractor가 planning을 방해하지 못한다.
- 공식 코드는 공개되지 않았다.

## Related Work

선행연구를 세 조건으로 비교한다: ① controllable / uncontrollable 분리, ② uncontrollable이 controllable에 주는 영향(u → c) 모델링, ③ reconstruction·reward 없이 학습해 planning.

| Work | ① c/u 분리 | ② u → c interaction | ③ Recon·reward 없이 planning |
|---|---|---|---|
| TIA (ICML 2021) | △ reward 관련 / 무관 | ✗ | ✗ |
| Denoised MDP (ICML 2022) | ✓ | ✗ | ✗ |
| Iso-Dream++ (2023) | ✓ | △ 반대 방향 (c → u) | ✗ |
| SPARTAN (NeurIPS 2025) | ✗ object 단위 | ✓ sparse attention | ✗ 예측만 |
| Causal-JEPA (2026.02) | ✗ object 단위 | △ masking으로 암묵적 | ✓ slot encoder 필요 |
| SMWM (2026.06) | △ controllable만 | ✗ | ✓ |
| CF-JEPA (2026.10) | ✓ | ✗ 독립 가정 | ✓ |
| **Ours (목표)** | ✓ | ✓ gate `α_t` | ✓ |

- **Iso-Dream++**가 가장 가까운 선행연구다. Sparse gate가 있지만 방향이 controllable → noncontrollable이고, noncontrollable 미래는 policy의 attention으로만 쓰인다. 본 연구는 반대 방향(`z^u → z^c`)을 world model 예측 안에서 학습한다.
- **SPARTAN**은 object token 간 attention에 sparsity penalty를 건다. 본 연구의 sparsity term `ρ · E[α_t]`는 같은 계열의 idea다.
- 세 조건을 동시에 만족하는 연구는 찾지 못했다 (2026.10.09 검색 기준).

## Problem Statement

CF-JEPA §V (Conclusion)에 다음과 같이 적혀 있다:

> "we assume that the uncontrollable part of the scene never affects the controllable part"
>
> = "주변 object는 agent에 절대 영향을 주지 않는다고 가정했다"

이 가정 때문에 생기는 문제는 두 가지다.
- **Architectural independence:** CF-JEPA는 §III-A Eq. 1에서 환경을 `T(s′|s, a) = T_c(s^c′|s^c, a) · T_u(s^u′|s^u)`로 정의하고, 이 구조를 exogenous block MDP(Efroni et al., 2022)라 부른다. `T_c`에는 `s^u`가 들어가지 않는다. 같은 구조로 `z^c`를 예측하는 `g`는 `z^u`를 아예 보지 않는다. 주변 object가 agent를 밀어도 모델은 그 사실을 알 수 없다. 그래서 collision이 있으면 `g`를 아무리 잘 학습해도 `E[Var(z^c_{t+1} | z^c_t, a_t)]`만큼의 오차가 남는다.
- **Evaluation gap:** 논문 실험의 distractor는 화면 가장자리를 지나가는 색 사각형뿐이다. 부딪히는 상황은 시험되지 않았다.

## Proposed Method

기존 예측은 그대로 두고, "주변 object의 영향" 항을 하나 더한다. 그 항 앞에는 0~1 사이의 gate를 둔다.

```
ẑ^c_{t+1} = g(z^c_t, a_t)  +  α_t · h_int(z^c_t, z^u_t, a_t)
            ─────────────     ───   ─────────────────────────
            기존 CF-JEPA 예측   gate   object가 agent에 주는 영향

α_t = σ(MLP([z^c_t, z^u_t, a_t]))      # 0 = 무시, 1 = 반영
L   = L_CF-JEPA + ρ · E[α_t]           # sparsity: gate가 괜히 열리면 penalty
```

- **Robustness preservation:** 부딪히지 않는 distractor는 예측에 도움이 되지 않는다. 그래서 sparsity penalty 때문에 `α ≈ 0`이 되고, 모델은 CF-JEPA와 같아진다.
- **Interaction learning:** 부딪히는 순간에는 `z^u` 없이 agent의 다음 위치를 맞힐 수 없다. 그래서 prediction loss가 gate를 연다.
- **Planning:** CF-JEPA와 똑같이 `z^c`로 CEM을 수행한다. 다만 예측이 collision을 반영하므로, obstacle을 피하는 action이 선택된다.

## Experimental Setup

| | |
|---|---|
| **Env. ① Robustness** | CF-JEPA와 같은, 부딪히지 않는 distractor (2·8·16개). TwoRoom, PushT. 기존 강점이 **유지**되는지 확인 |
| **Env. ② Interaction (new)** | TwoRoom에 움직이는 obstacle을 추가. 닿으면 agent를 밀거나 막는다. **개선**되는지 확인 |
| **Baselines** | LeWM, CF-JEPA (직접 재구현) |
| **Ablations** | gate 없이 항상 반영 (`α = 1`), sparsity 제거, `z^u` stop-gradient |
| **Metrics** | **Success rate** (goal 도달 비율), contact / non-contact 구간별 **prediction error**, `α_t`의 contact 구분 **AUROC** |
| **Protocol** | 3 seeds × 50 episodes |

**Hypotheses:**
- **H1.** Env. ②에서 CF-JEPA보다 success rate가 +10%p 이상 높다.
- **H2.** Env. ①에서 success rate 하락이 3%p 이내다.
- **H3.** AUROC가 0.8 이상이다.

## Timeline

| Date | Milestone |
|---|---|
| 10/12 | Proposal. Env. ② prototype, preliminary study (CF-JEPA가 contact 구간에서 실제로 틀리는지 확인) |
| ~10/26 | CF-JEPA 재구현. Env. ① 결과 재현 (TwoRoom, distractor 2개에서 ≈ 0.95) |
| 11/5 | Interim. Env. ② 완성, interaction gate 초기 결과 |
| 11/26 | Final. 두 환경의 main results, ablations, `α_t` visualization |

## Repository Structure

```
src/        model (encoder, predictors, interaction gate, losses)
envs/       interaction (collision) environments
configs/    training / evaluation configs
notebooks/  Colab experiments
scripts/    data collection, training, evaluation
```

Dataset과 checkpoint는 커밋하지 않는다 (`.gitignore` 참고).

## Team

| Name | GitHub |
|---|---|
| 강연우 | [@yeonooe](https://github.com/yeonooe) |
| 조하은 | [@gkdms04](https://github.com/gkdms04) |
| 설규원 | [@seolcu](https://github.com/seolcu) |

## References

1. Maes, Le Lidec, Scieur, LeCun, Balestriero. **LeWorldModel: Stable End-to-End JEPA from Pixels (LeWM)**. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026.
2. Byrd, Wright, Ha. **CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization**. [arXiv:2610.00727](https://arxiv.org/abs/2610.00727), 2026.
3. Ivashkov, Balestriero, Schölkopf. **Sensorimotor World Models: Perception for Action via Inverse Dynamics (SMWM)**. [arXiv:2606.20104](https://arxiv.org/abs/2606.20104), 2026.
4. Nam, Le Lidec, Maes, LeCun, Balestriero. **Causal-JEPA: Learning World Models through Object-Level Latent Masking**. [arXiv:2602.11389](https://arxiv.org/abs/2602.11389), 2026.
5. Lei, Schölkopf, Posner. **SPARTAN: A Sparse Transformer World Model Attending to What Matters**. NeurIPS 2025. [arXiv:2411.06890](https://arxiv.org/abs/2411.06890).
6. Pan, Zhu, Zheng, Wang, Yang. **Model-Based Reinforcement Learning with Isolated Imaginations (Iso-Dream++)**. [arXiv:2303.14889](https://arxiv.org/abs/2303.14889), 2023.
7. Wang, Du, Torralba, Isola, Zhang, Tian. **Denoised MDPs: Learning World Models Better Than the World Itself**. ICML 2022. [arXiv:2206.15477](https://arxiv.org/abs/2206.15477).
8. Fu, Yang, Agrawal, Jaakkola. **Learning Task Informed Abstractions (TIA)**. ICML 2021. [arXiv:2106.15612](https://arxiv.org/abs/2106.15612).
9. Efroni, Misra, Krishnamurthy, Agarwal, Langford. **Provably Filtering Exogenous Distractors using Multistep Inverse Dynamics**. ICLR 2022. [arXiv:2110.08847](https://arxiv.org/abs/2110.08847).
10. You et al. **A Control Theory of Predictability in Latent World Models**. [arXiv:2607.10362](https://arxiv.org/abs/2607.10362), 2026.
11. Sawada. **Disentangling Controllable and Uncontrollable Factors of Variation by Interacting with the World**. [arXiv:1804.06955](https://arxiv.org/abs/1804.06955), 2018.
