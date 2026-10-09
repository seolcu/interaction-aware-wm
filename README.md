# Interaction-Aware CF-JEPA: Learning When Uncontrollable Objects Matter

**상호작용 인지 CF-JEPA: 조종할 수 없는 물체가 중요해지는 순간 학습하기**

Research proposal · Artificial Intelligence, Ajou University (Fall 2026)

## Abstract

CF-JEPA는 JEPA world model의 latent를 **controllable** 부분 `z^c`와 **uncontrollable** 부분 `z^u`로 factorize한다. 이를 통해 visual distractor에 대한 robustness를 크게 높였다. 그러나 CF-JEPA는 uncontrollable 부분이 controllable 부분의 dynamics에 영향을 주지 않는다고 가정한다. 그래서 distractor가 agent와 물리적으로 interaction하는 상황, 예를 들어 collision은 모델링할 수 없다.

본 연구는 controllable predictor에 learnable **interaction gate**를 추가한다. 이 gate는 non-interacting 구간에서는 CF-JEPA의 robustness를 보존하고, interaction이 발생하는 구간에서만 `z^u`의 정보를 `z^c`의 transition에 반영한다.

## Research Questions

- **RQ1.** Interaction gate를 추가하면, collision이 있는 환경에서 latent prediction error와 planning success rate가 개선되는가?
- **RQ2.** 학습된 gate value `α_t`는 실제 contact event와 정렬되는가? (interpretability)

## Background

- **LeWM** ([arXiv:2603.19312](https://arxiv.org/abs/2603.19312))
  - pixel observation에서 end-to-end로 학습하는 JEPA world model이다 (~15M parameters).
  - representation collapse는 SIGReg로 방지한다.
  - planning은 latent space에서 CEM으로 수행한다.
- **CF-JEPA** ([arXiv:2610.00727](https://arxiv.org/abs/2610.00727))
  - LeWM의 192-dim latent를 `z^c` (128-dim)와 `z^u` (64-dim)로 분리한다.
  - **Controllable predictor** `g(z^c_t, a_t)`: action-conditioned이다.
  - **Uncontrollable predictor** `h(z^u_t)`: action-free이다.
  - **Inverse dynamics loss**: `z^c`가 action 정보를 담도록 한다.
  - **Adversarial dynamics loss** (gradient reversal layer 사용): `z^u`에서 action 정보를 제거한다.
  - Planning cost는 `z^c`에 대해서만 계산한다.
  - 공식 코드는 공개되지 않았다.

## Problem Statement

CF-JEPA §V (Conclusion):

> "we assume that the uncontrollable part of the scene never affects the controllable part"

- **Architectural independence:** `g`는 `z^u`를 input으로 받지 않는다. 그래서 `z^u`에서 `z^c`로 가는 information path가 존재하지 않는다.
- **Evaluation gap:** 실험에 쓰인 distractor는 이미지 가장자리를 따라 움직이는 colored rectangle이다. agent와 physical contact가 없다.

## Proposed Method

```
ẑ^c_{t+1} = g(z^c_t, a_t) + α_t · h_int(z^c_t, z^u_t, a_t)
α_t       = σ(MLP([z^c_t, z^u_t, a_t]))
L         = L_CF-JEPA + ρ · E[α_t]          # sparsity regularization
```

- **`α_t = 0`인 경우:** 모델은 CF-JEPA와 동일해진다. 이 경우 sparsity regularization이 non-interacting distractor에 대한 robustness를 보존한다.
- **Collision 구간:** `z^u` 없이는 `z^c`의 next state를 예측할 수 없다. 따라서 prediction loss가 gate를 연다.
- **Planning:** CF-JEPA와 동일하게 `z^c`에 대한 CEM을 사용한다.

## Experimental Setup

| | |
|---|---|
| **Env. ① (robustness)** | CF-JEPA와 동일한 non-interacting distractor (N = 2, 8, 16). TwoRoom, PushT |
| **Env. ② (interaction)** | TwoRoom에 moving obstacle을 추가. contact 시 agent를 push 또는 block. scripted policy로 data를 수집 |
| **Baselines** | LeWM, CF-JEPA (reimplementation) |
| **Ablations** | `α = 1` (no gating), sparsity term 제거, `z^u` stop-gradient |
| **Metrics** | planning success rate, contact / non-contact 구간의 1-step latent prediction error, `α_t`의 contact detection AUROC |
| **Protocol** | 3 seeds × 50 episodes |

**Hypotheses:**
- Env. ②에서 CF-JEPA 대비 success rate가 +10%p 이상 높다.
- Env. ①에서 success rate 하락이 3%p 이내다.
- Contact detection AUROC가 0.8 이상이다.

## Timeline

| Date | Milestone |
|---|---|
| 10/12 | Proposal. Env. ② prototype, preliminary study |
| ~10/26 | CF-JEPA reimplementation. Env. ① 결과 재현 (TwoRoom, N=2에서 ≈ 0.95) |
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

1. **LeWorldModel (LeWM)**. [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), 2026.
2. Byrd, Wright, Ha. **CF-JEPA: Improving Robustness of JEPA World Models via Controllability Factorization**. [arXiv:2610.00727](https://arxiv.org/abs/2610.00727), 2026.
3. Ivashkov, Balestriero, Schölkopf. **Sensorimotor World Models (SMWM)**. [arXiv:2606.20104](https://arxiv.org/abs/2606.20104), 2026.
4. **A Control Theory of Predictability in Latent World Models**. [arXiv:2607.10362](https://arxiv.org/abs/2607.10362), 2026.
5. **Orthogonal JEPA: Factorized Predictive States for Latent World Models**. [arXiv:2608.20065](https://arxiv.org/abs/2608.20065), 2026.
6. **Disentangling Controllable and Uncontrollable Factors of Variation by Interacting with the World**. [arXiv:1804.06955](https://arxiv.org/abs/1804.06955), 2018.
