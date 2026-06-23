---
layout: post
title: "Why variance matters in reinforcement learning (and why most people ignore it)"
subtitle: "On the gap between what RL algorithms optimize and what we actually want."
date: 2026-06-23
tags: [RL, risk]
---

Most RL algorithms optimize for expected return. That's the Bellman equation, that's Q-learning, that's policy gradient — all of it.

$$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_t \,\Big|\, S_0 = s \right]$$

The expectation is doing a lot of work there. It's telling you: *we will average over everything that can happen, and optimize that average.* Which is fine if you're playing an Atari game thousands of times. But in any setting where you care about individual trajectories — manufacturing, clinical intervention, financial control — averaging over outcomes is exactly the wrong thing to do.

## The problem with expectation

Here's a simple example. Consider two policies:

- **Policy A**: Always returns exactly $10.
- **Policy B**: Returns $20 with probability 0.5, and $0 with probability 0.5.

Both have expected return $10. Any standard RL algorithm will treat them identically. But if you're running a manufacturing process where a zero-return outcome means a failed production run, Policy A is strictly better.

The formal object we want to optimize is something like:

$$J_\text{var}(\pi) = \mathbb{E}[G^\pi] - \beta \cdot \text{Var}(G^\pi)$$

where $\beta > 0$ is a risk-sensitivity parameter. This is the mean-variance objective — old in finance (Markowitz, 1952), underused in RL.

## Why variance is hard to handle in RL

The core difficulty is that variance doesn't satisfy the Bellman equation.

Recall that expected return does:

$$V^\pi(s) = \mathbb{E}[R(s,a)] + \gamma \mathbb{E}_{s'}[V^\pi(s')]$$

This recursive structure is what makes Q-learning and TD methods work — you can decompose a long-horizon problem into single-step updates.

But now consider the variance of the return:

$$\text{Var}(G^\pi) = \mathbb{E}[(G^\pi)^2] - (\mathbb{E}[G^\pi])^2$$

The second-moment term $\mathbb{E}[(G^\pi)^2]$ has its own Bellman-like recursion:

$$W^\pi(s) = \mathbb{E}[R^2 + 2\gamma R V^\pi(s') + \gamma^2 W^\pi(s')]$$

where $W^\pi(s) = \mathbb{E}[(G^\pi)^2 \mid S_0 = s]$. This requires tracking *both* $V^\pi$ and $W^\pi$ simultaneously — two coupled estimators — which is why most risk-sensitive actor-critic methods end up with three timescales: one for the policy, one for the value baseline, one for the variance estimate.

## The random scaling trick

In my own work on [VPAC-RS](https://saunakp123.github.io/pandasaunak.github.io/), I noticed something: if you're already computing asymptotic inference for Q-learning (which I was, for the [SA-QL paper](https://doi.org/10.1109/TIT.2025.3569421)), you naturally get a non-parametric variance estimator as a byproduct.

The random scaling estimator for variance looks like:

$$\hat{V}_{\text{RS},i} = \frac{A_{\text{diag},i} - 2\bar{Q}_i \cdot b_{\text{mat},i} + S_i [\bar{Q}_i]^2}{(i+1)^2}$$

where $S_i = (i+1)(i+2)(2i+3)/6$ and the terms track the running sum-of-squares of Q-iterates. This gives you a consistent variance estimate without any hyperparameters — no bandwidth, no kernel, no separate critic head.

The key insight for contraction: $\hat{\sigma}_\text{clip}$ appears identically in $KQ_1$ and $KQ_2$, so it cancels:

$$\|KQ_1 - KQ_2\|_\infty \leq \gamma \|Q_1 - Q_2\|_\infty$$

The contraction holds regardless of how accurate $\hat{\sigma}$ is — consistency not required. This collapses the three-timescale problem back to two timescales.

## What this buys you practically

In a High-Temperature Superconductor manufacturing process (58,240 real sensor measurements, 970-minute production run), VPAC-RS achieved **83% variance reduction** in the Solid-State Ic metric compared to standard DDPG — while maintaining output quality under hard safety constraints.

The point isn't the number. The point is that a standard RL policy deployed on that process would have produced high-variance behavior that, in practice, would have been unacceptable to operators — even if the *average* output looked fine on paper.

## Where this matters most

Risk-sensitive RL is worth caring about whenever:

1. **Individual trajectories matter more than averages** — clinical trials, manufacturing runs, financial positions
2. **Failure is asymmetric** — a single catastrophic outcome can outweigh many good ones
3. **Operators need predictability** — high variance in a deployed system erodes trust, even if mean performance is good

Most benchmark environments (OpenAI Gym, Atari) are explicitly *not* in this regime — they average over thousands of episodes, which washes out variance. This is probably why risk-sensitive RL remains underexplored despite being practically important.

---

*If you're curious about the theory, the VPAC-RS paper goes into the full ODE convergence proof and two-timescale analysis. The SA-QL paper covers the FCLT-based inference framework that the variance estimator is borrowed from.*
