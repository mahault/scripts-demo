"""Experimental framework for stochastic multi-trial evaluation.

Provides:
- episode_generator: Synthetic PerceptBundle generation (no Webots dependency)
- stochastic_runner: Multi-trial experiment runner with noise injection
- statistical_analysis: Bootstrap CIs, permutation tests, effect sizes
- baselines: Four comparison baselines (lookup, no-gating, random, uniform)
"""
