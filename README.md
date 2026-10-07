# Causal Wave Kinematics in the "Vacuum of Prime Numbers"
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Academic Preprint](https://img.shields.io/badge/Status-Academic_Preprint-success.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the numerical simulation source code accompanying the theoretical physics paper:
**"Numerical synthesis and scaling algorithm of the covariant non-Hermitian coupling matrix on the Hopf bundle $S^3 \to S^2$"**
*(Author: Miroslav Aleksandrovich Kurkov).*

The code resolves a fundamental problem in relativistic quantum field theory formulated by F. I. Fedorov regarding the universal nonlinear equation for interacting fields. Historically, phenomenological attempts to describe the dynamics of high-spin particles ($s \ge 3/2$) in external fields led to the **Velo-Zwanziger paradox**: the emergence of tachyonic instabilities, loss of hyperbolicity, and the violation of microscopic causality.

This project introduces a fundamentally novel topological approach. The canonical coupling operator $\Lambda$ is rigorously extracted *directly from the sequence of natural prime numbers* using covariant geometry on the Hopf fibration and inverse moment algorithms.

## The Mathematical Framework

The script performs a "bottom-up" synthesis of the pseudo-Hermitian Hamiltonian $\hat{H} \equiv \hat{J}_N$:

1. **Prime-to-Moment Convolution:** 
   Extracting macroscopic integer moments $c_n \in \mathbb{Z}$ directly from the prime numbers $\{p_k\}$ without division or factoring:
   $$c_1 = p_1, \quad c_n = p_n + \sum_{k=1}^{n-1} p_k \cdot c_{n-k}$$
2. **Spectral Normalization:** 
   Scaling moments using the fundamental invariant of the Hopf torus $\rho \approx 3.523...$:
   $$m_n = c_n / \rho^n$$
3. **Stieltjes-Krein Inversion:** 
   Solving the inverse generalized Chebyshev-Markov problem to build a complex tridiagonal Jacobi matrix $J_N$. Because of the alternating sign of the Hankel determinants for higher harmonics, the squared off-diagonal elements $b_k^2$ naturally acquire negative values. This implies imaginary coupling constants and rigorously defines a non-Hermitian Krein space.

## Physical Simulation & Key Discoveries

The script `Kurkov-of-a-covariant-non-Hermitian-coupling-matrix-visualisation.py` numerically integrates the Schrödinger-type evolution of a wave packet in the spin-harmonic basis:
$$ i\partial_t |\Psi(t)\rangle = \hat{J}_N |\Psi(t)\rangle, \quad |\Psi(0)\rangle = |0\rangle $$

**Visual outputs explicitly prove:**
1. **Macroscopic Causality:** The operator $\hat{J}_N$ exhibits a strictly bounded spectral radius ($|\lambda| \le 1.0$). Consequently, the wave packet never breaches the effective theoretical light cone ($c_{\text{eff}}$). This computationally falsifies the Velo-Zwanziger tachyonic propagation for this specific operator space.
2. **Topological Staircase Effect:** The probability density propagates via discrete quasi-bound states (solitonic steps) corresponding to Maslov indices, rather than a continuous continuum.
3. **$\mathcal{PT}$-Symmetry Breaking:** Driven by the imaginary cross-couplings of the Krein metric, the system undergoes a spontaneous $\mathcal{PT}$-symmetry breaking transition at higher modes, visualized as a macroscopic exponential pulsation of the Euclidean state norm.

## Installation and Requirements

To run the simulation, you need Python 3.8+ and standard scientific computing libraries:

```bash
pip install numpy scipy matplotlib sympy mpmath

To run the simulation, you need Python 3.8+ and standard scientific computing libraries:

```bash
pip install numpy scipy matplotlib sympy mpmath

