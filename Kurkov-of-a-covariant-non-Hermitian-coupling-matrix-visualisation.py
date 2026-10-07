#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ФЁДОРОВ–ХОПФ: КАУЗАЛЬНАЯ СИМУЛЯЦИЯ И СОХРАНЕНИЕ МЕТРИКИ КРЕЙНА
Академический расчетный комплекс с выводом физического паспорта
================================================================================
"""

import sys
import time
import numpy as np
import mpmath as mp
import sympy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

# ==============================================================================
# 1. ВВОД ПАРАМЕТРОВ СИСТЕМЫ
# ==============================================================================

def get_user_params():
    """Чтение параметров матрицы N и времени T из консоли или аргументов."""
    default_N = 32
    default_T = 40.0

    if len(sys.argv) >= 3:
        try:
            n_val = int(sys.argv[1])
            t_val = float(sys.argv[2])
            return n_val, t_val
        except ValueError:
            pass

    print("=" * 80)
    print("  ПАРАМЕТРЫ КВАНТОВОЙ СИСТЕМЫ ФЁДОРОВА–ХОПФА")
    print("=" * 80)
    try:
        user_n = input(f"Введите размер матрицы N [по умолчанию {default_N}]: ").strip()
        n_val = int(user_n) if user_n else default_N
    except (ValueError, EOFError):
        n_val = default_N

    try:
        user_t = input(f"Введите максимальное время T [по умолчанию {default_T}]: ").strip()
        t_val = float(user_t) if user_t else default_T
    except (ValueError, EOFError):
        t_val = default_T

    return max(4, n_val), max(1.0, t_val)

N_SPINS, MAX_TIME = get_user_params()

# ==============================================================================
# 2. РАСЧЕТ И ВЫВОД ФУНДАМЕНТАЛЬНЫХ КОНСТАНТ
# ==============================================================================

RHO_STR = '3.523019107092335542748196400366218388934326396843662720745233990171677478324985987702197964937828374'
needed_n = 2 * N_SPINS
dps_calc = int(0.5618 * needed_n) + 80
mp.mp.dps = dps_calc

rho = mp.mpf(RHO_STR)
Z_rho = (1 - rho**2) / (1 + rho**2)
k2 = 1 / rho
K_ell = mp.ellipk(k2)
E_ell = mp.ellipe(k2)
K_prime = mp.ellipk(1 - k2)
y = K_prime / K_ell
q_mod = mp.exp(-2 * mp.pi * y)
S2 = (K_ell**2 - E_ell**2) / rho * (1 - 10*q_mod - 196*q_mod**2 + 696*q_mod**3)
K_spec = (1 - rho**3 + mp.sqrt((rho**3 - 1)**2 + 4 * S2 * rho**3 * (rho - 1))) / 2
Q_spec = K_spec * (1 - K_spec)
kappa_0 = 1 / mp.sqrt(1 - rho**(-2))
kappa_1 = kappa_0 * mp.mpf(5) / 9
c_eff = 1.5

print("\n" + "=" * 80)
print("             ПАСПОРТ ФУНДАМЕНТАЛЬНЫХ КОНСТАНТ СИСТЕМЫ")
print("=" * 80)
print(f"  • Спектральный масштаб (ρ):        {mp.nstr(rho, 25)}")
print(f"  • Параллель тора Клиффорда (Z_ρ):  {float(Z_rho):.10f}")
print(f"  • Модулярный параметр Якоби (q):   {float(q_mod):.6e}")
print(f"  • Функция замыкания Стилтьеса (S₂): {float(S2):.10f}")
print(f"  • Спектральная амплитуда (K):      {float(K_spec):.10f}")
print(f"  • Калибровочный инвариант (Q):     {float(Q_spec):.10f}")
print(f"  • Редукционный фактор Хаара (κ₀):  {float(kappa_0):.8f}")
print(f"  • Проекционный фактор связи (κ₁):  {float(kappa_1):.8f}")
print(f"  • Предел Либа–Робинсона (c_eff):   {c_eff:.2f}")
print(f"  • Рабочая разрядность (dps):       {dps_calc} знаков (для 2N = {needed_n})")
print("=" * 80)

# ==============================================================================
# 3. ПОСТРОЕНИЕ ОПЕРАТОРА J_N И КРЕЙН-СИГНАТУРЫ
# ==============================================================================

def get_true_primes(count):
    return list(sp.primerange(2, sp.prime(count) + 1))

def compute_c_moments(primes):
    M = len(primes)
    c = [0] * (M + 1)
    c[1] = int(primes[0])
    for n in range(2, M + 1):
        s = sum(int(primes[k - 1]) * c[n - k] for k in range(1, n))
        c[n] = int(primes[n - 1]) + s
    return c

primes = get_true_primes(needed_n)
c_int = compute_c_moments(primes)
moments = [mp.mpf(1)] + [mp.mpf(c_int[i]) / (rho**i) for i in range(1, needed_n + 1)]

M_order = len(moments) - 1
sigma = {(-1, n): mp.mpf(0) for n in range(M_order + 1)}
for n in range(M_order + 1):
    sigma[(0, n)] = moments[n]

a_list, b_sq_list = [], []
for k in range(N_SPINS):
    if k > 0:
        b_sq_list.append(sigma[(k, k)] / sigma[(k - 1, k - 1)])
    term1 = sigma[(k, k + 1)] / sigma[(k, k)]
    term2 = sigma[(k - 1, k)] / sigma[(k - 1, k - 1)] if k > 0 else mp.mpf(0)
    ak = term1 - term2
    a_list.append(ak)
    for n in range(k + 1, M_order - k):
        prev_b2 = b_sq_list[-1] if k > 0 else mp.mpf(0)
        sigma[(k + 1, n)] = sigma[(k, n + 1)] - ak * sigma[(k, n)] - prev_b2 * sigma[(k - 1, n)]

J_mat = np.zeros((N_SPINS, N_SPINS), dtype=np.complex128)
eta_diag = np.ones(N_SPINS, dtype=np.float64)
current_sign = 1.0

for k in range(N_SPINS):
    J_mat[k, k] = complex(a_list[k])
    if k < N_SPINS - 1:
        val = float(b_sq_list[k])
        b_k = np.sqrt(val) if val >= 0 else 1j * np.sqrt(-val)
        J_mat[k, k + 1] = b_k
        J_mat[k + 1, k] = b_k
        if val < 0:
            current_sign *= -1.0
        eta_diag[k + 1] = current_sign

eta_mat = np.diag(eta_diag)

print("\n" + "=" * 80)
print(f"         СТРУКТУРА УРОВНЕЙ МАТРИЦЫ J_{N_SPINS}x{N_SPINS} (ФЁДОРОВ–ХОПФ)")
print("=" * 80)
print(f" {'k':>3} | {'Диагональ a_k':>15} | {'Поддиагональ b_k²':>18} | {'Тип связи':>10} | {'Сигнатура η_k':>14}")
print(" " + "─" * 70)

for k in range(N_SPINS):
    ak_str = f"{float(a_list[k]):15.8f}"
    if k < N_SPINS - 1:
        bk2_val = float(b_sq_list[k])
        bk2_str = f"{bk2_val:18.8f}"
        bond_type = "Крейн (i)" if bk2_val < 0 else "Евклид"
    else:
        bk2_str = f"{'---':>18}"
        bond_type = "---"
    eta_str = f"{'+1' if eta_diag[k] > 0 else '-1':>14}"
    print(f" {k:3d} | {ak_str} | {bk2_str} | {bond_type:>10} | {eta_str}")

print(" " + "─" * 70)

# ==============================================================================
# 4. ПОШАГОВАЯ СИМУЛЯЦИЯ И ВЫВОД ВРЕМЕННЫХ ИТЕРАЦИЙ
# ==============================================================================

TIME_STEPS = 1000
time_grid = np.linspace(0, MAX_TIME, TIME_STEPS)

Psi_0 = np.zeros(N_SPINS, dtype=np.complex128)
Psi_0[0] = 1.0

density_matrix = np.zeros((N_SPINS, TIME_STEPS))
euclid_norm = np.zeros(TIME_STEPS)
krein_norm = np.zeros(TIME_STEPS)

print("\n" + "=" * 80)
print(f"   ДИНАМИЧЕСКАЯ ЭВОЛЮЦИЯ СИСТЕМЫ ВО ВРЕМЕНИ t ∈ [0, {MAX_TIME}]")
print("=" * 80)
print(f" {'Шаг':>5} | {'Время t':>8} | {'Евклид. норма ||Ψ||²':>22} | {'Крейн-норма <Ψ|η|Ψ>':>20} | {'Ведущий уровень k_max':>21}")
print(" " + "─" * 84)

# Определяем 20 контрольных точек времени для вывода в консоль
print_interval = max(1, TIME_STEPS // 20)

for t_idx, t in enumerate(time_grid):
    Psi_t = la.expm(-1j * J_mat * t) @ Psi_0
    prob_dist = np.abs(Psi_t)**2
    density_matrix[:, t_idx] = prob_dist
    
    e_norm = float(np.sum(prob_dist))
    k_norm = float(np.real(Psi_t.conj().T @ eta_mat @ Psi_t))
    
    euclid_norm[t_idx] = e_norm
    krein_norm[t_idx] = k_norm

    if t_idx % print_interval == 0 or t_idx == TIME_STEPS - 1:
        dominant_k = int(np.argmax(prob_dist))
        max_prob = float(prob_dist[dominant_k])
        print(f" {t_idx:5d} | {t:8.2f} | {e_norm:22.4e} | {k_norm:20.10f} | {dominant_k:>8d} (P = {max_prob:.3f})")

print(" " + "─" * 84)

# ==============================================================================
# 5. ГРАФИЧЕСКИЙ КОМПЛЕКС И СОХРАНЕНИЕ
# ==============================================================================

print("\n[*] Построение двухпанельного академического графика...")
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2, 1]})

# Верхняя панель: тепловая карта
im = ax1.imshow(density_matrix, aspect='auto', origin='lower', cmap='inferno',
                extent=[0, MAX_TIME, 0, N_SPINS - 1], vmax=0.3)
ax1.set_title(r'$|\Psi_k(t)|^2 = \left| \langle k | \exp(-i \hat{J}_N t) | 0 \rangle \right|^2$', fontsize=15, pad=12)
ax1.set_ylabel('Спиновые гармоники $k$', fontsize=11)

# Световой конус Либа–Робинсона
ax1.plot(time_grid, np.clip(c_eff * time_grid, 0, N_SPINS - 1), color='cyan', linestyle='--', linewidth=1.8, 
         label=rf'Конус Либа--Робинсона ($c_{{\mathrm{{eff}}}} = {c_eff}$)')
ax1.legend(loc='upper left', fontsize=10, framealpha=0.3)
for s in ax1.spines.values(): 
    s.set_visible(False)

cbar = fig.colorbar(im, ax=ax1, pad=0.01)
cbar.outline.set_visible(False)
cbar.set_label('Фазовая плотность', fontsize=10)

# Нижняя панель: нормы состояния
ax2.plot(time_grid, euclid_norm, color='magenta', linewidth=1.8, 
         label=r'Евклидова норма $\|\Psi(t)\|^2$ (некомпактная группа $SO(1,3)$)')
ax2.plot(time_grid, krein_norm, color='lime', linestyle='--', linewidth=2.0, 
         label=r'Инвариант Крейна $\langle\Psi(t)|\eta|\Psi(t)\rangle \equiv 1$ (сохранение)')
ax2.set_xlabel('Время $t$', fontsize=11)
ax2.set_ylabel('Норма состояния', fontsize=11)
ax2.set_ylim(-0.2, max(euclid_norm) * 1.1)
ax2.legend(loc='upper left', fontsize=10, framealpha=0.3)
ax2.grid(True, linestyle=':', alpha=0.3)
for s in ax2.spines.values(): 
    s.set_visible(False)

plt.tight_layout()

# Сохранение и интерактивный показ
output_img = 'causal_cone.png'
plt.savefig(output_img, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
print(f"[✓] График сохранен в файл: {output_img}")
print("[*] Открытие графического окна...")
plt.show()
print("=" * 80)
print("  РАСЧЕТ И СИМУЛЯЦИЯ ПОЛНОСТЬЮ ЗАВЕРШЕНЫ")
print("=" * 80)
