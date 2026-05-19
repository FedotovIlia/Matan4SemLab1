import numpy as np
import matplotlib.pyplot as plt

# МАТАН ЛАБА2

L = 3 * np.pi
PERIOD = 2 * L
def f_original(x):
    x = np.asarray(x)

    return np.where(
        (0 <= x) & (x < np.pi),
        1 - np.cos(x),
        2
    )
def f_even(x):
    x = ((x + L) % PERIOD) - L
    ax = np.abs(x)
    return np.where(
        ax < np.pi,
        1 - np.cos(ax),
        2
    )
def f_odd(x):

    x = ((x + L) % PERIOD) - L
    y = np.zeros_like(x)
    mask1 = (0 < x) & (x < np.pi)
    y[mask1] = 1 - np.cos(x[mask1])
    mask2 = (np.pi < x) & (x < 3 * np.pi)
    y[mask2] = 2
    mask3 = (-np.pi < x) & (x < 0)
    y[mask3] = -(1 - np.cos(x[mask3]))
    mask4 = (-3 * np.pi < x) & (x < -np.pi)
    y[mask4] = -2

    return y

def fourier_general_coeffs(N):

    x = np.linspace(-L, L, 50000)
    y = f_even(x)
    a0 = (1 / L) * np.trapezoid(y, x)
    an = np.zeros(N)
    bn = np.zeros(N)

    for n in range(1, N + 1):
        an[n - 1] = (1 / L) * np.trapezoid(
            y * np.cos(n * np.pi * x / L),
            x
        )
        bn[n - 1] = (1 / L) * np.trapezoid(
            y * np.sin(n * np.pi * x / L),
            x
        )

    return a0, an, bn

def cosine_coeffs(N):

    x = np.linspace(0, L, 50000)
    y = f_original(x)
    a0 = (2 / L) * np.trapezoid(y, x)
    an = np.zeros(N)
    for n in range(1, N + 1):
        an[n - 1] = (2 / L) * np.trapezoid(
            y * np.cos(n * np.pi * x / L),
            x
        )

    return a0, an

def sine_coeffs(N):

    x = np.linspace(0, L, 50000)
    y = f_original(x)
    bn = np.zeros(N)
    for n in range(1, N + 1):
        bn[n - 1] = (2 / L) * np.trapezoid(
            y * np.sin(n * np.pi * x / L),
            x
        )
    return bn

def partial_sum_general(x, N):
    a0, an, bn = fourier_general_coeffs(N)
    S = np.full_like(x, a0 / 2)
    for n in range(1, N + 1):
        S += (
            an[n - 1] * np.cos(n * np.pi * x / L)
            + bn[n - 1] * np.sin(n * np.pi * x / L)
        )

    return S

def partial_sum_cosine(x, N):
    a0, an = cosine_coeffs(N)
    S = np.full_like(x, a0 / 2)
    for n in range(1, N + 1):
        S += an[n - 1] * np.cos(n * np.pi * x / L)
    return S

def partial_sum_sine(x, N):
    bn = sine_coeffs(N)
    S = np.zeros_like(x)
    for n in range(1, N + 1):
        S += bn[n - 1] * np.sin(n * np.pi * x / L)
    return S

x_general = np.linspace(0, 7.5, 5000)
x_even_odd = np.linspace(-7.5, 7.5, 5000)
Ns = [3, 10, 30]

for N in Ns:
    plt.figure(figsize=(12, 5))
    plt.plot(
        x_general,
        f_even(x_general),
        linewidth=2,
        label='Исходная функция'
    )
    plt.plot(
        x_general,
        partial_sum_general(x_general, N),
        linewidth=2,
        label=f'Частичная сумма N={N}'
    )
    plt.title(f'Общий ряд Фурье, N={N}')
    plt.xlim(0, 7.5)
    plt.grid(True)
    plt.legend()

for N in Ns:
    plt.figure(figsize=(12, 5))
    plt.plot(
        x_even_odd,
        f_even(x_even_odd),
        linewidth=2,
        label='Четное продолжение'
    )
    plt.plot(
        x_even_odd,
        partial_sum_cosine(x_even_odd, N),
        linewidth=2,
        label=f'Частичная сумма N={N}'
    )
    plt.title(f'Косинусный ряд Фурье, N={N}')
    plt.xlim(-7.5, 7.5)
    plt.grid(True)
    plt.legend()

for N in Ns:
    plt.figure(figsize=(12, 5))
    plt.plot(
        x_even_odd,
        f_odd(x_even_odd),
        linewidth=2,
        label='Нечетное продолжение'
    )
    plt.plot(
        x_even_odd,
        partial_sum_sine(x_even_odd, N),
        linewidth=2,
        label=f'Частичная сумма N={N}'
    )
    plt.title(f'Синусный ряд Фурье, N={N}')
    plt.xlim(-7.5, 7.5)
    plt.grid(True)
    plt.legend()
plt.show()