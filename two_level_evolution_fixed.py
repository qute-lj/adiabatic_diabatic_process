#!/usr/bin/env python3
"""
二能级系统时间演化算法 (英文版)

实现了二能级系统在含时哈密顿量下的量子动力学演化，
用于研究绝热到非绝热转变的过程。

哈密顿量形式:
    H(t) = [[α*t, V],
            [V, -α*t]]

其中:
- α: 线性扫描速率
- V: 耦合强度(恒定)
- t: 时间

作者: ZCF Workflow
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from typing import Tuple, Callable, Optional
import argparse


class TwoLevelSystem:
    """
    二能级系统时间演化模拟器

    实现了基于薛定谔方程的时间演化算法，
    支持多种数值求解方法和分析工具。
    """

    def __init__(self, alpha: float = 1.0, V: float = 1.0):
        """
        初始化二能级系统

        参数:
            alpha (float): 线性扫描速率
            V (float): 耦合强度
        """
        self.alpha = alpha
        self.V = V
        self.hbar = 1.0  # 设置ℏ=1，自然单位制

    def hamiltonian(self, t: float) -> np.ndarray:
        """
        构造时间t时的哈密顿量矩阵

        H(t) = [[α*t, V],
                [V, -α*t]]

        参数:
            t (float): 时间

        返回:
            np.ndarray: 2×2哈密顿量矩阵
        """
        H = np.array([[self.alpha * t, self.V],
                      [self.V, -self.alpha * t]], dtype=complex)
        return H

    def schrodinger_equation(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        薛定谔方程的时间演化导数
        iℏ d|ψ⟩/dt = H(t)|ψ⟩

        参数:
            t (float): 时间
            state (np.ndarray): 当前量子态 [c1, c2]

        返回:
            np.ndarray: 状态的时间导数
        """
        H = self.hamiltonian(t)
        dstate_dt = -1j / self.hbar * (H @ state)
        return dstate_dt

    def evolve_runge_kutta_4(self, initial_state: np.ndarray,
                           time_points: np.ndarray) -> np.ndarray:
        """
        使用4阶Runge-Kutta方法进行时间演化

        参数:
            initial_state (np.ndarray): 初始量子态
            time_points (np.ndarray): 时间点数组

        返回:
            np.ndarray: 演化后的量子态序列
        """
        def rk4_step(t: float, state: np.ndarray, dt: float) -> np.ndarray:
            """单步4阶Runge-Kutta演化"""
            k1 = self.schrodinger_equation(t, state)
            k2 = self.schrodinger_equation(t + dt/2, state + dt*k1/2)
            k3 = self.schrodinger_equation(t + dt/2, state + dt*k2/2)
            k4 = self.schrodinger_equation(t + dt, state + dt*k3)

            return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        states = np.zeros((len(time_points), 2), dtype=complex)
        states[0] = initial_state

        for i in range(1, len(time_points)):
            dt = time_points[i] - time_points[i-1]
            states[i] = rk4_step(time_points[i-1], states[i-1], dt)

        return states

    def evolve_scipy(self, initial_state: np.ndarray,
                    time_points: np.ndarray,
                    method: str = 'RK45') -> np.ndarray:
        """
        使用SciPy的ODE求解器进行时间演化

        参数:
            initial_state (np.ndarray): 初始量子态
            time_points (np.ndarray): 时间点数组
            method (str): 求解方法 ('RK45', 'BDF', 'LSODA')

        返回:
            np.ndarray: 演化后的量子态序列
        """
        # 定义实数形式的ODE方程（SciPy odeint偏好实数）
        def real_ode(t, y_real):
            # 将实数数组转换为复数量子态
            state = y_real[:2] + 1j * y_real[2:]
            dstate = self.schrodinger_equation(t, state)
            # 返回实部和虚部
            return np.concatenate([np.real(dstate), np.imag(dstate)])

        # 初始条件转为实数形式
        y0_real = np.concatenate([np.real(initial_state), np.imag(initial_state)])

        # 求解ODE
        solution = solve_ivp(real_ode,
                           [time_points[0], time_points[-1]],
                           y0_real,
                           t_eval=time_points,
                           method=method,
                           rtol=1e-10, atol=1e-12)

        # 转换回复数形式
        states_real = solution.y[:2, :].T
        states_imag = solution.y[2:, :].T
        states = states_real + 1j * states_imag

        return states

    def evolve(self, initial_state: np.ndarray,
              time_points: np.ndarray,
              method: str = 'rk4') -> np.ndarray:
        """
        统一的时间演化接口

        参数:
            initial_state (np.ndarray): 初始量子态
            time_points (np.ndarray): 时间点数组
            method (str): 演化方法 ('rk4', 'scipy')

        返回:
            np.ndarray: 演化后的量子态序列
        """
        if method == 'rk4':
            return self.evolve_runge_kutta_4(initial_state, time_points)
        elif method == 'scipy':
            return self.evolve_scipy(initial_state, time_points)
        else:
            raise ValueError(f"未知的演化方法: {method}")

    def get_probabilities(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算各能级的布居概率

        参数:
            states (np.ndarray): 量子态序列

        返回:
            Tuple[np.ndarray, np.ndarray]: (能级1概率, 能级2概率)
        """
        prob1 = np.abs(states[:, 0])**2
        prob2 = np.abs(states[:, 1])**2
        return prob1, prob2

    def get_adiabatic_states(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算时刻t的绝热基矢

        参数:
            t (float): 时间

        返回:
            Tuple[np.ndarray, np.ndarray]: (基态绝热态, 激发态绝热态)
        """
        H = self.hamiltonian(t)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        ground_state = eigenvectors[:, 0]  # 基态
        excited_state = eigenvectors[:, 1]  # 激发态

        return ground_state, excited_state

    def adiabatic_parameter(self, t: float) -> float:
        """
        计算绝热参数
        γ(t) = |⟨e(t)|dH/dt|g(t)⟩| / (ΔE(t))²

        参数:
            t (float): 时间

        返回:
            float: 绝热参数
        """
        # 计算能级差
        delta_E = 2 * np.sqrt((self.alpha * t)**2 + self.V**2)

        # 计算矩阵元
        dH_dt = np.array([[self.alpha, 0], [0, -self.alpha]])
        ground, excited = self.get_adiabatic_states(t)

        matrix_element = np.abs(np.conj(excited) @ dH_dt @ ground)

        return matrix_element / (delta_E**2)


def calculate_landaue_zener_probability(alpha: float, V: float) -> float:
    """
    计算Landau-Zener跃迁概率

    P_LZ = exp(-2πV²/α)

    参数:
        alpha (float): 扫描速率
        V (float): 耦合强度

    返回:
        float: Landau-Zener跃迁概率
    """
    return np.exp(-2 * np.pi * V**2 / alpha)


def main():
    """主函数：演示二能级系统的时间演化"""
    parser = argparse.ArgumentParser(description='Two-Level System Quantum Evolution Simulation')
    parser.add_argument('--alpha', type=float, default=1.0, help='Linear sweep rate')
    parser.add_argument('--V', type=float, default=1.0, help='Coupling strength')
    parser.add_argument('--t_max', type=float, default=10.0, help='Maximum time')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of time steps')
    parser.add_argument('--method', type=str, default='rk4',
                       choices=['rk4', 'scipy'], help='Evolution method')

    args = parser.parse_args()

    # 创建二能级系统
    system = TwoLevelSystem(alpha=args.alpha, V=args.V)

    # 设置时间网格
    time_points = np.linspace(-args.t_max, args.t_max, args.n_steps)

    # 设置初始状态（基态）
    initial_state = np.array([1.0, 0.0], dtype=complex)

    print(f"Starting Two-Level System Evolution Simulation...")
    print(f"Parameters: α={args.alpha}, V={args.V}, Method={args.method}")

    # 进行时间演化
    states = system.evolve(initial_state, time_points, method=args.method)

    # 计算概率
    prob1, prob2 = system.get_probabilities(states)

    # 计算绝热参数
    adiabatic_params = np.array([system.adiabatic_parameter(t) for t in time_points])

    # 计算Landau-Zener概率
    lz_probability = calculate_landaue_zener_probability(args.alpha, args.V)

    # 绘制结果
    plt.figure(figsize=(15, 10))

    # 布居概率演化
    plt.subplot(2, 2, 1)
    plt.plot(time_points, prob1, label='|c₁|² (Level 1)', linewidth=2)
    plt.plot(time_points, prob2, label='|c₂|² (Level 2)', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Population Probability')
    plt.title('Level Population Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绝热参数
    plt.subplot(2, 2, 2)
    plt.plot(time_points, adiabatic_params, 'r-', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Adiabatic Parameter γ(t)')
    plt.title('Adiabatic Parameter Evolution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    # 能级差
    plt.subplot(2, 2, 3)
    energy_gaps = 2 * np.sqrt((args.alpha * time_points)**2 + args.V**2)
    plt.plot(time_points, energy_gaps, 'g-', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Energy Gap ΔE')
    plt.title('Two-Level Gap Evolution')
    plt.grid(True, alpha=0.3)

    # 相位演化（可选）
    plt.subplot(2, 2, 4)
    phases1 = np.angle(states[:, 0])
    phases2 = np.angle(states[:, 1])
    plt.plot(time_points, phases1, label='φ₁', linewidth=2)
    plt.plot(time_points, phases2, label='φ₂', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Phase')
    plt.title('Wavefunction Phase Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('two_level_evolution_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出统计信息
    final_prob2 = prob2[-1]
    max_adiabatic_param = np.max(adiabatic_params)

    print(f"\n=== Simulation Results ===")
    print(f"Final excited state population: {final_prob2:.4f}")
    print(f"Landau-Zener transition probability: {lz_probability:.4f}")
    print(f"Maximum adiabatic parameter: {max_adiabatic_param:.4f}")
    print(f"Adiabatic approximation {'valid' if max_adiabatic_param < 0.1 else 'may fail'}")

    # 保存数据
    np.savez('simulation_data_fixed.npz',
             time_points=time_points,
             states=states,
             prob1=prob1,
             prob2=prob2,
             adiabatic_params=adiabatic_params,
             alpha=args.alpha,
             V=args.V,
             lz_probability=lz_probability)

    print(f"Data saved to simulation_data_fixed.npz")


if __name__ == "__main__":
    main()