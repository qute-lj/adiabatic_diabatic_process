"""
二维狄拉克点含时演化系统

实现了基于二维波矢绝热参数的狄拉克点哈密顿量演化，
用于研究Berry相位和几何量子效应。

使用方法:
    from dirac_system import DiracSystem

    # 创建狄拉克系统
    system = DiracSystem(k_center=(0.0, 0.0), radius=0.1, alpha=0.1)

    # 执行时间演化
    system.evolve(dt=0.01)

    # 获取Berry相位
    berry_phase = system.compute_berry_phase()
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List, Optional, Union
import warnings

# 设置物理常数
HBAR = 1.0  # 约化普朗克常数（自然单位制）
VF = 1.0    # 费米速度


class DiracSystem:
    """
    二维狄拉克点系统（带质量项）

    实现H(k) = VF * (kx * sigma_x + ky * sigma_y) + V * sigma_z的哈密顿量，
    其中k = (kx, ky)是二维波矢，V是质量项（打开能隙）。

    物理背景：
    - 描述二维材料中的狄拉克点（如石墨烯）
    - V=0时在k=0处能带简并，呈线性色散
    - V≠0时打开能隙，Berry相位随V变化
    """

    def __init__(self,
                 k_center: Tuple[float, float] = (0.0, 0.0),
                 radius: float = 0.1,
                 alpha: float = 0.1,
                 V: float = 0.0,  # 质量项
                 n_points: int = 1000):
        """
        初始化狄拉克系统

        参数:
            k_center: 绕圈中心 (kx0, ky0)
            radius: 绕圈半径
            alpha: 角速度，控制绕圈速度
            V: 质量项（能隙参数）
            n_points: 时间演化点数
        """
        self.k_center = np.array(k_center, dtype=float)
        self.radius = float(radius)
        self.alpha = float(alpha)  # 角频率
        self.V = float(V)  # 质量项
        self.n_points = int(n_points)

        # 时间数组（完整一圈）
        self.t_max = 2 * np.pi / self.alpha if self.alpha > 0 else 10.0
        self.dt = self.t_max / self.n_points
        self.time = np.linspace(0, self.t_max, self.n_points)

        # 演化数据存储
        self.trajectory = []        # k(t)轨迹
        self.eigenvalues = []       # 本征值历史
        self.eigenvectors = []      # 本征矢历史
        self.state_evolution = []   # 态演化历史
        self.berry_phase = None     # Berry相位
        self.adiabatic_params = []  # 绝热参数历史

        # 泡利矩阵
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        # 初始化系统
        self._initialize()

    def _initialize(self):
        """初始化系统状态"""
        # 计算k(t)轨迹
        for t in self.time:
            kx, ky = self.k_trajectory(t)
            self.trajectory.append([kx, ky])

        self.trajectory = np.array(self.trajectory)

    def k_trajectory(self, t: float) -> Tuple[float, float]:
        """
        计算时刻t的波矢位置

        k(t) = k_center + radius * (cos(alpha*t), sin(alpha*t))

        参数:
            t: 时间

        返回:
            (kx, ky): 波矢坐标
        """
        kx = self.k_center[0] + self.radius * np.cos(self.alpha * t)
        ky = self.k_center[1] + self.radius * np.sin(self.alpha * t)
        return kx, ky

    def get_hamiltonian(self, kx: float, ky: float) -> np.ndarray:
        """
        获取狄拉克哈密顿量（带质量项）

        H(k) = VF * (kx * sigma_x + ky * sigma_y) + V * sigma_z

        参数:
            kx: x方向波矢
            ky: y方向波矢

        返回:
            2x2 复数矩阵
        """
        H = VF * (kx * self.sigma_x + ky * self.sigma_y) + self.V * self.sigma_z
        return H

    def get_eigenvalues(self, kx: float, ky: float) -> np.ndarray:
        """
        计算本征值（带质量项）

        E± = ± sqrt(VF²(kx² + ky²) + V²)

        参数:
            kx: x方向波矢
            ky: y方向波矢

        返回:
            [E-, E+] 本征值数组（从低到高）
        """
        E = np.sqrt((VF * (kx**2 + ky**2)) + self.V**2)
        return np.array([-E, E])

    def get_eigenvectors(self, kx: float, ky: float) -> np.ndarray:
        """
        计算本征矢

        参数:
            kx: x方向波矢
            ky: y方向波矢

        返回:
            2x2 复数矩阵，每列是一个本征矢
        """
        H = self.get_hamiltonian(kx, ky)
        _, eigvecs = eigh(H)
        return eigvecs

    def compute_adiabatic_parameter(self, t: float) -> float:
        """
        计算绝热参数 γ(t)

        γ(t) = |⟨m|∂ₜH|n⟩| / |E_m - E_n|²

        参数:
            t: 时间

        返回:
            绝热参数值
        """
        # 获取当前k(t)和dk/dt
        kx, ky = self.k_trajectory(t)
        dkx_dt = -self.radius * self.alpha * np.sin(self.alpha * t)
        dky_dt = self.radius * self.alpha * np.cos(self.alpha * t)

        # 获取本征矢和本征值
        eigvals, eigvecs = eigh(self.get_hamiltonian(kx, ky))
        E0, E1 = eigvals  # E0 < E1
        v0, v1 = eigvecs.T  # 转置得到行向量

        # 计算 ∂ₜH = VF * (dkx/dt * σx + dky/dt * σy)
        dH_dt = VF * (dkx_dt * self.sigma_x + dky_dt * self.sigma_y)

        # 计算矩阵元 |⟨0|∂ₜH|1⟩|
        matrix_element = np.abs(np.vdot(v0, dH_dt @ v1))

        # 计算绝热参数
        delta_E = E1 - E0
        if delta_E > 1e-10:
            gamma = matrix_element / (delta_E**2)
        else:
            gamma = np.inf

        return gamma

    def evolve(self, dt: Optional[float] = None, method: str = 'rk4'):
        """
        执行时间演化

        参数:
            dt: 时间步长（如果提供，覆盖默认值）
            method: 数值积分方法 ('rk4' 或 'scipy')
        """
        if dt is not None:
            self.dt = dt
            self.n_points = int(self.t_max / dt) + 1
            self.time = np.linspace(0, self.t_max, self.n_points)

        # 清空之前的数据
        self.eigenvalues = []
        self.eigenvectors = []
        self.state_evolution = []
        self.adiabatic_params = []

        # 获取初始态（基态）
        kx0, ky0 = self.k_trajectory(0)
        _, eigvecs0 = eigh(self.get_hamiltonian(kx0, ky0))
        current_state = eigvecs0[:, 0]  # 基态

        # 时间演化
        for i, t in enumerate(self.time):
            kx, ky = self.k_trajectory(t)

            # 存储本征值和本征矢
            eigvals, eigvecs = eigh(self.get_hamiltonian(kx, ky))
            self.eigenvalues.append(eigvals)
            self.eigenvectors.append(eigvecs)
            self.state_evolution.append(current_state.copy())

            # 计算绝热参数
            gamma = self.compute_adiabatic_parameter(t)
            self.adiabatic_params.append(gamma)

            # 如果不是最后一个点，演化到下一个时间步
            if i < len(self.time) - 1:
                if method == 'rk4':
                    current_state = self._rk4_step(current_state, t)
                else:
                    warnings.warn("Scipy ODE solver not implemented, using RK4")
                    current_state = self._rk4_step(current_state, t)

        # 转换为numpy数组
        self.eigenvalues = np.array(self.eigenvalues)
        self.eigenvectors = np.array(self.eigenvectors)
        self.state_evolution = np.array(self.state_evolution)
        self.adiabatic_params = np.array(self.adiabatic_params)

    def _rk4_step(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        RK4时间演化步骤

        |ψ(t+dt)⟩ = |ψ(t)⟩ - i/ℏ * dt * H(k(t)) * |ψ(t)⟩

        参数:
            state: 当前态矢量
            t: 当前时间

        返回:
            演化后的态矢量
        """
        kx, ky = self.k_trajectory(t)
        H = self.get_hamiltonian(kx, ky)

        # RK4方法（对于不含时系统，实际就是普通的时间演化）
        k1 = -1j / HBAR * H @ state
        k2 = -1j / HBAR * H @ (state + 0.5 * self.dt * k1)
        k3 = -1j / HBAR * H @ (state + 0.5 * self.dt * k2)
        k4 = -1j / HBAR * H @ (state + self.dt * k3)

        new_state = state + self.dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)

        # 归一化
        norm = np.linalg.norm(new_state)
        if norm > 1e-10:
            new_state = new_state / norm

        return new_state

    def compute_berry_phase(self) -> float:
        """
        计算Berry相位

        计算方法：将本征态投影到瞬时本征基，然后计算几何相位
        Berry相位 = ∮ i⟨n(t)|∂ₜn(t)⟩dt = 几何相位

        返回:
            Berry相位（弧度）
        """
        if len(self.eigenvectors) == 0:
            raise RuntimeError("需要先运行evolve()计算演化")

        # 获取连续的本征矢（基态）
        eigenvectors_phase_fixed = self._ensure_phase_continuity(
            self.eigenvectors[:, :, 0].copy()
        )

        # 计算每个时刻的几何相位
        geometric_phases = []

        for i, state in enumerate(self.state_evolution):
            # 投影到连续的本征态
            c_g = np.vdot(eigenvectors_phase_fixed[i], state)
            geometric_phases.append(np.angle(c_g))

        geometric_phases = np.array(geometric_phases)

        # 计算总几何相位变化
        # 使用相位展开来获得连续值
        geometric_phases_unwrapped = np.unwrap(geometric_phases)

        # Berry相位 = 终点 - 起点的相位变化
        berry_phase = geometric_phases_unwrapped[-1] - geometric_phases_unwrapped[0]

        # 归一化到 [-π, π]
        berry_phase = np.mod(berry_phase + np.pi, 2*np.pi) - np.pi

        # 存储结果
        self.berry_phase = berry_phase

        return berry_phase

    def _ensure_phase_continuity(self, eigenvectors):
        """
        确保本征矢的相位连续性
        """
        n = len(eigenvectors)
        for i in range(1, n):
            # 计算与前一时刻的重叠
            overlap = np.vdot(eigenvectors[i-1], eigenvectors[i])

            # 如果实部为负，调整相位
            if np.real(overlap) < 0:
                eigenvectors[i] *= -1

        return eigenvectors

    def get_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取时间演化过程中的系数分布

        返回:
            (|c₀|², |c₁|²): 基态和激发态的占据概率
        """
        if len(self.state_evolution) == 0:
            raise RuntimeError("需要先运行evolve()计算演化")

        # 将态矢量投影到瞬时本征态基矢
        c0_abs2 = []
        c1_abs2 = []

        for i, state in enumerate(self.state_evolution):
            eigvecs = self.eigenvectors[i]
            # cₙ = ⟨φₙ|ψ⟩
            c0 = np.vdot(eigvecs[:, 0], state)
            c1 = np.vdot(eigvecs[:, 1], state)
            c0_abs2.append(np.abs(c0)**2)
            c1_abs2.append(np.abs(c1)**2)

        return np.array(c0_abs2), np.array(c1_abs2)

    def get_phases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取时间演化过程中的相位

        返回:
            (arg(c₀), arg(c₁)): 基态和激发态的相位
        """
        if len(self.state_evolution) == 0:
            raise RuntimeError("需要先运行evolve()计算演化")

        phases_0 = []
        phases_1 = []

        for i, state in enumerate(self.state_evolution):
            eigvecs = self.eigenvectors[i]
            c0 = np.vdot(eigvecs[:, 0], state)
            c1 = np.vdot(eigvecs[:, 1], state)
            phases_0.append(np.angle(c0))
            phases_1.append(np.angle(c1))

        return np.array(phases_0), np.array(phases_1)