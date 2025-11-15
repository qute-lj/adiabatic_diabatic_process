"""
狄拉克点系统可视化工具

提供3D能带图、k空间轨迹图和综合分析图表的绘制功能。

使用方法:
    from visualization_tools import Visualizer
    viz = Visualizer(system)
    viz.create_comprehensive_four_panel()
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List, Union
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Use default fonts for better compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'


class Visualizer:
    """狄拉克点系统可视化器"""

    def __init__(self, system, figsize: Tuple[float, float] = (15, 10)):
        """
        初始化可视化器

        参数:
            system: DiracSystem实例
            figsize: 图形大小
        """
        self.system = system
        self.figsize = figsize

        # 颜色配置
        self.colors = {
            'ground': '#2E86AB',      # 基态 - 蓝色
            'excited': '#A23B72',     # 激发态 - 紫红色
            'trajectory': '#F18F01',  # 轨迹 - 橙色
            'dirac': '#C73E1D'        # 狄拉克点 - 红色
        }

    def plot_3d_band_structure(self, ax: Optional[plt.Axes] = None,
                              k_range: float = 0.3,
                              resolution: int = 50):
        """
        绘制3D能带结构

        参数:
            ax: matplotlib轴对象
            k_range: k空间范围
            resolution: 网格分辨率
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

        # 创建k空间网格
        kx = np.linspace(-k_range, k_range, resolution)
        ky = np.linspace(-k_range, k_range, resolution)
        KX, KY = np.meshgrid(kx, ky)

        # 计算能带 E± = ±√(kx² + ky²)
        E_plus = np.sqrt(KX**2 + KY**2)
        E_minus = -E_plus

        # 绘制能带表面
        surf1 = ax.plot_surface(KX, KY, E_plus, alpha=0.7,
                               cmap='viridis', label='Conduction Band')
        surf2 = ax.plot_surface(KX, KY, E_minus, alpha=0.7,
                               cmap='plasma', label='Valence Band')

        # 标记狄拉克点
        ax.scatter([0], [0], [0], color=self.colors['dirac'],
                  s=100, marker='o', label='Dirac Point', zorder=5)

        # 绘制演化路径（如果存在）
        if hasattr(self.system, 'trajectory') and len(self.system.trajectory) > 0:
            traj = self.system.trajectory
            # 计算路径上的能级
            E_traj = []
            for kx_t, ky_t in traj:
                E = self.system.get_eigenvalues(kx_t, ky_t)
                E_traj.append(E[1])  # 导带能量
            E_traj = np.array(E_traj)

            ax.plot(traj[:, 0], traj[:, 1], E_traj,
                   color=self.colors['trajectory'], linewidth=3,
                   label='Evolution Path', zorder=4)

        # 设置标签和标题
        ax.set_xlabel('$k_x$', fontsize=14)
        ax.set_ylabel('$k_y$', fontsize=14)
        ax.set_zlabel(r'$\mathbf{Energy}\ (E/v_F)$', fontsize=14)
        ax.set_title(r'$\mathbf{Dirac\ Point\ 3D\ Band\ Structure}$', fontsize=16, pad=20)

        # 设置视角
        ax.view_init(elev=30, azim=45)

        # 添加图例
        ax.legend(loc='upper left', fontsize=12)

        # 设置网格
        ax.grid(True, alpha=0.3)

        return ax

    def plot_k_trajectory(self, ax: Optional[plt.Axes] = None,
                         show_energy_contours: bool = True,
                         k_range: float = 0.3):
        """
        绘制k空间轨迹图

        参数:
            ax: matplotlib轴对象
            show_energy_contours: 是否显示能量等高线
            k_range: 显示范围
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制能量等高线
        if show_energy_contours:
            kx = np.linspace(-k_range, k_range, 200)
            ky = np.linspace(-k_range, k_range, 200)
            KX, KY = np.meshgrid(kx, ky)
            E = np.sqrt(KX**2 + KY**2)

            contours = ax.contour(KX, KY, E, levels=10,
                                 colors='gray', alpha=0.4, linewidths=0.5)
            ax.clabel(contours, inline=True, fontsize=8)

        # 绘制演化轨迹
        if hasattr(self.system, 'trajectory') and len(self.system.trajectory) > 0:
            traj = self.system.trajectory
            # 绘制完整路径
            ax.plot(traj[:, 0], traj[:, 1],
                   color=self.colors['trajectory'], linewidth=2,
                   label='演化路径', zorder=3)

            # 标记起点和终点
            ax.scatter(traj[0, 0], traj[1, 0],
                      color='green', s=100, marker='o',
                      label='起点', zorder=4, edgecolor='black', linewidth=1)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                      color='red', s=100, marker='s',
                      label='终点', zorder=4, edgecolor='black', linewidth=1)

            # 添加箭头表示方向
            n_arrows = 10
            arrow_indices = np.linspace(0, len(traj)-2, n_arrows, dtype=int)
            for idx in arrow_indices:
                dx = traj[idx+1, 0] - traj[idx, 0]
                dy = traj[idx+1, 1] - traj[idx, 1]
                ax.arrow(traj[idx, 0], traj[idx, 1],
                        dx*5, dy*5,  # 放大箭头以便观察
                        head_width=0.01, head_length=0.01,
                        fc=self.colors['trajectory'],
                        ec=self.colors['trajectory'],
                        alpha=0.7, zorder=2)

        # 标记狄拉克点
        ax.scatter([0], [0], color=self.colors['dirac'],
                  s=200, marker='*', label='狄拉克点',
                  zorder=5, edgecolor='black', linewidth=1)

        # 标记绕圈中心
        ax.scatter([self.system.k_center[0]], [self.system.k_center[1]],
                  color='black', s=50, marker='+',
                  label=f'中心 {self.system.k_center}',
                  zorder=5, linewidth=2)

        # 绘制绕圈圆
        circle = Circle(self.system.k_center, self.system.radius,
                       fill=False, edgecolor='gray',
                       linestyle='--', linewidth=1,
                       label=f'半径 r={self.system.radius}')
        ax.add_patch(circle)

        # 设置标签和标题
        ax.set_xlabel('$k_x$', fontsize=14)
        ax.set_ylabel('$k_y$', fontsize=14)
        ax.set_title('k空间演化轨迹', fontsize=16)
        ax.set_aspect('equal')

        # 设置范围
        ax.set_xlim(-k_range, k_range)
        ax.set_ylim(-k_range, k_range)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        return ax

    def plot_coefficients(self, ax: Optional[plt.Axes] = None):
        """
        绘制系数分布 |cₙ|²

        参数:
            ax: matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # 获取系数
        try:
            c0_abs2, c1_abs2 = self.system.get_coefficients()
            time = self.system.time

            # 绘制系数分布
            ax.plot(time, c0_abs2, color=self.colors['ground'],
                   linewidth=2, label='$|c_0|^2$ (基态)')
            ax.plot(time, c1_abs2, color=self.colors['excited'],
                   linewidth=2, label='$|c_1|^2$ (激发态)')

            # 添加理论值线
            ax.axhline(y=1.0, color='gray', linestyle='--',
                      alpha=0.5, label='理论值')
            ax.axhline(y=0.0, color='gray', linestyle='--',
                      alpha=0.5)

        except RuntimeError:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

        # 设置标签和标题
        ax.set_xlabel(r'$\mathbf{Time}\ (t)$', fontsize=14)
        ax.set_ylabel(r'$\mathbf{Occupation\ Probability}\ |c_n|^2$', fontsize=14)
        ax.set_title(r'$\mathbf{(b)\ Energy\ Occupation\ Probability}$', fontsize=16)

        # 设置范围
        ax.set_xlim(0, self.system.t_max)
        ax.set_ylim(-0.05, 1.05)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)

        return ax

    def plot_phases(self, ax: Optional[plt.Axes] = None):
        """
        绘制相位演化

        参数:
            ax: matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        try:
            # 获取相位
            phases_0, phases_1 = self.system.get_phases()
            time = self.system.time

            # 展开相位（避免跳跃）
            phases_0_unwrapped = np.unwrap(phases_0)
            phases_1_unwrapped = np.unwrap(phases_1)

            # 绘制相位
            ax.plot(time, phases_0_unwrapped, color=self.colors['ground'],
                   linewidth=2, label='$\\arg(c_0)$ (基态)')
            ax.plot(time, phases_1_unwrapped, color=self.colors['excited'],
                   linewidth=2, label='$\\arg(c_1)$ (激发态)')

            # 标记Berry相位（如果已计算）
            if hasattr(self.system, 'berry_phase') and self.system.berry_phase is not None:
                berry_phase_deg = np.degrees(self.system.berry_phase)
                ax.text(0.95, 0.95,
                       f'Berry相位: {berry_phase_deg:.1f}°',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=12)

        except RuntimeError:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

        # 设置标签和标题
        ax.set_xlabel(r'$\mathbf{Time}\ (t)$', fontsize=14)
        ax.set_ylabel(r'$\mathbf{Phase\ (rad)}$', fontsize=14)
        ax.set_title(r'$\mathbf{(c)\ Wave\ Function\ Phase\ Evolution}$', fontsize=16)

        # 设置范围
        ax.set_xlim(0, self.system.t_max)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=12)

        return ax

    def plot_adiabatic_parameter(self, ax: Optional[plt.Axes] = None):
        """
        绘制绝热参数演化

        参数:
            ax: matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if hasattr(self.system, 'adiabatic_params') and len(self.system.adiabatic_params) > 0:
            time = self.system.time
            gamma = self.system.adiabatic_params

            # 过滤无穷大值
            finite_mask = np.isfinite(gamma)
            gamma_finite = gamma[finite_mask]

            # 绘制绝热参数
            ax.plot(time, gamma, color='black', linewidth=2, label='绝热参数')

            # 绘制阈值线（绝热近似要求 γ << 1）
            ax.axhline(y=1.0, color='red', linestyle='--',
                      linewidth=2, alpha=0.7, label='绝热阈值')
            ax.axhline(y=0.1, color='orange', linestyle='--',
                      linewidth=1, alpha=0.5, label='强绝热区域')

            # 填充强绝热区域
            ax.fill_between(time, 0, 0.1, color='green',
                           alpha=0.2, label='绝热近似有效')

            # 设置对数坐标（如果需要）
            if np.max(gamma_finite) > 10:
                ax.set_yscale('log')
                ax.set_ylim(1e-2, np.max(gamma_finite) * 1.1)

            # 添加统计信息
            max_gamma = np.max(gamma_finite) if len(gamma_finite) > 0 else 0
            mean_gamma = np.mean(gamma_finite) if len(gamma_finite) > 0 else 0
            ax.text(0.95, 0.95,
                   f'最大值: {max_gamma:.3f}\n平均值: {mean_gamma:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)

        else:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

        # 设置标签和标题
        ax.set_xlabel(r'$\mathbf{Time}\ (t)$', fontsize=14)
        ax.set_ylabel(r'$\mathbf{Adiabatic\ Parameter}\ \gamma(t)$', fontsize=14)
        ax.set_title(r'$\mathbf{(d)\ Adiabatic\ Condition\ Check}$', fontsize=16)

        # 设置范围
        ax.set_xlim(0, self.system.t_max)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        return ax

    def create_comprehensive_four_panel(self, save_path: Optional[str] = None):
        """
        创建综合四面板分析图（与single_adia风格一致）

        布局：
        (a) 3D能级 + k轨迹（左上，大图）
        (b) 系数幅值 |cₙ|（右上）
        (c) Berry相位（右下）
        (d) 绝热参数 γ(t)（右下）

        参数:
            save_path: 保存路径（可选）
        """
        # 创建图形 - 使用与single_adia相同的尺寸
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 添加总标题 - 使用与single_adia相同的格式
        fig.suptitle(f'Dirac Point Berry Phase: k_center={self.system.k_center}, r={self.system.radius}, α={self.system.alpha}\n'
                    f'(initialized in ground state at k(0))',
                    fontsize=18, fontweight='bold', y=0.98)

        # 子图1: 3D能带图（替换为与single_adia一致的风格）
        ax1 = axes[0, 0]
        ax1.remove()
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        self.plot_3d_band_structure(ax1)
        ax1.set_title('(a) 3D Band Structure', fontsize=14, fontweight='bold', pad=20)

        # 添加Hamiltonian显示（与single_adia格式一致）
        hamiltonian_text = 'H(k) = v_F (k_x σ_x + k_y σ_y)\n' + \
                          '     = ⎡ 0    k_x - ik_y ⎤\n' + \
                          '       ⎣ k_x + ik_y   0   ⎦'

        ax1.text2D(0.02, 0.98, hamiltonian_text,
                  transform=ax1.transAxes,
                  fontsize=11,
                  verticalalignment='top',
                  family='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                           edgecolor='darkblue', alpha=0.9))

        # 子图2: 系数幅值演化
        ax2 = axes[0, 1]
        self._plot_coefficients_single_adia_style(ax2)

        # 子图3: Berry相位演化（动态相位扣除后）
        ax3 = axes[1, 0]
        self._plot_berry_phase_single_adia_style(ax3)

        # 子图4: 绝热参数γ(t)
        ax4 = axes[1, 1]
        self._plot_adiabatic_parameter_single_adia_style(ax4)

        # 调整子图间距
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

        # 保存图片（与single_adia相同的命名风格）
        if save_path is None:
            output_file = f'comprehensive_dirac_r{self.system.radius:.3f}_a{self.system.alpha:.3f}_analysis.png'
        else:
            output_file = save_path

        plt.savefig(output_file,
                    dpi=300, bbox_inches='tight', facecolor='white',
                    edgecolor='none', format='png')

        print(f"✅ 综合图表已生成: {output_file}")

        return fig

    def _plot_coefficients_single_adia_style(self, ax):
        """绘制系数幅值（single_adia风格）"""
        try:
            c0_abs, c1_abs = self.system.get_coefficients()
            time = self.system.time

            # 绘制幅值（使用single_adia的颜色和线型）
            ax.plot(time, np.sqrt(c0_abs), 'r-', linewidth=2.5,
                   label=r'$|c_g(t)|$ (ground state amplitude)')
            ax.plot(time, np.sqrt(c1_abs), 'b-', linewidth=2.5,
                   label=r'$|c_e(t)|$ (excited state amplitude)')

            ax.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
            ax.set_title('(b) Coefficient Amplitudes', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.system.t_max/2, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylim([0, 1.05])

        except RuntimeError:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

    def _plot_berry_phase_single_adia_style(self, ax):
        """绘制Berry相位（single_adia风格，扣除动态相位）"""
        try:
            # 获取相位并计算Berry相位（扣除动态相位）
            phases_0, phases_1 = self.system.get_phases()
            time = self.system.time

            # 计算动态相位
            E0 = self.system.eigenvalues[:, 0]
            E1 = self.system.eigenvalues[:, 1]
            dt = self.system.dt

            dynamic_phase_0 = np.zeros_like(time)
            dynamic_phase_1 = np.zeros_like(time)

            for i in range(1, len(time)):
                dynamic_phase_0[i] = dynamic_phase_0[i-1] - E0[i-1] * dt
                dynamic_phase_1[i] = dynamic_phase_1[i-1] - E1[i-1] * dt

            # Berry相位 = 总相位 - 动态相位
            berry_phase_0 = phases_0 - dynamic_phase_0
            berry_phase_1 = phases_1 - dynamic_phase_1

            # 展开相位
            berry_phase_0_unwrapped = np.unwrap(berry_phase_0)
            berry_phase_1_unwrapped = np.unwrap(berry_phase_1)

            # 绘制（使用single_adia的风格）
            ax.plot(time, berry_phase_0_unwrapped, 'r-', linewidth=2.5,
                   label=r'Berry Phase of ground state (unwrapped)')
            ax.plot(time, berry_phase_1_unwrapped, 'b-', linewidth=2.5,
                   label=r'Berry Phase of excited state (unwrapped)')

            # 标记最终的Berry相位
            final_berry_0 = berry_phase_0_unwrapped[-1] - berry_phase_0_unwrapped[0]
            final_berry_1 = berry_phase_1_unwrapped[-1] - berry_phase_1_unwrapped[0]

            ax.text(0.98, 0.98,
                   f'Ground state: {final_berry_0/np.pi:.3f}π\n'
                   f'Excited state: {final_berry_1/np.pi:.3f}π\n'
                   f'(Theoretical: π)',
                   transform=ax.transAxes,
                   fontsize=10,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                            edgecolor='orange', alpha=0.9))

            ax.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Berry Phase (π units)', fontsize=12, fontweight='bold')
            ax.set_title('(c) Berry Phases (Dynamic Phase Removed)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.system.t_max/2, color='gray', linestyle='--', alpha=0.5)

            # 设置y轴为π单位
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(
                lambda val, pos: f'{val/np.pi:.1f}π' if val != 0 else '0'
            ))
            ax.set_ylim([-3*np.pi, 3*np.pi])  # 设置合理的y轴范围

        except RuntimeError:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

    def _plot_adiabatic_parameter_single_adia_style(self, ax):
        """绘制绝热参数（single_adia风格）"""
        if hasattr(self.system, 'adiabatic_params') and len(self.system.adiabatic_params) > 0:
            time = self.system.time
            gamma = self.system.adiabatic_params

            # 使用对数坐标（与single_adia一致）
            finite_mask = np.isfinite(gamma)
            ax.semilogy(time[finite_mask], gamma[finite_mask], 'g-',
                       linewidth=2.5, label=r'$\gamma(t)$')

            # 找到最大值的位置
            max_idx = np.argmax(gamma[finite_mask])
            max_t = time[finite_mask][max_idx]
            max_gamma = gamma[finite_mask][max_idx]

            ax.axvline(x=max_t, color='red', linestyle=':', alpha=0.7)
            ax.text(max_t, max_gamma, f'  Max: {max_gamma:.3e} at t={max_t:.1f}',
                    fontsize=9, verticalalignment='bottom')

            # 添加绝热条件参考线
            ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='γ=1 (threshold)')
            ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='γ=0.1')

            ax.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
            ax.set_ylabel(r'$\gamma(t)$', fontsize=12, fontweight='bold')
            ax.set_title(r'(d) Adiabatic Parameter $\gamma(t) = \frac{|\langle e(t)|\partial_t H(t)|g(t)\rangle|}{[E_e(t) - E_g(t)]^2}$',
                         fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.system.t_max/2, color='gray', linestyle='--', alpha=0.5)

        else:
            ax.text(0.5, 0.5, '请先运行evolve()',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)

    def plot_energy_cut(self, ax: Optional[plt.Axes] = None,
                       cut_direction: str = 'x',
                       cut_value: float = 0.0,
                       k_range: float = 0.3):
        """
        绘制能带切面图

        参数:
            ax: matplotlib轴对象
            cut_direction: 切割方向 ('x' 或 'y')
            cut_value: 切割位置的值
            k_range: k空间范围
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # 创建切割线
        if cut_direction == 'x':
            k = np.linspace(-k_range, k_range, 500)
            kx = k
            ky = np.full_like(k, cut_value)
            xlabel = '$k_x$'
        else:  # cut_direction == 'y'
            k = np.linspace(-k_range, k_range, 500)
            kx = np.full_like(k, cut_value)
            ky = k
            xlabel = '$k_y$'

        # 计算能带
        E_plus = np.sqrt(kx**2 + ky**2)
        E_minus = -E_plus

        # 绘制能带
        ax.plot(k, E_plus, 'b-', linewidth=2, label='导带 $E_+$')
        ax.plot(k, E_minus, 'r-', linewidth=2, label='价带 $E_-$')

        # 标记狄拉克点
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.scatter([0], [0], color=self.colors['dirac'],
                  s=100, marker='o', zorder=5, label='狄拉克点')

        # 设置标签和标题
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel('能量 $E/\\hbar v_F$', fontsize=14)
        title = f'能带切面图 ({cut_direction}={cut_value:.2f})'
        ax.set_title(title, fontsize=16)

        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=12)

        return ax