#!/usr/bin/env python3
"""
综合多面板图表 - 可调alpha参数，修复Berry相位计算
包含能级、c_i幅值、Berry相位和绝热参数γ(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from two_level_evolution_fixed import TwoLevelSystem


def calculate_adiabatic_parameter(t_array, alpha, V):
    """
    计算绝热参数 γ(t) = |αV| / [2((αt)² + V²)^(3/2)]
    """
    numerator = abs(alpha * V)
    denominator = 2 * ((alpha * t_array)**2 + V**2)**(3/2)
    gamma = numerator / denominator
    return gamma


def create_comprehensive_plot(alpha=0.1, V=1.0):
    """创建可调alpha参数的综合多面板图表"""

    # 参数设置
    t_max = 30.0  # 更大的时间范围以看清缓慢变化
    n_points = 2000

    # 创建系统和时间点
    system = TwoLevelSystem(alpha=alpha, V=V)
    time_points = np.linspace(-t_max, t_max, n_points)

    print(f"🔬 计算α={alpha}, V={V}的综合分析...")

    # 1. 计算能级和基矢成分（用于第一个子图）
    E_plus = np.zeros(n_points)
    E_minus = np.zeros(n_points)
    ground_c0_sq = np.zeros(n_points)
    excited_c0_sq = np.zeros(n_points)

    # 存储本征矢以确保相位连续性
    eigenvectors_g = np.zeros((n_points, 2), dtype=complex)
    eigenvectors_e = np.zeros((n_points, 2), dtype=complex)

    for i, t in enumerate(time_points):
        H = system.hamiltonian(t)
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        E_minus[i] = eigenvalues[0]
        E_plus[i] = eigenvalues[1]

        # 确保本征矢的相位连续性
        ground_state = eigenvectors[:, 0]
        excited_state = eigenvectors[:, 1]

        if i > 0:
            # 确保与前一时刻的相位连续性
            overlap_g = np.conj(eigenvectors_g[i-1]) @ ground_state
            if np.real(overlap_g) < 0:
                ground_state = -ground_state

            overlap_e = np.conj(eigenvectors_e[i-1]) @ excited_state
            if np.real(overlap_e) < 0:
                excited_state = -excited_state

        eigenvectors_g[i] = ground_state
        eigenvectors_e[i] = excited_state

        ground_c0_sq[i] = abs(ground_state[0])**2
        excited_c0_sq[i] = abs(excited_state[0])**2

    # 2. 计算时间演化的c_i系数
    print("📈 计算时间演化系数...")

    # 设置初始态为t_start时的基态（最低本征态）
    t_start = time_points[0]
    H_start = system.hamiltonian(t_start)
    _, eigenvectors_start = np.linalg.eigh(H_start)
    initial_state = eigenvectors_start[:, 0]  # 从t_start时的基态开始

    print(f"  初始态: t={t_start:.2f}时的基态")
    print(f"  初始态系数: c₁={initial_state[0]:.4f}, c₂={initial_state[1]:.4f}")

    # 使用RK4方法求解时间演化
    solution_rk4 = system.evolve_runge_kutta_4(
        initial_state,
        time_points
    )

    # 提取c₁和c₂系数
    c1_coefficients = solution_rk4[:, 0]  # |0⟩的系数
    c2_coefficients = solution_rk4[:, 1]  # |1⟩的系数

    # 3. 计算幅值和相位
    print("📊 计算幅值和相位...")
    c1_amplitude = np.abs(c1_coefficients)
    c2_amplitude = np.abs(c2_coefficients)

    # 计算总相位并限制在[-π, π]范围内
    c1_phase = np.angle(c1_coefficients)
    c2_phase = np.angle(c2_coefficients)

    # 4. 计算Berry相位（改进版）
    print("🌀 计算Berry Phase（改进版）...")

    # 将时间演化态投影到连续的本征态基矢上
    c_g_coefficients = np.zeros(n_points, dtype=complex)  # 基态系数
    c_e_coefficients = np.zeros(n_points, dtype=complex)  # 激发态系数

    for i in range(n_points):
        # 投影: c_g(t) = ⟨g(t)|ψ(t)⟩，使用连续的本征矢
        c_g_coefficients[i] = np.conj(eigenvectors_g[i]) @ solution_rk4[i]
        c_e_coefficients[i] = np.conj(eigenvectors_e[i]) @ solution_rk4[i]

    # 计算动态相位: exp(-i∫E(t)dt/hbar)
    dt = time_points[1] - time_points[0]
    dynamic_phase_g = np.zeros(n_points)
    dynamic_phase_e = np.zeros(n_points)

    for i in range(1, n_points):
        dynamic_phase_g[i] = dynamic_phase_g[i-1] - E_minus[i-1] * dt / system.hbar
        dynamic_phase_e[i] = dynamic_phase_e[i-1] - E_plus[i-1] * dt / system.hbar

    # Berry Phase = 总相位 - 动态相位
    berry_phase_g = np.angle(c_g_coefficients) - dynamic_phase_g
    berry_phase_e = np.angle(c_e_coefficients) - dynamic_phase_e

    # 使用相位展开来获得连续的Berry相位，然后映射回[-π, π]
    berry_phase_g_unwrapped = np.unwrap(berry_phase_g)
    berry_phase_e_unwrapped = np.unwrap(berry_phase_e)

    # 计算相对于起始点的净变化
    berry_phase_g_relative = berry_phase_g_unwrapped - berry_phase_g_unwrapped[0]
    berry_phase_e_relative = berry_phase_e_unwrapped - berry_phase_e_unwrapped[0]

    # 最终限制在合理范围内用于显示
    berry_phase_g_final = np.angle(np.exp(1j * berry_phase_g_relative))
    berry_phase_e_final = np.angle(np.exp(1j * berry_phase_e_relative))

    # 5. 计算绝热参数
    print("⚡ 计算绝热参数γ(t)...")
    gamma_array = calculate_adiabatic_parameter(time_points, alpha, V)

    # 创建图形和子图
    print("🎨 创建综合图表...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Adiabatic Evolution: α={alpha}, V={V}\n(initialized in ground state at t={time_points[0]:.1f})',
                 fontsize=18, fontweight='bold', y=0.98)

    # 平滑映射函数
    def smooth_mapping(x):
        t = 0.5
        x = np.array(x)
        result = np.zeros_like(x)
        mask_low = x <= t
        mask_high = x > t
        result[mask_low] = (x[mask_low] / t) ** 2 * t
        result[mask_high] = 1 - ((1 - x[mask_high]) / (1 - t)) ** 2 * (1 - t)
        return result

    # 应用平滑映射
    ground_c0_mapped = smooth_mapping(ground_c0_sq)
    excited_c0_mapped = smooth_mapping(excited_c0_sq)

    # 子图1: 能级（使用现有的渐变色方法）
    ax1 = axes[0, 0]

    # 绘制渐变色能级线
    for i in range(n_points-1):
        # 基态线段
        color_ground = (ground_c0_mapped[i], 0, 1-ground_c0_mapped[i], 1.0)
        ax1.plot([time_points[i], time_points[i+1]],
                [E_minus[i], E_minus[i+1]],
                color=color_ground, linewidth=3)

        # 激发态线段
        color_excited = (excited_c0_mapped[i], 0, 1-excited_c0_mapped[i], 1.0)
        ax1.plot([time_points[i], time_points[i+1]],
                [E_plus[i], E_plus[i+1]],
                color=color_excited, linewidth=3)

    ax1.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Energy Levels', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 添加Hamiltonian显示 - 精确对齐矩阵符号（来自corrected_final_plot.py）
    hamiltonian_text = 'H(t) = ⎡ αt   V ⎤\n' + \
                       '       ⎣ V  -αt ⎦'

    ax1.text(0.02, 0.98, hamiltonian_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue',
                     edgecolor='darkblue', alpha=0.9))

    # 添加物理过程说明（来自corrected_final_plot.py，但更新为绝热演化信息）
    physics_text = f'Adiabatic Evolution\n' + \
                  f'α={alpha}, V={V}\n' + \
                  f'Start: ground state at t={time_points[0]:.1f}\n' + \
                  f't = 0: |g⟩=(|0⟩-|1⟩)/√2\n' + \
                  f'End: |g⟩→|1⟩'

    ax1.text(0.98, 0.02, physics_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                     edgecolor='orange', alpha=0.9))

    # 子图2: c_i幅值演化
    ax2 = axes[0, 1]
    ax2.plot(time_points, c1_amplitude, 'r-', linewidth=2.5, label=r'$|c_1(t)|$ (|0⟩ coefficient)')
    ax2.plot(time_points, c2_amplitude, 'b-', linewidth=2.5, label=r'$|c_2(t)|$ (|1⟩ coefficient)')
    ax2.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Coefficient Amplitudes', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylim([0, 1.05])

    # 子图3: Berry Phase演化（使用连续版本）
    ax3 = axes[1, 0]

    # 展示净Berry相位变化
    ax3.plot(time_points, berry_phase_g_relative, 'r-', linewidth=2.5,
             label=r'Berry Phase of $|g(t)\rangle$ (unwrapped)')
    ax3.plot(time_points, berry_phase_e_relative, 'b-', linewidth=2.5,
             label=r'Berry Phase of $|e(t)\rangle$ (unwrapped)')

    ax3.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Berry Phase (rad)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Berry Phases (Dynamic Phase Removed)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 子图4: 绝热参数γ(t)
    ax4 = axes[1, 1]
    ax4.semilogy(time_points, gamma_array, 'g-', linewidth=2.5, label=r'$\gamma(t)$')
    ax4.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'$\gamma(t)$', fontsize=12, fontweight='bold')
    ax4.set_title(r'(d) Adiabatic Parameter $\gamma(t) = \frac{|\langle e(t)|\partial_t H(t)|g(t)\rangle|}{[E_e(t) - E_g(t)]^2}$',
                  fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 找到最大值的位置
    max_idx = np.argmax(gamma_array)
    max_t = time_points[max_idx]
    max_gamma = gamma_array[max_idx]
    ax4.axvline(x=max_t, color='red', linestyle=':', alpha=0.7)
    ax4.text(max_t, max_gamma, f'  Max: {max_gamma:.3e} at t={max_t:.1f}',
             fontsize=9, verticalalignment='bottom')

    # 调整子图间距
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

    # 保存图片
    output_file = f'comprehensive_alpha{alpha}_V{V}_analysis.png'
    plt.savefig(output_file,
                dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', format='png')

    print(f"✅ 综合图表已生成: {output_file}")

    return {
        'time_points': time_points,
        'c1_amplitude': c1_amplitude,
        'c2_amplitude': c2_amplitude,
        'c1_phase': c1_phase,
        'c2_phase': c2_phase,
        'berry_phase_g': berry_phase_g_relative,
        'berry_phase_e': berry_phase_e_relative,
        'gamma_array': gamma_array,
        'max_gamma': max_gamma,
        'max_t': max_t,
        'alpha': alpha,
        'V': V
    }


def analyze_berry_phase(results):
    """分析Berry相位"""
    print("\n🌀 Berry相位分析:")
    print("=" * 50)

    berry_g = results['berry_phase_g']
    berry_e = results['berry_phase_e']
    time_points = results['time_points']
    alpha = results['alpha']
    V = results['V']

    # 计算总Berry相位积累（从-t_max到+t_max）
    total_berry_g = berry_g[-1] - berry_g[0]
    total_berry_e = berry_e[-1] - berry_e[0]

    print(f"初始条件: 从t_start={time_points[0]:.1f}时的基态开始绝热演化")
    print(f"总Berry相位积累:")
    print(f"  基态 |g(t)⟩: {total_berry_g:.4f} rad ({total_berry_g/np.pi:.3f}π)")
    print(f"  激发态 |e(t)⟩: {total_berry_e:.4f} rad ({total_berry_e/np.pi:.3f}π)")

    # 寻找Berry相位的变化特征
    print(f"\nBerry相位特征:")
    print(f"  基态最大值: {np.max(berry_g):.4f} rad, 最小值: {np.min(berry_g):.4f} rad")
    print(f"  激发态最大值: {np.max(berry_e):.4f} rad, 最小值: {np.min(berry_e):.4f} rad")

    # 检查t=0附近的连续性
    t_zero_idx = np.argmin(np.abs(time_points))
    print(f"\nt=0附近的Berry相位:")
    print(f"  基态 (t={time_points[t_zero_idx-1]:.2f}): {berry_g[t_zero_idx-1]:.4f} rad")
    print(f"  基态 (t={time_points[t_zero_idx]:.2f}): {berry_g[t_zero_idx]:.4f} rad")
    print(f"  基态 (t={time_points[t_zero_idx+1]:.2f}): {berry_g[t_zero_idx+1]:.4f} rad")
    print(f"  激发态 (t={time_points[t_zero_idx-1]:.2f}): {berry_e[t_zero_idx-1]:.4f} rad")
    print(f"  激发态 (t={time_points[t_zero_idx]:.2f}): {berry_e[t_zero_idx]:.4f} rad")
    print(f"  激发态 (t={time_points[t_zero_idx+1]:.2f}): {berry_e[t_zero_idx+1]:.4f} rad")

    # Landau-Zener模型的理论Berry相位
    print(f"\n理论预期:")
    print(f"  α={alpha}, V={V}的Landau-Zener系统，从基态开始绝热演化")
    print(f"  在绝热条件下，Berry相位应该表现出平滑的几何性质")
    print(f"  预期基态保持在瞬时基态上，激发态占据为零")


def analyze_adiabaticity(results):
    """分析绝热性"""
    print("\n🔍 绝热性分析:")
    print("=" * 50)

    max_gamma = results['max_gamma']
    max_t = results['max_t']
    alpha = results['alpha']
    V = results['V']

    print(f"绝热参数最大值: γ_max = {max_gamma:.6e} at t = {max_t:.2f}")
    print(f"绝热条件: γ(t) ≪ 1")

    if max_gamma < 0.1:
        print("✅ 系统行为高度绝热")
    elif max_gamma < 1.0:
        print("⚠️  系统行为中等绝热")
    else:
        print("❌ 系统行为非绝热")

    # 分析Landau-Zener跃迁概率
    P_LZ = np.exp(-2 * np.pi * V**2 / alpha)  # Landau-Zener跃迁概率

    print(f"\nLandau-Zener跃迁概率: P_LZ = exp(-2πV²/α) = {P_LZ:.6e}")
    print("由于α很小，系统几乎完全绝热演化")


if __name__ == "__main__":
    # 可以调整alpha参数
    alpha_values = [0.01, 0.1, 0.2, 0.5]

    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"🚀 开始创建α={alpha}的综合分析图表...")
        print(f"{'='*60}")

        # 创建综合图表
        results = create_comprehensive_plot(alpha=alpha, V=1.0)

        # 分析Berry相位
        analyze_berry_phase(results)

        # 分析绝热性
        analyze_adiabaticity(results)

        print(f"\n📊 生成的文件:")
        print(f"  • comprehensive_alpha{alpha}_V1.0_analysis.png - 综合多面板分析图表")

    print(f"\n🎯 所有图表完成！可以比较不同alpha值的影响。")