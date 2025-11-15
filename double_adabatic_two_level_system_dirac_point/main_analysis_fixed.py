#!/usr/bin/env python3
"""
Dirac Point Berry Phase Analysis - Fixed Version
解决所有显示问题的综合分析脚本
"""

import numpy as np
import matplotlib.pyplot as plt
from dirac_system import DiracSystem
from visualization_tools import Visualizer


def create_dirac_analysis_fixed(V=0.0):
    """创建修复版的狄拉克点Berry相位分析图

    Parameters:
        V: 质量项（能隙参数）
    """

    # 优化的绝热演化参数
    k_center = (0.0, 0.0)
    radius = 0.3
    alpha = 0.01

    print(f"Dirac Point Analysis: k_center={k_center}, r={radius}, α={alpha}, V={V}")

    # 创建系统
    system = DiracSystem(k_center=k_center, radius=radius, alpha=alpha, V=V)
    print(f"  Evolution time: T = {system.t_max:.2f}")

    # 执行时间演化
    print("\nExecuting time evolution...")
    system.evolve()

    # 计算Berry相位
    print("Computing Berry Phase...")
    berry_phase = system.compute_berry_phase()
    print(f"  Berry phase = {np.degrees(berry_phase):.2f}° ({berry_phase/np.pi:.3f}π)")

    # 获取数据
    time = system.time
    c0_abs2, c1_abs2 = system.get_coefficients()
    phases_0, phases_1 = system.get_phases()
    gamma = system.adiabatic_params

    # 创建图形
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Dirac Point Berry Phase Analysis\n' +
                f'k_center=({k_center[0]}, {k_center[1]}), r={radius}, α={alpha}, V={V}\n' +
                f'Initialized in ground state',
                fontsize=18, fontweight='bold', y=0.98)

    # 子图1: 3D能带结构
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    viz = Visualizer(system)
    viz.plot_3d_band_structure(ax1)
    ax1.set_title('(a) 3D Band Structure', fontsize=14, fontweight='bold', pad=40)

    # 添加哈密顿量文本（避免重叠）
    ax1.text2D(0.02, 0.02, f'H(k) = v_F(k_x σ_x + k_y σ_y) + {V} σ_z',
               transform=ax1.transAxes,
               fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                        edgecolor='orange', alpha=0.9))

    # 子图2: 系数振幅
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(time, np.sqrt(c0_abs2), 'r-', linewidth=2.5,
            label=r'$|c_g(t)|$ (ground state)')
    ax2.plot(time, np.sqrt(c1_abs2), 'b-', linewidth=2.5,
            label=r'$|c_e(t)|$ (excited state)')
    ax2.set_xlabel(r'Time (t)', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'Amplitude', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Coefficient Amplitudes', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # 子图3: Berry相位（完全重写）
    ax3 = fig.add_subplot(2, 2, 3)

    # 计算动态相位
    E0 = system.eigenvalues[:, 0]
    dt = system.dt
    dynamic_phase = np.zeros_like(time)
    for i in range(1, len(time)):
        dynamic_phase[i] = dynamic_phase[i-1] - E0[i-1] * dt

    # 计算几何相位（Berry相位）
    berry_phase_t = phases_0 - dynamic_phase
    berry_phase_unwrapped = np.unwrap(berry_phase_t)

    # 绘制Berry相位
    ax3.plot(time, berry_phase_unwrapped, 'r-', linewidth=2.5,
            label='Berry Phase')

    # 设置y轴为π单位
    ax3.set_ylabel('Berry Phase (π units)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Berry Phase Evolution', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 设置y轴刻度为π单位
    y_min, y_max = ax3.get_ylim()
    y_ticks = np.arange(np.floor(y_min/np.pi), np.ceil(y_max/np.pi)+1) * np.pi
    ax3.set_yticks(y_ticks)
    ax3.set_yticklabels([f'{y/np.pi:.0g}π' if abs(y) > 1e-10 else '0' for y in y_ticks])

    # 添加理论值参考线
    if V == 0:
        expected = np.pi
        label = 'Expected: π'
    else:
        expected = 0
        label = 'Expected: 0'
    ax3.axhline(y=expected, color='green', linestyle='--', alpha=0.7,
               linewidth=2, label=label)

    # 添加最终Berry相位值文本
    final_berry = berry_phase_unwrapped[-1] - berry_phase_unwrapped[0]
    ax3.text(0.95, 0.05, f'Berry Phase:\n{final_berry/np.pi:.3f}π\n({np.degrees(final_berry):.1f}°)',
             transform=ax3.transAxes,
             fontsize=11,
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax3.legend(loc='upper left', fontsize=10)

    # 子图4: 绝热参数
    ax4 = fig.add_subplot(2, 2, 4)
    finite_mask = np.isfinite(gamma)
    ax4.semilogy(time[finite_mask], gamma[finite_mask], 'g-',
                linewidth=2.5, label=r'$\gamma(t)$')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
               linewidth=2, label=r'$\gamma=1$ (threshold)')
    ax4.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5,
               linewidth=2, label=r'$\gamma=0.1$')
    ax4.set_xlabel(r'Time (t)', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'Adiabatic Parameter $\gamma(t)$', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Adiabatic Condition', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 添加结果文本框
    max_gamma = np.max(gamma[finite_mask])
    result_text = f'Berry Phase: {final_berry/np.pi:.3f}π\n'
    result_text += f'Max γ: {max_gamma:.3e}\n'
    if max_gamma < 0.1:
        result_text += 'Status: Highly Adiabatic ✓'
        color = 'lightgreen'
    elif max_gamma < 1.0:
        result_text += 'Status: Moderately Adiabatic ~'
        color = 'lightyellow'
    else:
        result_text += 'Status: Non-adiabatic ✗'
        color = 'lightcoral'

    ax4.text(0.95, 0.95, result_text,
             transform=ax4.transAxes,
             fontsize=11,
             ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 保存图形
    output_file = f'dirac_analysis_fixed_r{radius:.3f}_a{alpha:.3f}_V{V:.3f}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"\n✅ Plot generated: {output_file}")
    print(f"  Berry phase: {np.degrees(final_berry):.2f}°")
    print(f"  Max adiabatic parameter: {max_gamma:.3e}")

    plt.show()

    return fig, final_berry


if __name__ == "__main__":
    # 测试无能隙狄拉克点 (V=0)
    print("=" * 60)
    print("Testing gapless Dirac point (V = 0)")
    print("=" * 60)
    fig1, berry1 = create_dirac_analysis_fixed(V=0.0)

    # 测试有能隙系统 (V=0.1)
    print("\n" + "=" * 60)
    print("Testing gapped system (V = 0.1)")
    print("=" * 60)
    fig2, berry2 = create_dirac_analysis_fixed(V=0.1)