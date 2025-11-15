#!/usr/bin/env python3
"""
Dirac Point Berry Phase Analysis - Main Script
Visualizes 3D band structure and Berry phase accumulation
"""

import numpy as np
import matplotlib.pyplot as plt
from dirac_system import DiracSystem
from visualization_tools import Visualizer


def create_dirac_analysis(V=0.0):
    """Create Dirac point Berry phase analysis plot

    Parameters:
        V: mass term (gap parameter)
    """

    # Use optimized parameters for adiabatic evolution
    k_center = (0.0, 0.0)
    radius = 0.3
    alpha = 0.01

    print(f"Dirac Point Analysis: k_center={k_center}, r={radius}, α={alpha}, V={V}")

    # Create system
    system = DiracSystem(k_center=k_center, radius=radius, alpha=alpha, V=V)
    print(f"  Evolution time: T = {system.t_max:.2f}")

    # Execute time evolution
    print("\nExecuting time evolution...")
    system.evolve()

    # Calculate Berry phase
    print("Computing Berry Phase...")
    berry_phase = system.compute_berry_phase()
    print(f"  Berry phase = {np.degrees(berry_phase):.2f}° (theoretical: 180°)")

    # Create visualization
    print("Creating comprehensive plot...")
    viz = Visualizer(system)

    # Create figure with proper 3D subplot handling
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(r'$\mathbf{Dirac\ Point\ Berry\ Phase\ Analysis:}$\n' +
                r'$k_{center}=' + f'({k_center[0]}, {k_center[1]}), r={radius}, α={alpha}, V={V}\n' +
                r'$\mathbf{Initialized\ in\ ground\ state}$',
                fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: 3D band structure
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    viz.plot_3d_band_structure(ax1)
    ax1.set_title(r'$\mathbf{(a)\ 3D\ Band\ Structure}$', fontsize=14, fontweight='bold', pad=40)

    # Add Hamiltonian display (positioned to avoid overlap)
    hamiltonian_text = r'$H(k) = v_F(k_x σ_x + k_y σ_y) + V σ_z$'

    ax1.text2D(0.98, 0.02, hamiltonian_text,
              transform=ax1.transAxes,
              fontsize=10,
              horizontalalignment='right',
              verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                       edgecolor='orange', alpha=0.9))

    # Subplot 2: Coefficient amplitudes
    ax2 = fig.add_subplot(2, 2, 2)
    c0_abs2, c1_abs2 = system.get_coefficients()
    time = system.time
    ax2.plot(time, np.sqrt(c0_abs2), 'r-', linewidth=2.5,
            label=r'$|c_g(t)|$ (ground state)')
    ax2.plot(time, np.sqrt(c1_abs2), 'b-', linewidth=2.5,
            label=r'$|c_e(t)|$ (excited state)')
    ax2.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$\mathbf{Amplitude}$', fontsize=12, fontweight='bold')
    ax2.set_title(r'$\mathbf{(b)\ Coefficient\ Amplitudes}$', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    # Subplot 3: Berry phase (simplified)
    ax3 = fig.add_subplot(2, 2, 3)
    phases_0, phases_1 = system.get_phases()

    # Calculate dynamic phase
    E0 = system.eigenvalues[:, 0]
    dt = system.dt
    dynamic_phase_0 = np.zeros_like(time)
    for i in range(1, len(time)):
        dynamic_phase_0[i] = dynamic_phase_0[i-1] - E0[i-1] * dt

    # Berry phase = total phase - dynamic phase
    berry_phase_0 = phases_0 - dynamic_phase_0
    berry_phase_0_unwrapped = np.unwrap(berry_phase_0)

    ax3.plot(time, berry_phase_0_unwrapped, 'r-', linewidth=2.5,
            label=r'Berry Phase (ground state)')
    ax3.set_xlabel('Time (t)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Berry Phase (π units)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Berry Phases (Dynamic Removed)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Set y-axis in π units
    ax3.yaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda val, pos: f'{val/np.pi:.2g}π' if abs(val) > 1e-10 else '0'
    ))

    # Adaptive ylim based on actual Berry phase values
    berry_min = np.min(berry_phase_0_unwrapped)
    berry_max = np.max(berry_phase_0_unwrapped)
    berry_range = berry_max - berry_min
    margin = 0.2 * berry_range  # 20% margin
    ax3.set_ylim([berry_min - margin, berry_max + margin])

    # Add horizontal line at expected Berry phase (depends on V)
    if V == 0:
        expected_phase = np.pi
        label_text = r'$\pi$ (gapless Dirac)'
    else:
        # For V ≠ 0, Berry phase depends on whether the path encloses the gap
        expected_phase = 0  # Typically 0 for gapped system
        label_text = r'$0$ (gapped system)'

    ax3.axhline(y=expected_phase, color='green', linestyle='--', alpha=0.7,
               label=label_text)
    if berry_min < 0 < berry_max:
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax3.legend(loc='best', fontsize=10)

    # Subplot 4: Adiabatic parameter
    ax4 = fig.add_subplot(2, 2, 4)
    gamma = system.adiabatic_params
    finite_mask = np.isfinite(gamma)
    ax4.semilogy(time[finite_mask], gamma[finite_mask], 'g-',
                linewidth=2.5, label=r'$\gamma(t)$')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label=r'$\gamma=1$ (threshold)')
    ax4.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label=r'$\gamma=0.1$')
    ax4.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'$\mathbf{\gamma(t)}$', fontsize=12, fontweight='bold')
    ax4.set_title(r'$\mathbf{(d)\ Adiabatic\ Parameter}$', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Add result text box
    max_gamma = np.max(gamma[finite_mask])
    result_text = f'Berry Phase: {berry_phase/np.pi:.3f}π\n'
    result_text += f'Max γ: {max_gamma:.3e}\n'
    if max_gamma < 0.1:
        result_text += 'Status: Highly Adiabatic ✓'
    elif max_gamma < 1.0:
        result_text += 'Status: Moderately Adiabatic ~'
    else:
        result_text += 'Status: Non-adiabatic ✗'

    ax4.text(0.95, 0.05, result_text,
            transform=ax4.transAxes,
            fontsize=11,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_file = f'dirac_analysis_r{radius:.3f}_a{alpha:.3f}_V{V:.3f}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"\n✅ Plot generated: {output_file}")
    print(f"  Berry phase: {np.degrees(berry_phase):.2f}°")
    print(f"  Max adiabatic parameter: {max_gamma:.3e}")

    plt.show()

    return


if __name__ == "__main__":
    create_dirac_analysis()