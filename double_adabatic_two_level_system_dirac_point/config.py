"""
狄拉克点系统配置参数

包含所有可调参数的默认值、物理常数和实验配置。

使用方法:
    from config import *
    system = DiracSystem(k_center=K_CENTER, radius=RADIUS, alpha=ALPHA)
"""

import numpy as np
from typing import Tuple, List, Dict, Any

# ======================
# 物理常数
# ======================
HBAR = 1.0      # 约化普朗克常数（自然单位制）
VF = 1.0        # 费米速度
EV_TO_JOULE = 1.602176634e-19  # eV到焦耳转换

# ======================
# 默认系统参数
# ======================
# 波矢空间配置
K_CENTER: Tuple[float, float] = (0.0, 0.0)  # 绕圈中心
RADIUS: float = 0.1                         # 绕圈半径
ALPHA: float = 0.1                          # 角速度

# 时间演化参数
DT: float = 0.01                            # 时间步长
N_POINTS: int = 1000                        # 时间演化点数
T_MAX: float = None                         # 最大时间（自动计算）

# ======================
# 可视化参数
# ======================
# 图形设置
FIGSIZE: Tuple[float, float] = (15, 10)     # 主图形大小
DPI: int = 300                               # 图像分辨率
FONTSIZE: int = 14                           # 默认字体大小

# 颜色方案
COLORS: Dict[str, str] = {
    'ground': '#2E86AB',      # 基态 - 蓝色
    'excited': '#A23B72',     # 激发态 - 紫红色
    'trajectory': '#F18F01',  # 轨迹 - 橙色
    'dirac': '#C73E1D'        # 狄拉克点 - 红色
}

# 3D能带图参数
K_RANGE: float = 0.3          # k空间显示范围
RESOLUTION: int = 50          # 网格分辨率
ELEVATION: float = 30         # 3D视角仰角
AZIMUTH: float = 45           # 3D视角方位角

# ======================
# 数值计算参数
# ======================
# 数值精度
TOLERANCE: float = 1e-10      # 数值容差
MAX_ITERATIONS: int = 10000   # 最大迭代次数

# RK4积分参数
RK4_STEPS: int = 4            # RK4子步数
ADAPTIVE_DT: bool = False     # 是否使用自适应步长

# ======================
# 实验配置
# ======================
# 参数扫描配置
PARAMETER_SCAN: Dict[str, Any] = {
    'enabled': True,
    'radius_range': (0.05, 0.2, 5),      # (最小值, 最大值, 点数)
    'alpha_range': (0.05, 0.5, 5),       # (最小值, 最大值, 点数)
    'center_positions': [(0, 0)],        # 中心位置列表
}

# Berry相位计算
BERRY_PHASE_CONFIG: Dict[str, Any] = {
    'unwrap_phase': True,        # 是否展开相位
    'subtract_dynamic': True,    # 是否扣除动力学相位
    'reference_point': 0,        # 参考点索引
}

# ======================
# 预设实验
# ======================

def get_experiment_preset(name: str) -> Dict[str, Any]:
    """
    获取预设实验参数

    参数:
        name: 预设名称

    返回:
        参数字典
    """
    presets = {
        'small_loop': {
            'k_center': (0.0, 0.0),
            'radius': 0.05,
            'alpha': 0.1,
            'dt': 0.01,
            'description': '小半径慢速绕圈，验证绝热近似'
        },
        'large_loop': {
            'k_center': (0.0, 0.0),
            'radius': 0.2,
            'alpha': 0.1,
            'dt': 0.005,
            'description': '大半径绕圈，观察非绝热效应'
        },
        'fast_loop': {
            'k_center': (0.0, 0.0),
            'radius': 0.1,
            'alpha': 0.5,
            'dt': 0.002,
            'description': '快速绕圈，强非绝热过程'
        },
        'off_center': {
            'k_center': (0.1, 0.1),
            'radius': 0.1,
            'alpha': 0.1,
            'dt': 0.01,
            'description': '偏离狄拉克点的绕圈'
        },
        'figure_eight': {
            'k_center': (0.0, 0.0),
            'radius': 0.1,
            'alpha': 0.1,
            'dt': 0.01,
            'trajectory_type': 'figure_eight',
            'description': '8字形轨迹，研究路径依赖性'
        }
    }

    if name not in presets:
        raise ValueError(f"未知的预设实验: {name}")

    return presets[name]

# ======================
# 路径配置
# ======================
# 文件路径配置
OUTPUT_DIR: str = './output'       # 输出目录
DATA_DIR: str = './data'           # 数据目录
FIGURE_DIR: str = './figures'      # 图形目录

# 文件命名格式
FILENAME_FORMATS: Dict[str, str] = {
    'main_plot': 'dirac_berry_phase_r{radius:.3f}_a{alpha:.3f}.png',
    '3d_band': '3d_band_structure.png',
    'k_trajectory': 'k_trajectory.png',
    'energy_cut': 'energy_cut_{direction:.1f}.png'
}

# ======================
# 验证函数
# ======================

def validate_parameters(params: Dict[str, Any]) -> bool:
    """
    验证参数的合理性

    参数:
        params: 参数字典

    返回:
        是否有效
    """
    # 检查半径
    if 'radius' in params:
        if params['radius'] <= 0 or params['radius'] > 1.0:
            print(f"警告：半径 {params['radius']} 可能不合理")
            return False

    # 检查角速度
    if 'alpha' in params:
        if params['alpha'] <= 0 or params['alpha'] > 10.0:
            print(f"警告：角速度 {params['alpha']} 可能不合理")
            return False

    # 检查时间步长
    if 'dt' in params:
        if params['dt'] <= 0 or params['dt'] > 0.1:
            print(f"警告：时间步长 {params['dt']} 可能不合适")
            return False

    return True

# ======================
# 调试和日志配置
# ======================
DEBUG: bool = False                      # 调试模式
LOG_LEVEL: str = 'INFO'                  # 日志级别
VERBOSE: bool = True                      # 详细输出

# 打印配置
PRINT_CONFIG: Dict[str, bool] = {
    'show_progress': True,        # 显示进度
    'show_warnings': True,        # 显示警告
    'show_statistics': True,      # 显示统计信息
    'show_energy': True,          # 显示能量信息
    'show_berry_phase': True      # 显示Berry相位
}

# ======================
# 性能优化配置
# ======================
# 并行计算
USE_MULTIPROCESSING: bool = False      # 是否使用多进程
N_WORKERS: int = None                  # 工作进程数（None表示自动）

# 内存管理
CHUNK_SIZE: int = 1000                 # 数据块大小
CACHE_EIGENSTATES: bool = True         # 是否缓存本征态

# ======================
# 导出配置
# ======================
# 数据导出
EXPORT_FORMATS: List[str] = ['png', 'pdf']  # 支持的导出格式
EXPORT_DPI: int = 300                       # 导出分辨率
TRANSPARENT_BACKGROUND: bool = False        # 透明背景

# 数据保存
SAVE_EVOLUTION_DATA: bool = True       # 是否保存演化数据
SAVE_EIGENSTATES: bool = True          # 是否保存本征态
COMPRESS_DATA: bool = False            # 是否压缩数据