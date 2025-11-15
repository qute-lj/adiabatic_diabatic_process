# 二维狄拉克点Berry相位绕圈积分模块

> 📁 **模块路径**: `double_adabatic_two_level_system_dirac_point/`
> 🔗 **返回[根目录](../../CLAUDE.md)** | 📊 **查看[模块索引](#模块文件结构)**

**创建时间**: 2025-11-15
**模块类型**: 物理模拟与数值计算
**主要功能**: 研究二维狄拉克点哈密顿量的Berry相位几何量子效应

---

## 🎯 模块概述

本模块实现了基于二维波矢绝热参数的狄拉克点系统含时演化，专门用于研究能带理论中的Berry相位现象。与单绝热变量模块相比，本模块将绝热参数扩展到二维波矢空间 **k = (kx, ky)**，通过在k空间绕狄拉克点进行闭合回路积分来观测Berry相位。

### 核心物理模型

**狄拉克哈密顿量**:
```
H(k) = v_F (kx σx + ky σy)
```

其中：
- `k = (kx, ky)`: 二维波矢（绝热参数）
- `v_F = 1`: 费米速度（自然单位制）
- `σx, σy`: 泡利矩阵

**本征值**:
```
E±(k) = ± v_F √(kx² + ky²)
```

**Berry相位**:
绕狄拉克点一周获得 Berry 相位 **γ_B = π**

### 物理意义

1. **拓扑性质**: Berry相位的量子化体现了狄拉克点的拓扑特性
2. **几何相位**: 纯粹由参数空间几何路径获得的相位
3. **陈数关系**: γ_B = 2π × C，其中陈数 C = 1/2

## 📁 模块文件结构

| 文件 | 主要功能 | 核心类/函数 | 依赖关系 |
|------|---------|------------|----------|
| [`dirac_system.py`](dirac_system.py) | 核心物理引擎 | `DiracSystem` | numpy, scipy |
| [`visualization_tools.py`](visualization_tools.py) | 可视化工具 | `Visualizer` | matplotlib, numpy |
| [`comprehensive_analyzer.py`](comprehensive_analyzer.py) | 主分析脚本 | `DiracAnalyzer` | 本地模块 |
| [`config.py`](config.py) | 配置管理 | `get_experiment_preset()` | 无 |
| [`CLAUDE.md`](CLAUDE.md) | 模块文档 | - | - |

## 🧮 核心数据模型

### DiracSystem 类

```python
class DiracSystem:
    """二维狄拉克点系统"""

    def __init__(self, k_center=(0,0), radius=0.1, alpha=0.1, n_points=1000):
        """
        参数:
            k_center: 绕圈中心坐标 (kx0, ky0)
            radius: 绕圈半径
            alpha: 角速度（控制演化快慢）
            n_points: 时间演化点数
        """
```

### 关键方法

1. **时间演化**:
   ```python
   system.evolve(dt=0.01)  # RK4方法积分
   ```

2. **Berry相位计算**:
   ```python
   berry_phase = system.compute_berry_phase()  # Wilson圈积分
   ```

3. **轨迹生成**:
   ```python
   kx, ky = system.k_trajectory(t)  # k(t)轨迹
   ```

### 物理量计算

- **绝热参数**: `γ(t) = |⟨m|∂ₜH|n⟩| / |E_m - E_n|²`
- **占据概率**: `P_n(t) = |c_n(t)|²`
- **相位演化**: `φ_n(t) = arg(c_n(t))`

## 🎨 可视化特性

### 四面板综合图

1. **(a) 3D能带结构**: `energy levels vs (kx, ky)`
   - 上色面：导带 E₊(k)
   - 下色面：价带 E₋(k)
   - 橙色线：k(t)演化轨迹

2. **(b) 系数分布**: `|c₀|², |c₁|²`
   - 蓝线：基态占据概率
   - 紫红线：激发态占据概率

3. **(c) 相位演化**: `arg(c₀), arg(c₁)`
   - 展开后的连续相位曲线
   - 显示Berry相位累积

4. **(d) 绝热参数**: `γ(t)`
   - 检验绝热条件 `γ ≪ 1`
   - 绿色区域：绝热近似有效

### 特色可视化

- **k空间轨迹图**: 带能量等高线和方向箭头
- **能带切面图**: E(kx) 或 E(ky) 一维色散
- **参数扫描热图**: Berry相位随参数变化

## 🚀 使用方法

### 快速开始

```python
# 导入模块
from dirac_system import DiracSystem
from visualization_tools import Visualizer

# 创建系统
system = DiracSystem(
    k_center=(0.0, 0.0),  # 绕圈中心在狄拉克点
    radius=0.1,           # 绕圈半径
    alpha=0.1             # 角速度
)

# 执行时间演化
system.evolve(dt=0.01)

# 计算Berry相位
berry_phase = system.compute_berry_phase()
print(f"Berry相位 = {np.degrees(berry_phase):.1f}°")

# 创建可视化
viz = Visualizer(system)
fig = viz.create_comprehensive_four_panel()
plt.show()
```

### 运行主分析脚本

```bash
# 基本运行
python comprehensive_analyzer.py

# 使用预设
python comprehensive_analyzer.py --preset small_loop

# 自定义参数
python comprehensive_analyzer.py --radius 0.15 --alpha 0.2

# 参数扫描
python comprehensive_analyzer.py --sweep
```

### 预设实验

| 预设名称 | 描述 | 适用场景 |
|----------|------|----------|
| `small_loop` | 小半径慢速绕圈 | 验证绝热近似 |
| `large_loop` | 大半径绕圈 | 观察非绝热效应 |
| `fast_loop` | 快速绕圈 | 强非绝热过程 |
| `off_center` | 偏离狄拉克点 | 位置依赖性 |
| `figure_eight` | 8字形轨迹 | 路径依赖性 |

## 📊 物理结果分析

### 理论预期

1. **Berry相位**: 绕狄拉克点一周应为 **π** (180°)
2. **绝热近似**: 当 `γ(t) ≪ 1` 时系统保持在基态
3. **陈数**: `C = γ_B / 2π = 1/2`

### 数值验证

- **收敛性**: 减小时间步长 `dt` 验证数值稳定性
- **路径无关性**: 不同绕圈路径应得到相同Berry相位
- **绝热性检验**: 监测绝热参数 `γ(t)` 的演化

### 常见问题排查

1. **Berry相位偏离π**:
   - 检查绕圈是否闭合
   - 减小时间步长 `dt`
   - 增加演化点数 `n_points`

2. **非绝热跃迁**:
   - 减小角速度 `alpha`
   - 增大绕圈半径 `radius`
   - 检查绝热参数 `gamma(t)`

3. **数值不稳定**:
   - 使用更高精度积分方法
   - 检查本征矢连续性
   - 展开相位避免跳跃

## 🔧 扩展指南

### 添加新轨迹

在 `dirac_system.py` 中扩展 `k_trajectory()` 方法：

```python
def k_trajectory(self, t):
    if self.trajectory_type == 'ellipse':
        kx = self.k_center[0] + self.a * np.cos(self.alpha * t)
        ky = self.k_center[1] + self.b * np.sin(self.alpha * t)
    elif self.trajectory_type == 'figure_eight':
        kx = self.radius * np.sin(2 * self.alpha * t)
        ky = self.radius * np.sin(self.alpha * t)
    return kx, ky
```

### 添加质量项

```python
def get_hamiltonian(self, kx, ky):
    # 添加质量项打开能隙
    H = VF * (kx * self.sigma_x + ky * self.sigma_y) + \
        self.mass * self.sigma_z
    return H
```

### 多狄拉克点系统

扩展到多个狄拉克点：

```python
class MultiDiracSystem:
    def __init__(self, dirac_points):
        self.dirac_points = dirac_points

    def total_hamiltonian(self, kx, ky):
        # 叠加多个狄拉克点的贡献
        pass
```

## 📚 相关理论

1. **Berry相位原始文献**:
   - M.V. Berry, Proc. R. Soc. A 392, 45 (1984)

2. **狄拉克点物理**:
   - Castro Neto, Rev. Mod. Phys. 81, 109 (2009) - 石墨烯综述

3. **拓扑绝缘体**:
   - M.Z. Hasan & C.L. Kane, Rev. Mod. Phys. 82, 3045 (2010)

4. **数值方法**:
   - Sakurai, Modern Quantum Mechanics (第2章)

## 🔍 测试与验证

### 单元测试

```python
def test_berry_phase():
    """验证Berry相位为π"""
    system = DiracSystem(radius=0.1, alpha=0.05)
    system.evolve(dt=0.001)
    berry_phase = system.compute_berry_phase()
    assert np.abs(berry_phase - np.pi) < 0.01
```

### 收敛性测试

```python
def test_convergence():
    """测试数值收敛性"""
    dts = [0.01, 0.005, 0.001]
    berry_phases = []

    for dt in dts:
        system = DiracSystem()
        system.evolve(dt=dt)
        berry_phases.append(system.compute_berry_phase())

    # Berry相位应收敛
    assert np.std(berry_phases) < 0.01
```

---

**模块维护者**: ZCF Workflow
**物理理论**: 狄拉克点、Berry相位、拓扑绝缘体
**数值方法**: RK4积分、Wilson圈积分、本征值问题
**最后更新**: 2025-11-15