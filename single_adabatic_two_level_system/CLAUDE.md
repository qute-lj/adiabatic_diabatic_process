[根目录](../../CLAUDE.md) > **single_adabatic_two_level_system**

# 单绝热变量二能级系统模块

**变更记录 (Changelog):**
- 2025-11-15 14:15:42: 初始化模块文档，完成核心代码分析

## 模块职责

本模块专注于研究单绝热变量控制的二能级量子系统，实现Landau-Zener模型的完整数值模拟。通过分析不同扫描速率α下的系统动力学，深入理解绝热到非绝热转变的物理机制，并精确计算Berry相位等几何效应。

## 入口与启动

### 主要入口文件
- **`two_level_evolution_fixed.py`**: 核心模拟引擎，提供完整的类接口
- **`comprehensive_adjustable_plot.py`**: 可视化分析工具，生成多面板综合图表

### 启动方式
```bash
# 运行核心演化模拟
python two_level_evolution_fixed.py --alpha 0.1 --V 1.0 --method rk4

# 生成综合分析图表
python comprehensive_adjustable_plot.py
```

## 对外接口

### TwoLevelSystem 类
```python
class TwoLevelSystem:
    def __init__(self, alpha: float = 1.0, V: float = 1.0)
    def hamiltonian(self, t: float) -> np.ndarray
    def evolve(self, initial_state, time_points, method='rk4') -> np.ndarray
    def get_probabilities(self, states) -> Tuple[np.ndarray, np.ndarray]
    def get_adiabatic_states(self, t: float) -> Tuple[np.ndarray, np.ndarray]
    def adiabatic_parameter(self, t: float) -> float
```

### 关键函数
- `create_comprehensive_plot(alpha=0.1, V=1.0)`: 生成四面板综合分析
- `calculate_landaue_zener_probability(alpha, V)`: 计算Landau-Zener跃迁概率
- `calculate_adiabatic_parameter(t_array, alpha, V)`: 绝热参数计算

## 关键依赖与配置

### 外部依赖
- **numpy**: 数值计算和矩阵操作
- **matplotlib**: 科学可视化
- **scipy**: 高级ODE求解器 (可选)

### 物理模型配置
```python
# 哈密顿量形式
H(t) = [[α*t, V],
        [V, -α*t]]

# 参数范围
α ∈ [0.01, 0.5]  # 扫描速率
V = 1.0           # 耦合强度
t ∈ [-30, 30]     # 时间范围
```

### 数值方法配置
- **时间演化**: 4阶Runge-Kutta (默认) / SciPy solve_ivp
- **时间步长**: 自适应，默认2000个时间点
- **收敛精度**: rtol=1e-10, atol=1e-12

## 数据模型

### 核心数据结构
```python
# 量子态表示
state = np.array([c1, c2], dtype=complex)  # c1|0⟩ + c2|1⟩

# 时间演化数据
states = np.array([[c1_1, c2_1], [c1_2, c2_2], ...])

# 物理观测量
probabilities = (prob1, prob2)  # |c1|², |c2|²
phases = (phase1, phase2)       # arg(c1), arg(c2)
adiabatic_params = [γ(t1), γ(t2), ...]
```

### Berry相位计算
- **动态相位分离**: 通过数值积分减去动态相位贡献
- **相位连续性处理**: 确保本征态相位平滑连续
- **几何相位提取**: 总相位 - 动态相位 = Berry相位

## 测试与质量

### 验证方法
1. **Landau-Zener公式对比**: `P_LZ = exp(-2πV²/α)`
2. **绝热条件检查**: `γ_max ≪ 1` 验证绝热近似
3. **概率守恒**: `|c1|² + |c2|² = 1` 数值验证
4. **对称性检查**: 时间反演对称性验证

### 性能指标
- **计算精度**: 绝热参数计算精度 > 1e-6
- **数值稳定性**: 长时间演化误差累积 < 1e-4
- **内存效率**: 支持2000+时间点的高精度计算

## 常见问题 (FAQ)

**Q: 如何选择合适的α参数范围？**
A: α < 0.1 为绝热区域，α > 0.3 为非绝热区域，α ∈ [0.1, 0.3] 为过渡区域。

**Q: Berry相位计算出现跳跃怎么办？**
A: 检查本征态相位连续性处理，确保相邻时刻本征态重叠积分为正。

**Q: Runge-Kutta与SciPy方法如何选择？**
A: RK4精度足够且速度快，SciPy适合刚性方程或需要更高精度时使用。

**Q: 如何验证计算结果的正确性？**
A: 对比Landau-Zener理论公式，检查绝热参数，验证概率守恒。

## 相关文件清单

### 核心文件
- `two_level_evolution_fixed.py` - 主要模拟引擎 (350行)
- `comprehensive_adjustable_plot.py` - 可视化工具 (300+行)

### 生成文件
- `comprehensive_alpha*.png` - 多α值分析图表
- `simulation_data_fixed.npz` - 演化数据存档
- `two_level_evolution_fixed.png` - 单次运行结果图

### 数据文件格式
```python
# .npz 文件内容
{
    'time_points': 时间数组,
    'states': 量子态演化,
    'prob1': 能级1概率,
    'prob2': 能级2概率,
    'adiabatic_params': 绝热参数,
    'alpha': 扫描速率,
    'V': 耦合强度,
    'lz_probability': Landau-Zener概率
}
```

---

**模块负责人**: ZCF Workflow
**物理模型**: Landau-Zener二能级系统
**数值方法**: Runge-Kutta + 本征值分析
**最后更新**: 2025-11-15 14:15:42