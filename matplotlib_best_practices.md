# Matplotlib 可视化最佳实践

基于项目经验总结的matplotlib使用核心要点。

---

## 1. 图上文本书写

### LaTeX公式使用

**基本原则**：使用 `r'$\latex$'` 格式，避免复杂环境

```python
# ✅ 简单公式
ax.set_xlabel(r'$\mathbf{Time\ (t)}$', fontsize=12)
ax.set_ylabel(r'$\mathbf{Berry\ Phase\ (\pi\ units)}$')

# ✅ 物理量定义
ax.text(0.5, 0.5, r'$\gamma(t) = \frac{|\langle m|\partial_t H|n\rangle|}{|E_m - E_n|^2}$',
        ha='center', va='center')
```

**避免复杂LaTeX环境**

```python
# ❌ 复杂矩阵环境（易出错）
ax.text(0.5, 0.5, r'$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$')

# ✅ 使用Unicode字符对齐
hamiltonian_text = 'H(t) = ⎡ αt   V ⎤\n' + \
                   '       ⎣ V  -αt ⎦'
ax.text(0.5, 0.5, hamiltonian_text,
        fontfamily='monospace', ha='center', va='center')
```

### 文本格式化规范

```python
# 一般文本使用 \text
ax.text(0.5, 0.8, r'$\text{Ground State}$')

# 重要标签使用 \mathbf
ax.set_xlabel(r'$\mathbf{Time\ (t)}$')
ax.set_ylabel(r'$\mathbf{Berry\ Phase}$')

# 混合文本和公式（中间公式不用格式）
ax.text(0.5, 0.5, r'$\text{Berry Phase: } \gamma_B = \pi$')
```

### 字符编码规范

```python
# ✅ 使用英文标签
ax.set_xlabel('Time (t)')
ax.set_ylabel('Energy (eV)')
ax.legend(['Ground State', 'Excited State'])

# ❌ 避免中文（除非正确配置字体）
# ax.set_xlabel('时间 (t)')  # 可能显示为方框
```

### 公式书写原则

```python
# ✅ 给出定义式（通用）
ax.text(0.5, 0.5, r'$\gamma(t) = \frac{|\langle m|\partial_t H|n\rangle|}{|E_m - E_n|^2}$')

# ✅ 给出物理意义
ax.text(0.5, 0.5, r'$\text{Landau-Zener: } P_{LZ} = \exp\left(-\frac{2\pi V^2}{\alpha}\right)$')

# ❌ 避免具体数值代入（除非展示特定结果）
# ax.text(0.5, 0.5, r'$\gamma(0.1) = 5.2 \times 10^{-3}$')  # 太具体
```

---

## 2. text2D 使用技巧

### 基本用法

```python
from mpl_toolkits.mplot3d import Axes3D

ax = fig.add_subplot(111, projection='3d')

# 使用 text2D 添加2D文本到3D图
ax.text2D(0.02, 0.02,
          r'$H(k) = v_F(k_x \sigma_x + k_y \sigma_y)$',
          transform=ax.transAxes,
          fontsize=10,
          ha='left', va='bottom')
```

### 位置选择策略

```python
# 左下角（最常用）
ax.text2D(0.02, 0.02, text, transform=ax.transAxes,
          ha='left', va='bottom')

# 右下角（用于结果展示）
ax.text2D(0.98, 0.02, result_text, transform=ax.transAxes,
          ha='right', va='bottom',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
```

### 避免遮挡的技巧

```python
# 1. 设置透明度
ax.text2D(0.5, 0.5, text, alpha=0.8)

# 2. 使用小的内边距
ax.text2D(0.5, 0.5, text,
          bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# 3. 检查数据范围，放在空白区域
data_range = ax.get_xlim()
if min(data_range) > 0:
    # 数据在右侧，文本放左侧
    position = (0.02, 0.5)
else:
    position = (0.98, 0.5)
```

### 完整示例：3D图注释函数

```python
def add_3d_annotations(ax):
    """向3D图添加注释，避免遮挡"""

    # 哈密顿量（左下角）
    ax.text2D(0.02, 0.02,
              r'$H(k) = v_F(k_x \sigma_x + k_y \sigma_y) + V \sigma_z$',
              transform=ax.transAxes,
              fontsize=9,
              ha='left', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightyellow', alpha=0.9))

    # Berry相位结果（右下角）
    berry_text = r'$\gamma_B = \pi$ (gapless)' + '\n' + \
                 r'$\gamma_B \to 0$ (gapped)'
    ax.text2D(0.98, 0.02,
              berry_text,
              transform=ax.transAxes,
              fontsize=9,
              ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='lightblue', alpha=0.9))
```

---

## 3. 核心要点总结

1. **LaTeX**：简单公式用 `r'$...$'`，复杂环境改用Unicode
2. **格式**：一般文本用 `\text`，重要标签用 `\mathbf`
3. **编码**：避免中文，使用英文标签
4. **公式**：给出定义式而非具体数值
5. **位置**：`text2D` 放在边角，避免遮挡数据
6. **透明**：适当使用alpha和bbox美化文本显示