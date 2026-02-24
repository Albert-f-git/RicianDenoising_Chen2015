import numpy as np

def gradient(u):
    """
    计算图像的一阶向前差分

    u : ndarray_like, 输入图像
    dx, dy : ndarray, 图像水平方向和垂直方向的一阶向前差分
    """
    dx = np.zeros_like(u)
    dy = np.zeros_like(u)

    # 水平方向的一阶向前差分：dx[i, j] = u[i, j+1] - u[i, j]
    dx[:, :-1] = u[:, 1:] - u[:, :-1]
    # 垂直方向的一阶向前差分：dy[i, j] = u[i+1, j] - u[i, j]
    dy[:-1, :] = u[1:, :] - u[:-1, :]
    return dx, dy

def divergence(dx, dy):
    """
    计算图像的一阶向后散度

    dx, dy : ndarray_like, 输入的一阶向前差分
    div : ndarray, 图像的一阶向后散度  
    """
    p, q = dx.shape
    div_val = np.zeros((p, q))

    # --- 处理水平方向 dx ---
    # 1. 中间部分：向后差分 (对应数学上的伴随关系)
    div_val[:, 1:-1] = dx[:, 1:-1] - dx[:, :-2]
    # 2. 左边界补偿：必须直接等于 dx 的第一列
    div_val[:, 0] = dx[:, 0]
    # 3. 右边界补偿：必须等于 -dx 的倒数第二列 (假设 dx 最后一列为 0)
    div_val[:, -1] = -dx[:, -2]

    # --- 处理垂直方向 dy ---
    # 1. 中间部分
    div_val[1:-1, :] += dy[1:-1, :] - dy[:-2, :]
    # 2. 上边界补偿
    div_val[0, :] += dy[0, :]
    # 3. 下边界补偿
    div_val[-1, :] += -dy[-2, :]

    return div_val

