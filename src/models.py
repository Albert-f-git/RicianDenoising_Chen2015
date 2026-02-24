import numpy as np
from scipy import special

def G_prime(u, f, sigma):
    """
    计算目标函数数据保真项 G(u) 的导数。
    使用 scipy.special.ive 防止 I1(x)/I0(x) 在大数值下溢出。
    """
    eps = 1e-8
    u_safe = np.maximum(u, eps)
    x = (f * u_safe) / (sigma**2)
    
    # 计算 I1(x) / I0(x)
    ratio = special.ive(1, x) / np.maximum(special.ive(0, x), eps)
    
    term1 = u_safe / (sigma**2)
    term2 = (f / (sigma**2)) * ratio
    term3 = (1 / sigma) * (1 - np.sqrt(f / u_safe))
    
    return term1 - term2 + term3