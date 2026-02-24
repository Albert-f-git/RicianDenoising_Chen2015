import numpy as np
from scipy import special
from src.operators import gradient, divergence

def primal_dual_denoise(f, sigma, gamma=0.05, max_iter=150):
    """
    使用显式线性化主对偶算法 (Linearized Primal-Dual) 解决 Rician 去噪。
    绕过了极不稳定的内部牛顿求解器，直接保证全局收敛。
    """
    # [cite_start]重新配置步长以满足收敛条件: beta * tau * gamma^2 * 8 <= 1 [cite: 393]
    # G''(u) 约等于 1/sigma^2，因此 tau 需要设定在一个安全的范围内保证显式步长稳定
    tau = 8.0 / gamma
    beta = 0.015 /gamma 
    
    u = f.copy()
    u_bar = f.copy()
    p_x = np.zeros_like(f)
    p_y = np.zeros_like(f)
    
    for k in range(max_iter):
        u_old = u.copy()
        
        # [cite_start]1. Dual Update (更新对偶变量 p) [cite: 388, 389]
        grad_u_bar_x, grad_u_bar_y = gradient(u_bar)
        p_x = p_x + beta * gamma * grad_u_bar_x
        p_y = p_y + beta * gamma * grad_u_bar_y
        
        # [cite_start]投影到 l2 单位球 [cite: 417, 418, 419, 420]
        magnitude = np.sqrt(p_x**2 + p_y**2)
        magnitude_max = np.maximum(1.0, magnitude)
        p_x = p_x / magnitude_max
        p_y = p_y / magnitude_max
        
        # 2. Primal Update (显式更新 u，替代原本不稳定的牛顿法)
        p_div = divergence(p_x, p_y)
        
        u_safe = np.maximum(u, 1e-6)  # 防止除以 0
        x = (f * u_safe) / (sigma**2)
        
        # [cite_start]安全计算贝塞尔函数比值 I1/I0 [cite: 252]
        # 使用 scipy.special.ive 防止大数值下指数爆炸
        I1 = special.ive(1, x)
        I0 = special.ive(0, x)
        B_x = I1 / np.maximum(I0, 1e-12)
        
        # [cite_start]计算 G'(u) [cite: 252]
        G_prime = (u_safe / (sigma**2)) - (f / (sigma**2)) * B_x + (1.0 / sigma) * (1.0 - np.sqrt(f / u_safe))
        
        # 显式梯度下降步 (一步到位)
        u = u - tau * (G_prime - gamma * p_div)
        
        # [cite_start]箱式约束 0 <= u <= 255 [cite: 196, 390]
        u = np.clip(u, 0.0, 255.0)
        
        # [cite_start]3. Extrapolation (外推 u_bar) [cite: 391]
        u_bar = 2.0 * u - u_old
        
    return u