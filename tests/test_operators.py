import numpy as np
from src.operators import gradient, divergence

def adjoint_test(shape=(100, 100)):
    # 1. 生成随机输入
    u = np.random.randn(*shape)
    # 向量场 p = (px, py)
    px = np.random.randn(*shape)
    py = np.random.randn(*shape)

    # 2. 计算左式: <grad u, p>
    # 先计算梯度
    grad_ux, grad_uy = gradient(u)
    # 计算内积 (两个分量分别相乘再求和)
    lhs = np.sum(grad_ux * px + grad_uy * py)

    # 3. 计算右式: <u, -div p>
    # 计算散度
    div_p = divergence(px, py)
    # 注意这里公式里通常带个负号，取决于你对 div 的定义
    rhs = np.sum(u * (-div_p))

    # 4. 比较结果
    diff = np.abs(lhs - rhs)
    print(f"LHS: {lhs:.10f}")
    print(f"RHS: {rhs:.10f}")
    print(f"Difference: {diff:.10e}")
    
    return diff < 1e-10

if __name__ == "__main__":
    if adjoint_test():
        print("Adjoint test PASSED! 🚀")
    else:
        print("Adjoint test FAILED. ❌ 请检查散度的边界处理。")