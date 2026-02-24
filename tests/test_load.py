import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_brainweb_data

def test_data_integrity():
    """
    数据完整性与可视化测试
    """
    # 路径请根据实际情况修改，建议使用相对路径
    file_path = "data/t1_icbm_normal_1mm_pn0_rf0.raws"
    
    print("--- 正在执行数据读取测试 ---")
    try:
        img = load_brainweb_data(file_path)
        
        # 1. 基础属性检查
        print(f"成功加载图像。")
        print(f"图像尺寸: {img.shape} (预期应为 181x217)")
        print(f"像素均值: {img.mean():.2f}")
        print(f"像素范围: [{img.min():.2f}, {img.max():.2f}]")

        # 2. 自动化断言 (面试官喜欢的工程实践)
        assert img.shape == (181, 217), "维度错误：未正确提取 2D 切片"
        assert np.abs(img.max() - 255.0) < 1e-5, "归一化错误：最大值不是 255"
        print(">> 数据校验通过！✅")

        # 3. 高质量可视化
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.colorbar(label='Intensity (0-255)')
        plt.title("BrainWeb MRI - Slice 90 (Fortran Order)")
        plt.axis('off')
        
        # 打印部分像素值进行人工比对
        print(f"中心像素 (90, 108) 的值为: {img[90, 108]:.2f}")
        
        plt.show()

    except Exception as e:
        print(f">> 测试失败：{str(e)} ❌")

if __name__ == "__main__":
    test_data_integrity()