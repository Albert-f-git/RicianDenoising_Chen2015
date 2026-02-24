import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 引入 src 目录下的模块
from src.utils import add_rician_noise, bias_correction, load_brainweb_data, compute_metrics_foreground
from src.solvers import primal_dual_denoise

def main():
    # 加载测试图像 (Cameraman) 并转为 float [0, 255]
    file_path = "data/t1_icbm_normal_1mm_pn0_rf0.raws"

    print("Step 1: 正在加载原始 BrainWeb 数据...")
    u_true = load_brainweb_data(file_path)
    
    # 设置噪声标准差和超参数
    sigma = 25.0
    gamma = 0.035
    
    # 生成含噪图像
    print(f"Adding Rician noise with sigma={sigma}...")
    f_noisy = add_rician_noise(u_true, sigma)
    
    # 使用提出的模型进行去噪
    print(f"Running proposed Primal-Dual algorithm (gamma={gamma})...")
    u_restored = primal_dual_denoise(f_noisy, sigma, gamma=gamma, max_iter=25)
    
    # 偏差校正
    print("Applying bias correction...")
    u_final = bias_correction(u_restored, f_noisy, sigma, c=1.2)
    
    # 结果评估
    psnr_noisy, ssim_noisy = compute_metrics_foreground(u_true, f_noisy)
    psnr_restored, ssim_restored = compute_metrics_foreground(u_true, u_final)

    print(f"\n[前景区域评估 (Threshold > 5)]")
    print(f"含噪图像: PSNR = {psnr_noisy:.2f}dB, SSIM = {ssim_noisy:.4f}")
    print(f"去噪图像: PSNR = {psnr_restored:.2f}dB, SSIM = {ssim_restored:.4f}")
    print(f"指标提升: PSNR:{psnr_restored - psnr_noisy:.2f}dB, SSIM:{ssim_restored - ssim_noisy:.4f}")
    
    print(f"Noisy Image    - PSNR: {psnr_noisy:.2f} dB, SSIM: {ssim_noisy:.4f}")
    print(f"Restored Image - PSNR: {psnr_restored:.2f} dB, SSIM: {ssim_restored:.4f}")
    
    # 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(u_true, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(f_noisy, cmap='gray')
    axes[1].set_title(f"Noisy (sigma={sigma})\nPSNR: {psnr_noisy:.2f}")
    axes[1].axis('off')
    
    axes[2].imshow(u_final, cmap='gray')
    axes[2].set_title(f"Restored\nPSNR: {psnr_restored:.2f}")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()