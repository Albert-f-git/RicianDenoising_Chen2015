import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def load_brainweb_data(file_path):
    """从原始二进制文件加载 BrainWeb 数据并归一化"""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # BrainWeb 标准尺寸: 181x217x181
    shape = (181, 217, 181)
    
    # 1. 以小端序 16位无符号整型读取
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype='<u2')
    
    # 2. 关键：使用 Fortran 顺序还原 3D 卷
    data_3d = data.reshape(shape, order='F')
    
    # 3. 提取论文对应的第 90 层切片
    img_slice = data_3d[:, :, 90].astype(np.float32)
    img_slice = np.rot90(img_slice, k=1)

    # 4. 线性归一化到 [0, 255]
    img_min, img_max = img_slice.min(), img_slice.max()
    if img_max > img_min:
        img_normalized = (img_slice - img_min) / (img_max - img_min) * 255.0
    else:
        img_normalized = img_slice
        
    return img_normalized

def add_rician_noise(u, sigma):
    """
    向图像 u 添加 Rician 噪声
    u : ndarray, 原始图像
    sigma : float, 噪声标准差
    f : ndarray, 添加噪声后的图像
    """
    np.random.seed(45)
    n1 = np.random.normal(0, sigma, u.shape)
    n2 = np.random.normal(0, sigma, u.shape)
    f = np.sqrt((u + n1)**2 + n2**2)
    return f

def bias_correction(u_restored, f_noisy, sigma, c=1.2):
    """
    偏差校正模块。
    根据公式 E(u) ≈ E(sqrt(max(f^2 - c*sigma^2, 0))) 估算均值差异并补偿。
    """
    f_squared = f_noisy**2
    estimated_clean = np.sqrt(np.maximum(f_squared - c * (sigma**2), 0))
    a = np.mean(estimated_clean) - np.mean(u_restored)
    u_corrected = u_restored + a
    return np.clip(u_corrected, 0, 255)

def compute_metrics(clean, denoised):
    """计算 PSNR 和 SSIM"""
    p = psnr(clean, denoised, data_range=255)
    s = ssim(clean, denoised, data_range=255)
    return p, s

def compute_metrics_foreground(u_true, u_denoised, threshold=5):
    """
    仅计算前景区域的 PSNR 和 SSIM
    """
    # 1. 生成掩码 (True 代表脑组织，False 代表背景)
    mask = u_true > threshold
    
    # 2. 计算前景 MSE
    mse = np.mean((u_true[mask] - u_denoised[mask])**2)
    
    if mse == 0:
        return float('inf'), 1.0
        
    # 3. 计算前景 PSNR
    # 注意：即便只算前景，data_range 依然取 255
    psnr_fg = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # 4. 计算前景 SSIM (可选，需注意 SSIM 窗口处理)
    # 简单做法是将背景置零后再计算，虽然不完全精确但能反应趋势
    u_true_masked = u_true * mask
    u_denoised_masked = u_denoised * mask
    ssim_fg = ssim(u_true_masked, u_denoised_masked, data_range=255)
    
    return psnr_fg, ssim_fg