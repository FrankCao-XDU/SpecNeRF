import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Optional


# 定义一个简单的U-Net模型
class SpectralUNet(nn.Module):
    """U-Net用于将光谱图像转换为RGB图像"""

    def __init__(self, in_channels=33, out_channels=3, base_channels=32):
        super(SpectralUNet, self).__init__()

        # 编码器部分
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # 解码器部分
        self.dec4 = self._upconv_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._upconv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._upconv_block(base_channels * 2, base_channels)
        self.dec1 = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # 解码路径（带跳跃连接）
        dec4 = self.dec4(enc4)
        dec3 = self.dec3(dec4 + enc3)  # 跳跃连接
        dec2 = self.dec2(dec3 + enc2)  # 跳跃连接
        dec1 = self.dec1(dec2 + enc1)  # 跳跃连接

        return dec1


def fetch_net_loss_criternion(params: Any, device: str = 'cuda') -> Tuple[nn.Module, Optional[Any], nn.Module]:
    """
    构建网络、损失函数和评估标准

    Args:
        params: 参数配置对象，包含模型和训练配置
        device: 设备类型 ('cuda' 或 'cpu')

    Returns:
        Unet: U-Net模型
        loss_fn: 损失函数（可选）
        criterion: 评价标准/损失函数
    """
    # 从参数中获取配置
    # 这里假设params有相应的属性，根据实际需要调整
    try:
        in_channels = getattr(params, 'spectrum_num', 11) # 默认11个光谱
        out_channels = 3  # RGB输出
        base_channels = getattr(params, 'unet_base_channels', 32)
    except:
        # 如果参数对象不是期望的类型，使用默认值
        in_channels = 33  # 11 * 3
        out_channels = 3
        base_channels = 32

    # 创建U-Net模型
    unet_model = SpectralUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels
    ).to(device)

    # 创建损失函数（这里可以根据参数选择不同的损失函数）
    # 例如：L1损失、MSE损失、感知损失等

    # 创建criterion（评价标准）
    act_func_name = getattr(params, 'act_func', 'leaky_relu')
    if act_func_name == 'leaky_relu':
        criterion = nn.LeakyReLU()

    # 第二个返回值可能是其他类型的损失函数或None
    # 根据原始代码中的使用方式，这里返回None
    return unet_model, None, criterion


# 可选：如果代码中需要其他损失函数，可以在这里定义
class SpectralLoss(nn.Module):
    """光谱损失函数"""

    def __init__(self, weights=None):
        super(SpectralLoss, self).__init__()
        self.weights = weights
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        if self.weights is not None:
            # 加权MSE损失
            weighted_loss = 0
            for i, w in enumerate(self.weights):
                # 假设输入是(batch, channels, h, w)，其中channels按光谱排列
                start_idx = i * 3
                end_idx = (i + 1) * 3
                weighted_loss += w * self.mse_loss(
                    pred[:, start_idx:end_idx, :, :],
                    target[:, start_idx:end_idx, :, :]
                )
            return weighted_loss / len(self.weights)
        else:
            return self.mse_loss(pred, target)