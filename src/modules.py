import torch
import torch.nn as nn

# ----------------- 通道注意力模块 (Channel Attention Module) -----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 通道注意力：平均池化和最大池化结果送入共享的全连接层
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 2. 元素级相加后经过 Sigmoid 激活
        out = avg_out + max_out
        return self.sigmoid(out)

# ----------------- 空间注意力模块 (Spatial Attention Module) -----------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 卷积层，将通道维度压缩为 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 对输入特征图在通道维度进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 2. 沿通道维度拼接
        x_out = torch.cat([avg_out, max_out], dim=1)
        # 3. 经过卷积和 Sigmoid 激活
        x_out = self.conv1(x_out)
        return self.sigmoid(x_out)

# ----------------- CBAM 整体模块 -----------------
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 1. 通道注意力
        self.ca = ChannelAttention(in_planes, ratio)
        # 2. 空间注意力
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # x -> 通道注意力 -> 乘积 -> 空间注意力 -> 乘积 -> 输出
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out