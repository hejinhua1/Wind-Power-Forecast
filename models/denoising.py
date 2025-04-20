import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim, cond_dim, embed_dim=256):
        super().__init__()
        self.query_proj = nn.Conv1d(query_dim, embed_dim, 1)
        self.cond_proj = nn.Conv1d(cond_dim, embed_dim, 1)
        self.key = nn.Conv1d(embed_dim, embed_dim, 1)
        self.value = nn.Conv1d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv1d(embed_dim, query_dim, 1)

    def forward(self, x, cond):
        # x: [B, D, T], cond: [B, C, T]
        query = self.query_proj(x).transpose(1, 2)  # [B, T, E]
        cond_proj = self.cond_proj(cond)
        key = self.key(cond_proj).transpose(1, 2)  # [B, T, E]
        value = self.value(cond_proj).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(query, key.transpose(1, 2)) / (query.size(-1) ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, value).transpose(1, 2)  # [B, E, T]

        return self.out_proj(output) + x  # 残差连接


class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.attn = CrossAttention(dim, cond_dim)
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, x, cond):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.attn(x, cond)
        return self.norm(x + residual)


class ResUNetCondition(nn.Module):
    def __init__(self, in_channels=9, cond_channels=36, base_dim=64):
        super().__init__()
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_dim, 3, padding=1),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )

        # 下采样部分
        self.down = nn.Sequential(
            nn.Conv1d(base_dim, base_dim * 2, 3, stride=2, padding=1),
            nn.BatchNorm1d(base_dim * 2),
            nn.ReLU()
        )

        # 条件处理
        self.cond_proj = nn.Conv1d(cond_channels, base_dim * 2, 1)
        self.res_block = ResidualBlock(base_dim * 2, base_dim * 2)

        # 上采样部分
        self.up = nn.Sequential(
            nn.ConvTranspose1d(base_dim * 2, base_dim, 3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(base_dim),
            nn.ReLU()
        )

        # 输出层
        self.final_conv = nn.Conv1d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, condition):
        # 输入处理
        B, N, T = x.shape
        x = self.init_conv(x)  # [B, 64, T]

        # 下采样
        x_down = self.down(x)  # [B, 128, T//2]

        # 条件处理
        cond = self.cond_proj(F.avg_pool1d(condition, 2))  # [B, 128, T//2]

        # 残差块+交叉注意力
        x_down = self.res_block(x_down, cond)

        # 上采样
        x_up = self.up(x_down)  # [B, 64, T]

        # 跳过连接
        x = x + x_up  # 融合浅层和深层特征

        # 最终输出
        return self.final_conv(x)


# 测试代码
if __name__ == "__main__":
    B, N, C, T = 4, 9, 144, 96
    x = torch.randn(B, N, T)
    cond = torch.randn(B, C, T)

    model = ResUNetCondition(cond_channels=C)
    y = model(x, cond)
    print(y.shape)  # 应该输出 [4, 9, 96]