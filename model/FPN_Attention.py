import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNWithCrossAttention(nn.Module):
    def __init__(self, in_channels=256, base_channels=256, n_head=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)
        self.layer2 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)
        self.layer3 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)
        self.layer4 = nn.Conv1d(base_channels, base_channels, 3, stride=2, padding=1)

        self.bn_stem = nn.BatchNorm1d(base_channels)
        self.bn_layer1 = nn.BatchNorm1d(base_channels)
        self.bn_layer2 = nn.BatchNorm1d(base_channels)
        self.bn_layer3 = nn.BatchNorm1d(base_channels)
        self.bn_layer4 = nn.BatchNorm1d(base_channels)

        self.lat_layer4 = nn.Conv1d(base_channels, base_channels, 1)
        self.lat_layer3 = nn.Conv1d(base_channels, base_channels, 1)
        self.lat_layer2 = nn.Conv1d(base_channels, base_channels, 1)
        self.lat_layer1 = nn.Conv1d(base_channels, base_channels, 1)

        self.ca4 = LocalCrossAttention1D(base_channels, n_head=n_head, win_size=8)
        self.ca3 = LocalCrossAttention1D(base_channels, n_head=n_head, win_size=8)
        self.ca2 = LocalCrossAttention1D(base_channels, n_head=n_head, win_size=8)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1 = self.relu(self.bn_stem(self.stem(x)))
        c2 = self.relu(self.bn_layer1(self.layer1(c1)))
        c3 = self.relu(self.bn_layer2(self.layer2(c2)))
        c4 = self.relu(self.bn_layer3(self.layer3(c3))) 
        c5 = self.relu(self.bn_layer4(self.layer4(c4)))

        p5 = self.lat_layer4(c5)  # (2,256,32)
        p5_up = F.interpolate(p5, size=self.lat_layer3(c4).shape[-1], mode='nearest')
        p4 = self.ca4(self.lat_layer3(c4), p5_up)

        p4_up = F.interpolate(p4, size=self.lat_layer2(c3).shape[-1], mode='nearest')
        p3 = self.ca3(self.lat_layer2(c3), p4_up)

        p3_up = F.interpolate(p3, size=self.lat_layer1(c2).shape[-1], mode='nearest')
        p2 = self.ca2(self.lat_layer1(c2), p3_up)
        p2_up = F.interpolate(p2, size=x.shape[-1], mode='nearest')

        return p2_up, p3, p4, p5


class Conv(nn.Module):
    def __init__(self, inc, ouc, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv1d(inc, ouc, k, s, padding=k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm1d(ouc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# class MSBlockLayer(nn.Module):
#     def __init__(self, c, g, k):
#         super().__init__()
#         self.conv = nn.Conv1d(c, g, k, padding=k//2, groups=4)
#         self.bn = nn.BatchNorm1d(g)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class MSBlock(nn.Module):
#     def __init__(self, inc, ouc=512, kernel_sizes=[1,3,5,7], layers_num=3):
#         super().__init__()
#         hidden = inc * 3
#         self.in_conv = Conv(inc, hidden, 1)
#         self.scale_conv = nn.Sequential(
#             nn.Conv1d(inc, hidden//len(kernel_sizes), kernel_size=1),
#             nn.BatchNorm1d(hidden//len(kernel_sizes)),
#             nn.ReLU(inplace=True)
#         )
#         self.mid_convs = nn.ModuleList()
#         for k in kernel_sizes:
#             if k == 1:
#                 self.mid_convs.append(nn.Identity())
#                 continue
#             layers = [MSBlockLayer(hidden // len(kernel_sizes), hidden // len(kernel_sizes), k) for _ in range(layers_num)]
#             self.mid_convs.append(nn.Sequential(*layers))
#         self.out_conv = Conv(hidden, ouc, 1)

#     def forward(self, feats_list):
#         x = self.in_conv(feats_list[0])  # (B, 768, T)，feats_list[0]是p2_v 256通道
#         channels = []
#         weights = nn.Parameter(torch.ones(len(self.mid_convs), dtype=torch.float32))
#         branch_channels = x.shape[1] // len(self.mid_convs)  # 768//4=192
#         for i, conv in enumerate(self.mid_convs):
#             start = i * branch_channels
#             end = (i + 1) * branch_channels
#             channel = x[:, start:end]  # (B, 192, T)
#             if i > 0:
#                 up = F.interpolate(feats_list[i], size=x.shape[-1], mode='nearest')
#                 up = self.scale_conv(up)  # (B, 192, T)
#                 channel = channel + up  # (B, 192, T)
#             channel = conv(channel)
#             channels.append(channel * weights[i])
#         out = torch.cat(channels, dim=1)  # (B, 192*4=768, T)
#         return self.out_conv(out)  # (B, ouc, T)


class LocalCrossAttention1D(nn.Module):
    def __init__(self, dim, n_head=8, win_size=11, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.win_size = win_size
        self.head_dim = dim // n_head
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        self.relative_pos_embed = nn.Parameter(torch.zeros(2 * win_size - 1, n_head))
        nn.init.trunc_normal_(self.relative_pos_embed, std=0.02)

    def forward(self, q_feat, kv_feat):
        # q_feat: (B, C, T_q) → (B, T_q, C)
        B, C, T_q = q_feat.shape
        q = q_feat.permute(0, 2, 1).contiguous()
        kv = kv_feat.permute(0, 2, 1).contiguous()

        q = self.to_q(q).view(B, T_q, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T_q, D)
        kv = self.to_kv(kv).view(B, T_q, 2, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, H, T_q, D)

        q = self._window_partition(q, T_q)  # (B*num_win, H, win_size, D)
        k = self._window_partition(k, T_q)
        v = self._window_partition(v, T_q)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B*num_win, H, win_size, win_size)
        pos_idx = torch.arange(self.win_size, device=q.device)
        rel_pos = pos_idx[:, None] - pos_idx[None, :] + self.win_size - 1
        rel_pos_embed = self.relative_pos_embed[rel_pos.view(-1)].view(self.win_size, self.win_size, self.n_head)
        attn += rel_pos_embed.permute(2, 0, 1).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, C)
        out = self.proj(out).permute(0, 2, 1).contiguous()  # (B, C, T_q)

        residual = out + q_feat
        residual = residual.permute(0, 2, 1).contiguous()
        residual = self.norm(residual)
        residual = residual.permute(0, 2, 1).contiguous()
        return residual

    def _window_partition(self, x, T):
        # x: (B, H, T, D) → (B*num_win, H, win_size, D)
        B, H, T, D = x.shape
        num_win = T // self.win_size
        x = x.view(B, self.n_head, num_win, self.win_size, self.head_dim)
        return x.permute(0, 2, 1, 3, 4).contiguous().view(-1, self.n_head, self.win_size, self.head_dim)


class FPN_CA_MSBlock_Fusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=512):
        super().__init__()
        self.fpn_video = FPNWithCrossAttention(in_channels, base_channels=256, n_head=8)
        # self.msblock = MSBlock(inc=256, ouc=out_channels)

    def forward(self, video_feat):
        # (B, 256, T)
        p2_v, p3_v, p4_v, p5_v = self.fpn_video(video_feat)
        # video_fused = self.msblock([p2_v, p3_v, p4_v, p5_v])

        # return video_fused
        return p2_v

class CrossAttention1D(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.1):
        super().__init__()
        assert dim % n_head == 0, f"dim {dim} should be divided by n_head {n_head}"
        self.n_head = n_head
        self.dim_head = dim // n_head
        self.scale = self.dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, q_feat, kv_feat):
        """
        q_feat: (B, C, T_q)   # 高分辨率作为 Query
        kv_feat: (B, C, T_k)   # 低分辨率作为 Key/Value 会上采样到 T_q
        """
        B, C, T_q = q_feat.shape

        kv_feat = F.interpolate(kv_feat, size=T_q, mode='nearest')  # (B, C, T_q)

        q = q_feat.transpose(1, 2)   # (B, T_q, C)
        k = kv_feat.transpose(1, 2)  # (B, T_q, C)
        v = kv_feat.transpose(1, 2)  # (B, T_q, C)

        q = self.to_q(q)  # (B, T_q, C)
        k = self.to_k(k)  # (B, T_q, C)
        v = self.to_v(v)  # (B, T_q, C)

        q = q.view(B, T_q, self.n_head, self.dim_head).transpose(1, 2)  # (B, H, T_q, D)
        k = k.view(B, T_q, self.n_head, self.dim_head).transpose(1, 2)  # (B, H, T_q, D)
        v = v.view(B, T_q, self.n_head, self.dim_head).transpose(1, 2)  # (B, H, T_q, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_q)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T_q, C)  # (B, T_q, C)
        out = self.proj(out)  # (B, T_q, C)
        out = out.transpose(1, 2)  # (B, C, T_q)

        residual = out + q_feat  # (B, C, T_q)

        residual = residual.transpose(1, 2)  # (B, T_q, 256)
        residual = self.norm(residual)
        residual = residual.transpose(1, 2)  # 转回 (B, 256, T_q)

        return residual
    

if __name__ == '__main__':
    video_tensor = torch.randn((2, 256, 512))
    fpn_video = FPN_CA_MSBlock_Fusion(in_channels=256, out_channels=256)
    output_tensor = fpn_video(video_tensor)
    print(output_tensor.shape)