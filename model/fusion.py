import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalFusionModule(nn.Module):
    def __init__(
            self,
            audio_channels=256,
            video_channels=256,
            d_model=128,
            num_heads=4,
            out_channels=256
    ):
        super().__init__()
        self.d_model = d_model
        self.proj_audio = nn.Conv1d(audio_channels, d_model, kernel_size=1)  # F1→(B, d_model, T)
        self.proj_video = nn.Conv1d(video_channels, d_model, kernel_size=1)  # F2→(B, d_model, T)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.conv_fusion = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
    def forward(self, audio_feat, video_feat):
        B, _, T = audio_feat.shape 
        audio_proj = self.proj_audio(audio_feat)  # (B, d_model, T_audio)
        video_proj = self.proj_video(video_feat)  # (B, d_model, T_video)
        audio_v = audio_proj.permute(0, 2, 1)     # (B, T, d_model)
        video_v = video_proj.permute(0, 2, 1)     # (B, T, d_model)

        attn_audio, _ = self.attention(
            query=audio_v,
            key=video_v,
            value=video_v
        )  # (B, T, d_model)
        attn_video, _ = self.attention(
            query=video_v, 
            key=audio_v,
            value=audio_v
        )  # (B, T, d_model)
        f4_audio = attn_audio.permute(0, 2, 1)  # (B, d_model, T)
        f4_video = attn_video.permute(0, 2, 1)  # (B, d_model, T)
        f_concat = torch.cat([f4_audio, f4_video], dim=1)  # (B, 2*d_model, T)
        fusion_feat = self.conv_fusion(f_concat)
        return fusion_feat  # (B, out_channels, T)


if __name__ == "__main__":
    audio_feat = torch.randn(2, 256, 512)  # (B=2, C=257, T=512)
    video_feat = torch.randn(2, 256, 512)  # (B=2, C=257, T=512)

    fusion_module = ModalFusionModule2(
        audio_channels=256,
        video_channels=256,
        out_channels=256
    )

    fusion_feat = fusion_module(audio_feat, video_feat)
    print(f"Input Audio Feature Shape:{audio_feat.shape}")
    print(f"Input Video Feature Shape:{video_feat.shape}")
    print(f"Output Fusion Shape:{fusion_feat.shape}")  # (2, 256, 512)