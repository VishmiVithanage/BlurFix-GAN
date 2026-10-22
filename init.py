import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        # hidden_dim = int(round(inp * expand_ratio))
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0))
        self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, act=False))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, dropout=0, attn_dropout=0):
        super().__init__()
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            LinearSelfAttention(embed_dim, attn_dropout),
            nn.Dropout(dropout)
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # self attention
        x = x + self.pre_norm_attn(x)
        # Feed Forward network
        x = x + self.pre_norm_ffn(x)
        return x

class MobileViTBlockv3_v2(nn.Module):
    def __init__(self, inp, attn_dim, ffn_multiplier, attn_blocks, patch_size):
        super(MobileViTBlockv3_v2, self).__init__()
        self.patch_h, self.patch_w = patch_size

        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp))
        self.local_rep.add_module('conv_1x1', conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))

        # global representation
        self.global_rep = nn.Sequential()
        ffn_dims = [int((ffn_multiplier*attn_dim)//16*16)] * attn_blocks
        for i in range(attn_blocks):
            ffn_dim = ffn_dims[i]
            self.global_rep.add_module(f'LinearAttnFFN_{i}', LinearAttnFFN(attn_dim, ffn_dim))
        self.global_rep.add_module('LayerNorm2D', nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1))

        self.conv_proj = conv_2d(2*attn_dim, inp, kernel_size=1, stride=1, padding=0, act=False)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape
        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward(self, x):
        res = x.clone()
        fm_conv = self.local_rep(x)
        x, output_size = self.unfolding_pytorch(fm_conv)
        x = self.global_rep(x)
        x = self.folding_pytorch(patches=x, output_size=output_size)
        x = self.conv_proj(torch.cat((x, fm_conv), dim=1))
        x = x + res
        return x

# Reusing the existing MobileViTv3_v2 components as a base
# Only showing the new/modified components for brevity

class DeblurGeneratorMobileViT(nn.Module):
    def __init__(self, image_size, width_multiplier=1.0, patch_size=(2, 2)):
        super().__init__()
        # Check image size
        ih, iw = image_size
        self.ph, self.pw = patch_size
        assert ih % self.ph == 0 and iw % self.pw == 0

        # Model size
        channels = []
        channels.append(int(max(16, min(64, 32 * width_multiplier))))
        channels.append(int(64 * width_multiplier))
        channels.append(int(128 * width_multiplier))
        channels.append(int(256 * width_multiplier))
        channels.append(int(384 * width_multiplier))
        channels.append(int(512 * width_multiplier))

        attn_dim = []
        attn_dim.append(int(128 * width_multiplier))
        attn_dim.append(int(192 * width_multiplier))
        attn_dim.append(int(256 * width_multiplier))

        # Default parameters
        ffn_multiplier = 2
        mv2_exp_mult = 2

        # Encoder - similar to original MobileViTv3
        self.conv_in = conv_2d(3, channels[0], kernel_size=3, stride=1, padding=1)  # Modified to stride=1 to maintain resolution

        self.encoder_stage1 = nn.Sequential(
            InvertedResidual(channels[0], channels[1], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.encoder_stage2 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=mv2_exp_mult),
            InvertedResidual(channels[2], channels[2], stride=1, expand_ratio=mv2_exp_mult)
        )
        self.encoder_stage3 = nn.Sequential(
            InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv3_v2(channels[3], attn_dim[0], ffn_multiplier, 2, patch_size=patch_size)
        )
        self.encoder_stage4 = nn.Sequential(
            InvertedResidual(channels[3], channels[4], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv3_v2(channels[4], attn_dim[1], ffn_multiplier, 4, patch_size=patch_size)
        )
        self.encoder_stage5 = nn.Sequential(
            InvertedResidual(channels[4], channels[5], stride=2, expand_ratio=mv2_exp_mult),
            MobileViTBlockv3_v2(channels[5], attn_dim[2], ffn_multiplier, 3, patch_size=patch_size)
        )

        # Decoder - upsampling path with skip connections
        self.decoder_stage5 = nn.Sequential(
            conv_2d(channels[5], channels[4], kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder_stage4 = nn.Sequential(
            conv_2d(channels[4] * 2, channels[3], kernel_size=1, stride=1),  # *2 for skip connection
            MobileViTBlockv3_v2(channels[3], attn_dim[0], ffn_multiplier, 2, patch_size=patch_size),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder_stage3 = nn.Sequential(
            conv_2d(channels[3] * 2, channels[2], kernel_size=1, stride=1),  # *2 for skip connection
            MobileViTBlockv3_v2(channels[2], attn_dim[0]//2, ffn_multiplier, 1, patch_size=patch_size),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder_stage2 = nn.Sequential(
            conv_2d(channels[2] * 2, channels[1], kernel_size=1, stride=1),  # *2 for skip connection
            InvertedResidual(channels[1], channels[1], stride=1, expand_ratio=mv2_exp_mult),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.decoder_stage1 = nn.Sequential(
            conv_2d(channels[1] * 2, channels[0], kernel_size=1, stride=1),  # *2 for skip connection
            InvertedResidual(channels[0], channels[0], stride=1, expand_ratio=mv2_exp_mult)
        )

        # Output conv to generate deblurred image
        self.conv_out = nn.Sequential(
            conv_2d(channels[0], 32, kernel_size=3, stride=1, padding=1),
            conv_2d(32, 3, kernel_size=1, stride=1, act=False)
        )

        # Residual connection (original blurry image + predicted residual)
        self.use_residual = True

    def forward(self, x):
        input_img = x

        # Encoder
        x1 = self.conv_in(x)
        x1 = self.encoder_stage1(x1)
        x2 = self.encoder_stage2(x1)
        x3 = self.encoder_stage3(x2)
        x4 = self.encoder_stage4(x3)
        x5 = self.encoder_stage5(x4)

        # Decoder with skip connections
        d5 = self.decoder_stage5(x5)
        d5 = torch.cat([d5, x4], dim=1)

        d4 = self.decoder_stage4(d5)
        d4 = torch.cat([d4, x3], dim=1)

        d3 = self.decoder_stage3(d4)
        d3 = torch.cat([d3, x2], dim=1)

        d2 = self.decoder_stage2(d3)
        d2 = torch.cat([d2, x1], dim=1)

        d1 = self.decoder_stage1(d2)

        # Final output
        out = self.conv_out(d1)

        # Residual connection (helps preserve image details)
        if self.use_residual:
            out = out + input_img

        return torch.clamp(out, 0, 1)