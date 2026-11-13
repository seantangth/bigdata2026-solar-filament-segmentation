"""Layer 2, Layer 3 & Layer 4 Neural Architecture & Multi-Task Topological Loss (E-120):
TopoFlowNet — Pretrained ConvNeXt-Base IN22K + Parallel Native Stride-1/2 Residual Fine Stem
+ Axial/PPM Global PIL Bottleneck + GroupNorm/scSE Learnable Upsampling Decoder
+ Annotator Granularity FiLM Conditioning + 5-Channel Topological Spine-Flow Head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FastGroupNorm(nn.Module):
    """GroupNorm computed in fp32 for numerical precision, cast back to input dtype so bf16 stays bf16."""

    def __init__(self, num_groups: int, num_channels: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.float16, torch.bfloat16):
            return F.group_norm(
                x.float(), self.gn.num_groups,
                self.gn.weight.float() if self.gn.weight is not None else None,
                self.gn.bias.float() if self.gn.bias is not None else None,
                self.gn.eps,
            ).to(x.dtype)
        return self.gn(x)


def _gn(ch: int) -> FastGroupNorm:
    groups = 8
    while ch % groups != 0 and groups > 1:
        groups //= 2
    return FastGroupNorm(groups, ch)


class ScSE(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cse(x) + x * self.sse(x)


class ResConvGN(nn.Module):
    """Residual 3x3 Conv + GroupNorm + GELU block (resolution-invariant, zero BatchNorm drift)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _gn(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = _gn(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _gn(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return self.act(h + self.skip(x))


class GlobalPILContext(nn.Module):
    """Bottleneck Pyramid Pooling + Axial Long-Range Strip Context for cross-disk magnetic channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        p_ch = out_ch // 4
        self.pool_sizes = (1, 2, 4, 8)
        self.pools = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, p_ch, 1, bias=False), _gn(p_ch), nn.GELU())
            for _ in self.pool_sizes
        ])
        self.proj_in = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), _gn(out_ch), nn.GELU())
        # Axial horizontal & vertical strip context (captures full-row and full-column neutral lines)
        self.strip_h = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(1, 11), padding=(0, 5), groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            _gn(out_ch),
        )
        self.strip_v = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(11, 1), padding=(5, 0), groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            _gn(out_ch),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.GELU(),
            ScSE(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        base = self.proj_in(x)
        axial = F.gelu(base + 0.5 * (self.strip_h(base) + self.strip_v(base)))
        ppm_feats = []
        for sz, proj in zip(self.pool_sizes, self.pools):
            if sz == 1:
                pooled = x.mean(dim=(-2, -1), keepdim=True)
            elif h % sz == 0 and w % sz == 0:
                pooled = F.avg_pool2d(x, kernel_size=(h // sz, w // sz), stride=(h // sz, w // sz))
            else:
                pooled = F.interpolate(x, size=(sz, sz), mode="bilinear", align_corners=False)
            ppm_feats.append(F.interpolate(proj(pooled), size=(h, w), mode="bilinear", align_corners=False))
        ppm = torch.cat(ppm_feats, dim=1)
        return self.fuse(torch.cat([axial, ppm], dim=1))


class UpDecodeStage(nn.Module):
    """Learnable smooth upsampling (Bilinear + 3x3 Anti-Alias Conv) + Skip Fusion + ResConvGN + scSE."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            _gn(out_ch),
            nn.GELU(),
        )
        self.block = ResConvGN(out_ch + skip_ch, out_ch)
        self.attn = ScSE(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        u = self.up_conv(u)
        return self.attn(self.block(torch.cat([u, skip], dim=1)))


class SnakeConvGN(nn.Module):
    """Resolution-invariant GroupNorm Dynamic Snake Convolution (Qi et al., ICCV 2023).

    Uses FastGroupNorm(1, kernel_size) on offsets (zero BatchNorm resolution drift between
    512x512 training tiles and 2048x2048 whole-disk inference) and exact fp32 coordinate sampling.
    """

    def __init__(self, channels: int, kernel_size: int = 9, axis: int = 0, extend_scope: float = 1.0):
        super().__init__()
        if kernel_size % 2 == 0 or axis not in (0, 1):
            raise ValueError("Odd kernel_size and axis in {0, 1} required")
        self.k = kernel_size
        self.axis = axis
        self.scope = extend_scope
        self.offset = nn.Sequential(
            nn.Conv2d(channels, kernel_size, 3, padding=1),
            FastGroupNorm(1, kernel_size),
        )
        self.conv = nn.Conv2d(channels * kernel_size, channels, 1, bias=False)
        self.norm = _gn(channels)

    def _sample_grid(self, offset: torch.Tensor) -> torch.Tensor:
        b, _, h, w = offset.shape
        step = torch.tanh(offset.float()) * self.scope
        c = self.k // 2
        before = torch.flip(torch.cumsum(torch.flip(step[:, :c], [1]), 1), [1])
        after = torch.cumsum(step[:, c + 1 :], 1)
        perp = torch.cat([before, torch.zeros_like(step[:, c : c + 1]), after], 1)
        along = (torch.arange(self.k, device=offset.device, dtype=torch.float32) - c).view(1, -1, 1, 1)
        yy = torch.arange(h, device=offset.device, dtype=torch.float32).view(1, 1, -1, 1)
        xx = torch.arange(w, device=offset.device, dtype=torch.float32).view(1, 1, 1, -1)
        if self.axis == 0:
            px, py = xx + along, yy + perp
        else:
            px, py = xx + perp, yy + along
        px, py = torch.broadcast_tensors(px, py)
        gx = 2.0 * px / max(w - 1, 1) - 1.0
        gy = 2.0 * py / max(h - 1, 1) - 1.0
        return torch.stack([gx, gy], dim=-1).reshape(b, self.k * h, w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = x.shape
        in_dtype = x.dtype
        offset = self.offset(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            grid = self._sample_grid(offset)
            samples = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        samples = samples.view(b, ch, self.k, h, w).reshape(b, ch * self.k, h, w).to(in_dtype)
        return F.gelu(self.norm(self.conv(samples)))


class SnakeBranchGN(nn.Module):
    """Zero-initialized residual x- and y-axis Dynamic Snake branch for curvilinear filaments & barbs."""

    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        self.snake_x = SnakeConvGN(channels, kernel_size, axis=0)
        self.snake_y = SnakeConvGN(channels, kernel_size, axis=1)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sx = self.snake_x(x)
        sy = self.snake_y(x)
        return x + self.fuse(torch.cat([sx, sy], dim=1))


class BarbStripResidual(nn.Module):
    """Lightweight zero-init anisotropic strip + dilated residual at 1/2 scale for 1.5-4 px barbs."""

    def __init__(self, channels: int):
        super().__init__()
        self.dw_h = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False)
        self.dw_v = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False)
        self.dw_d = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.norm = _gn(channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm(self.dw_h(x) + self.dw_v(x) + self.dw_d(x)))
        return x + self.proj(h)


class TopoFlowNet(nn.Module):
    """Physics-Normalized Full-Skip Spine-Flow Panoptic Network.

    Outputs dict:
      - logits_3ch : (B, 3, H, W) -> [logit_body, logit_spine, logit_sep]
      - flow_2ch   : (B, 2, H, W) -> bounded vector field (vx, vy) in [-1, 1]
    """

    def __init__(
        self,
        encoder_name: str = "convnext_base.fb_in22k_ft_in1k",
        pretrained: bool = True,
        in_channels: int = 5,
        num_annotators: int = 40,
        drop_path_rate: float = 0.20,
        use_snake: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.use_snake = bool(use_snake)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            drop_path_rate=drop_path_rate,
        )
        enc_chs = self.encoder.feature_info.channels()  # [128, 256, 512, 1024] for convnext_base
        assert len(enc_chs) == 4, f"Expected 4-stage encoder, got {enc_chs}"
        c1, c2, c3, c4 = enc_chs

        # Careful initialization of 5-ch stem when loading 3-ch pretrained weights:
        # keep mean RGB projection on physics channels 0..2, small init on channels 3..4
        if pretrained and hasattr(self.encoder, "stem_0") and isinstance(self.encoder.stem_0, nn.Conv2d):
            with torch.no_grad():
                w = self.encoder.stem_0.weight
                if w.shape[1] == in_channels and in_channels > 3:
                    w[:, 3:] *= 0.25

        # Parallel Native High-Res Stem preserving 1/1 (barbs) and 1/2 (2-px inter-filament gaps)
        self.fine_s1 = ResConvGN(in_channels, 24, stride=1)
        self.fine_s2 = ResConvGN(24, 48, stride=2)

        # Bottleneck & Full-Skip U-Decoder (1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1)
        self.bottleneck = GlobalPILContext(c4, 512)
        self.dec16 = UpDecodeStage(512, c3, 256)
        self.dec8 = UpDecodeStage(256, c2, 160)
        self.dec4 = UpDecodeStage(160, c1, 96)
        self.dec2 = UpDecodeStage(96, 48, 48)
        if self.use_snake:
            self.snake16 = SnakeBranchGN(256, kernel_size=9)
            self.snake8 = SnakeBranchGN(160, kernel_size=9)
            self.snake4 = SnakeBranchGN(96, kernel_size=9)
            self.barb2 = BarbStripResidual(48)
        self.dec1_fuse = nn.Sequential(
            nn.Conv2d(48 + 24, 32, 3, padding=1, bias=False),
            _gn(32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            _gn(32),
            nn.GELU(),
        )

        # Layer 4: Annotator Granularity FiLM Conditioning (index 0 = Unbiased Consensus Token)
        self.annot_emb = nn.Embedding(num_annotators, 32)
        nn.init.normal_(self.annot_emb.weight, std=0.02)
        self.film_proj = nn.Linear(32, 32 * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        # Decoupled Heads at true 1/1 native resolution
        self.head_masks = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            _gn(32),
            nn.GELU(),
            nn.Conv2d(32, 3, 1),  # [body, spine, sep]
        )
        self.head_flow = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1, bias=False),
            _gn(24),
            nn.GELU(),
            nn.Conv2d(24, 2, 1),  # [vx, vy]
        )
        # Initialize mask bias slightly negative (~0.05 prior) for clean early training stability
        with torch.no_grad():
            self.head_masks[-1].bias.fill_(-2.5)
            nn.init.zeros_(self.head_flow[-1].bias)

    def decode_heads(self, feat_1x: torch.Tensor, annot_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Run lightweight FiLM + 1/1 heads on shared decoder feature map."""
        if annot_ids is None:
            annot_ids = torch.zeros(feat_1x.shape[0], dtype=torch.long, device=feat_1x.device)
        emb = self.annot_emb(annot_ids)
        gamma, beta = self.film_proj(emb).chunk(2, dim=-1)
        mod = feat_1x * (1.0 + gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        logits_3ch = self.head_masks(mod)
        raw_flow = self.head_flow(mod)
        # Smoothly bound flow magnitude in [-1, 1] while allowing zero vector on spine fixed points
        norm = torch.sqrt((raw_flow * raw_flow).sum(dim=1, keepdim=True) + 1e-8)
        scale = torch.tanh(norm) / norm
        flow_2ch = raw_flow * scale
        return logits_3ch, flow_2ch

    def forward_features_1x(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.fine_s1(x)       # 1/1, 24 ch
        s2 = self.fine_s2(s1)      # 1/2, 48 ch
        f4, f8, f16, f32 = self.encoder(x)
        b = self.bottleneck(f32)   # 1/32, 512 ch
        d16 = self.dec16(b, f16)   # 1/16, 256 ch
        if self.use_snake:
            d16 = self.snake16(d16)
        d8 = self.dec8(d16, f8)    # 1/8, 160 ch
        if self.use_snake:
            d8 = self.snake8(d8)
        d4 = self.dec4(d8, f4)     # 1/4, 96 ch
        if self.use_snake:
            d4 = self.snake4(d4)
        d2 = self.dec2(d4, s2)     # 1/2, 48 ch
        if self.use_snake:
            d2 = self.barb2(d2)
        u1 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1_fuse(torch.cat([u1, s1], dim=1))  # 1/1, 32 ch
        return d1

    def forward(
        self,
        x: torch.Tensor,
        annot_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d1 = self.forward_features_1x(x)
        return self.decode_heads(d1, annot_ids)


def _soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    tgt = targets.float()
    inter = (probs * tgt).sum(dim=(1, 2))
    den = (probs * probs).sum(dim=(1, 2)) + (tgt * tgt).sum(dim=(1, 2))
    return (1.0 - (2.0 * inter + eps) / (den + eps)).mean()


def _sobel_edge_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Multiscale Sobel gradient alignment loss preserving 1-3 px barbs and crisp boundaries."""
    probs = torch.sigmoid(logits.float()).unsqueeze(1)
    tgt = targets.float().unsqueeze(1)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=logits.device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(2, 3)

    loss = logits.new_zeros((), dtype=torch.float32)
    for scale in (1, 2):
        p_s = F.avg_pool2d(probs, scale, scale) if scale > 1 else probs
        t_s = F.avg_pool2d(tgt, scale, scale) if scale > 1 else tgt
        pg = torch.sqrt(F.conv2d(p_s, kx, padding=1) ** 2 + F.conv2d(p_s, ky, padding=1) ** 2 + 1e-6)
        tg = torch.sqrt(F.conv2d(t_s, kx, padding=1) ** 2 + F.conv2d(t_s, ky, padding=1) ** 2 + 1e-6)
        loss = loss + F.l1_loss(pg, tg)
    return 0.5 * loss


class TopoFlowLoss(nn.Module):
    """Joint topological supervision for Body, Spine, Touching-Sep, and Geodesic Flow."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits_3ch: torch.Tensor,
        flow_2ch: torch.Tensor,
        tgt_body: torch.Tensor,
        tgt_spine: torch.Tensor,
        tgt_sep: torch.Tensor,
        tgt_flow: torch.Tensor,
        fg_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_body = logits_3ch[:, 0].float()
        l_spine = logits_3ch[:, 1].float()
        l_sep = logits_3ch[:, 2].float()
        pred_f = flow_2ch.float()

        tb = tgt_body.float()
        ts = tgt_spine.float()
        tsep = tgt_sep.float()
        tf = tgt_flow.float()
        w = fg_weight.float()

        # 1. Body loss: thin/barb-weighted BCE + SoftDice + Multiscale Sobel Edge
        bce_b = (F.binary_cross_entropy_with_logits(l_body, tb, reduction="none") * w).mean()
        dice_b = _soft_dice_loss(l_body, tb)
        edge_b = _sobel_edge_loss(l_body, tb)
        loss_body = 0.45 * bce_b + 0.40 * dice_b + 0.15 * edge_b

        # 2. Spine centerline & gap-bridge loss
        w_sp = 1.0 + 4.0 * ts
        bce_sp = (F.binary_cross_entropy_with_logits(l_spine, ts, reduction="none") * w_sp).mean()
        dice_sp = _soft_dice_loss(l_spine, ts)
        loss_spine = 0.50 * bce_sp + 0.50 * dice_sp

        # 3. Inter-instance repulsive wall loss (sparse high-priority barrier!)
        w_sep = 1.0 + 8.0 * tsep
        bce_sep = (F.binary_cross_entropy_with_logits(l_sep, tsep, reduction="none") * w_sep).mean()
        dice_sep = _soft_dice_loss(l_sep, tsep)
        loss_sep = 0.55 * bce_sep + 0.45 * dice_sep

        # 4. Geodesic flow loss on foreground + mildly outside (keeps background flow near 0)
        fg_mask = (tb > 0.5).float()
        fg_den = fg_mask.sum().clamp_min(1.0)
        l1_vec = F.smooth_l1_loss(pred_f, tf, reduction="none", beta=0.1).sum(dim=1)
        fg_l1 = (l1_vec * fg_mask * w).sum() / fg_den
        bg_l1 = (l1_vec * (1.0 - fg_mask)).mean()

        # Angular cosine alignment where target vector has non-trivial magnitude (> 0.25)
        tf_mag = torch.sqrt((tf * tf).sum(dim=1) + 1e-8)
        dir_mask = ((tb > 0.5) & (tf_mag > 0.25)).float()
        dir_den = dir_mask.sum().clamp_min(1.0)
        pred_unit = pred_f / torch.sqrt((pred_f * pred_f).sum(dim=1, keepdim=True) + 1e-8)
        tgt_unit = tf / tf_mag.unsqueeze(1).clamp_min(1e-4)
        cos_err = (1.0 - (pred_unit * tgt_unit).sum(dim=1)) * dir_mask
        loss_cos = cos_err.sum() / dir_den
        loss_flow = 0.55 * fg_l1 + 0.35 * loss_cos + 0.10 * bg_l1

        total = loss_body + 0.65 * loss_spine + 0.75 * loss_sep + 0.80 * loss_flow
        metrics = {
            "loss": float(total.detach().item()),
            "body": float(loss_body.detach().item()),
            "spine": float(loss_spine.detach().item()),
            "sep": float(loss_sep.detach().item()),
            "flow": float(loss_flow.detach().item()),
        }
        return total, metrics
