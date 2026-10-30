import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(value, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        dilation=1,
        act=True,
    ):
        padding = dilation * (kernel_size // 2)
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale

def channel_shuffle(x, groups=2):
    batch_size, channels, height, width = x.size()
    if channels % groups != 0:
        return x
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)

class ShuffleMBBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        expand_ratio=2.0,
        dropout=0.0,
        dilation=1,
    ):
        super().__init__()
        hidden_channels = make_divisible(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvBNAct(in_channels, hidden_channels, 1)
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            groups=hidden_channels,
            dilation=dilation,
        )
        self.se = SqueezeExcite(hidden_channels)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return channel_shuffle(out, groups=2)

class MultiScaleContext(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        branch_channels = max(8, channels // 3)
        self.branch3 = nn.Sequential(
            ConvBNAct(channels, branch_channels, 1),
            ConvBNAct(branch_channels, branch_channels, 3, groups=branch_channels),
        )
        self.branch5 = nn.Sequential(
            ConvBNAct(channels, branch_channels, 1),
            ConvBNAct(branch_channels, branch_channels, 5, groups=branch_channels),
        )
        self.branch7 = nn.Sequential(
            ConvBNAct(channels, branch_channels, 1),
            ConvBNAct(branch_channels, branch_channels, 7, groups=branch_channels),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(branch_channels * 3, channels, 1),
            SqueezeExcite(channels),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        features = [self.branch3(x), self.branch5(x), self.branch7(x)]
        out = torch.cat(features, dim=1)
        out = self.fuse(out)
        return channel_shuffle(out, groups=2)

class LogChromaHistogram(nn.Module):
    def __init__(self, bins=64, min_value=-2.5, max_value=2.5, input_normalized=True):
        super().__init__()
        self.bins = int(bins)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.input_normalized = bool(input_normalized)
        centers = torch.linspace(
            self.min_value + (self.max_value - self.min_value) / (2 * self.bins),
            self.max_value - (self.max_value - self.min_value) / (2 * self.bins),
            self.bins,
        )
        self.register_buffer("centers", centers)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _denormalize(self, x):
        if not self.input_normalized:
            return x.clamp(0.0, 1.0)
        rgb = x * self.std + self.mean
        return rgb.clamp(0.0, 1.0)

    def forward(self, x):
        rgb = self._denormalize(x)
        red = rgb[:, 0]
        green = rgb[:, 1]
        blue = rgb[:, 2]
        valid = (rgb.sum(dim=1) > 0.05).float()
        eps = 1e-4
        u = torch.log((green + eps) / (red + eps))
        v = torch.log((green + eps) / (blue + eps))

        scale = (self.bins - 1) / (self.max_value - self.min_value)
        u_idx = ((u - self.min_value) * scale).round().long().clamp(0, self.bins - 1)
        v_idx = ((v - self.min_value) * scale).round().long().clamp(0, self.bins - 1)
        flat_idx = (u_idx * self.bins + v_idx).flatten(1)
        weights = valid.flatten(1)

        hist = x.new_zeros(x.size(0), self.bins * self.bins)
        hist.scatter_add_(1, flat_idx, weights)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1.0)
        return hist.view(x.size(0), 1, self.bins, self.bins)

class DepthwiseHistBlock(nn.Module):
    def __init__(self, channels, stride=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, 3, stride=stride, groups=channels),
            ConvBNAct(channels, channels, 1),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)

class LogChromaBranch(nn.Module):
    def __init__(self, bins=64, channels=40, embed_dim=96, dropout=0.1, input_normalized=True):
        super().__init__()
        self.histogram = LogChromaHistogram(bins=bins, input_normalized=input_normalized)
        self.encoder = nn.Sequential(
            ConvBNAct(1, channels, 3),
            DepthwiseHistBlock(channels, stride=2, dropout=dropout),
            DepthwiseHistBlock(channels, stride=2, dropout=dropout),
            ConvBNAct(channels, embed_dim, 1),
            SqueezeExcite(embed_dim),
        )
        self.heatmap = nn.Conv2d(embed_dim, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        illum, embed, confidence, hist, prob, _ = self.forward_with_logits(x)
        return illum, embed, confidence, hist, prob

    def forward_with_logits(self, x):
        hist = self.histogram(x)
        features = self.encoder(hist)
        logits = F.interpolate(self.heatmap(features), size=hist.shape[-2:], mode="bilinear", align_corners=False)
        prob = logits.flatten(1).softmax(dim=1).view(x.size(0), hist.size(2), hist.size(3))
        u_prob = prob.sum(dim=2)
        v_prob = prob.sum(dim=1)
        centers = self.histogram.centers.to(dtype=x.dtype)
        u = (u_prob * centers.view(1, -1)).sum(dim=1)
        v = (v_prob * centers.view(1, -1)).sum(dim=1)
        illum = torch.stack([torch.exp(-u), torch.ones_like(u), torch.exp(-v)], dim=1)
        illum = F.normalize(illum, dim=1)

        embed = self.pool(features).flatten(1)
        embed = self.embed(embed)
        entropy = -(prob.flatten(1) * prob.flatten(1).clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / torch.log(x.new_tensor(float(hist.size(2) * hist.size(3))))
        confidence = 1.0 - entropy
        return illum, embed, confidence, hist, prob.unsqueeze(1), logits

class LightLocalLogChromaEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, bottleneck_channels=None, dropout=0.1):
        super().__init__()
        if bottleneck_channels is None:
            bottleneck_channels = max(16, hidden_channels // 2)
        self.proj = ConvBNAct(in_channels, bottleneck_channels, 1)
        self.depthwise = ConvBNAct(bottleneck_channels, bottleneck_channels, 3, groups=bottleneck_channels)
        self.mix = nn.Sequential(
            ConvBNAct(bottleneck_channels, hidden_channels, 1),
            SqueezeExcite(hidden_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.depthwise(x)
        return self.mix(x)

class DenseLocalLogChromaBranch(nn.Module):
    def __init__(
        self,
        grid_size=16,
        embed_dim=96,
        hidden_channels=48,
        dropout=0.1,
        input_normalized=True,
        encoder_type="default",
        bottleneck_channels=None,
        local_pooling="softmax",
        local_topk=32,
        local_topk_boost=1.0,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.input_normalized = bool(input_normalized)
        self.local_pooling = str(local_pooling).lower()
        self.local_topk = int(local_topk)
        self.local_topk_boost = float(local_topk_boost)
        if self.local_pooling not in {"softmax", "topk", "soft_topk"}:
            raise ValueError(f"Unsupported local_pooling: {local_pooling}")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        encoder_type = str(encoder_type).lower()
        if encoder_type in {"default", "standard"}:
            self.encoder = nn.Sequential(
                ConvBNAct(8, hidden_channels, 1),
                ConvBNAct(hidden_channels, hidden_channels, 3, groups=hidden_channels),
                ConvBNAct(hidden_channels, hidden_channels, 1),
                SqueezeExcite(hidden_channels),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )
        elif encoder_type in {"light", "bottleneck"}:
            self.encoder = LightLocalLogChromaEncoder(
                in_channels=8,
                hidden_channels=hidden_channels,
                bottleneck_channels=bottleneck_channels,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported dense local encoder_type: {encoder_type}")
        self.local_illum = nn.Conv2d(hidden_channels, 3, 1)
        self.local_conf = nn.Conv2d(hidden_channels, 1, 1)
        self.embed_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _denormalize(self, x):
        if not self.input_normalized:
            return x.clamp(0.0, 1.0)
        return (x * self.std + self.mean).clamp(0.0, 1.0)

    @staticmethod
    def _pool(x, size):
        return F.adaptive_avg_pool2d(x, (size, size))

    def forward(self, x):
        rgb = self._denormalize(x)
        eps = 1e-4
        red = rgb[:, 0:1]
        green = rgb[:, 1:2]
        blue = rgb[:, 2:3]
        intensity = rgb.mean(dim=1, keepdim=True)
        valid = (rgb.sum(dim=1, keepdim=True) > 0.05).float()
        u = torch.log((green + eps) / (red + eps))
        v = torch.log((green + eps) / (blue + eps))
        valid_count = self._pool(valid, self.grid_size).clamp_min(1e-4)
        u_mean = self._pool(u * valid, self.grid_size) / valid_count
        v_mean = self._pool(v * valid, self.grid_size) / valid_count
        u_var = self._pool((u - F.interpolate(u_mean, size=u.shape[-2:], mode="nearest")) ** 2 * valid, self.grid_size)
        v_var = self._pool((v - F.interpolate(v_mean, size=v.shape[-2:], mode="nearest")) ** 2 * valid, self.grid_size)
        local_rgb = self._pool(rgb * valid, self.grid_size) / valid_count
        local_intensity = self._pool(intensity * valid, self.grid_size) / valid_count
        features = torch.cat(
            [
                u_mean,
                v_mean,
                u_var.clamp(0.0, 4.0),
                v_var.clamp(0.0, 4.0),
                local_intensity,
                valid_count,
                local_rgb[:, 0:1],
                local_rgb[:, 2:3],
            ],
            dim=1,
        )
        encoded = self.encoder(features)
        local_illum = F.normalize(F.softplus(self.local_illum(encoded)) + 1e-6, dim=1)
        conf_logits = self.local_conf(encoded).flatten(1)
        if self.local_pooling in {"topk", "soft_topk"}:
            total = conf_logits.size(1)
            k = max(1, min(self.local_topk, total))
            topk_values, topk_indices = torch.topk(conf_logits, k=k, dim=1)
            if self.local_pooling == "topk":
                topk_weights = topk_values.softmax(dim=1)
                flat_weights = conf_logits.new_zeros(conf_logits.shape)
                topk_weights = topk_weights.to(dtype=flat_weights.dtype)
                flat_weights.scatter_(1, topk_indices, topk_weights)
                weights = flat_weights.view(x.size(0), 1, self.grid_size, self.grid_size)
                entropy_norm = torch.log(x.new_tensor(float(k)))
            else:
                boost = conf_logits.new_zeros(conf_logits.shape)
                boost.scatter_(1, topk_indices, conf_logits.new_full(topk_indices.shape, self.local_topk_boost))
                weights = (conf_logits + boost).softmax(dim=1).view(x.size(0), 1, self.grid_size, self.grid_size)
                entropy_norm = torch.log(x.new_tensor(float(self.grid_size * self.grid_size)))
        else:
            weights = conf_logits.softmax(dim=1).view(x.size(0), 1, self.grid_size, self.grid_size)
            entropy_norm = torch.log(x.new_tensor(float(self.grid_size * self.grid_size)))
        illum = (local_illum * weights).sum(dim=(2, 3))
        illum = F.normalize(illum, dim=1)
        embed = self.embed_head(encoded)
        entropy = -(weights.flatten(1) * weights.flatten(1).clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / entropy_norm.clamp_min(1e-6)
        confidence = 1.0 - entropy
        return illum, embed, confidence, local_illum, weights

class TinyVisualBranch(nn.Module):
    def __init__(
        self,
        dropout=0.1,
        width_mult=1.0,
        stem_channels=16,
        stage_channels=(24, 48, 80, 112),
        embed_dim=128,
    ):
        super().__init__()
        c1 = make_divisible(stem_channels * width_mult)
        c2 = make_divisible(stage_channels[0] * width_mult)
        c3 = make_divisible(stage_channels[1] * width_mult)
        c4 = make_divisible(stage_channels[2] * width_mult)
        c5 = make_divisible(stage_channels[3] * width_mult)

        self.stem = nn.Sequential(
            ConvBNAct(3, c1, 3, stride=2),
            ConvBNAct(c1, c1, 3, stride=1, groups=c1),
        )
        self.stage1 = nn.Sequential(
            ShuffleMBBlock(c1, c2, stride=2, kernel_size=3, expand_ratio=1.75, dropout=dropout),
            ShuffleMBBlock(c2, c2, stride=1, kernel_size=3, expand_ratio=1.75, dropout=dropout),
        )
        self.stage2 = nn.Sequential(
            ShuffleMBBlock(c2, c3, stride=2, kernel_size=5, expand_ratio=2.0, dropout=dropout),
            ShuffleMBBlock(c3, c3, stride=1, kernel_size=5, expand_ratio=2.0, dropout=dropout),
        )
        self.stage3 = nn.Sequential(
            ShuffleMBBlock(c3, c4, stride=2, kernel_size=5, expand_ratio=2.25, dropout=dropout),
            ShuffleMBBlock(c4, c4, stride=1, kernel_size=3, expand_ratio=2.25, dropout=dropout),
        )
        self.stage4 = nn.Sequential(
            ShuffleMBBlock(c4, c5, stride=1, kernel_size=5, expand_ratio=2.5, dropout=dropout, dilation=2),
            ShuffleMBBlock(c5, c5, stride=1, kernel_size=3, expand_ratio=2.5, dropout=dropout, dilation=2),
        )
        self.context = MultiScaleContext(c5, dropout=dropout)
        self.local_illum = nn.Conv2d(c5, 3, 1)
        self.local_conf = nn.Conv2d(c5, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Linear(c5, embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        features = self.stem(x)
        features = self.stage1(features)
        features = self.stage2(features)
        features = self.stage3(features)
        features = self.stage4(features)
        features = self.context(features)

        local_illum = F.normalize(self.local_illum(features), dim=1)
        conf_logits = self.local_conf(features).flatten(1)
        weights = conf_logits.softmax(dim=1).view(x.size(0), 1, features.size(2), features.size(3))
        illum = (local_illum * weights).sum(dim=(2, 3))
        illum = F.normalize(illum, dim=1)

        embed = self.pool(features).flatten(1)
        embed = self.embed(embed)
        entropy = -(weights.flatten(1) * weights.flatten(1).clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        entropy = entropy / torch.log(x.new_tensor(float(features.size(2) * features.size(3))))
        confidence = 1.0 - entropy
        return illum, embed, confidence

class ColorConstancyDenseLocalLogChromaFusion(nn.Module):
    """Fuse visual local estimates, global log-chroma, and dense local log-chroma estimates."""

    def __init__(
        self,
        pretrained=False,
        dropout=0.1,
        width_mult=1.0,
        stem_channels=16,
        visual_embed=128,
        hist_bins=64,
        hist_channels=40,
        hist_embed=96,
        local_grid=16,
        local_embed=96,
        local_channels=48,
        local_encoder="default",
        local_bottleneck_channels=None,
        local_pooling="softmax",
        local_topk=32,
        local_topk_boost=1.0,
        gate_hidden=96,
        residual_hidden=64,
        residual_confidence_clamp=False,
        residual_disagree_scale=4.0,
        residual_confidence_power=1.0,
        residual_min_gate=0.25,
        global_suppression=False,
        global_disagree_scale=4.0,
        global_warm_suppression=0.5,
        global_warm_blue_threshold=0.34,
        global_suppression_min=0.20,
        input_normalized=True,
        **kwargs,
    ):
        super().__init__()
        del pretrained, kwargs
        self.visual = TinyVisualBranch(
            dropout=dropout,
            width_mult=width_mult,
            stem_channels=stem_channels,
            embed_dim=visual_embed,
        )
        self.global_chroma = LogChromaBranch(
            bins=hist_bins,
            channels=hist_channels,
            embed_dim=hist_embed,
            dropout=dropout,
            input_normalized=input_normalized,
        )
        self.local_chroma = DenseLocalLogChromaBranch(
            grid_size=local_grid,
            embed_dim=local_embed,
            hidden_channels=local_channels,
            dropout=dropout,
            input_normalized=input_normalized,
            encoder_type=local_encoder,
            bottleneck_channels=local_bottleneck_channels,
            local_pooling=local_pooling,
            local_topk=local_topk,
            local_topk_boost=local_topk_boost,
        )
        gate_dim = visual_embed + hist_embed + local_embed + 6
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, gate_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 3),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, residual_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 3),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.04))
        self.residual_confidence_clamp = bool(residual_confidence_clamp)
        self.residual_disagree_scale = float(residual_disagree_scale)
        self.residual_confidence_power = float(residual_confidence_power)
        self.residual_min_gate = float(residual_min_gate)
        self.global_suppression = bool(global_suppression)
        self.global_disagree_scale = float(global_disagree_scale)
        self.global_warm_suppression = float(global_warm_suppression)
        self.global_warm_blue_threshold = float(global_warm_blue_threshold)
        self.global_suppression_min = float(global_suppression_min)

    def forward(self, x):
        illum, _ = self.forward_with_aux(x)
        return illum

    def forward_with_aux(self, x):
        visual_illum, visual_embed, visual_conf = self.visual(x)
        global_illum, global_embed, global_conf, global_hist, global_prob, global_logits = self.global_chroma.forward_with_logits(x)
        local_illum, local_embed, local_conf, _, _ = self.local_chroma(x)
        vg_disagree = 1.0 - (visual_illum * global_illum).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        vl_disagree = 1.0 - (visual_illum * local_illum).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        gl_disagree = 1.0 - (global_illum * local_illum).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        reliability = torch.cat(
            [visual_conf, global_conf, local_conf, vg_disagree, vl_disagree, gl_disagree],
            dim=1,
        )
        gate_input = torch.cat([visual_embed, global_embed, local_embed, reliability], dim=1)
        weights = self.gate(gate_input).softmax(dim=1)
        if self.global_suppression:
            global_disagree = 0.5 * (vg_disagree + gl_disagree)
            disagree_factor = torch.exp(-self.global_disagree_scale * global_disagree.clamp_min(0.0))
            global_blue = global_illum[:, 2:3]
            visual_blue_gap = (visual_illum[:, 2:3] - global_blue).clamp_min(0.0)
            local_blue_gap = (local_illum[:, 2:3] - global_blue).clamp_min(0.0)
            low_blue = ((self.global_warm_blue_threshold - global_blue) / max(self.global_warm_blue_threshold, 1e-6)).clamp(0.0, 1.0)
            blue_gap = torch.maximum(visual_blue_gap, local_blue_gap).clamp(0.0, 1.0)
            warm_risk = torch.sigmoid(10.0 * (low_blue + blue_gap - 0.35))
            warm_factor = 1.0 - self.global_warm_suppression * warm_risk
            min_factor = max(0.0, min(1.0, self.global_suppression_min))
            global_factor = (disagree_factor * warm_factor).clamp(min=min_factor, max=1.0)
            weights = torch.cat(
                [
                    weights[:, 0:1],
                    weights[:, 1:2] * global_factor,
                    weights[:, 2:3],
                ],
                dim=1,
            )
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        illum = (
            weights[:, 0:1] * visual_illum
            + weights[:, 1:2] * global_illum
            + weights[:, 2:3] * local_illum
        )
        residual = torch.tanh(self.residual_head(gate_input))
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.12)
        if self.residual_confidence_clamp:
            confidence = (visual_conf + global_conf + local_conf) / 3.0
            disagreement = (vg_disagree + vl_disagree + gl_disagree) / 3.0
            confidence_gate = confidence.clamp(0.0, 1.0).pow(self.residual_confidence_power)
            disagreement_gate = torch.exp(-self.residual_disagree_scale * disagreement.clamp_min(0.0))
            residual_gate = confidence_gate * disagreement_gate
            min_gate = max(0.0, min(1.0, self.residual_min_gate))
            residual_gate = min_gate + (1.0 - min_gate) * residual_gate
            residual = residual * residual_gate
        illum = illum + scale * residual
        aux = {
            "global_heatmap_logits": global_logits,
            "global_heatmap_prob": global_prob,
            "global_hist": global_hist,
        }
        return F.normalize(illum, dim=1), aux
