import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


"""Dynamic Embedding from DOFA paper.
Reference:
- https://arxiv.org/abs/2403.15356
- https://github.com/zhu-xlab/DOFA
"""

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, fused_attn=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.fused_attn = fused_attn

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = attn.softmax(dim=-1)
            x = torch.matmul(attn, v)

        x = rearrange(x, "b h n d -> b n (h d)")
        return self.to_out(x)


class Transformer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        fused_attn,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim, heads=heads, dim_head=dim_head, fused_attn=fused_attn
                        ),
                        FeedForward(dim, mlp_dim),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FCBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.l1 = nn.Linear(size, size)
        self.l2 = nn.Linear(size, size)

    def forward(self, x):
        y = F.gelu(self.l1(x))
        y = F.gelu(self.l2(y))
        return x + y


class WavesTransformer(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        wave_dim,
        output_dim,
        num_latent_tokens,
        embed_dim,
        is_decoder,
        num_heads=4,
        num_layers=1,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.is_decoder = is_decoder
        layer = nn.TransformerEncoderLayer(
            d_model=wave_dim,
            nhead=num_heads,
            activation="gelu",
            dropout=0,
            norm_first=False,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

        self.fc_weight = nn.Linear(wave_dim, output_dim)
        self.fc_bias = None if self.is_decoder else nn.Linear(wave_dim, embed_dim)

        self.weight_tokens = nn.Parameter(
            torch.randn(self.num_latent_tokens, wave_dim) * 0.02
        )
        self.bias_token = nn.Parameter(torch.randn(1, wave_dim) * 0.02)

    def forward(self, x):
        x = torch.cat([self.weight_tokens, x, self.bias_token], dim=0)
        out = self.encoder(x)
        weights = self.fc_weight(
            out[self.num_latent_tokens : -1] + x[self.num_latent_tokens : -1]
        )
        bias = None if self.is_decoder else self.fc_bias(out[-1])
        return weights, bias


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        wave_dim,
        num_latent_tokens,
        patch_size,
        embed_dim,
        is_decoder=False,
    ):
        super().__init__()
        self.wave_dim = wave_dim
        self.num_latent_tokens = num_latent_tokens
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.is_decoder = is_decoder
        self.output_dim = (patch_size**2) * embed_dim

        self.weight_generator = WavesTransformer(
            wave_dim,
            self.output_dim,
            self.num_latent_tokens,
            self.embed_dim,
            is_decoder,
        )
        self.fclayer = FCBlock(self.wave_dim)

        self.initialize_weights()

    def forward(self, batch, waves):
        waves = posemb_sincos_1d(waves, self.wave_dim)
        waves = waves.to(batch.device)
        waves = self.fclayer(waves)
        weight, bias = self.weight_generator(waves)

        if self.is_decoder:
            dynamic_weight = rearrange(
                weight,
                "cin (k1 k2 cout) -> (cin k1 k2) cout",
                k1=self.patch_size,
                k2=self.patch_size,
                cout=self.embed_dim,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.linear(batch, dynamic_weight * 0.02, bias=bias)
            x = dynamic_out
        else:
            dynamic_weight = rearrange(
                weight,
                "cin (cout k1 k2) -> cout cin k1 k2",
                k1=self.patch_size,
                k2=self.patch_size,
            )
            if bias is not None:
                bias = rearrange(bias, "b -> (b)")
            dynamic_out = F.conv2d(
                batch, dynamic_weight * 0.02, bias=bias, stride=self.patch_size
            )
            x = rearrange(dynamic_out, "b c h w -> b (h w) c")

        return x, waves

    def initialize_weights(self):
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d_with_gsd(
    h, w, dim, gsd=1.0, temperature: int = 10000, dtype=torch.float32
):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

    gsd = gsd.to(x.device)
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** (2 * omega / dim)) * (gsd / 1.0)  # Adjusted for g

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_1d(waves, dim, temperature: int = 10000, dtype=torch.float32):
    assert (
        dim % 2 == 0
    ), "Feature dimension must be a multiple of 2 for sincos embedding"

    waves = torch.arange(waves) if isinstance(waves, int) else waves

    omega = torch.arange(dim // 2, device=waves.device) / (dim // 2 - 1)
    omega = 1.0 / (temperature**omega)

    scaled_waves = waves[:, None] * omega[None, :]
    pe = torch.cat((scaled_waves.sin(), scaled_waves.cos()), dim=1)

    return pe.type(dtype)
