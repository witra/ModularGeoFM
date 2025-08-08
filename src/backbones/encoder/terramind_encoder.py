
import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial
import math
from terratorch.models.backbones.terramind.model.terramind import TerraMind
from terratorch.models.backbones.terramind.model.encoder_embeddings import ImageEncoderEmbedding
from terratorch.models.backbones.terramind.model.terramind import Block, DecoderBlock, LayerNorm

"""
ref: https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/terramind/model/terramind.py#L118
"""

class TerramindEncoder(nn.Module):
    def __init__(self,
                 encoder_embeddings: dict[str, nn.Module],
                 dim,
                 num_heads,
                 mlp_ratio,
                 encoder_depth=12,
                 drop_path_rate_encoder=0,
                 qkv_bias=True,
                 mlp_bias=True,
                 # shared_drop_path=False,
                 qk_norm: bool = False,
                 proj_bias=True,
                 gated_mlp: bool = False, # Make the feedforward gated for e.g. SwiGLU
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: partial | nn.Module = partial(LayerNorm, eps=1e-6),):
        super().__init__()

        # Encoder embeddings & init
        # in this case the encoder embedding refers to image (S2 only) --> ImageEncoderEmbedding
        self.encoder_embeddings = nn.ModuleDict(encoder_embeddings)

        ## Transformer encoder
        dpr_encoder = [x.item() for x in
                       torch.linspace(0, drop_path_rate_encoder, encoder_depth)]  # stochastic depth decay rule

        self.encoder = nn.ModuleList([
            Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_bias=proj_bias,
                  mlp_bias=mlp_bias,
                  drop_path=dpr_encoder[i], act_layer=act_layer, norm_layer=norm_layer, gated_mlp=gated_mlp,
                  qk_norm=qk_norm)
            for i in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(dim)

        # Weight init
        self.init_weights()

    def init_weights(self):
        """Weight initialization following MAE's initialization scheme"""

        for name, m in self.named_modules():
            # Skipping tokenizers to avoid reinitializing them
            if "tokenizer" in name:
                continue
            # Linear
            elif isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                elif 'kv' in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 2 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Embedding
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.init_std)
            # Conv2d
            elif isinstance(m, nn.Conv2d):
                if '.proj' in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def get_num_layers_encoder(self):
        return len(self.encoder)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = set()

        for mod, emb_module in self.encoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'encoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for mod, emb_module in self.decoder_embeddings.items():
            if hasattr(emb_module, 'no_weight_decay'):
                to_skip = emb_module.no_weight_decay()
                to_skip = set([f'decoder_embeddings.{mod}.{name}' for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def forward(self,
                mod_dict: dict[str, dict[str, torch.Tensor]],
                num_encoder_tokens: int, ):
        # Mod dicts
        encoder_mod_dict = {mod: self.encoder_embeddings[mod](d)
                            for mod, d in mod_dict.items()
                            if mod in self.encoder_embeddings}
        encoder_tokens, encoder_emb, encoder_mask, encoder_mod_mask = self.forward_mask_encoder(encoder_mod_dict, num_encoder_tokens)
        # Encoder
        x = encoder_tokens + encoder_emb
        x = self.forward_encoder(x, encoder_mask=encoder_mask)
        return x

    def forward_encoder(self,
                        x: torch.Tensor,
                        encoder_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder.

        Args:
            x (torch.Tensor): Encoder input tokens. Shape (B, N, D) where N is the number of encoder tokens.
            encoder_mask (torch.Tensor): Encoder mask indicating which tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)

        Returns:
            torch.Tensor: Encoder output. Shape (B, N, D)
        """

        for blk in self.encoder:
            x = blk(x, mask=encoder_mask)

        x = self.encoder_norm(x)

        return x

    def forward_mask_encoder(self, mod_dict: dict[str, dict[str, torch.Tensor]], num_encoder_tokens: int) -> tuple[
        torch.Tensor]:
        """Concatenates and mask encoder tensors based on provided modality information.

        This function consolidates encoder tokens from multiple modalities, then selects a specified number of them based on modality information (i.e. masking).

        Args:
            mod_dict (dict): dictionary containing tensors for different modalities.
                            It is expected to have keys for each modality and values
                            containing the modalities' associated tensors.
            num_encoder_tokens (int): Number of encoder tokens to retain after masking.

        Returns:
            tuple:
                - encoder_tokens (torch.Tensor): Selected encoder tokens from all modalities. Shape (B, N, D) where N is the number of selected encoder tokens.
                - encoder_emb (torch.Tensor): Corresponding embeddings for encoder tokens. Shape (B, N, D)
                - encoder_mask (torch.Tensor): A boolean mask indicating which encoder tokens are valid (set to 0 for valid tokens, 1 otherwise). Shape (B, 1, N)
                - mod_mask (torch.Tensor): An integer mask marking the modality type for each encoder token (with -1 indicating unassigned pad tokens). Shape (B, N)

        Notes:
            - If `num_register_tokens` is set and greater than 0, register tokens are added at the beginning of the sequence.
        """
        B = list(mod_dict.values())[0]['tensor'].shape[0]

        encoder_tokens_all, emb_all, encoder_mask_all, mod_mask_all = self.cat_encoder_tensors(mod_dict)

        # Add arange multiplied by small constant to mask so they get sorted in a deterministic way
        mask_arange = torch.arange(encoder_mask_all.shape[1], device=encoder_mask_all.device).unsqueeze(0) * 1e-6
        ids_shuffle = torch.argsort(encoder_mask_all + mask_arange, dim=1)
        # ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_encoder_tokens]

        encoder_tokens = torch.gather(encoder_tokens_all, dim=1,
                                      index=repeat(ids_keep, "b n -> b n d", d=encoder_tokens_all.shape[2]))
        encoder_emb = torch.gather(emb_all, dim=1, index=repeat(ids_keep, "b n -> b n d", d=emb_all.shape[2]))
        encoder_mask = torch.gather(encoder_mask_all, dim=1, index=ids_keep)
        mod_mask = torch.gather(mod_mask_all, dim=1, index=ids_keep)

        if self.num_register_tokens > 0:
            register_tokens = repeat(self.register_tokens, '() n d -> b n d', b=B)
            # We add register tokens at the beginning of the sequence
            encoder_tokens = torch.cat([register_tokens, encoder_tokens], dim=1)
            encoder_emb = torch.cat([torch.zeros_like(register_tokens), encoder_emb], dim=1)
            encoder_mask = torch.cat(
                [torch.zeros((B, register_tokens.shape[1]), dtype=torch.bool, device=encoder_mask.device),
                 encoder_mask], dim=1)
            mod_mask = torch.cat(
                [torch.full((B, register_tokens.shape[1]), -1, dtype=torch.int16, device=mod_mask.device), mod_mask],
                dim=1)

        encoder_tokens[encoder_mask] = 0.
        encoder_emb[encoder_mask] = 0.
        mod_mask[encoder_mask] = -1
        # Mask could be of shape 'b n1 n2' but not needed for masked_fill
        # This means this mask can then be re-used for decoder cross-attention
        encoder_mask = rearrange(encoder_mask, 'b n2 -> b 1 n2')

        return encoder_tokens, encoder_emb, encoder_mask, mod_mask

    def freeze_encoder(self, freeze_embeddings=True):
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        if freeze_embeddings:
            for param in self.encoder_embeddings.parameters():
                param.requires_grad = False

    def freeze_encoder_except_specific_embeddings(self, frozen_embedding_domain):
        frozen_embedding_domain = frozen_embedding_domain.split('-')
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_norm.parameters():
            param.requires_grad = False

        for name, param in self.encoder_embeddings.named_parameters():
            if name.split('.')[0] in frozen_embedding_domain:
                param.requires_grad = False