import torch
from configs.harmonizer.stage0_embed.conf_embed_pretrain import get_config
from brain_datasets.datasets import get_dataset
import torch.nn as nn
from functools import partial


def get_pos_embed(device, name, **kwargs):
    import libs.position_embedding as pos_embeds

    return getattr(pos_embeds, name)(device, kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name, **kwargs):
    import libs.model as model

    return getattr(model, name)(pos_embed=pos_embed, cls_token=cls_token, **kwargs)


def get_decoder(patch_size, **kwargs):
    from libs.flex_transformer import VisionTransformerDecoder

    # TODO (will): if this bad, try just the VisionTransformerPredictor class. Also figure out the diff! Something about predictor projection layer?
    return VisionTransformerDecoder(
        patch_size=patch_size,
        # embed_dim=768,
        # depth=12,
        # num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


device = "cuda"
dtype = torch.bfloat16


def main():
    with torch.autocast(device_type=device, dtype=dtype):
        config = get_config()
        print("config", config)

        print("encoder", config.encoder)

        print("loading pos embed")
        fmri_encoder_pos_embed = get_pos_embed(device, **config.pos_embed)
        print(fmri_encoder_pos_embed)

        print("loading fmri encoder")
        fmri_encoder = get_encoder(fmri_encoder_pos_embed, None, **config.encoder).to(
            device
        )
        print(fmri_encoder)

        print("Loading fmri decoder")
        fmri_decoder = get_decoder(
            pos_embed=fmri_encoder_pos_embed, **config.decoder
        ).to(device)
        print(fmri_decoder)

        print("Loading fmri checkpoint")
        ckpt_pth = torch.load(
            config.fmri_pretrain_weights, map_location=device, weights_only=True
        )

        prefix = "encoder_ema."
        ema_state_dict = {
            k[len(prefix) :]: v for k, v in ckpt_pth.items() if k.startswith(prefix)
        }
        encoder_state_dict = ema_state_dict.copy()
        encoder_model_state_dict = fmri_encoder.state_dict()

        # NOTE (will): This (and the below with decoder) deeply scares me but it's copied from the original!
        for key in list(encoder_state_dict.keys()):
            if (
                key in encoder_model_state_dict
                and encoder_state_dict[key].size()
                != encoder_model_state_dict[key].size()
            ):
                print(
                    f"[encoder] skip param {key} due to size mismatch{encoder_state_dict[key].size()} vs {encoder_model_state_dict[key].size()}"
                )
                del encoder_state_dict[key]

        msg = fmri_encoder.load_state_dict(encoder_state_dict, strict=False)
        print(msg)

        decoder_model_state_dict = fmri_decoder.state_dict()
        decoder_prefix = "decoder."
        decoder_state_dict = {}
        for k, v in ckpt_pth.items():
            if k.startswith(decoder_prefix):
                decoder_state_dict[k.split(decoder_prefix)[1]] = v

        for key in list(decoder_state_dict.keys()):
            if (
                key in decoder_model_state_dict
                and decoder_model_state_dict[key].size()
                != decoder_state_dict[key].size()
            ):
                print(
                    f"[decoder] skip param {key} due to size mismatch{decoder_state_dict[key].size()} vs {decoder_model_state_dict[key].size()}"
                )
                del decoder_state_dict[key]

        decoder_msg = fmri_decoder.load_state_dict(decoder_state_dict, strict=False)
        print(decoder_msg)

        ds = get_dataset(**config.dataset_tr_07)

        row, _, batch_attention_patch_size, attn_mask = ds[0]

        row = row.unsqueeze(0)
        row = row.to(device).to()

        attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(device)

        print(
            f"Row shape {row.shape}, batch attention patch size {batch_attention_patch_size}, attn mask shape, {attn_mask.shape}, attn mask dtype, {attn_mask.dtype}"
        )

        out = fmri_encoder(
            row,
            batch_attention_patch_size,
            attention_mask=attn_mask,
        )
        print("out shape", out.shape)

        # B = out.shape[0]
        # N = out.shape[1]
        # all_indices = torch.arange(N).unsqueeze(0).expand(B, -1).to("cuda")  # [B, N]
        # masks_x = [all_indices]  # all patches as context
        # masks = [all_indices]  # predict all positions
        # print(f"masks_x shape {masks_x[0].shape}, masks shape {masks[0].shape}")
        # decoded = fmri_decoder(out, masks=masks, masks_x=masks_x)
        # print("decoded shape", decoded.shape)


if __name__ == "__main__":
    main()
