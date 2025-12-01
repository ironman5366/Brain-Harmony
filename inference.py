import torch
from configs.harmonizer.stage0_embed.conf_embed_pretrain import get_config
from brain_datasets.datasets import get_dataset


def get_pos_embed(device, name, **kwargs):
    import libs.position_embedding as pos_embeds

    return getattr(pos_embeds, name)(device, kwargs["model_args"])


def get_encoder(pos_embed, cls_token, name, **kwargs):
    import libs.model as model

    return getattr(model, name)(pos_embed=pos_embed, cls_token=cls_token, **kwargs)


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

        print("Loading fmri checkpoint")
        ckpt_pth = torch.load(
            config.fmri_pretrain_weights, map_location=device, weights_only=True
        )

        prefix = "encoder_ema."
        ema_state_dict = {
            k[len(prefix) :]: v for k, v in ckpt_pth.items() if k.startswith(prefix)
        }
        state_dict = ema_state_dict.copy()
        model_state_dict = fmri_encoder.state_dict()

        for key in list(state_dict.keys()):
            if (
                key in model_state_dict
                and state_dict[key].size() != model_state_dict[key].size()
            ):
                print(
                    f"skip param {key} due to size mismatch{state_dict[key].size()} vs {model_state_dict[key].size()}"
                )
                del state_dict[key]

        msg = fmri_encoder.load_state_dict(state_dict, strict=False)
        print(msg)

        ds = get_dataset(**config.dataset_tr_07)

        row, _, batch_attention_patch_size, attn_mask = ds[0]

        row = row.unsqueeze(0)
        row = row.to(device).to()

        attn_mask = attn_mask.unsqueeze(0)
        attn_mask = attn_mask.to(device)

        print(
            f"Row shape {row.shape}, batch attention patch size {batch_attention_patch_size}, attn mask shape, {attn_mask.shape}, attn mask dtype, {attn_mask.dtype}"
        )

        out = fmri_encoder(row, batch_attention_patch_size, attention_mask=attn_mask)
        print("out shape", out.shape)


if __name__ == "__main__":
    main()
