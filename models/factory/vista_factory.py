from models.vista.modeling import _build_vista2pt5d


def build_vista2pt5d_vit_h(
    checkpoint=None, image_size=1024, encoder_in_chans=3, clip_class_label_prompt=False, patch_embed_3d=False, **kwargs
):
    return _build_vista2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        image_size=image_size,
        clip_class_label_prompt=clip_class_label_prompt,
        patch_embed_3d=patch_embed_3d,
        **kwargs
    )


def build_vista2pt5d_vit_l(
    checkpoint=None, image_size=1024, encoder_in_chans=3, clip_class_label_prompt=False, patch_embed_3d=False, **kwargs
):
    return _build_vista2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        image_size=image_size,
        clip_class_label_prompt=clip_class_label_prompt,
        patch_embed_3d=patch_embed_3d,
        **kwargs
    )


def build_vista2pt5d_vit_b(
    checkpoint=None, image_size=1024, encoder_in_chans=3, clip_class_label_prompt=False, patch_embed_3d=False, **kwargs
):
    return _build_vista2pt5d(
        encoder_in_chans=encoder_in_chans,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        image_size=image_size,
        clip_class_label_prompt=clip_class_label_prompt,
        patch_embed_3d=patch_embed_3d,
        **kwargs
    )


vista_model_registry = {
    "default": build_vista2pt5d_vit_h,
    "vit_h": build_vista2pt5d_vit_h,
    "vit_l": build_vista2pt5d_vit_l,
    "vit_b": build_vista2pt5d_vit_b,
}