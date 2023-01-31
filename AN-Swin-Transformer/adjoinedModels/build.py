from .swin_transformer import SwinTransformer

def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm
    
    if config.DAN_TRAINING:
        alpha_downsample = config.DAN_DOWNSAMPLE_COMPRESSION_FACTORS
        alpha_l = config.DAN_COMPRESSION_FACTORS
    else:
        alpha_downsample = config.ADJOINED_DOWNSAMPLE_COMPRESSION_FACTORS
        alpha_l = [[], [], [], []]
        for i, alpha in enumerate(config.ADJOINED_COMPRESSION_FACTORS):
            for j in config.MODEL.SWIN.DEPTHS:
                alpha_l[i].append(alpha)
        alpha_l[0][1] = 1
        alpha_l[3][-1] = 1


    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS,
                                alpha_downsample=alpha_downsample,
                                alpha_l=alpha_l)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
