from .segmenter import DETRIS
from loguru import logger


def build_segmenter(args):
    model = DETRIS(args)
    backbone = []
    head = []
    neck = []
    decoder = []
    proj = []
    fix = []
    for k, v in model.named_parameters():
        if (k.startswith('txt_backbone') and 'positional_embedding' not in k  or 'dinov2' in k) and v.requires_grad:
            backbone.append(v)
        elif v.requires_grad:
            head.append(v)
            if 'neck' in k:
                neck.append(v)
            elif 'decoder' in k:
                decoder.append(v)
            elif 'proj' in k:
                proj.append(v)
        else:
            fix.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    
    n_backbone_parameters = sum(p.numel() for p in backbone)
    logger.info(f'number of updated params (Backbone): {n_backbone_parameters}.')
    n_head_parameters = sum(p.numel() for p in head)
    logger.info(f'number of updated params (Head)    : {n_head_parameters}')
    n_neck_parameters = sum(p.numel() for p in neck)
    logger.info(f'number of updated params (neck)    : {n_neck_parameters}')
    n_decoder_parameters = sum(p.numel() for p in decoder)
    logger.info(f'number of updated params (decoder)    : {n_decoder_parameters}')
    n_proj_parameters = sum(p.numel() for p in proj)
    logger.info(f'number of updated params (proj)    : {n_proj_parameters}')
    n_fixed_parameters = sum(p.numel() for p in fix)
    logger.info(f'number of fixed params             : {n_fixed_parameters}')
    return model, param_list
