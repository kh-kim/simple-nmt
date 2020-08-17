import sys
import os.path

import torch

from dual_train import define_argparser
from dual_train import main

from continue_train import overwrite_config


if __name__ == '__main__':
    config = define_argparser(is_continue=True)

    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(config.load_fn, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)
