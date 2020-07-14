import sys
import os.path

import torch

from train import define_argparser


def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for key in vars(prev_config).keys():
        if '--%s' % key not in sys.argv:
            if vars(prev_config).get(key) is not None:
                vars(config)[key] = vars(prev_config)[key]
            else:
                # Missing argument
                print('WARNING!!! Argument "--%s" is not found in current argument parser.\tSaved value:' % key, vars(prev_config)[key])
        else:
            # Argument value is change from saved model.
            print('WARNING!!! Argument "--%s" is not loaded from saved model.\tCurrent value:' % key, vars(config)[key])

    return config


if __name__ == '__main__':
    config = define_argparser(is_continue=True)

    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(config.load_fn, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)

        model_weight = saved_data['model']
        if 'lr' in vars(prev_config).keys():
            opt_weight = None
        else:
            opt_weight = saved_data['opt']

        from train import main
        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)
