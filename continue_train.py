def define_argparser():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--load_fn', required=True,)
    p.add_argument('--model_fn', type=str,)
    p.add_argument('--train', type=str,)
    p.add_argument('--valid', type=str,)
    p.add_argument('--lang', type=str,)
    p.add_argument('--gpu_id', type=int, default=-1,)

    p.add_argument('--batch_size', type=int, default=32,)
    p.add_argument('--n_epochs', type=int, default=15,)
    p.add_argument('--verbose', type=int, default=2,)

    p.add_argument('--max_length', type=int, default=80,)
    p.add_argument('--dropout', type=float, default=.2,)
    p.add_argument('--word_vec_size', type=int, default=512,)
    p.add_argument('--hidden_size', type=int, default=768,)
    p.add_argument('--n_layers', type=int, default=4,)
    p.add_argument('--max_grad_norm', type=float, default=5.,)

    p.add_argument('--rl_lr', type=float, default=.01,)
    p.add_argument('--rl_n_samples', type=int, default=1,)
    p.add_argument('--rl_n_epochs', type=int, default=10,)
    p.add_argument('--rl_n_gram', type=int, default=6,)

    p.add_argument('--dsl', action='store_true',)
    p.add_argument('--lm_n_epochs', type=int, default=10,)
    p.add_argument('--dsl_n_epochs', type=int, default=10,)
    p.add_argument('--dsl_lambda', type=float, default=1e-3,)
    p.add_argument('--dsl_retrain_lm', action='store_true',)
    p.add_argument('--dsl_continue_train_lm', action='store_true',)

    config = p.parse_args()

    return config

def overwrite_config(config, prev_config):
    import sys

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
    config = define_argparser()

    import os.path
    import torch

    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.load_fn):
        saved_data = torch.load(config.load_fn, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)

        model_weight = saved_data['model']
        opt_weight = saved_data['opt']

        from train import main
        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)
