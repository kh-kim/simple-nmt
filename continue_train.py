def define_argparser():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument(
        '--load_fn',
        required=True,
    )
    p.add_argument(
        '--model_fn',
        type=str,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        type=str,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        type=str,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        type=str,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=-1'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=32'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=18,
        help='Number of epochs to train. Default=18'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=80,
        help='Maximum length of the training sequence. Default=80'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=0.2'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=256,
        help='Word embedding vector dimension. Default=512'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=512,
        help='Hidden size of LSTM. Default=768'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=4'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=5.0'
    )

    p.add_argument(
        '--rl_n_samples',
        type=int,
        default=1,
        help='Number of samples to get baseline. Default=1'
    )
    p.add_argument(
        '--rl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for reinforcement learning. Default=10'
    )
    p.add_argument(
        '--rl_n_gram',
        type=int,
        default=6,
        help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=6'
    )

    p.add_argument(
        '--dsl',
        action='store_true',
        help='Training with Dual Supervised Learning method.'
    )
    p.add_argument(
        '--lm_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for language model training. Default=5'
    )
    p.add_argument(
        '--dsl_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for Dual Supervised Learning. \'--n_epochs\' - \'--dsl_n_epochs\' will be number of epochs for pretraining (without regularization term).'
    )
    p.add_argument(
        '--dsl_lambda',
        type=float,
        default=1e-3,
        help='Lagrangian Multiplier for regularization term. Default=1e-3'
    )
    p.add_argument(
        '--dsl_retrain_lm',
        action='store_true',
        help='Retrain the language models whatever.'
    )
    p.add_argument(
        '--dsl_continue_train_lm',
        action='store_true',
        help='Continue to train the language models watever.'
    )

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
        opt_weight = saved_data['opt'].state_dict()

        from train_w_ignite import main
        main(config, model_weight=model_weight, opt_weight=opt_weight)
    else:
        print('Cannot find file %s' % config.load_fn)
