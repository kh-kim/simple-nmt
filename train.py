import argparse, sys

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer
import simple_nmt.rl_trainer as rl_trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True)
    p.add_argument('-train', required = True)
    p.add_argument('-valid', required = True)
    p.add_argument('-lang', required = True)
    p.add_argument('-gpu_id', type = int, default = -1)

    p.add_argument('-batch_size', type = int, default = 32)
    p.add_argument('-n_epochs', type = int, default = 10)
    p.add_argument('-print_every', type = int, default = 50)
    p.add_argument('-early_stop', type = int, default = -1)

    p.add_argument('-max_length', type = int, default = 80)
    p.add_argument('-dropout', type = float, default = .2)
    p.add_argument('-word_vec_dim', type = int, default = 512)
    p.add_argument('-hidden_size', type = int, default = 1024)
    p.add_argument('-n_layers', type = int, default = 4)   
    
    p.add_argument('-max_grad_norm', type = float, default = 5.)
    p.add_argument('-adam', action = 'store_true', help = 'Use Adam instead of using SGD.')
    p.add_argument('-lr', type = float, default = 1.)
    p.add_argument('-min_lr', type = float, default = .000001)
    p.add_argument('-lr_decay_start_at', type = int, default = 10, help = 'Start learning rate decay from this epoch.')
    p.add_argument('-lr_slow_decay', action = 'store_true', help = 'Decay learning rate only if there is no improvement on last epoch.')
    p.add_argument('-lr_decay_rate', type = float, default = .5)

    p.add_argument('-rl_lr', type = float, default = .01)
    p.add_argument('-n_samples', type = int, default = 5)
    p.add_argument('-rl_n_epochs', type = int, default = 5)
    p.add_argument('-rl_ratio_per_epoch', type = float, default = .1)

    config = p.parse_args()

    return config

def overwrite_config(config, prev_config):
    for key in vars(prev_config).keys():
            if '-%s' % key not in sys.argv or key == 'model':
                if vars(config).get(key) is not None:
                    vars(config)[key] = vars(prev_config)[key]
                else:
                    print('WARNING!!! Argument "-%s" is not found in current argument parser.\tSaved value:' % key, vars(prev_config)[key])
            else:
                print('WARNING!!! Argument "-%s" is not loaded from saved model.\tCurrent value:' % key, vars(config)[key])

    return config


if __name__ == "__main__":
    config = define_argparser()

    import os.path
    if os.path.isfile(config.model):
        saved_data = torch.load(config.model)
    
        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)
        config.lr = saved_data['current_lr']
    else:
        saved_data = None
    
    loader = DataLoader(config.train, 
                        config.valid, 
                        (config.lang[:2], config.lang[-2:]), 
                        batch_size = config.batch_size, 
                        device = config.gpu_id, 
                        max_length = config.max_length
                        )

    input_size = len(loader.src.vocab)
    output_size = len(loader.tgt.vocab)
    model = Seq2Seq(input_size, 
                    config.word_vec_dim, 
                    config.hidden_size, 
                    output_size, 
                    n_layers = config.n_layers, 
                    dropout_p = config.dropout
                    )

    loss_weight = torch.ones(output_size)
    loss_weight[data_loader.PAD] = 0
    criterion = nn.NLLLoss(weight = loss_weight, size_average = False)

    print(model)
    print(criterion)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        criterion.cuda(config.gpu_id)

    if saved_data is not None:
        model.load_state_dict(saved_data['model'])

    trainer.train_epoch(model, 
                        criterion, 
                        loader.train_iter, 
                        loader.valid_iter, 
                        config,
                        start_epoch = saved_data['epoch'] if saved_data is not None else 1,
                        others_to_save = {'src_vocab': loader.src.vocab, 'tgt_vocab': loader.tgt.vocab}
                        )

    if config.rl_n_epochs > 0:
        rl_trainer.train_epoch(model,
                            criterion,
                            loader.train_iter,
                            loader.valid_iter,
                            config,
                            start_epoch = (saved_data['epoch'] - config.n_epochs) if saved_data is not None else 1,
                            others_to_save = {'src_vocab': loader.src.vocab, 'tgt_vocab': loader.tgt.vocab}
                            )
