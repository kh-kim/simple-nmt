import argparse

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True)
    p.add_argument('-train', required = True)
    p.add_argument('-valid', required = True)
    p.add_argument('-lang', required = True)
    p.add_argument('-gpu_id', type = int, default = -1)

    p.add_argument('-batch_size', type = int, default = 32)
    p.add_argument('-n_epochs', type = int, default = 20)
    p.add_argument('-print_every', type = int, default = 50)
    p.add_argument('-early_stop', type = int, default = 3)

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

    config = p.parse_args()

    return config

if __name__ == "__main__":
    config = define_argparser()
    
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

    trainer.train_epoch(model, 
                        criterion, 
                        loader.train_iter, 
                        loader.valid_iter, 
                        config,
                        others_to_save = {'src_vocab': loader.src.vocab, 'tgt_vocab': loader.tgt.vocab}
                        )
