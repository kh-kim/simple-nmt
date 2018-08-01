import argparse, sys
from operator import itemgetter

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader
from simple_nmt.seq2seq import Seq2Seq
import simple_nmt.trainer as trainer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('-model', required = True, help = 'Model file name to use')
    p.add_argument('-gpu_id', type = int, default = -1, help = 'GPU ID to use. -1 for CPU. Default = -1')

    p.add_argument('-batch_size', type = int, default = 128, help = 'Mini batch size for parallel inference. Default = 128')
    p.add_argument('-max_length', type = int, default = 255, help = 'Maximum sequence length for inference. Default = 255')
    p.add_argument('-n_best', type = int, default = 1, help = 'Number of best inference result per sample. Default = 1')
    p.add_argument('-beam_size', type = int, default = 5, help = 'Beam size for beam search. Default = 5')
    
    config = p.parse_args()

    return config

def read_text():
    lines = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

    return lines

def to_text(indice, vocab):
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                #line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines

if __name__ == '__main__':
    config = define_argparser()

    saved_data = torch.load(config.model)
    
    train_config = saved_data['config']
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)
    input_size = len(loader.src.vocab)
    output_size = len(loader.tgt.vocab)

    model = Seq2Seq(input_size,
                    train_config.word_vec_dim,
                    train_config.hidden_size,
                    output_size,
                    n_layers = train_config.n_layers,
                    dropout_p = train_config.dropout
                    )
    model.load_state_dict(saved_data['model'])
    model.eval()

    torch.set_grad_enabled(False)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    lines = read_text()
    
    with torch.no_grad():
        while len(lines) > 0:
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            sorted_lines = lines[:config.batch_size]
            lines = lines[config.batch_size:]
            lengths = [len(_) for _ in sorted_lines]        
            orders = [i for i in range(len(sorted_lines))]
            
            sorted_tuples = sorted(zip(sorted_lines, lengths, orders), key = itemgetter(1), reverse = True)
            sorted_lines = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            orders = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            x = loader.src.numericalize(loader.src.pad(sorted_lines), device = 'cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu')

            if config.beam_size == 1:
                y_hat, indice = model.search(x)
                output = to_text(indice, loader.tgt.vocab)

                sorted_tuples = sorted(zip(output, orders), key = itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                batch_indice, _ = model.batch_beam_search(x, 
                                                            beam_size = config.beam_size, 
                                                            max_length = config.max_length, 
                                                            n_best = config.n_best
                                                            )

                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, orders), key = itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')