import argparse
import sys

import torch
import torch.nn as nn

from data_loader import DataLoader
import data_loader

from simple_nmt.seq2seq import Seq2Seq
from simple_nmt.rnnlm import LanguageModel

from simple_nmt.trainer import Trainer
from simple_nmt.rl_trainer import MinimumRiskTrainer
from simple_nmt.dual_trainer import DualTrainer
from simple_nmt.lm_trainer import LanguageModelTrainer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model',
                   required=True,
                   help='Model file name to save. Additional information would be annotated to the file name.'
                   )
    p.add_argument('--train',
                   required=True,
                   help='Training set file name except the extention. (ex: train.en --> train)'
                   )
    p.add_argument('--valid',
                   required=True,
                   help='Validation set file name except the extention. (ex: valid.en --> valid)'
                   )
    p.add_argument('--lang',
                   required=True,
                   help='Set of extention represents language pair. (ex: en + ko --> enko)'
                   )
    p.add_argument('--gpu_id',
                   type=int,
                   default=-1,
                   help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=-1'
                   )

    p.add_argument('--batch_size',
                   type=int,
                   default=32,
                   help='Mini batch size for gradient descent. Default=32'
                   )
    p.add_argument('--n_epochs',
                   type=int,
                   default=18,
                   help='Number of epochs to train. Default=18'
                   )
    p.add_argument('--verbose',
                   type=int,
                   default=2,
                   help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2'
                   )
    p.add_argument('--early_stop',
                   type=int,
                   default=-1,
                   help='The training will be stopped if there is no improvement this number of epochs. Default=-1'
                   )

    p.add_argument('--max_length',
                   type=int,
                   default=80,
                   help='Maximum length of the training sequence. Default=80'
                   )
    p.add_argument('--dropout',
                   type=float,
                   default=.2,
                   help='Dropout rate. Default=0.2'
                   )
    p.add_argument('--word_vec_dim',
                   type=int,
                   default=256,
                   help='Word embedding vector dimension. Default=512'
                   )
    p.add_argument('--hidden_size',
                   type=int,
                   default=512,
                   help='Hidden size of LSTM. Default=768'
                   )
    p.add_argument('--n_layers',
                   type=int,
                   default=4,
                   help='Number of layers in LSTM. Default=4'
                   )

    p.add_argument('--max_grad_norm',
                   type=float,
                   default=5.,
                   help='Threshold for gradient clipping. Default=5.0'
                   )
    p.add_argument('--adam',
                   action='store_true',
                   help='Use Adam instead of using SGD.'
                   )
    p.add_argument('--lr',
                   type=float,
                   default=1.,
                   help='Initial learning rate. Default=1.0'
                   )
    p.add_argument('--min_lr',
                   type=float,
                   default=.000001,
                   help='Minimum learning rate. Default=.000001'
                   )
    p.add_argument('--lr_decay_start_at',
                   type=int,
                   default=10,
                   help='Start learning rate decay from this epoch.'
                   )
    p.add_argument('--lr_slow_decay',
                   action='store_true',
                   help='Decay learning rate only if there is no improvement on last epoch.'
                   )
    p.add_argument('--lr_decay_rate',
                   type=float,
                   default=.5,
                   help='Learning rate decay rate. Default=0.5'
                   )
    p.add_argument('--train_ratio_per_epoch',
                   type=float,
                   default=1.,
                   help='Ratio of using train-set in one training epoch. Default=1.'
                   )

    p.add_argument('--dsl',
                   action='store_true',
                   help='Training with Dual Supervised Learning method.'
                   )
    p.add_argument('--lm_n_epochs',
                   type=int,
                   default=10,
                   help='Number of epochs for language model training. Default=5'
                   )
    p.add_argument('--dsl_n_epochs',
                   type=int,
                   default=10,
                   help='Number of epochs for Dual Supervised Learning. \'--n_epochs\' - \'--dsl_n_epochs\' will be number of epochs for pretraining (without regularization term).'
                   )
    p.add_argument('--dsl_lambda',
                   type=float,
                   default=1e-3,
                   help='Lagrangian Multiplier for regularization term. Default=1e-3')
    p.add_argument('--dsl_retrain_lm',
                   action='store_true',
                   help='Retrain the language models whatever.')
    p.add_argument('--dsl_continue_train_lm',
                   action='store_true',
                   help='Continue to train the language models watever.')

    p.add_argument('--rl_lr',
                   type=float,
                   default=.01,
                   help='Learning rate for reinforcement learning. Default=.01'
                   )
    p.add_argument('--n_samples',
                   type=int,
                   default=1,
                   help='Number of samples to get baseline. Default=1'
                   )
    p.add_argument('--rl_n_epochs',
                   type=int,
                   default=10,
                   help='Number of epochs for reinforcement learning. Default=10'
                   )
    p.add_argument('--rl_n_gram',
                   type=int,
                   default=6,
                   help='Maximum number of tokens to calculate BLEU for reinforcement learning. Default=6'
                   )

    config = p.parse_args()

    return config


def overwrite_config(config, prev_config):
    # This method provides a compatibility for new or missing arguments.
    for key in vars(prev_config).keys():
        if '--%s' % key not in sys.argv or key == 'model':
            if vars(config).get(key) is not None:
                vars(config)[key] = vars(prev_config)[key]
            else:
                # Missing argument
                print('WARNING!!! Argument "-%s" is not found in current argument parser.\tSaved value:' % key, vars(prev_config)[key])
        else:
            # Argument value is change from saved model.
            print('WARNING!!! Argument "-%s" is not loaded from saved model.\tCurrent value:' % key, vars(config)[key])

    return config


if __name__ == "__main__":
    config = define_argparser()

    import os.path
    # If the model exists, load model and configuration to continue the training.
    if os.path.isfile(config.model):
        saved_data = torch.load(config.model, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)

        prev_config = saved_data['config']
        config = overwrite_config(config, prev_config)
        config.lr = saved_data['current_lr'] if saved_data.get('current_lr') is not None else config.lr
    else:
        saved_data = None

    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    _print_config(config)

    # Load training and validation data set.
    loader = DataLoader(config.train,
                        config.valid,
                        (config.lang[:2], config.lang[-2:]),
                        batch_size=config.batch_size,
                        device=config.gpu_id,
                        max_length=config.max_length,
                        dsl=config.dsl
                        )

    if config.dsl: # In case of the dual supervised training mode is turn-on.
        # Because we must train both models in same time, we need to declare both models.
        models = [Seq2Seq(len(loader.src.vocab),
                          config.word_vec_dim,
                          config.hidden_size,
                          len(loader.tgt.vocab),
                          n_layers=config.n_layers,
                          dropout_p=config.dropout
                          ),
                  Seq2Seq(len(loader.tgt.vocab),
                          config.word_vec_dim,
                          config.hidden_size,
                          len(loader.src.vocab),
                          n_layers=config.n_layers,
                          dropout_p=config.dropout
                          )
                  ]
        # Because we also need to get P(src) and P(tgt), we need language models consist of LSTM.
        language_models = [LanguageModel(len(loader.tgt.vocab),
                                         config.word_vec_dim,
                                         config.hidden_size,
                                         n_layers=config.n_layers,
                                         dropout_p=config.dropout
                                         ),
                           LanguageModel(len(loader.src.vocab),
                                         config.word_vec_dim,
                                         config.hidden_size,
                                         n_layers=config.n_layers,
                                         dropout_p=config.dropout
                                         )
                           ]
        loss_weights = [torch.ones(len(loader.tgt.vocab)),
                        torch.ones(len(loader.src.vocab))
                        ]

        loss_weights[0][data_loader.PAD] = .0
        loss_weights[1][data_loader.PAD] = .0
        # Both loss weights have bais about PAD index.

        # Because language model and translation model have same output size, 
        # (if the output language is the same)
        # we can use same loss function for each direction.
        crits = [nn.NLLLoss(weight=loss_weights[0],
                            reduction='none'
                            ),
                 nn.NLLLoss(weight=loss_weights[1],
                            reduction='none'
                            )
                 ]

        print(models)
        print(language_models)
        print(crits)

        if config.gpu_id >= 0:
            for model, language_model, crit in zip(models, language_models, crits):
                model.cuda(config.gpu_id)
                language_model.cuda(config.gpu_id)
                crit.cuda(config.gpu_id)

        if saved_data is None or config.dsl_retrain_lm is True or config.dsl_continue_train_lm is True:
            # In case of resuming the language model training
            if saved_data is not None and config.dsl_continue_train_lm is True:
                for lm, lm_weight in zip(language_models, saved_data['lms']):
                    lm.load_state_dict(lm_weight)

            # Start to train the language models for --lm_n_epochs
            for language_model, crit, is_src in zip(language_models, crits, [False, True]):
                lm_trainer = LanguageModelTrainer(language_model,
                                                  crit,
                                                  config=config,
                                                  is_src=is_src
                                                  )
                lm_trainer.train(loader.train_iter,
                                 loader.valid_iter,
                                 verbose=config.verbose
                                 )

                language_model.load_state_dict(lm_trainer.best['model']) # Pick the best one.
                print('A language model from best epoch %d is loaded.' % lm_trainer.best['epoch'])

            if saved_data is not None:
                saved_data['lms'] = [lm.state_dict() for lm in language_models]
        else:
            print('Skip the langauge model training.')

        # Start dual supervised learning
        trainer = DualTrainer(models,
                              crits,
                              language_models,
                              config=config,
                              src_vocab=loader.src.vocab,
                              tgt_vocab=loader.tgt.vocab
                              )
        
        if saved_data is not None:
            trainer.best = saved_data
            trainer.get_best_model()

        trainer.train(loader.train_iter, loader.valid_iter, verbose=config.verbose)
    else:
        # Encoder's embedding layer input size
        input_size = len(loader.src.vocab)
        # Decoder's embedding layer input size and Generator's softmax layer output size
        output_size = len(loader.tgt.vocab)
        # Declare the model
        model = Seq2Seq(input_size,
                        config.word_vec_dim,  # Word embedding vector size
                        config.hidden_size,  # LSTM's hidden vector size
                        output_size,
                        n_layers=config.n_layers,  # number of layers in LSTM
                        dropout_p=config.dropout  # dropout-rate in LSTM
                        )

        # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
        # Thus, set a weight for PAD to zero.
        loss_weight = torch.ones(output_size)
        loss_weight[data_loader.PAD] = 0.
        # Instead of using Cross-Entropy loss, we can use Negative Log-Likelihood(NLL) loss with log-probability.
        crit = nn.NLLLoss(weight=loss_weight, 
                          reduction='sum'
                          )

        print(model)
        print(crit)

        # Pass models to GPU device if it is necessary.
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        # Start training. This function maybe equivalant to 'fit' function in Keras.
        trainer = Trainer(model,
                          crit,
                          config=config,
                          src_vocab=loader.src.vocab,
                          tgt_vocab=loader.tgt.vocab
                          )
        # If we have loaded model weight parameters, use that weights for declared model.
        if saved_data is not None:
            trainer.best = saved_data
            trainer.get_best_model()

        trainer.train(loader.train_iter, loader.valid_iter, verbose=config.verbose)

        # Start reinforcement learning.
        if config.rl_n_epochs > 0:
            rl_trainer = MinimumRiskTrainer(model,
                                            crit,
                                            config=config,
                                            src_vocab=loader.src.vocab,
                                            tgt_vocab=loader.tgt.vocab,
                                            )

            if saved_data is not None:
                rl_trainer.best = saved_data
                if rl_trainer.best['epoch'] == config.n_epochs:
                    rl_trainer.best['current_lr'] = config.rl_lr
                rl_trainer.get_best_model()

            rl_trainer.train(loader.train_iter,
                            loader.valid_iter,
                            verbose=config.verbose
                            )
