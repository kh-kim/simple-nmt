import argparse

import torch
from torch import optim
import torch.nn as nn

from data_loader import DataLoader
import data_loader

from simple_nmt.seq2seq import Seq2Seq
from simple_nmt.rnnlm import LanguageModel


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=True,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=True,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=True,
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
        default=15,
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
        default=512,
        help='Word embedding vector dimension. Default=512'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
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
        '--rl_lr',
        type=float,
        default=.01,
        help='Learning rate for reinforcement learning. Default=.01'
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
        '--lm_batch_size',
        type=int,
        default=512,
        help='Batch size for language model training. Default=512',
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

    config = p.parse_args()

    return config

def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    if config.dsl:
        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:2], config.lang[-2:]),
            batch_size=config.lm_batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl,
        )

        from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer

        language_models = [
            LanguageModel(
                len(loader.tgt.vocab),
                config.word_vec_size,
                config.hidden_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            LanguageModel(
                len(loader.src.vocab),
                config.word_vec_size,
                config.hidden_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]

        models = [
            Seq2Seq(
                len(loader.src.vocab),
                config.word_vec_size,
                config.hidden_size,
                len(loader.tgt.vocab),
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            Seq2Seq(
                len(loader.tgt.vocab),
                config.word_vec_size,
                config.hidden_size,
                len(loader.src.vocab),
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]

        loss_weights = [
            torch.ones(len(loader.tgt.vocab)),
            torch.ones(len(loader.src.vocab)),
        ]
        loss_weights[0][data_loader.PAD] = .0
        loss_weights[1][data_loader.PAD] = .0

        crits = [
            nn.NLLLoss(weight=loss_weights[0], reduction='none'),
            nn.NLLLoss(weight=loss_weights[1], reduction='none'),
        ]

        print(language_models)
        print(models)
        print(crits)

        if model_weight is not None:
            for model, w in zip(models + language_models, model_weight):
                model.load_state_dict(w)

        if config.gpu_id >= 0:
            for lm, seq2seq, crit in zip(language_models, models, crits):
                lm.cuda(config.gpu_id)
                seq2seq.cuda(config.gpu_id)
                crit.cuda(config.gpu_id)

        for lm, crit in zip(language_models, crits):
            optimizer = optim.Adam(lm.parameters())
            lm_trainer = LMTrainer(config)

            lm_trainer.train(
                lm, crit, optimizer,
                train_loader=loader.train_iter,
                valid_loader=loader.valid_iter,
                src_vocab=loader.src.vocab if lm.vocab_size == len(loader.src.vocab) else None,
                tgt_vocab=loader.tgt.vocab if lm.vocab_size == len(loader.tgt.vocab) else None,
                n_epochs=config.lm_n_epochs,
            )

        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:2], config.lang[-2:]),
            batch_size=config.batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl
        )

        from simple_nmt.dual_trainer import DualSupervisedTrainer as DSLTrainer
        dsl_trainer = DSLTrainer(config)

        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

        if opt_weight is not None:
            for opt, w in zip(optimizers, opt_weight):
                opt.load_state_dict(w)

        dsl_trainer.train(
            models,
            language_models,
            crits,
            optimizers,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            vocabs=[loader.src.vocab, loader.tgt.vocab],
            n_epochs=config.n_epochs + config.dsl_n_epochs,
        )
    else:
        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:2], config.lang[-2:]),
            batch_size=config.batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl
        )

        from simple_nmt.trainer import MaximumLikelihoodEstimationTrainer as MLETrainer
        # Encoder's embedding layer input size
        input_size = len(loader.src.vocab)
        # Decoder's embedding layer input size and Generator's softmax layer output size
        output_size = len(loader.tgt.vocab)
        # Declare the model
        model = Seq2Seq(input_size,
                        config.word_vec_size,  # Word embedding vector size
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

        if model_weight is not None:
            model.load_state_dict(model_weight)

        # Pass models to GPU device if it is necessary.
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        optimizer = optim.Adam(model.parameters())     
        if opt_weight is not None:
            optimizer.load_state_dict(opt_weight)        
        print(optimizer)

        # Start training. This function maybe equivalant to 'fit' function in Keras.
        mle_trainer = MLETrainer(config)
        mle_trainer.train(
            model,
            crit,
            optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab,
            tgt_vocab=loader.tgt.vocab,
            n_epochs=config.n_epochs,
        )

        if config.rl_n_epochs > 0:
            optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)

            from simple_nmt.rl_trainer import MinimumRiskTrainer
            mrt_trainer = MinimumRiskTrainer(config)

            mrt_trainer.train(
                model,
                crit,
                optimizer,
                train_loader=loader.train_iter,
                valid_loader=loader.valid_iter,
                src_vocab=loader.src.vocab,
                tgt_vocab=loader.tgt.vocab,
                n_epochs=config.rl_n_epochs,
            )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
