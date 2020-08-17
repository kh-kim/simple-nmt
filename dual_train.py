import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

from transformers import get_linear_schedule_with_warmup

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.models.seq2seq import Seq2Seq
from simple_nmt.models.transformer import Transformer
from simple_nmt.models.rnnlm import LanguageModel

from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer
from simple_nmt.dual_trainer import DualSupervisedTrainer as DSLTrainer


def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        required=not is_continue,
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        required=not is_continue,
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        required=not is_continue,
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        required=not is_continue,
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=15,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=80,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )

    p.add_argument(
        '--lm_n_epochs',
        type=int,
        default=10,
        help='Number of epochs for language model training. Default=%(default)s'
    )
    p.add_argument(
        '--lm_batch_size',
        type=int,
        default=512,
        help='Batch size for language model training. Default=%(default)s',
    )

    p.add_argument(
        '--dsl_n_warmup_epochs',
        type=int,
        default=2,
        help='Number of warmup epochs for Dual Supervised Learning. Default=%(default)s'
    )
    p.add_argument(
        '--dsl_lambda',
        type=float,
        default=1e-3,
        help='Lagrangian Multiplier for regularization term. Default=%(default)s'
    )

    config = p.parse_args()

    return config


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.lm_batch_size,
        device=config.gpu_id,
        max_length=config.max_length,
        dsl=True,
    )

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
        dsl=True,
    )

    dsl_trainer = DSLTrainer(config)

    optimizers = [
        optim.Adam(models[0].parameters()),
        optim.Adam(models[1].parameters()),
    ]

    if config.verbose >= 2:
        print(language_models)
        print(models)
        print(crits)
        print(optimizers)

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
        n_epochs=config.n_epochs,
        lr_schedulers=None,
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
