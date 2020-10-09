import argparse
import pprint

import torch
from torch import optim
import torch.nn as nn

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
        '--lm_fn',
        required=not is_continue,
        help='LM file name, which is trained with lm_train.py.'
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
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
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
        default=20,
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
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
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
        default=1e+8,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=1,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
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

    p.add_argument(
        '--use_transformer',
        action='store_true',
        help='Set model architecture as Transformer.',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='Number of heads in multi-head attention in Transformer. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def load_lm(fn, language_models):
    saved_data = torch.load(fn, map_location='cpu')

    model_weight = saved_data['model']
    language_models[0].load_state_dict(model_weight[0])
    language_models[1].load_state_dict(model_weight[1])


def get_models(src_vocab_size, tgt_vocab_size, config):
    language_models = [
        LanguageModel(
            tgt_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
        LanguageModel(
            src_vocab_size,
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
    ]

    if config.use_transformer:
        models = [
            Transformer(
                src_vocab_size,
                config.hidden_size,
                tgt_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout,
            ),
            Transformer(
                tgt_vocab_size,
                config.hidden_size,
                src_vocab_size,
                n_splits=config.n_splits,
                n_enc_blocks=config.n_layers,
                n_dec_blocks=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]
    else:
        models = [
            Seq2Seq(
                src_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                tgt_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            Seq2Seq(
                tgt_vocab_size,
                config.word_vec_size,
                config.hidden_size,
                src_vocab_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]

    return language_models, models


def get_crits(src_vocab_size, tgt_vocab_size, pad_index):
    loss_weights = [
        torch.ones(tgt_vocab_size),
        torch.ones(src_vocab_size),
    ]
    loss_weights[0][pad_index] = .0
    loss_weights[1][pad_index] = .0

    crits = [
        nn.NLLLoss(weight=loss_weights[0], reduction='none'),
        nn.NLLLoss(weight=loss_weights[1], reduction='none'),
    ]

    return crits


def get_optimizers(models, config):
    if config.use_transformer:
        optimizers = [
            optim.Adam(models[0].parameters(), betas=(.9, .98)),
            optim.Adam(models[1].parameters(), betas=(.9, .98)),
        ]
    else:
        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

    return optimizers


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length,
        dsl=True,
    )

    src_vocab_size = len(loader.src.vocab)
    tgt_vocab_size = len(loader.tgt.vocab)

    language_models, models = get_models(
        src_vocab_size,
        tgt_vocab_size,
        config
    )

    crits = get_crits(
        src_vocab_size,
        tgt_vocab_size,
        pad_index=data_loader.PAD
    )

    if model_weight is not None:
        for model, w in zip(models + language_models, model_weight):
            model.load_state_dict(w)

    load_lm(config.lm_fn, language_models)

    if config.gpu_id >= 0:
        for lm, seq2seq, crit in zip(language_models, models, crits):
            lm.cuda(config.gpu_id)
            seq2seq.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

    dsl_trainer = DSLTrainer(config)

    optimizers = get_optimizers(models, config)

    if opt_weight is not None:
        for opt, w in zip(optimizers, opt_weight):
            opt.load_state_dict(w)

    if config.verbose >= 2:
        print(language_models)
        print(models)
        print(crits)
        print(optimizers)

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
