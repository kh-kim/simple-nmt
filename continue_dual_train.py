import sys
import os.path

import torch

from dual_train import define_argparser
from dual_train import main

from continue_train import overwrite_config
from continue_train import continue_main


if __name__ == '__main__':
    config = define_argparser(is_continue=True)
    continue_main(config, main)
