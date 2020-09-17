import argparse
import os


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        nargs='+',
    )
    p.add_argument(
        '--script_fn',
        required=True,
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
    )
    
    config = p.parse_args()

    return config


if __name__ == '__main__':
    config = define_argparser()

    for fn in config.model_fn:
        cmd = "%s %s %d" % (config.script_fn, fn, config.gpu_id)
        os.system('echo "%s"' % cmd)
        os.system(cmd)
