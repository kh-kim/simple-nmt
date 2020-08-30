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
    
    config = p.parse_args()

    return config


if __name__ == '__main__':
    config = define_argparser()

    for fn in config.model_fn:
        cmd = "%s %s" % (config.script_fn, fn)
        print(cmd)
        os.system(cmd)
