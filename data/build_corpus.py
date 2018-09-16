import sys, argparse
from random import shuffle

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--input', required=True)
    p.add_argument('--lang', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--valid_ratio', type=float, default=.02)
    p.add_argument('--test_ratio', type=float, default=.0)
    p.add_argument('--no_shuffle', action='store_true')

    config = p.parse_args()

    return config

def read(fn):
    f = open(fn, 'r')

    lines = []
    for line in f:
        lines += [line.strip()]

    f.close()

    return lines

def write(fn, lines):
    if len(lines) > 0:
        print('write %d lines to %s' % (len(lines), fn))
        f = open(fn, 'w')

        for line in lines:
            f.write(line + '\n')

        f.close()

if __name__ == '__main__':
    config = define_argparser()
    
    src_lang = config.lang[:2]
    tgt_lang = config.lang[-2:]

    src_fn = config.input + '.' + src_lang
    tgt_fn = config.input + '.' + tgt_lang
    train_src_fn = config.output + '.train.' + src_lang
    train_tgt_fn = config.output + '.train.' + tgt_lang
    valid_src_fn = config.output + '.valid.' + src_lang
    valid_tgt_fn = config.output + '.valid.' + tgt_lang
    test_src_fn = config.output + '.test.' + src_lang
    test_tgt_fn = config.output + '.test.' + tgt_lang
    
    src_lines = read(src_fn)
    tgt_lines = read(tgt_fn)

    print('total src lines: %d' % len(src_lines))
    print('total tgt lines: %d' % len(tgt_lines))

    assert len(src_lines) == len(tgt_lines)

    combined_lines = list(zip(src_lines, tgt_lines))
    if not config.no_shuffle:
        shuffle(combined_lines)

    length = len(combined_lines)
    if config.test_ratio > 0:
        test_index = int(length * config.test_ratio)
        test_src_lines, test_tgt_lines = list(zip(*(list(combined_lines)[:test_index])))

        write(test_src_fn, test_src_lines)
        write(test_tgt_fn, test_tgt_lines)
    else:
        test_index = 0

    if config.valid_ratio > 0:
        valid_index = test_index + int(length * config.valid_ratio)
        valid_src_lines, valid_tgt_lines = list(zip(*(list(combined_lines)[test_index:valid_index])))

        write(valid_src_fn, valid_src_lines)
        write(valid_tgt_fn, valid_tgt_lines)
    else:
        valid_index = test_index

    train_src_lines, train_tgt_lines = list(zip(*(list(combined_lines)[valid_index:])))
    write(train_src_fn, train_src_lines)
    write(train_tgt_fn, train_tgt_lines)