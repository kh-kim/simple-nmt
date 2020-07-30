import torch
from operator import itemgetter


@torch.no_grad()
def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.grad.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


@torch.no_grad()
def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


def sort_by_length(x, lengths):
    batch_size = x.size(0)
    x = [x[i] for i in range(batch_size)]
    lengths = [lengths[i] for i in range(batch_size)]
    orders = [i for i in range(batch_size)]

    sorted_tuples = sorted(zip(x, lengths, orders), key=itemgetter(1), reverse=True)
    sorted_x = torch.stack([sorted_tuples[i][0] for i in range(batch_size)])
    sorted_lengths = torch.stack([sorted_tuples[i][1] for i in range(batch_size)])
    sorted_orders = [sorted_tuples[i][2] for i in range(batch_size)]

    return sorted_x, sorted_lengths, sorted_orders


def sort_by_order(x, orders):
    batch_size = x.size(0)
    x = [x[i] for i in range(batch_size)]
    
    sorted_tuples = sorted(zip(x, orders), key=itemgetter(1))
    sorted_x = torch.stack([sorted_tuples[i][0] for i in range(batch_size)])

    return sorted_x
