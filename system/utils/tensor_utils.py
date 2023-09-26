from pkg_resources import require
import torch

def l2_squared_diff(w1, w2, requires_grad=True):
    """ Return the sum of squared difference between two models. """
    diff = 0.0
    for p1, p2 in zip(w1.parameters(), w2.parameters()):
        if requires_grad:
            diff += torch.sum(torch.pow(p1-p2, 2))
        else:
            diff += torch.sum(torch.pow(p1.data-p2.data, 2))
    return diff

def model_dot_product(w1, w2, requires_grad=True):
    """ Return the sum of squared difference between two models. """
    dot_product = 0.0
    for p1, p2 in zip(w1.parameters(), w2.parameters()):
        if requires_grad:
            dot_product += torch.sum(p1 * p2)
        else:
            dot_product += torch.sum(p1.data * p2.data)
    return dot_product