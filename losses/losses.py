import torch
from torch import nn
import torch.nn.functional as F


def get_loss(name):
    if  name == "arcface":
        return arcface_loss()
    else:
        raise ValueError()

def arcface_loss(cosine, targ, num_classes, m=.4):
    # this prevents nan when a value slightly crosses 1.0 due to numerical error
    cosine = cosine.clip(-1+1e-7, 1-1e-7)
    # Step 3:
    arcosine = cosine.arccos()
    # Step 4:
    arcosine += F.one_hot(targ, num_classes = num_classes) * m
    # Step 5:
    cosine2 = arcosine.cos()
    # Step 6:
    return F.cross_entropy(cosine2, targ)
