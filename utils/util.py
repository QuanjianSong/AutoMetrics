import torch


def get_mean(input_list):
    return torch.Tensor(input_list).mean().item()
