import torch


def argmax(vec):
    # Return the argmax as an integer
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec):
    # Compute log sum exp in a numerically stable way for the forward algorithm
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))