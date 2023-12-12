import torch
import json


def prepare_sequence(seq, to_ix):
    # Parse sequence from list to torch tensor 
    idxs = [to_ix[w] if w in to_ix else to_ix["<UNK>"] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def read_data(file):
    TRAIN_DATA = []
    with open(file) as file:
        data = json.load(file)
    for id in data.keys():
        TRAIN_DATA.append((data[id]["sentence"], data[id]["labels"]))
    return TRAIN_DATA