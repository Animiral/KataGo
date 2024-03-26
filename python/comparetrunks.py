# compare trunk computed by pytorch-model vs trunk from feature cache
import argparse
import math
import sys
import numpy as np
import torch
from datetime import datetime

from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

from strength_dataset import StrengthDataset, StrengthDataLoader
from strengthmodel import StrengthModel
from load_model import load_model as load_katamodel
from model_pytorch import Model as KataModel

device = "cuda"


def read_floats_to_tensor(file_path):
    with open(file_path, 'r') as file:
        floats = [float(line.strip()) for line in file]
    return torch.tensor(floats)


def to_device(*args):
    if 1 == len(args):
        if type(args[0]) == torch.Tensor:
            return args[0].to(device)
        else:
            return to_device(*args[0])
    else:
        return tuple(to_device(a) for a in args)

def main(args):
    file_path_1 = args[1]
    file_path_2 = args[2]

    tensor_1 = read_floats_to_tensor(file_path_1)
    tensor_2 = read_floats_to_tensor(file_path_2)

    if tensor_1.shape != tensor_2.shape:
        raise ValueError("The tensors are of different shapes.")

    mse = torch.mean((tensor_1 - tensor_2) ** 2)
    print(f"MSE: {mse.item()}")



    # recentmovefile = args["recentmovefile"]  # contains input tensors
    # trunkfile = args["trunkfile"]  # contains precomputed trunk output
    # katamodelfile = args["katamodel"]

    # print(f"Load recent moves from {recentmovefile}")
    # print(f"Load precomputed trunk from {trunkfile}")
    # print(f"KataGo model at {katamodelfile}")
    # print(f"Device: {device}")

    # with np.load(recentmovefile) as npz:
    #     binaryInputNCHW = npz["binaryInputNCHW"]
    #     locInputNCHW = npz["locInputNCHW"]
    #     globalInputNC = npz["globalInputNC"]
    # del npz

    # with np.load(trunkfile) as npz:
    #     trunkOutputNCHW = npz["trunkOutputNCHW"]
    #     locInputNCHW2 = npz["locInputNCHW"]
    # del npz

    # locInputNCHW = torch.from_numpy(locInputNCHW)
    # locInputNCHW2 = torch.from_numpy(locInputNCHW2)
    # print(f"locs equal? {torch.allclose(locInputNCHW, locInputNCHW2)}")

    # katamodel, _, other_state_dict = load_katamodel(katamodelfile, use_swa=False, device=device)

    # binaryInputNCHW = torch.from_numpy(binaryInputNCHW)
    # globalInputNC = torch.from_numpy(globalInputNC)
    # trunkOutputNCHW = torch.from_numpy(trunkOutputNCHW)

    # binaryInputNCHW = binaryInputNCHW[:5]
    # globalInputNC = globalInputNC[:5]
    # trunkOutputNCHW = trunkOutputNCHW[:5]

    # print(f"shapes: {binaryInputNCHW.shape}, {globalInputNC.shape}, {trunkOutputNCHW.shape}")

    # katamodel.eval()
    # with torch.no_grad():
    #     for b, g, t in zip(binaryInputNCHW, globalInputNC, trunkOutputNCHW):
    #         b, g, t = map(lambda x: x.unsqueeze(0), (b, g, t))
    #         b, g, t = b.to(device), g.to(device), t.to(device)
    #         print(f"b g t shapes: {b.shape}, {g.shape}, {t.shape}")
    #         out = katamodel(b, g)
    #         diff = out - t
    #         print(f"diff shape: {diff.shape}")
    #         print(torch.sum(abs(diff), dim=(0, 1, 2, 3)))

    print("Done!")

if __name__ == "__main__":
    description = """
    Compare trunk results for debugging.
    """

    # parser = argparse.ArgumentParser(description=description,add_help=False)
    # required_args = parser.add_argument_group('required arguments')
    # optional_args = parser.add_argument_group('optional arguments')
    # optional_args.add_argument(
    #     '-h',
    #     '--help',
    #     action='help',
    #     default=argparse.SUPPRESS,
    #     help='show this help message and exit'
    # )
    # required_args.add_argument('recentmovefile', help='NPZ of recent moves', type=str)
    # required_args.add_argument('trunkfile', help='NPZ with precomputed trunk output of recent moves', type=str)
    # required_args.add_argument('katamodel', help='KataGo model file', type=str)

    # args = vars(parser.parse_args())
    args = sys.argv
    main(args)
