import math
import numpy as np
import torch
from model_pytorch import Model as KataModel
from torch import nn

def embedMove(katamodel: KataModel, x_spatial, x_next, x_global):
    if 0 == len(x_spatial):
        return x_spatial.new_empty((0, katamodel.c_trunk))

    katamodel.eval()
    with torch.no_grad():
        out = katamodel(x_spatial, x_global)
        n = out.shape[0]  # nr of parallel boards/moves
        c = out.shape[1]
        wh = out.shape[2] * out.shape[3]
        out = out.reshape(n, c, wh)  # [N, 384, 19*19]
        x_next = x_next.reshape(n, 1, wh)  # [N, 1, 19*19]
        out = out * x_next  # select played locations to extract the feature vectors as embeddings
        return out.sum(dim=2)  # [N, 384]

class StrengthModel(torch.nn.Module):
    def __init__(self, katamodel: KataModel):
        super(StrengthModel, self).__init__()
        self.katamodel = katamodel
        self.at_once = 30  # how many boards max to feed through the kata network in one go (limited memory)

        # strength model weights: learn to interpret the set of move embeddings
        featureDims = self.katamodel.c_trunk
        hiddenDims = 32
        self.layer1 = nn.Sequential(
            nn.Linear(featureDims, hiddenDims),
            nn.ReLU()
        )
        self.rating = nn.Linear(hiddenDims, 1)
        self.weights = nn.Linear(hiddenDims, 1)
        self.softmax = nn.Softmax(dim=0)
        self.SCALE = 400 / math.log(10)  # Scale outputs to Elo/Glicko-like numbers

    def forward(self, x_spatial, x_next, x_global, batch_lengths = None):
        """
        If batch_lengths are [2, 3, 4], we expect x_spatial.shape == [9, *, 19, 19].
        These are 9 positions from a batch of 3 inputs, the first has 2, the second 3
        and the third 4 positions.
        """

        # We cannot pass the entire input to kata network at once, especially a whole minibatch,
        # as that requires gigabytes upon gigabytes of GPU memory.
        split_up = lambda t: torch.split(t, self.at_once, dim=0)
        split_spatial, split_next, split_global = map(split_up, (x_spatial, x_next, x_global))
        # import pdb; pdb.set_trace()

        # Run each sub-batch through the model
        embeddings = (embedMove(self.katamodel, p, l, g) for p, l, g in zip(split_spatial, split_next, split_global))
        embeddings = torch.cat(list(embeddings), dim=0)

        # batch_lengths specifies length of manually packed sequences in the batch
        if batch_lengths is not None:
            clens = np.cumsum([0] + batch_lengths)
        else:
            clens = [0, len(embeddings)]  # assume one input (eg in evaluation)
        h = self.layer1(embeddings)
        r = self.rating(h).squeeze(-1)
        z = self.weights(h).squeeze(-1)

        # predict one rating for each part in the batch
        parts = zip(clens[:-1], clens[1:])
        preds = [self._sumBySoftmax(r, z, start, end) for start, end in parts]
        return self.SCALE * torch.stack(preds)

    def _sumBySoftmax(self, r, z, start, end):
        if start == end:
            DEFAULT_PRED = 7.6699353278706015  # default prediction = 1332.40 Glicko
            return torch.tensor(DEFAULT_PRED, device=r.device)
        rslice = r[start:end]
        zslice = z[start:end]
        zslice = self.softmax(zslice)
        return torch.sum(zslice * rslice)

