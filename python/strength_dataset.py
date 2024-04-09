from __future__ import annotations
import os
from os.path import exists
import csv
import itertools
import re
from typing import List, Optional
import numpy as np
import torch
from sgfmill import sgf
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class PlayerGameEntry:
    """Represents the data for one of the players in a game"""
    def __init__(self, name: str, rating: float, prevGame: Optional[GameEntry]):
        self.name = name
        self.rating = rating
        self.predictedRating = None
        self.prevGame = prevGame
        self.features = None
        self.recentMoves = None

class GameEntry:
    """Represents one game in the list"""
    def __init__(self, sgfPath: str, black: PlayerGameEntry, white: PlayerGameEntry, score: float, marker: str):
        self.sgfPath = sgfPath
        self.black = black
        self.white = white
        self.score = score
        self.predictedScore = None
        self.marker = marker

    def playerEntry(self, name: str):
        if self.black.name == name:
            return self.black
        elif self.white.name == name:
            return self.white
        else:
            raise Exception(f"Player {name} does not occur in game {self.sgfPath}.")

class StrengthDataset(Dataset):
    """Load the dataset from a CSV list file"""
    featureDims = 6  # TODO, adapt to whichever features we currently use

    def __init__(self, listpath: str, featuredir: str, marker: str):
        self.featuredir = featuredir
        self.players: Dict[str, GameEntry] = {}  # stores last occurrence of player
        self.games = List[GameEntry]

        with open(listpath, 'r') as listfile:
            reader = csv.DictReader(listfile)
            self.games = [self._makeGameEntry(r) for r in reader]

        self.marked = [g for g in self.games if g.marker == marker]

    def __len__(self):
        return len(self.marked)

    def __getitem__(self, idx):
        """Load recent move features on demand"""
        game = self.marked[idx]
        if game.black.recentMoves is None:
            self._fillRecentMoves(game.black, game)
        if game.white.recentMoves is None:
            self._fillRecentMoves(game.white, game)
        return (game.black.recentMoves, game.white.recentMoves, game.black.rating, game.white.rating, game.score)

    def write(self, outpath: str):
        """Write to CSV file including predictions data where applicable"""
        with open(outpath, 'w') as outfile:
            fieldnames = ['File','Player White','Player Black','Score','BlackRating','WhiteRating','PredictedScore','PredictedBlackRating','PredictedWhiteRating','Set']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for game in self.marked:
                row = {
                    'File': game.sgfPath,
                    'Player White': game.white.name,
                    'Player Black': game.black.name,
                    'Score': game.score,
                    'BlackRating': game.black.rating,
                    'WhiteRating': game.white.rating,
                    'PredictedScore': game.predictedScore,
                    'PredictedBlackRating': game.black.predictedRating,
                    'PredictedWhiteRating': game.white.predictedRating,
                    'Set': game.marker
                }
                writer.writerow(row)

    def writeRecentMoves(self):
        """Write to CSV files in the feature directory all the recent move specifications"""
        for game in self.marked:
            print(game.sgfPath)
            self._writeRecentMovesIfNotExists(game.black.name, game)
            self._writeRecentMovesIfNotExists(game.white.name, game)

    @staticmethod
    def _getScore(row):
        if "Score" in row.keys():
            return float(row["Score"])
        elif "Winner" in row.keys():
            winner = row["Winner"]
        elif "Judgement" in row.keys():
            winner = row["Judgement"]

        w = winner[0].lower()
        if 'b' == w:
            return 1
        elif 'w' == w:
            return 0
        else:
            print(f"Warning! Undecided game in dataset: {row['File']}")
            return 0.5  # Jigo and undecided cases

    @staticmethod
    def _isSelected(self, row, setmarker):
        if "Set" in row.keys() and '*' != setmarker:
            return setmarker == row["Set"]
        else:
            return True

    def _makePlayerGameEntry(self, row, color):
        name = row['Player ' + color]
        rating = float(row[color + 'Rating'])
        prevGame = self.players.get(name, None)
        return PlayerGameEntry(name, rating, prevGame)

    def _makeGameEntry(self, row):
        sgfPath = row['File']
        black = self._makePlayerGameEntry(row, 'Black')
        white = self._makePlayerGameEntry(row, 'White')
        score = StrengthDataset._getScore(row)
        marker = row["Set"]
        game = GameEntry(sgfPath, black, white, score, marker)
        self.players[black.name] = game  # set last occurrence
        self.players[white.name] = game  # set last occurrence
        return game

    def _loadFeatures(self, game: GameEntry):
        sgfPathWithoutExt, _ = os.path.splitext(game.sgfPath)
        game.black.features = StrengthDataset._readFeaturesFromFile(f"{self.featuredir}/{sgfPathWithoutExt}_BlackFeatures.bin");
        game.white.features = StrengthDataset._readFeaturesFromFile(f"{self.featuredir}/{sgfPathWithoutExt}_WhiteFeatures.bin");

    @staticmethod
    def _readFeaturesFromFile(path: str):
        FEATURE_HEADER = 0xfea70235  # feature file needs to start with this marker

        with open(path, 'rb') as file:
            # Read and validate the header
            header = np.fromfile(file, dtype=np.uint32, count=1)
            if header.size == 0 or header[0] != FEATURE_HEADER:
                raise IOError("Failed to read from feature file " + path)

            features_flat = np.fromfile(file, dtype=np.float32)

        count = len(features_flat) // StrengthDataset.featureDims
        return torch.from_numpy(features_flat).reshape(count, StrengthDataset.featureDims)

    def _fillRecentMoves(self, entry: PlayerGameEntry, game: GameEntry):
        """
        Place strength model input tensors in game.black/white.recentMoves.
        The data must exist in the feature cache directory.
        """
        if entry is game.black:
            color = 'Black'
        elif entry is game.white:
            color = 'White'
        else:
            raise Exception(f"Entry for player {entry.name} is not from game {game.sgfPath}.")

        sgfPathWithoutExt, _ = os.path.splitext(game.sgfPath)
        recentpath = f"{self.featuredir}/{sgfPathWithoutExt}_{color}RecentMoves.npz"

        with np.load(recentpath) as npz:
            binaryInputNCHW = npz["binaryInputNCHW"]
            locInputNCHW = npz["locInputNCHW"]
            globalInputNC = npz["globalInputNC"]
        del npz

        binaryInputNCHW = torch.from_numpy(binaryInputNCHW) #.to(device)
        locInputNCHW = torch.from_numpy(locInputNCHW) #.to(device)
        globalInputNC = torch.from_numpy(globalInputNC) #.to(device)
        entry.recentMoves = (binaryInputNCHW, locInputNCHW, globalInputNC)

    def _fillRecentMovesFromPocFeatures(self, player: str, game: GameEntry, window: int = 1000):
        recentMoves = torch.empty(0, StrengthDataset.featureDims)
        count = 0
        gamePlayerEntry = game.playerEntry(player)
        historic = gamePlayerEntry.prevGame

        while count < window and historic is not None:
            sgfPathWithoutExt, _ = os.path.splitext(historic.sgfPath)
            entry = historic.playerEntry(player)
            if entry.features is None:
                color = 'Black' if historic.black.name == player else 'White'
                featurepath = f"{self.featuredir}/{sgfPathWithoutExt}_{color}Features.bin"
                entry.features = StrengthDataset._readFeaturesFromFile(featurepath);
                recentMoves = torch.cat((entry.features, recentMoves), dim=0)
                count += entry.features.shape[0]

            # trim to window size if necessary
            if count > window:
                recentMoves = recentMoves[slice(-window, None), slice(None)]

            historic = entry.prevGame

        gamePlayerEntry.recentMoves = recentMoves

    @staticmethod
    def _countGameMoves(path: str):
        # We don't want to spend the time and really parse the SGF here, so let's do crude main-variation parsing.
        # From the first move indicated by "B[", the main variation always comes first.
        # Alternative variations may be present after a close paren to the main variation, so stop at that.
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.read()

        count = 0
        bpattern, wpattern = re.compile(r'\WB\[\w'), re.compile(r'\WW\[\w')
        black_next = True
        match = bpattern.search(contents)
        if not match:
            import pdb; pdb.set_trace()
        limit = contents.find(')', match.end())
        contents = contents[:limit] if limit >= 0 else contents

        while match:
            count += 1
            contents = contents[match.end():]
            black_next = not black_next
            pattern = bpattern if black_next else wpattern
            match = pattern.search(contents)

        return count

    def _writeRecentMovesIfNotExists(self, player: str, game: GameEntry, window: int = 1000):
        recentMoves = torch.empty(0, StrengthDataset.featureDims)
        count = 0
        gamePlayerEntry = game.playerEntry(player)
        historic = gamePlayerEntry.prevGame

        color = 'Black' if game.black.name == player else 'White'
        sgfPathWithoutExt, _ = os.path.splitext(game.sgfPath)
        recentpath = f"{self.featuredir}/{sgfPathWithoutExt}_{color}RecentMoves.csv"

        if exists(recentpath):
            return  # this allows us to resume previously interrupted recent moves extraction

        os.makedirs(os.path.dirname(recentpath), exist_ok=True) # ensure dir exists

        with open(recentpath, 'w') as recentfile:
            writer = csv.DictWriter(recentfile, fieldnames=['File','StartMove','Count'])
            writer.writeheader()
            while count < window and historic is not None:
                entry = historic.playerEntry(player)
                color = 'Black' if historic.black.name == player else 'White'
                gamemoves = StrengthDataset._countGameMoves(historic.sgfPath)
                base = 0 if 'Black' == color else 1
                mymoves = range(base, gamemoves, 2)
                newcount = count + len(mymoves)
                overshoot = max(0, newcount - window)
                count = min(newcount, window)
                startmove = mymoves[overshoot]
                writer.writerow({'File': historic.sgfPath, 'StartMove': startmove, 'Count': len(mymoves)-overshoot})
                historic = entry.prevGame

def pad_collate(batch):
    brecent, wrecent, brating, wrating, score = zip(*batch)
    b_spatial, b_next, b_global = zip(*brecent)
    w_spatial, w_next, w_global = zip(*wrecent)
    blens = [r.shape[0] for r in b_spatial]
    wlens = [r.shape[0] for r in w_spatial]
    b_spatial, b_next, b_global, w_spatial, w_next, w_global = map(lambda t: torch.cat(t, dim=0), (b_spatial, b_next, b_global, w_spatial, w_next, w_global))
    brating, wrating, score = map(torch.Tensor, (brating, wrating, score))
    return (b_spatial, b_next, b_global), (w_spatial, w_next, w_global), blens, wlens, brating, wrating, score

class StrengthDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = pad_collate
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    from load_model import load_model as load_katamodel
    from model_pytorch import Model as KataModel
    device = "cuda"
    modelPath = "/home/user/source/katago/models/kata1-b18c384nbt-s9131461376-d4087399203.ckpt"


    # with np.load(trunkNpzPath) as npz:
    #     trunkOutputNCHW = torch.from_numpy(npz["trunkOutputNCHW"]).to(device)
    # del npz

    katamodel, _, other_state_dict = load_katamodel(modelPath, use_swa=False, device=device)
    print("Test trunks using model " + modelPath)

    def dumpTrunkOutput(npzPath, outPath, katamodel):
        with np.load(npzPath) as npz:
            binaryInputNCHW = torch.from_numpy(npz["binaryInputNCHW"]).to(device)
            # locInputNCHW = torch.from_numpy(npz["locInputNCHW"]).to(device)
            globalInputNC = torch.from_numpy(npz["globalInputNC"]).to(device)
        del npz

        katamodel.eval()
        with torch.no_grad():
            out = katamodel(binaryInputNCHW, globalInputNC)

        with open(outPath, 'w') as file:
            for value in out.flatten():
                file.write(f"{value.item():.6f}\n")


    dumpTrunkOutput("featurecache/dataset/2006/02/10/13281-DaoLin-udhar_nabresh_BlackInputs.npz", "stuff/13281-PyOutput.txt", katamodel)
    # dumpTrunkOutput("stuff/13056-Input.npz", "stuff/13056-PyOutput.txt", katamodel)
    # dumpTrunkOutput("stuff/13788-Input.npz", "stuff/13788-PyOutput.txt", katamodel)
    # dumpTrunkOutput("stuff/13801-Input.npz", "stuff/13801-PyOutput.txt", katamodel)
