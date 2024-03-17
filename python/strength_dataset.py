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

from board import Board
from features import Features

max_board_size = 19  # used for filling net input tensors

# only required to build input tensors of recent moves
class GameState:
    def __init__(self,board_size):
        self.board_size = board_size
        self.board = Board(size=board_size)
        self.moves = []
        self.boards = [self.board.copy()]

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

def makeTensor(board, pla, loc):
    """Build an input to the KataGo network"""

def generateBoards(sgfPath, startMove, count):
    """Generator for all position data as specified in a recent move record"""
    with open(sgfPath, 'r') as f:
        sgf_data = f.read()
    game = sgf.Sgf_game.from_string(sgf_data)
    ms = game.get_main_sequence()
    gs = GameState(max_board_size)

    # get to starting position, then generate every 2nd board position
    for i in range(startMove + 2*count - 1):
        color, moveloc = ms[i].get_move()
        if moveloc is None:
            pla = Board.EMPTY
            loc = Board.PASS_LOC
        else:
            pla = {'b': Board.BLACK, 'w': Board.WHITE}[color]
            loc = gs.board.loc(*moveloc)
        if i > startMove and (i % 2) == (startMove % 2):
            yield gs, pla, loc

        if loc != Board.PASS_LOC:
            gs.board.play(pla,loc)
            gs.moves.append((pla,loc))
            gs.boards.append(gs.board.copy())

class StrengthDataset(Dataset):
    """Load the dataset from a CSV list file"""
    featureDims = 6  # TODO, adapt to whichever features we currently use

    def __init__(self, listpath: str, featuredir: str, marker: str, katamodel):
        self.featuredir = featuredir
        self.players: Dict[str, GameEntry] = {}  # stores last occurrence of player
        self.games = List[GameEntry]
        self.bin_input_shape = katamodel.bin_input_shape        # one input to Kata model
        self.global_input_shape = katamodel.global_input_shape  # one input to Kata model
        self.features = Features(katamodel.config, max_board_size)

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
            self._fillRecentMoves(game.black.name, game)
        if game.white.recentMoves is None:
            self._fillRecentMoves(game.white.name, game)
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

    def _fillRecentMoves(self, player: str, game: GameEntry):
        """Place strength model input tensors in game.black/white.recentMoves"""

        # rules are arbitrary because most moves evaluate the same regardless of rules
        rules = {
            "koRule": "KO_POSITIONAL",
            "scoringRule": "SCORING_AREA",
            "taxRule": "TAX_NONE",
            "multiStoneSuicideLegal": True,
            "hasButton": False,
            "encorePhase": 0,
            "passWouldEndPhase": False,
            "whiteKomi": 7.5,
            "asymPowersOfTwo": 0.0,
        }

        color = 'Black' if game.black.name == player else 'White'
        sgfPathWithoutExt, _ = os.path.splitext(game.sgfPath)
        recentpath = f"{self.featuredir}/{sgfPathWithoutExt}_{color}RecentMoves.csv"

        if not exists(recentpath):
            raise IOError(f"Precomputed recent moves not found in {recentpath}")

        with open(recentpath, 'r') as recentfile:
            reader = csv.DictReader(recentfile)
            recents = [(row['File'], int(row['StartMove']), int(row['Count'])) for row in reader]

        playerdata = {game.black.name: game.black, game.white.name: game.white}[player]
        total = sum(r[2] for r in recents)
        bin_input_data = np.zeros(shape=[total]+self.bin_input_shape, dtype=np.float32)
        global_input_data = np.zeros(shape=[total]+self.global_input_shape, dtype=np.float32)
        if 0 == total:
            playerdata.recentMoves = (bin_input_data, global_input_data)  # ToDo: one-hot tensor for next move
            return

        # fill_row_features assumes N(HW)C order but we actually use NCHW order, so work with it and revert after
        bin_input_data = np.transpose(bin_input_data,axes=(0,2,3,1))
        bin_input_data = bin_input_data.reshape([total,max_board_size*max_board_size,-1])
        loc_input_data = np.zeros(shape=[*bin_input_data.shape[:-1], 1], dtype=np.float32)  # 1-hot of next move to strength-evaluate

        i = 0
        for sgfPath, startMove, count in recents:
            for gs, pla, loc in generateBoards(sgfPath, startMove, count):
                # print(f"{gs.board.to_string()}\npla={pla}, loc={loc}\n")
                opp = Board.get_opp(pla)
                move_idx = len(gs.moves)
                self.features.fill_row_features(gs.board,pla,opp,gs.boards,gs.moves,move_idx,rules,bin_input_data,global_input_data,idx=i)
                if Board.PASS_LOC != loc:
                    loc_pos = self.features.loc_to_tensor_pos(loc, gs.board)
                    loc_input_data[i,loc_pos,0] = 1.0
                i += 1
        bin_input_data = bin_input_data.reshape([total,max_board_size,max_board_size,-1])
        bin_input_data = np.transpose(bin_input_data,axes=(0,3,1,2))

        playerdata.recentMoves = (bin_input_data, loc_input_data, global_input_data)

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
                if overshoot < 0 or overshoot >= len(mymoves):
                    import pdb; pdb.set_trace()
                startmove = mymoves[overshoot]
                writer.writerow({'File': historic.sgfPath, 'StartMove': startmove, 'Count': len(mymoves)-overshoot})
                historic = entry.prevGame

def pad_collate(batch):
    brecent, wrecent, brating, wrating, score = zip(*batch)
    blens = [r.shape[0] for r in brecent]
    wlens = [r.shape[0] for r in wrecent]
    brecent, wrecent = torch.cat(brecent, dim=0), torch.cat(wrecent, dim=0)
    brating, wrating, score = map(torch.Tensor, (brating, wrating, score))
    return brecent, wrecent, blens, wlens, brating, wrating, score

class StrengthDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = pad_collate
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    print("Test moves_dataset.py")
    listpath = 'csv/games_labels.csv'
    featuredir = 'featurecache'
    marker = 'V'
    dataset = StrengthDataset(listpath, featuredir, marker)
    print(f"Loaded {len(dataset.games)} games, {len(dataset)} of type {marker}.")

    loader = DataLoader(dataset, batch_size=3, collate_fn=pad_collate)

    for bx, wx, blens, wlens, by, wy, score in loader:
        print(f"Got batch size bx: {bx.shape} ({blens}), wx: {wx.shape} ({wlens}); {by}; {wy}; {score}.")
