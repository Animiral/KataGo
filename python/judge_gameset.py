"""
Given an input file listing SGFs to analyze, run each of them through KataGo's analysis engine.
Output the winner based on the winning chances at the end position of the main variation.
Whoever has a winning percentage greater than 60% is declared winner. Otherwise, the result is Jigo.
Output the evaluation results to a new file with the declared winner as a new CSV column.
"""

import argparse
import json
import csv
import os
import subprocess
import time
from threading import Thread
import sgfmill
import sgfmill.boards
from sgfmill import sgf
from sgfmill import sgf_moves
from typing import Tuple, List, Optional, Union, Literal

Color = Union[Literal["b"],Literal["w"]]
Move = Union[Literal["pass"],Tuple[int,int]]

def processed_games(outfile: str):
    try:
        with open(outfile, 'r') as file:
            reader = csv.DictReader(file)
            return set(r['File'] for r in reader)

    except FileNotFoundError:
        return set()

def sgfmill_to_str(move: Move) -> str:
    if move == "pass":
        return "pass"
    (y,x) = move
    return "ABCDEFGHJKLMNOPQRSTUVWXYZ"[x] + str(y+1)

class KataGo:

    def __init__(self, katago_path: str, config_path: str, model_path: str):
        katago = subprocess.Popen(
            [katago_path, "analysis", "-config", config_path, "-model", model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.katago = katago
        def printforever():
            while katago.poll() is None:
                data = katago.stderr.readline()
                time.sleep(0)
                if data:
                    print("KataGo: ", data.decode(), end="")
            data = katago.stderr.read()
            if data:
                print("KataGo: ", data.decode(), end="")
        self.stderrthread = Thread(target=printforever)
        self.stderrthread.start()

    def query(self, query_id, game: sgfmill.sgf.Sgf_game, max_visits=None):
        """Queue up one query"""
        query = {}
        query["id"] = str(query_id)

        board, moves = sgf_moves.get_setup_and_moves(game)
        query["moves"] = [(color,sgfmill_to_str(move)) for color, move in moves if move is not None]
        query["initialStones"] = []
        for y in range(board.side):
            for x in range(board.side):
                color = board.get(y,x)
                if color:
                    query["initialStones"].append((color,sgfmill_to_str((y,x))))
        root = game.get_root()
        query["rules"] = root.get('RU')
        if query["rules"] == "ogs":  # compatibility with a few old records from 2005
            query["rules"] = "Chinese"
        query["komi"] = root.get('KM')
        query["boardXSize"] = board.side
        query["boardYSize"] = board.side
        query["includePolicy"] = False
        if max_visits is not None:
            query["maxVisits"] = max_visits

        self.katago.stdin.write((json.dumps(query) + "\n").encode())
        # print(json.dumps(query))

    def finish(self):
        """Flush stream with pending queries and close input to signal end"""
        self.katago.stdin.close()

    def response(self):
        """
        Return the next response from the engine as pair of id, winrate
        Return id None when KataGo has no more responses
        Return winrate None when KataGo cannot judge the game (e.g. corrupt moves in record)
        """
        line = self.katago.stdout.readline()
        line = line.decode().strip()
        if line == "":
            return None, None
        response = json.loads(line)
        query_id = int(response["id"])
        if "moveInfos" in response:
            winrate = response["moveInfos"][0]["winrate"]
        else:
            winrate = None
        # print(response)
        return query_id, winrate

def analyze(sgf_file, katago, query_id, max_visits):
    """Evaluate the end position and return the winner B+XX%, W+XX%, Jigo"""
    with open(sgf_file, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    katago.query(query_id, game, max_visits=max_visits)

def judge(winrate, threshold = 0.6):
    """Return the game score based on the given black winrate"""
    assert(threshold >= 0.5)
    if winrate is None:
        return "error"  # cannot judge this game
    elif winrate < (1-threshold):
        return "0"
    elif winrate <= threshold:
        return "0.5"
    else:
        return "1"

def process_one_response(katago, writer):
    query_id, winrate = katago.response()
    print(f"Got response {query_id}: winrate {winrate}.")
    if query_id is None or winrate is None:
        return False
    score = judge(winrate)
    if "0.5" != score or args["keep_undecided"]:
        row = row_ids[query_id]
        row['Score'] = score
        if has_winner:
            del row['Winner']
        writer.writerow(row)
    return True

if __name__ == "__main__":
    description = """
    Run a KataGo query on every move in the given SGF(s).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--keep-undecided",
        help="Output games which cannot be decided for either black or white",
        type=bool,
        required=False,
    )
    parser.add_argument(
        "--max-visits",
        help="Number of maxVisits to be passed to KataGo",
        type=int,
        default=50,
        required=False,
    )
    parser.add_argument(
        "--katago-path",
        help="Path to katago executable",
        required=True,
    )
    parser.add_argument(
        "--config-path",
        help="Path to KataGo analysis config (e.g. cpp/configs/analysis_example.cfg in KataGo repo)",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        help="Path to neural network .bin.gz file",
        required=True,
    )
    parser.add_argument(
        "-o", "--outfile",
        metavar="OUTFILE",
        default="games_judged.csv",
        type=str,
        help="Output file (appends if it exists)",
    )
    parser.add_argument(
        "-i", "--input",
        default="games.csv",
        type=str,
        help="Input file containing list of game record(s) in first CSV column",
    )
    args = vars(parser.parse_args())
    print(args)

    katago = KataGo(args["katago_path"], args["config_path"], args["model_path"])
    resumed_games = processed_games(args["outfile"])
    if resumed_games:
        print(f"Resuming from {len(resumed_games)} already judged.")

    # Input CSV format (title row):
    # File,Player White,Player Black,Winner
    with open(args["input"], 'r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    row_ids = dict(enumerate(rows))
    max_visits = args["max_visits"]

    # write output CSV file
    writemode = 'a' if resumed_games else 'w'
    outfile = open(args["outfile"], writemode)
    fieldnames = list(rows[0].keys())
    fieldnames.append('Score')
    has_winner = 'Winner' in rows[0].keys()
    if has_winner:
        fieldnames.remove('Winner')  # replaced by Score
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    if not resumed_games:
        writer.writeheader()

    # launch queries to the engine, pick up responses with 100 queries in queue
    queued = 0
    for i, row in row_ids.items():
        sgf_file = row['File']
        if sgf_file in resumed_games:
            continue

        print(f"Submit query {i}...")
        analyze(sgf_file, katago, i, max_visits)
        queued += 1
        if queued > 100:
            process_one_response(katago, writer)

    katago.finish()
    print("Finished writing queries.")

    # process remaining responses
    while process_one_response(katago, writer):
        pass

    outfile.close()
    print("Finished writing output file.")
