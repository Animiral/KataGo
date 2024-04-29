#include "tests.h"

#include <string>
#include "core/fileutils.h"
#include "core/using.h"
#include "neuralnet/nninterface.h"
#include "strmodel/dataset.h"

namespace {
  void runAliceRecent3(const Dataset& dataset);
  void runBobRecent3(const Dataset& dataset);
  void runMergeSelectedMoves();
}

namespace Tests {

void runPrecomputeTests() {
  cout << "Running precompute tests" << endl;
  Board::initHash();
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  Dataset dataset;
  dataset.load("precomputetest/games_labels.csv");

  runAliceRecent3(dataset);
  runBobRecent3(dataset);
  runMergeSelectedMoves();

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}

}

namespace {

void runAliceRecent3(const Dataset& dataset) {
  cout << "- Recent moves of " << dataset.games[2].sgfPath << " for white\n";
  // alices 3 recent moves are parts from game 1 and all from 2
  SelectedMoves selectedMoves = dataset.getRecentMoves(P_WHITE, 2, 3); // alice moves = game 0 black + game 1 black
  const char* sgf1 = "sgfs/precomputetests/1-alice-bob.sgf";
  const char* sgf2 = "sgfs/precomputetests/2-alice-claire.sgf";
  testAssert(contains(selectedMoves.bygame, sgf1));
  testAssert(contains(selectedMoves.bygame, sgf2));
  SelectedMoves::Moveset moves1 = selectedMoves.bygame[sgf1];
  testAssert(1 == moves1.moves.size());
  testAssert(2 == moves1.moves[0].index); // prefer moves from the back of the game
  SelectedMoves::Moveset moves2 = selectedMoves.bygame[sgf2];
  testAssert(2 == moves2.moves.size());
  testAssert(0 == moves2.moves[0].index);
  testAssert(2 == moves2.moves[1].index);
}

void runBobRecent3(const Dataset& dataset) {
  cout << "- Recent moves of " << dataset.games[2].sgfPath << " for black\n";
  // alices 3 recent moves are parts from game 1 and all from 2
  SelectedMoves selectedMoves = dataset.getRecentMoves(P_BLACK, 2, 3); // bob moves = game 0 white
  const char* sgf1 = "sgfs/precomputetests/1-alice-bob.sgf";
  testAssert(contains(selectedMoves.bygame, sgf1));
  SelectedMoves::Moveset moveset = selectedMoves.bygame[sgf1];
  testAssert(1 == moveset.moves.size());
  testAssert(1 == moveset.moves[0].index);
}

void runMergeSelectedMoves() {
  cout << "- Merge selected move sets\n";
  const char* sgf1 = "game1.sgf";
  const char* sgf2 = "game2.sgf";
  SelectedMoves sel;
  sel.bygame[sgf1].insert(3, P_WHITE);
  sel.bygame[sgf1].insert(5, P_WHITE);
  sel.bygame[sgf2].insert(2, P_BLACK);
  SelectedMoves sel2;
  sel2.bygame[sgf1].insert(3, P_WHITE);
  sel2.bygame[sgf1].insert(4, P_BLACK);
  sel2.bygame[sgf2].insert(2, P_BLACK);
  sel.merge(sel2);

  testAssert(contains(sel.bygame, sgf1));
  testAssert(contains(sel.bygame, sgf2));
  SelectedMoves::Moveset moves1 = sel.bygame[sgf1];
  testAssert(3 == moves1.moves.size());
  testAssert(3 == moves1.moves[0].index);  // moves need to be in order
  testAssert(4 == moves1.moves[1].index);
  testAssert(5 == moves1.moves[2].index);
  SelectedMoves::Moveset moves2 = sel.bygame[sgf2];
  testAssert(1 == moves2.moves.size());
  testAssert(2 == moves2.moves[0].index);
}

}