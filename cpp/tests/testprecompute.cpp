#include "tests.h"

#include <string>
#include "core/fileutils.h"
#include "core/using.h"
#include "neuralnet/nninterface.h"
#include "strmodel/dataset.h"

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
  SelectedMoves selectedMoves = dataset.getRecentMoves(P_WHITE, 2, 3); // alice moves = game 0 black + game 1 black
  const char* sgf1 = "sgfs/precomputetests/1-alice-bob.sgf";
  const char* sgf2 = "sgfs/precomputetests/2-alice-claire.sgf";
  testAssert(contains(selectedMoves.bygame, sgf1));
  testAssert(contains(selectedMoves.bygame, sgf2));
  SelectedMoves::Moveset moves1 = selectedMoves.bygame[sgf1];
  testAssert(1 == moves1.moves.size());
  testAssert(2 == moves1.moves[0].index); // prefer moves from the back of the game
  // TODO: get recent moves of bob, then merge selected moves

  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}

}
