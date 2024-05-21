#include "tests.h"

#include <string>
#include "core/fileutils.h"
#include "core/using.h"
#include "neuralnet/nninterface.h"
#include "strmodel/dataset.h"
#include "strmodel/precompute.h"

namespace {
  void runAliceRecent3(const Dataset& dataset);
  void runBobRecent3(const Dataset& dataset);
  void runMergeSelectedMoves();
  void runTrunksPicks(LoadedModel& loadedModel);
  void runSaveLoadMoveset();
  std::shared_ptr<vector<float>> fakeData(size_t elements, int variant);
  bool approxEqual(const vector<float>& expected, const vector<float>& actual);
}

namespace Tests {

void runPrecomputeTests(const string& modelFile) {
  cout << "Running precompute tests" << endl;
  Board::initHash();
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  Dataset dataset;
  dataset.load("precomputetest/games_labels.csv");
  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile,"");

  runAliceRecent3(dataset);
  runBobRecent3(dataset);
  runMergeSelectedMoves();
  runTrunksPicks(*loadedModel);
  runSaveLoadMoveset();

  NeuralNet::freeLoadedModel(loadedModel);
  NeuralNet::globalCleanup();
  cout << "Done" << endl;
}

}

namespace {

constexpr static int nnXLen = 19;
constexpr static int nnYLen = 19;
constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;

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

void runTrunksPicks(LoadedModel& loadedModel) {
  cout << "- Precompute picks should match with precompute trunks\n";

  // we should evaluate a batch that covers every move position;
  // it should be greater than the number of GPU threads to test blocks.
  PrecomputeFeatures precompute(loadedModel, 722);

  // generate test data
  SelectedMoves::Moveset movesetA;
  SelectedMoves::Moveset movesetB;
  for(auto evaluate : {&PrecomputeFeatures::evaluateTrunks, &PrecomputeFeatures::evaluatePicks}){
    for(auto game_set : {make_pair("Game A", std::ref(movesetA)), make_pair("Game B", std::ref(movesetB))}) {
      precompute.startGame(game_set.first);
      Rules rules = Rules::getTrompTaylorish();
      Board board = Board(19,19);
      Player pla = P_BLACK;
      BoardHistory history = BoardHistory(board,pla,rules,0);
      for(int i = 0; i < 19; i++)
        for(int j = 0; j < 19; j++) {
          testAssert(!precompute.isFull());
          Loc loc = Location::getLoc(i, j, 19);
          Move move(loc, pla);
          precompute.addBoard(board, history, move);
          game_set.second.insert(i*19+j, pla);
          history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr);
          pla = getOpp(pla);
        }
      precompute.endGame();
    }
    testAssert(precompute.isFull());
    std::vector<PrecomputeFeatures::Result> results = (precompute.*evaluate)();
    testAssert(2 == results.size());
    PrecomputeFeatures::writeResultToMoveset(results[0], movesetA);
    PrecomputeFeatures::writeResultToMoveset(results[1], movesetB);
  }

  // compare trunks with picks
  for(const auto& moveset : {movesetA, movesetB}) {
    for(const auto& move : moveset.moves) {
      testAssert(move.trunk);
      testAssert(move.pick);
      vector<float> expectedPick(numTrunkFeatures);
      for(int c = 0; c < numTrunkFeatures; c++)
        expectedPick[c] = move.trunk->at(c*19*19 + move.pos);
      testAssert(approxEqual(*move.pick, expectedPick));
    }
  }
}

void runSaveLoadMoveset() {
  cout << "- Moveset data should correctly save/load to ZIP file\n";

  using Move = SelectedMoves::Move;
  using Moveset = SelectedMoves::Moveset;

  // generate test data
  Moveset moveset{
    {
      Move{0, 0, fakeData(trunkSize, 0), fakeData(numTrunkFeatures, 0), 10},
      Move{1, 0, fakeData(trunkSize, 1), fakeData(numTrunkFeatures, 1), 20},
      Move{2, 0, fakeData(trunkSize, 2), fakeData(numTrunkFeatures, 2), 30}
    }
  };

  string filePath = "test-moveset.zip";
  moveset.writeToZip(filePath);
  Moveset actual = Moveset::readFromZip(filePath);

  testAssert(3 == actual.moves.size());
  for(int i = 0; i < 3; i++) {
    testAssert(moveset.moves[i].index == actual.moves[i].index);
    testAssert(moveset.moves[i].pla == actual.moves[i].pla);
    testAssert(moveset.moves[i].pos == actual.moves[i].pos);
    testAssert(approxEqual(*moveset.moves[i].trunk, *actual.moves[i].trunk));
    testAssert(approxEqual(*moveset.moves[i].pick, *actual.moves[i].pick));
  }
}

std::shared_ptr<vector<float>> fakeData(size_t elements, int variant) {
  auto data = std::make_shared<vector<float>>(elements);
  for(size_t i = 0; i < elements; i++) {
    (*data)[i] = float(i+1) + variant * 10.f;
  }
  return data;
}

bool approxEqual(const vector<float>& expected, const vector<float>& actual) {
  assert(expected.size() == actual.size());
  for(size_t i = 0; i < expected.size(); i++) {
    if(fabs(expected[i] - actual[i]) > 0.001)
      return false;
  }
  return true;
}

}