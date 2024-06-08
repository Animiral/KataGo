#include "tests.h"

#include <string>
#include "core/fileutils.h"
#include "core/global.h"
#include "neuralnet/nninterface.h"
#include "strmodel/dataset.h"
#include "strmodel/precompute.h"

using namespace StrModel;
using std::cout;

namespace {
  // we should evaluate a batch that covers every move position;
  // it should be greater than the number of GPU threads to test blocks.
  constexpr int batchSize = 722;
  constexpr int evaluatorThreads = 2;

  std::unique_ptr<NNEvaluator> createEvaluator(const string& modelFile);
  void runAliceRecent3(const Dataset& dataset);
  void runBobRecent3(const Dataset& dataset);
  // void runTrunksPicks(NNEvaluator& evaluator);
  void runSaveLoadMoveset();
  std::shared_ptr<vector<float>> fakeData(size_t elements, int variant);
  bool approxEqual(const vector<float>& expected, const vector<float>& actual);
}

namespace Tests {

void runPrecomputeTests(const string& modelFile) {
  cout << "Running precompute tests\n";
  Board::initHash();
  NeuralNet::globalInitialize();

  const bool logToStdout = true;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);

  DatasetFiles files;
  Dataset dataset("precomputetest/games_labels.csv", files);
  auto evaluator = createEvaluator(modelFile);

  runAliceRecent3(dataset);
  runBobRecent3(dataset);
  // runTrunksPicks(*evaluator);
  runSaveLoadMoveset();

  NeuralNet::globalCleanup();
  cout << "Done\n";
}

}

namespace {

constexpr static int nnXLen = 19;
constexpr static int nnYLen = 19;
constexpr static int numTrunkFeatures = 384;  // strength model is limited to this size
constexpr static int numHeadFeatures = 6;
constexpr static int trunkSize = nnXLen*nnYLen*numTrunkFeatures;

std::unique_ptr<NNEvaluator> createEvaluator(const string& modelFile) {
  constexpr int maxConcurrentEvals = evaluatorThreads*2;
  vector<int> gpuIdxByServerThread(evaluatorThreads, -1);
  auto evaluator = std::make_unique<NNEvaluator>(
    modelFile,
    modelFile,
    "", // expectedSha256
    nullptr, // logger
    batchSize,
    maxConcurrentEvals,
    nnXLen,
    nnYLen,
    true, // requireExactNNLen
    false, // inputsUseNHWC
    23, // nnCacheSizePowerOfTwo
    17, // nnMutexPoolSizePowerOfTwo
    false, // debugSkipNeuralNet
    "", // openCLTunerFile
    "", // homeDataDirOverride
    false, // openCLReTunePerBoardSize
    enabled_t::False, // useFP16Mode
    enabled_t::False, // useNHWCMode
    evaluatorThreads, // numNNServerThreadsPerModel
    gpuIdxByServerThread,
    "", // nnRandSeed
    false, // doRandomize (for symmetry)
    0 // defaultSymmetry
  );
  evaluator->spawnServerThreads();
  return evaluator;
}

void runAliceRecent3(const Dataset& dataset) {
  cout << "- Recent moves of " << dataset.games[2].sgfPath << " for white\n";
  // alices 3 recent moves are parts from game 1 and all from 2
  GamesTurns gamesTurns = dataset.getRecentMoves(P_WHITE, 2, 3); // alice moves = game 0 black + game 1 black
  testAssert(contains(gamesTurns.bygame, 0));
  testAssert(contains(gamesTurns.bygame, 1));
  vector<int> moves1 = gamesTurns.bygame[0];
  testAssert(1 == moves1.size());
  testAssert(2 == moves1[0]); // prefer moves from the back of the game
  vector<int> moves2 = gamesTurns.bygame[1];
  testAssert(2 == moves2.size());
  testAssert(0 == moves2[0]);
  testAssert(2 == moves2[1]);
}

void runBobRecent3(const Dataset& dataset) {
  cout << "- Recent moves of " << dataset.games[2].sgfPath << " for black\n";
  // alices 3 recent moves are parts from game 1 and all from 2
  GamesTurns gamesTurns = dataset.getRecentMoves(P_BLACK, 2, 3); // bob moves = game 0 white
  testAssert(contains(gamesTurns.bygame, 0));
  vector<int> moveset = gamesTurns.bygame[0];
  testAssert(1 == moveset.size());
  testAssert(1 == moveset[0]);
}

namespace {

// condense vec to a single printable number, likely different from vecs (trunks) of other positions,
// but also close in value to very similar vecs (tolerant of float inaccuracies)
float vecChecksum(const vector<float>& vec) {
  float sum = 0.0f;
  float sos = 0.0f;
  float weight = 1.0f;
  float decay = 0.9999667797285222f; // = pow(0.01, (1/(vec.size()-1))) -> smallest weight is 0.01

  for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i] * weight;
      sos += vec[i] * vec[i];
      weight *= decay;
  }

  return sum + std::sqrt(sos);
}

}

// void runTrunksPicks(NNEvaluator& evaluator) {
//   cout << "- Precompute picks should match with precompute trunks\n";

//   // we should evaluate a batch that covers every move position;
//   // it should be greater than the number of GPU threads to test blocks.
//   PrecomputeFeatures precompute(evaluator, batchSize);
//   precompute.selection.trunk = true;
//   precompute.selection.pick = true;

//   // generate test data
//   SelectedMoves::Moveset movesetA;
//   SelectedMoves::Moveset movesetB;
//   for(auto game_set : {make_pair("Game A", std::ref(movesetA)), make_pair("Game B", std::ref(movesetB))}) {
//     precompute.startGame(game_set.first);
//     Rules rules = Rules::getTrompTaylorish();
//     Board board = Board(19,19);
//     Player pla = P_BLACK;
//     BoardHistory history = BoardHistory(board,pla,rules,0);
//     for(int i = 0; i < 19; i++)
//       for(int j = 0; j < 19; j++) {
//         testAssert(!precompute.isFull());
//         Loc loc = Location::getLoc(i, j, 19);
//         Move move(loc, pla);
//         precompute.addBoard(board, history, move, precompute.selection);
//         game_set.second.insert(i*19+j, precompute.selection);
//         history.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr);
//         pla = getOpp(pla);
//       }
//     precompute.endGame();
//   }
//   testAssert(precompute.isFull());
//   std::vector<PrecomputeFeatures::Result> results = precompute.evaluate();
//   testAssert(2 == results.size());
//   PrecomputeFeatures::writeResultToMoveset(results[0], movesetA);
//   PrecomputeFeatures::writeResultToMoveset(results[1], movesetB);

//   // compare trunks with picks
//   for(const auto& moveset : {movesetA, movesetB}) {
//     for(const auto& move : moveset.moves) {
//       testAssert(move.trunk);
//       testAssert(move.pick);
//       vector<float> expectedPick(numTrunkFeatures);
//       for(int c = 0; c < numTrunkFeatures; c++)
//         expectedPick[c] = move.trunk->at(c*19*19 + move.pos);
//       testAssert(approxEqual(*move.pick, expectedPick));
//     }
//   }
// }

void runSaveLoadMoveset() {
  cout << "- Moveset data should correctly save/load to ZIP file\n";

  // generate test data
  vector<BoardFeatures> moveset{
    BoardFeatures{0, C_EMPTY, 10, fakeData(trunkSize, 0), fakeData(numTrunkFeatures, 0), fakeData(numHeadFeatures, 0)},
    BoardFeatures{1, C_EMPTY, 20, fakeData(trunkSize, 1), fakeData(numTrunkFeatures, 1), fakeData(numHeadFeatures, 1)},
    BoardFeatures{2, C_EMPTY, 30, fakeData(trunkSize, 2), fakeData(numTrunkFeatures, 2), fakeData(numHeadFeatures, 2)}
  };

  DatasetFiles files(".");
  string filePath = "test-moveset.zip";
  files.storeFeatures(moveset, filePath);
  vector<BoardFeatures> actual = files.loadFeatures(filePath);

  testAssert(3 == actual.size());
  for(int i = 0; i < 3; i++) {
    testAssert(moveset[i].turn == actual[i].turn);
    testAssert(moveset[i].pos == actual[i].pos);
    testAssert(approxEqual(*moveset[i].trunk, *actual[i].trunk));
    testAssert(approxEqual(*moveset[i].pick, *actual[i].pick));
    testAssert(approxEqual(*moveset[i].head, *actual[i].head));
  }
}

shared_ptr<FeatureVector> fakeData(size_t elements, int variant) {
  auto data = make_shared<FeatureVector>(elements);
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