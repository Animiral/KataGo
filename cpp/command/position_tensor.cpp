#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../strmodel/dataset.h"
#include "../strmodel/strengthmodel.h"
#include "../strmodel/precompute.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <zip.h>

using namespace std;

void getTensorsForPosition(const Board& board, const BoardHistory& history, Player pla, NNEvaluator* nnEval);

namespace
{

  constexpr int maxMoves = 1000; // capacity for moves

  // unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
  unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, Logger& logger);
  SelectedMoves::Moveset readFromSgf(const string& sgfPath, const string& modelFile, int moveNumber);
  SelectedMoves::Moveset readFromZip(const string& zipPath);
  void dumpTensor(string path, float* data, size_t N);

  ConfigParser cfg;
  Logger* theLogger = nullptr;

}

int MainCmds::position_tensor(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string modelFile;
  string sgfPath;
  string zipPath;
  int moveNumber;
  bool summary;

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> sgfArg("","sgf","SGF file with the game to extract.",false,"","SGF_FILE",cmd);
    TCLAP::ValueArg<string> zipArg("","zip","ZIP file with precomputed trunk to extract.",false,"","ZIP_FILE", cmd);
    TCLAP::ValueArg<int> moveNumberArg("","move-number","Extract the position or trunk at this move number/index, starting at 0.",false,0,"MOVE_NUMBER", cmd);
    TCLAP::SwitchArg summaryArg("s","summary","Print general info on all moves in the SGF or ZIP.",cmd,false);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    sgfPath = sgfArg.getValue();
    zipPath = zipArg.getValue();
    moveNumber = moveNumberArg.getValue();
    summary = summaryArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  theLogger = &logger;

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write(Version::getKataGoVersionForHelp());
  if(!logger.isLoggingToStderr())
    cerr << Version::getKataGoVersionForHelp() << endl;

  if(!sgfPath.empty()) {
    auto moveset = readFromSgf(sgfPath, modelFile, moveNumber);
    if(summary) {
      moveset.printSummary(std::cout);
    }
    else {
      string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
      // precompute.writeInputsToNpz(sgfPathWithoutExt + "_Inputs.npz");
      // precompute.writeOutputsToNpz(sgfPathWithoutExt + "_Trunk.npz");
      // precompute.writePicksToNpz(sgfPathWithoutExt + "_Pick.npz");
      dumpTensor(sgfPathWithoutExt + "_Trunk.txt", moveset.moves.at(moveNumber).trunk->data(), PrecomputeFeatures::trunkSize);
      // dumpTensor(sgfPathWithoutExt + "_Pick.txt", pickNC->data, pickNC->dataLen);
    }
  }

  if(!zipPath.empty()) {
    auto moveset = readFromZip(zipPath);
    if(summary) {
      moveset.printSummary(std::cout);
    }
    else {
      string zipPathWithoutExt = Global::chopSuffix(zipPath, ".zip");
      dumpTensor(zipPathWithoutExt + "_TrunkUnzip.txt", moveset.moves.at(moveNumber).trunk->data(), PrecomputeFeatures::trunkSize);
      // dumpTensor(zipPathWithoutExt + "_PickUnzip.txt", pickNC->data, pickNC->dataLen);
    }
  }

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

// unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file) {
//   return unique_ptr<LoadedModel, void(*)(LoadedModel*)>(NeuralNet::loadModelFile(file, ""), &NeuralNet::freeLoadedModel);
// }

unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, Logger& logger) {
  constexpr int evaluatorThreads = 32;
  constexpr int maxConcurrentEvals = evaluatorThreads*2;
  constexpr int batchSize = 100;
  constexpr int nnXLen = 19;
  constexpr int nnYLen = 19;
  vector<int> gpuIdxByServerThread(evaluatorThreads, -1);
  auto evaluator = make_unique<NNEvaluator>(
    modelFile,
    modelFile,
    "", // expectedSha256
    &logger,
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

SelectedMoves::Moveset readSgfToPrecompute(const string& sgfPath, PrecomputeFeatures& precompute, int moveNumber) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  SelectedMoves::Moveset moveset;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  precompute.startGame(sgfPath);
  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];
    if(precompute.isFull())
      throw StringError("Partial game results not implemented");
    if(turnIdx >= moveNumber) {
      moveset.insert(turnIdx, move.pla);
      precompute.addBoard(board, history, move);
    }
    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  precompute.endGame();

  return moveset; // blank moveset to receive results
}

SelectedMoves::Moveset readFromSgf(const string& sgfPath, const string& modelFile, int moveNumber) {
  auto evaluator = createEvaluator(modelFile, *theLogger);
  theLogger->write("Loaded model "+ modelFile);
  PrecomputeFeatures precompute(*evaluator, maxMoves);
  precompute.includeTrunk = true;
  precompute.includePick = true;
  theLogger->write("Starting to extract tensors from " + sgfPath + "...");
  SelectedMoves::Moveset moveset = readSgfToPrecompute(sgfPath, precompute, moveNumber);
  vector<PrecomputeFeatures::Result> results = precompute.evaluate();
  assert(1 == results.size());
  PrecomputeFeatures::writeResultToMoveset(results.front(), moveset);
  return moveset;
}

SelectedMoves::Moveset readFromZip(const string& zipPath) {
  theLogger->write("Starting to extract tensors from " + zipPath + "...");
  return SelectedMoves::Moveset::readFromZip(zipPath);
}

void dumpTensor(string path, float* data, size_t N) {
  std::ofstream file(path);
  for(int i = 0; i < N; i++) {
    file << data[i] << "\n";
  }
  file.close();
}

}
