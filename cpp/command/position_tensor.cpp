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

  unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
  void readFromSgf(const string& sgfPath, int moveNumber, const string& modelFile);
  void readFromZip(const string& zipPath, int moveNumber);
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

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> sgfArg("","sgf","SGF file with the game to extract.",false,"","SGF_FILE");
    cmd.add(sgfArg);
    TCLAP::ValueArg<string> zipArg("","zip","ZIP file with precomputed trunk to extract.",false,"","ZIP_FILE");
    cmd.add(zipArg);
    TCLAP::ValueArg<int> moveNumberArg("","move-number","Extract the position or trunk at this move number/index, starting at 0.",true,10,"MOVE_NUMBER");
    cmd.add(moveNumberArg);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    sgfPath = sgfArg.getValue();
    zipPath = zipArg.getValue();
    moveNumber = moveNumberArg.getValue();

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

  if(!sgfPath.empty())
    readFromSgf(sgfPath, moveNumber, modelFile);

  if(!zipPath.empty())
    readFromZip(zipPath, moveNumber);

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file) {
  return unique_ptr<LoadedModel, void(*)(LoadedModel*)>(NeuralNet::loadModelFile(file, ""), &NeuralNet::freeLoadedModel);
}

void readFromSgf(const string& sgfPath, int moveNumber, const string& modelFile) {
  auto model = loadModel(modelFile);
  theLogger->write("Loaded model "+ modelFile);
  PrecomputeFeatures extractor(*model, maxMoves);

  theLogger->write("Starting to extract tensors from move " + Global::intToString(moveNumber) + " in " + sgfPath + "...");

  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  for(int turnIdx = 0; turnIdx < moveNumber; turnIdx++) {
    Move move = moves[turnIdx];
    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  Move move = moves[moveNumber];
  extractor.addBoard(board, history, move);
  extractor.endGame(sgfPath);
  extractor.evaluate();
  assert(extractor.hasResult());
  PrecomputeFeatures::Result result = extractor.nextResult();

  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  // extractor.writeInputsToNpz(sgfPathWithoutExt + "_Inputs.npz");
  // extractor.writeOutputsToNpz(sgfPathWithoutExt + "_Trunk.npz");
  // extractor.writePicksToNpz(sgfPathWithoutExt + "_Pick.npz");
  dumpTensor(sgfPathWithoutExt + "_Trunk.txt", result.trunk, PrecomputeFeatures::trunkSize);
  // dumpTensor(sgfPathWithoutExt + "_Pick.txt", pickNC->data, pickNC->dataLen);
}

void readFromZip(const string& zipPath, int moveNumber) {
  theLogger->write("Starting to extract tensors from move " + Global::intToString(moveNumber) + " in " + zipPath + "...");
  auto moveset = SelectedMoves::Moveset::readFromZip(zipPath);
  string zipPathWithoutExt = Global::chopSuffix(zipPath, ".zip");
  dumpTensor(zipPathWithoutExt + "_TrunkUnzip.txt", moveset.moves.at(moveNumber).trunk->data(), PrecomputeFeatures::trunkSize);
  // dumpTensor(zipPathWithoutExt + "_PickUnzip.txt", pickNC->data, pickNC->dataLen);
}

void dumpTensor(string path, float* data, size_t N) {
  std::ofstream file(path);
  for(int i = 0; i < N; i++) {
    file << data[i] << "\n";
  }
  file.close();
}

}
