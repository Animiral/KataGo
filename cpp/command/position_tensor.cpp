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
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>

using namespace std;

void getTensorsForPosition(const Board& board, const BoardHistory& history, Player pla, NNEvaluator* nnEval);

namespace
{

  void loadParams(ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
    params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS);
    perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
      params.conservativePass = true;
  }

  struct RecentMovesTensor {
    int count; // first dimension size actually written
    unique_ptr<NumpyBuffer<float>> binaryInputNCHW; // game position tensors
    unique_ptr<NumpyBuffer<float>> locInputNCHW; // 1-hot move location tensors
    unique_ptr<NumpyBuffer<float>> globalInputNC; // global position tensors
    unique_ptr<NumpyBuffer<float>> trunkOutputNCHW; // NN state after last batch norm, without heads
    unique_ptr<NumpyBuffer<float>> pickNC; // trunk output at indicated location
  };

  void writeRecentMovesNpz(const RecentMovesTensor& tensors, const string& filePath);
  void writeTrunkNpz(const RecentMovesTensor& tensors, const string& filePath);
  void writePickNpz(const RecentMovesTensor& tensors, const string& filePath);
  void dumpTensor(string path, float* data, size_t N);

}

int MainCmds::position_tensor(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string sgfPath;
  int moveNumber;

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> sgfArg("","sgf","SGF file with the game to extract.",true,"","SGF_FILE");
    cmd.add(sgfArg);
    TCLAP::ValueArg<int> moveNumberArg("","move-number","Extract the position at this move number, starting at 0.",true,10,"MOVE_NUMBER");
    cmd.add(moveNumberArg);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    sgfPath = sgfArg.getValue();
    moveNumber = moveNumberArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  const int numAnalysisThreads = cfg.getInt("numAnalysisThreads",1,16384);
  if(numAnalysisThreads <= 0 || numAnalysisThreads > 16384)
    throw StringError("Invalid value for numAnalysisThreads: " + Global::intToString(numAnalysisThreads));

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());

  // LOAD NN MODEL
  SearchParams searchParams;
  Player defaultPerspective;
  loadParams(cfg, searchParams, defaultPerspective, C_EMPTY);
  NNEvaluator* nnEval;
  {
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = numAnalysisThreads * searchParams.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
    const int expectedConcurrentEvals = numAnalysisThreads * searchParams.numThreads;
    const bool defaultRequireExactNNLen = false;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = true;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }

  logger.write("Loaded model "+ modelFile);
  cmd.logOverrides(logger);

  logger.write("Starting to extract tensors from move " + Global::intToString(moveNumber) + " in " + sgfPath + "...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  if(moveNumber >= moves.size()) {
    logger.write(Global::strprintf("Error: Requested move %d, but game only has %d moves.", moveNumber, moves.size()));
    return 0;
  }
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  for(int i = 0; i < moveNumber; i++) {
    Move move = moves[i];
    history.makeBoardMoveTolerant(board, move.loc, move.pla);
  }

  RecentMovesTensor rmt;
  Move move = moves[moveNumber];

  int nnXLen = 19;
  int nnYLen = 19;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  int numTrunkFeatures = 384;  // strength model is limited to this size

  rmt.binaryInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({1, numSpatialFeatures, nnXLen, nnYLen})));
  rmt.locInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({1, 1, nnXLen, nnYLen})));
  rmt.globalInputNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({1, numGlobalFeatures})));
  rmt.trunkOutputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({1, numTrunkFeatures, nnXLen, nnYLen})));
  rmt.pickNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({1, numTrunkFeatures})));
  rmt.count = 1;

  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  bool inputsUseNHWC = false;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, rmt.binaryInputNCHW->data, rmt.globalInputNC->data);

  NNResultBuf nnResultBuf;
  // evaluate initial board once for initial prev-features
  nnEval->evaluate(board, history, move.pla, nnInputParams, nnResultBuf, false, false);
  assert(nnResultBuf.hasResult);
  const NNOutput* nnout = nnResultBuf.result.get();
  memcpy(rmt.trunkOutputNCHW->data, nnout->trunkData, nnXLen * nnYLen * numTrunkFeatures * sizeof(float));

  float* pickNC = rmt.pickNC->data;
  int pos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  if(pos >= 0 && pos < nnXLen * nnYLen) {
    for(int i = 0; i < numTrunkFeatures; i++) {
      pickNC[i] = rmt.trunkOutputNCHW->data[pos + i * nnXLen * nnYLen];
    }
  }
  else {
    std::fill(pickNC, pickNC + numTrunkFeatures, 0);
  }

  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  writeRecentMovesNpz(rmt, sgfPathWithoutExt + "_Inputs.npz");
  writeTrunkNpz(rmt, sgfPathWithoutExt + "_Trunk.npz");
  writePickNpz(rmt, sgfPathWithoutExt + "_Pick.npz");
  dumpTensor(sgfPathWithoutExt + "_Trunk.txt", rmt.trunkOutputNCHW->data, rmt.trunkOutputNCHW->dataLen);
  dumpTensor(sgfPathWithoutExt + "_Pick.txt", rmt.pickNC->data, rmt.pickNC->dataLen);

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

void writeRecentMovesNpz(const RecentMovesTensor& tensors, const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = tensors.binaryInputNCHW->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("binaryInputNCHW", tensors.binaryInputNCHW->dataIncludingHeader, numBytes);
  numBytes = tensors.locInputNCHW->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("locInputNCHW", tensors.locInputNCHW->dataIncludingHeader, numBytes);
  numBytes = tensors.globalInputNC->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("globalInputNC", tensors.globalInputNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void writeTrunkNpz(const RecentMovesTensor& tensors, const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = tensors.trunkOutputNCHW->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("trunkOutputNCHW", tensors.trunkOutputNCHW->dataIncludingHeader, numBytes);
  numBytes = tensors.locInputNCHW->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("locInputNCHW", tensors.locInputNCHW->dataIncludingHeader, numBytes);
  zipFile.close();
}

void writePickNpz(const RecentMovesTensor& tensors, const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = tensors.pickNC->prepareHeaderWithNumRows(tensors.count);
  zipFile.writeBuffer("pickNC", tensors.pickNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void dumpTensor(string path, float* data, size_t N) {
  std::ofstream file(path);
  for(int i = 0; i < N; i++) {
    file << data[i] << "\n";
  }
  file.close();
}

}
