#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../program/strengthmodel.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>

using namespace std;

void testExtractSingleMoves();

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

  int readMovesIntoTensor(const string& sgfPath, Player pla, RecentMovesTensor& rmt, NNEvaluator* nnEval, int idx, int windowSize);
  RecentMovesTensor getRecentMovesTensor(const Dataset& dataset, const Dataset::Game& game, Player pla, NNEvaluator* nnEval, int windowSize);
  void writeRecentMovesNpz(const RecentMovesTensor& tensors, const string& filePath);
  void writeTrunkNpz(const RecentMovesTensor& tensors, const string& filePath);
  void writePickNpz(const RecentMovesTensor& tensors, const string& filePath);
}

int MainCmds::extract_features(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;
  // testExtractSingleMoves();
  // return 0;

  ConfigParser cfg;
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  int windowSize; // Extract up to this many recent moves

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be fed into the rating system.",true,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",true,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    TCLAP::ValueArg<int> windowSizeArg("","window-size","Extract up to this many recent moves.",true,1000,"WINDOW_SIZE");
    cmd.add(windowSizeArg);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    windowSize = windowSizeArg.getValue();

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

  logger.write("Starting to extract features...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  Dataset dataset;
  dataset.load(listFile); // deliberately omit passing featureDir; we want to compute features, not load them

  for(size_t i = 0; i < dataset.games.size(); i++) {
    const Dataset::Game& game = dataset.games[i];
    if(Dataset::Game::none != game.set) {
      logger.write(game.sgfPath);
      auto blackTensors = getRecentMovesTensor(dataset, game, C_BLACK, nnEval, windowSize);
      auto whiteTensors = getRecentMovesTensor(dataset, game, C_WHITE, nnEval, windowSize);

      string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
      string blackFeaturesPath = Global::strprintf("%s/%s_BlackRecentMoves.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteRecentMoves.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      writeRecentMovesNpz(blackTensors, blackFeaturesPath);
      writeRecentMovesNpz(whiteTensors, whiteFeaturesPath);
      string blackTrunkPath = Global::strprintf("%s/%s_BlackRecentMovesTrunk.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      string whiteTrunkPath = Global::strprintf("%s/%s_WhiteRecentMovesTrunk.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      writeTrunkNpz(blackTensors, blackTrunkPath);
      writeTrunkNpz(whiteTensors, whiteTrunkPath);
      string blackPickPath = Global::strprintf("%s/%s_BlackRecentMovesPick.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      string whitePickPath = Global::strprintf("%s/%s_WhiteRecentMovesPick.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      writePickNpz(blackTensors, blackPickPath);
      writePickNpz(whiteTensors, whitePickPath);
    }
  }

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

void readSingleMoveTensor(
  RecentMovesTensor& rmt,
  int idx,
  NNEvaluator* nnEval,
  Board& board,
  const BoardHistory& history,
  Move move
) {
  int nnXLen = 19;
  int nnYLen = 19;
  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  int numTrunkFeatures = 384;  // strength model is limited to this size
  bool inputsUseNHWC = false;
  // write to row
  float* binaryInputNCHW = rmt.binaryInputNCHW->data + idx * nnXLen * nnYLen * numSpatialFeatures;
  float* globalInputNC = rmt.globalInputNC->data + idx * numGlobalFeatures;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInputNCHW, globalInputNC);
  memset(&rmt.locInputNCHW->data[idx * nnXLen * nnYLen], 0, nnXLen * nnYLen * sizeof(float));

  NNResultBuf nnResultBuf;
  // evaluate initial board once for initial prev-features
  nnEval->evaluate(board, history, move.pla, nnInputParams, nnResultBuf, false, false);
  assert(nnResultBuf.hasResult);
  const NNOutput* nnout = nnResultBuf.result.get();
  float* trunkOutputNCHW = rmt.trunkOutputNCHW->data + idx * nnXLen * nnYLen * numTrunkFeatures;
  memcpy(trunkOutputNCHW, nnout->trunkData, nnXLen * nnYLen * numTrunkFeatures * sizeof(float));

  float* pickNC = rmt.pickNC->data + idx * numTrunkFeatures;
  int pos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  if(pos >= 0 && pos < nnXLen * nnYLen) {
    for(int i = 0; i < numTrunkFeatures; i++) {
      pickNC[i] = trunkOutputNCHW[pos + i * nnXLen * nnYLen];
    }
  }
  else {
    std::fill(pickNC, pickNC + numTrunkFeatures, 0);
  }
}

int readMovesIntoTensor(const string& sgfPath, Player pla, RecentMovesTensor& rmt, NNEvaluator* nnEval, int idx, int windowSize) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  int count = std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla; });
  int skip = max(0, idx + count - windowSize);
  count -= skip;

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];

    if(pla == move.pla && skip-- <= 0) {
      readSingleMoveTensor(rmt, idx, nnEval, board, history, move);
      idx++;
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }

  return count;
}

RecentMovesTensor getRecentMovesTensor(const Dataset& dataset, const Dataset::Game& game, Player pla, NNEvaluator* nnEval, int windowSize) {
  RecentMovesTensor rmt;

  int nnXLen = 19;
  int nnYLen = 19;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  int numTrunkFeatures = 384;  // strength model is limited to this size

  rmt.binaryInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numSpatialFeatures, nnXLen, nnYLen})));
  rmt.locInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, 1, nnXLen, nnYLen})));
  rmt.globalInputNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numGlobalFeatures})));
  rmt.trunkOutputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numTrunkFeatures, nnXLen, nnYLen})));
  rmt.pickNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numTrunkFeatures})));

  int idx = 0;
  size_t playerId;
  int historic; // index of prev game
  if(P_BLACK == pla) {
    playerId = game.black.player;
    historic = game.black.prevGame;
  } else if(P_WHITE == pla) {
    playerId = game.white.player;
    historic = game.white.prevGame;
  } else {
    throw StringError("getRecentMoves: player must be black or white");
  }

  while(idx < windowSize && historic >= 0) {
    const Dataset::Game& historicGame = dataset.games[historic];

    if(playerId == historicGame.black.player) {
      pla = P_BLACK;
      historic = historicGame.black.prevGame;
    } else if(playerId == historicGame.white.player) {
      pla = P_WHITE;
      historic = historicGame.white.prevGame;
    } else {
      throw StringError(Global::strprintf("Game %s does not contain player %d (name=%s)",
        historicGame.sgfPath.c_str(), playerId, dataset.players[playerId].name.c_str()));
    }
    int count = readMovesIntoTensor(historicGame.sgfPath, pla, rmt, nnEval, idx, windowSize);
    idx += count;
  }

  rmt.count = idx;
  return rmt;
}

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

}


void dumpTensor(string path, float* data, size_t N) {
  std::ofstream file(path);
  for(int i = 0; i < N; i++) {
    file << data[i] << "\n";
  }
  file.close();
}

void testExtractSingleMove(
  const string& sgfPath,
  int moveNumber,
  const string& inputNpzPath,
  const string& trunkNpzPath,
  const string& pickNpzPath,
  const string& trunkTxtPath,
  const string& pickTxtPath,
  NNEvaluator* nnEval
) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  RecentMovesTensor rmt;

  int nnXLen = 19;
  int nnYLen = 19;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  int numTrunkFeatures = 384;  // strength model is limited to this size
  int windowSize = 1;

  rmt.binaryInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numSpatialFeatures, nnXLen, nnYLen})));
  rmt.locInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, 1, nnXLen, nnYLen})));
  rmt.globalInputNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numGlobalFeatures})));
  rmt.trunkOutputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numTrunkFeatures, nnXLen, nnYLen})));
  rmt.pickNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numTrunkFeatures})));

  // get to the move
  for(int idx = 0; idx < moveNumber; idx++) {
    Move move = moves[idx];
    history.makeBoardMoveTolerant(board, move.loc, move.pla);
  }

  Move move = moves[moveNumber];
  int rowIdx = 0;
  readSingleMoveTensor(rmt, rowIdx, nnEval, board, history, move);

  {
    ZipFile zipFile(inputNpzPath);
    uint64_t numBytes = rmt.binaryInputNCHW->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("binaryInputNCHW", rmt.binaryInputNCHW->dataIncludingHeader, numBytes);
    numBytes = rmt.locInputNCHW->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("locInputNCHW", rmt.locInputNCHW->dataIncludingHeader, numBytes);
    numBytes = rmt.globalInputNC->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("globalInputNC", rmt.globalInputNC->dataIncludingHeader, numBytes);
    zipFile.close();
  }
  {
    ZipFile zipFile(trunkNpzPath);
    uint64_t numBytes = rmt.trunkOutputNCHW->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("trunkOutputNCHW", rmt.trunkOutputNCHW->dataIncludingHeader, numBytes);
    numBytes = rmt.locInputNCHW->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("locInputNCHW", rmt.locInputNCHW->dataIncludingHeader, numBytes);
    zipFile.close();
  }
  {
    ZipFile zipFile(pickNpzPath);
    uint64_t numBytes = rmt.pickNC->prepareHeaderWithNumRows(1);
    zipFile.writeBuffer("pickNC", rmt.pickNC->dataIncludingHeader, numBytes);
    zipFile.close();
  }

  dumpTensor(trunkTxtPath, rmt.trunkOutputNCHW->data, rmt.trunkOutputNCHW->dataLen);
  dumpTensor(pickTxtPath, rmt.pickNC->data, rmt.pickNC->dataLen);
}

void testExtractSingleMoves() {
  string configPath = "/home/user/source/katago/cpp/configs/analysis_example.cfg";
  string modelPath = "/home/user/source/katago/models/kata1-b18c384nbt-s9131461376-d4087399203.bin.gz";

  NNEvaluator* nnEval;
  {
    Rand rand;
    ConfigParser cfg;
    cfg.initialize(configPath);
    Logger logger(&cfg, false, true);
    Setup::initializeSession(cfg);
    const int maxConcurrentEvals = 16;
    const int expectedConcurrentEvals = 4;
    const bool defaultRequireExactNNLen = false;
    const int defaultMaxBatchSize = -1;
    const bool disableFP16 = true;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelPath,modelPath,expectedSha256,cfg,logger,rand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }

  testExtractSingleMove(
    "dataset/2005/12/28/13056-udhar_nabresh-Reepicheep.sgf",
    10,
    "stuff/13056-Input.npz",
    "stuff/13056-Trunk.npz",
    "stuff/13056-Pick.npz",
    "stuff/trunk13056cpp.txt",
    "stuff/pick13056cpp.txt",
    nnEval
  );

  testExtractSingleMove(
    "dataset/2006/04/04/13788-dobromila-Duff.sgf",
    20,
    "stuff/13788-Input.npz",
    "stuff/13788-Trunk.npz",
    "stuff/13788-Pick.npz",
    "stuff/trunk13788cpp.txt",
    "stuff/pick13788cpp.txt",
    nnEval
  );

  testExtractSingleMove(
    "dataset/2006/04/04/13801-saruman-shige.sgf",
    30,
    "stuff/13801-Input.npz",
    "stuff/13801-Trunk.npz",
    "stuff/13801-Pick.npz",
    "stuff/trunk13801cpp.txt",
    "stuff/pick13801cpp.txt",
    nnEval
  );
}
