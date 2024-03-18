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
  };

  int readMovesIntoTensor(const string& sgfPath, Player pla, RecentMovesTensor& rmt, int idx, int windowSize);
  RecentMovesTensor getRecentMovesTensor(const Dataset& dataset, const Dataset::Game& game, Player pla, int windowSize);
  void writeRecentMovesNpz(const RecentMovesTensor& tensors, const string& filePath);

}

int MainCmds::extract_features(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  int windowSize; // Extract up to this many recent moves

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be fed into the rating system.",true,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",true,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    TCLAP::ValueArg<int> windowSizeArg("","window-size","Extract up to this many recent moves.",true,1000,"WINDOW_SIZE");
    cmd.add(windowSizeArg);
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    windowSize = windowSizeArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
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
      auto blackTensors = getRecentMovesTensor(dataset, game, C_BLACK, windowSize);
      auto whiteTensors = getRecentMovesTensor(dataset, game, C_WHITE, windowSize);

      string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
      string blackFeaturesPath = Global::strprintf("%s/%s_BlackRecentMoves.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      string whiteFeaturesPath = Global::strprintf("%s/%s_WhiteRecentMoves.npz", featureDir.c_str(), sgfPathWithoutExt.c_str());
      writeRecentMovesNpz(blackTensors, blackFeaturesPath);
      writeRecentMovesNpz(whiteTensors, whiteFeaturesPath);
    }
  }

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

int readMovesIntoTensor(const string& sgfPath, Player pla, RecentMovesTensor& rmt, int idx, int windowSize) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);
  int nnXLen = 19;
  int nnYLen = 19;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  bool inputsUseNHWC = false;
  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;

  int count = std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla; });
  int skip = max(0, idx + count - windowSize);
  count -= skip;

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];

    if(pla == move.pla && skip-- <= 0) {
      // write to row
      float* binaryInputNCHW = rmt.binaryInputNCHW->data + idx * nnXLen * nnYLen * numSpatialFeatures;
      float* globalInputNC = rmt.globalInputNC->data + idx * numGlobalFeatures;
      NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInputNCHW, globalInputNC);
      memset(&rmt.locInputNCHW->data[idx * nnXLen * nnYLen], 0, nnXLen * nnYLen * sizeof(float));
      if(Board::PASS_LOC != move.loc) {
        int pos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
        rmt.locInputNCHW->data[idx * nnXLen * nnYLen + pos] = 1;
      }

      idx++;
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }

  return count;
}

RecentMovesTensor getRecentMovesTensor(const Dataset& dataset, const Dataset::Game& game, Player pla, int windowSize) {
  RecentMovesTensor rmt;

  int nnXLen = 19;
  int nnYLen = 19;
  int modelVersion = 14;
  int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);

  rmt.binaryInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numSpatialFeatures, nnXLen, nnYLen})));
  rmt.locInputNCHW.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, 1, nnXLen, nnYLen})));
  rmt.globalInputNC.reset(new NumpyBuffer<float>(std::vector<int64_t>({windowSize, numGlobalFeatures})));

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
    int count = readMovesIntoTensor(historicGame.sgfPath, pla, rmt, idx, windowSize);
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

}