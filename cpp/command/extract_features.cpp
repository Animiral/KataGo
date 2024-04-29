#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../strmodel/dataset.h"
#include "../strmodel/precompute.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <thread>
#include <mutex>
#include <functional>
#include <zip.h>
#include "../core/using.h"

namespace
{

using std::unique_ptr;
using std::make_unique;

constexpr int maxMoves = 1000; // capacity for moves

void loadParams(ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
  params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS);
  perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
  //Set a default for conservativePass that differs from matches or selfplay
  if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
    params.conservativePass = true;
}

// params of this command
struct Parameters {
  unique_ptr<ConfigParser> cfg;
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  int windowSize; // Extract up to this many recent moves
};

// takes a subset of the dataset games, feeds them into NN and prepares the results
class Worker {
 public:
  Worker() = default;
  Worker(
    SelectedMoves moves_,
    LoadedModel* loadedModel_,
    const string& featureDir_,
    std::function<void(const string&)> progressCallback
  );

  void operator()(); // to be called as its own thread

 private:

  SelectedMoves moves; // get trunks on all these
  LoadedModel* loadedModel;
  string featureDir;
  std::function<void(const string&)> progressCallback;
  unique_ptr<PrecomputeFeatures> extractor;

  void processGame(const string& sgfPath, SelectedMoves::Moveset& moveset);
  void processResults();
  void outputZip(const SelectedMoves::Moveset& result, const string& sgfPath, const char* player) const;

};

Parameters parseArgs(const vector<string>& args);
unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);

}

int MainCmds::extract_features(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  Parameters params = parseArgs(args);
  Logger logger(params.cfg.get(), false, true);
  auto loadedModel = loadModel(params.modelFile);

  logger.write("Starting to extract features...");
  logger.write(Version::getKataGoVersionForHelp());
  cerr << Version::getKataGoVersionForHelp() << endl;

  Dataset dataset;
  dataset.load(params.listFile); // deliberately omit passing featureDir; we want to compute features, not load them
  logger.write("Find recent moves of all train/eval/test games using window size " + Global::intToString(params.windowSize) + "...");

  struct RecentMoves {
    size_t game;
    Player pla;
    SelectedMoves sel;
  };
  vector<RecentMoves> recentByGame;
  for(size_t i = 0; i < dataset.games.size(); i++) {
    if(Dataset::Game::none != dataset.games[i].set) {
      recentByGame.push_back({i, P_BLACK, dataset.getRecentMoves(P_BLACK, i, params.windowSize)});
      recentByGame.push_back({i, P_WHITE, dataset.getRecentMoves(P_WHITE, i, params.windowSize)});
    }
  }
  SelectedMoves recentAll;
  for(auto& recent : recentByGame) {
    recentAll.merge(recent.sel);
  }

  size_t total = recentAll.bygame.size();
  logger.write(Global::intToString(total) + " games found. Extracting...");

  size_t progress = 0;
  std::mutex progressMutex;
  auto progressCallback = [&] (const string& sgfPath) {
    std::lock_guard<std::mutex> lock(progressMutex);
    progress++;
    logger.write(Global::strprintf("%d/%d: %s", progress, total, sgfPath.c_str()));
  };

  int threadCount = 1;
  vector<Worker> workers;
  vector<std::thread> threads;
  for(int i = 0; i < threadCount; i++) {
    // size_t firstIndex = i * recentAll.bygame.size() / threadCount;
    // size_t lastIndex = (i+1) * recentAll.bygame.size() / threadCount;
    workers.push_back({
      recentAll,
      // recentAll.bygame.begin() + firstIndex,
      // recentAll.bygame.begin() + lastIndex,  
      loadedModel.get(),
      params.featureDir,
      progressCallback
    });
    threads.emplace_back([&workers,i](){workers[i]();});
  }

  for(auto& thread : threads)
    thread.join();

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

Parameters parseArgs(const vector<string>& args) {
  Parameters params;
  params.cfg = make_unique<ConfigParser>();

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
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

  params.modelFile = cmd.getModelFile();
  params.listFile = listArg.getValue();
  params.featureDir = featureDirArg.getValue();
  params.windowSize = windowSizeArg.getValue();
  cmd.getConfig(*params.cfg);

  return params;
}

unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file) {
  return unique_ptr<LoadedModel, void(*)(LoadedModel*)>(NeuralNet::loadModelFile(file, ""), &NeuralNet::freeLoadedModel);
}

  // float* pickNC = rmt.pickNC->data + idx * numTrunkFeatures;
  // int pos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  // if(pos >= 0 && pos < nnXLen * nnYLen) {
  //   for(int i = 0; i < numTrunkFeatures; i++) {
  //     pickNC[i] = trunkOutputNCHW[pos + i * nnXLen * nnYLen];
  //   }
  // }
  // else {
  //   std::fill(pickNC, pickNC + numTrunkFeatures, 0);
  // }

Worker::Worker(
  SelectedMoves moves_,
  LoadedModel* loadedModel_,
  const string& featureDir_,
  std::function<void(const string&)> progressCallback_
)
: moves(std::move(moves_)),
  loadedModel(loadedModel_),
  featureDir(featureDir_),
  progressCallback(progressCallback_),
  extractor(nullptr)
{}

void Worker::operator()() {
  extractor.reset(new PrecomputeFeatures(*loadedModel, maxMoves));
  for(auto& gm : moves.bygame) {
    const string& sgfPath = gm.first;
    SelectedMoves::Moveset& moveset = gm.second;
    processGame(sgfPath, moveset);
    progressCallback(sgfPath);
  }
  processResults();
}

void Worker::processGame(const string& sgfPath, SelectedMoves::Moveset& moveset) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);
  auto moveIt = moveset.moves.begin(); // moveset is always in ascending order
  auto moveEnd = moveset.moves.end();

  for(int turnIdx = 0; moveIt != moveEnd; turnIdx++) {
    assert(turnIdx < moves.size()); // moves beyond end of game should never occur in moveset
    Move move = moves[turnIdx];

    if(turnIdx == moveIt->index) {
      extractor->addBoard(board, history, move);
      if(extractor->count >= extractor->capacity) {
        processResults();
        extractor->flip();
      }
      moveIt++;
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  extractor->endGame(sgfPath);
}

void Worker::processResults() {
  extractor->evaluate();
  while(extractor->hasResult()) {
    PrecomputeFeatures::Result result = extractor->nextResult();
    SelectedMoves::Moveset& moveset = moves.bygame[result.sgfPath];
    PrecomputeFeatures::writeResultToMoveset(result, moveset);
    auto splitSet = moveset.splitBlackWhite();
    outputZip(splitSet.first, result.sgfPath, "Black");
    outputZip(splitSet.second, result.sgfPath, "White");
  }
}

void Worker::outputZip(const SelectedMoves::Moveset& result, const string& sgfPath, const char* player) const
{
  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  string featuresPath = Global::strprintf("%s/%s_%sTrunk.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), player);
  result.writeToZip(featuresPath);
}

}
