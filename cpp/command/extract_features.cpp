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
#include <chrono>
#include <functional>
#include <zip.h>
#include "../core/using.h"

namespace
{

using std::unique_ptr;
using std::make_unique;
using Global::strprintf;

constexpr int maxMoves = 1000; // capacity for moves

// params of this command
struct Parameters {
  unique_ptr<ConfigParser> cfg;
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  int windowSize; // Extract up to this many recent moves
};

// stores recent moves of one player in one game in a Dataset
struct RecentMoves {
  size_t game;
  Player pla;
  SelectedMoves sel;
};

Parameters parseArgs(const vector<string>& args);
unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
string movesetToString(const SelectedMoves::Moveset& moveset);
vector<RecentMoves> getRecentMovesOfEveryGame(const Dataset& dataset, int windowSize, Logger& logger);
// output one specific moveset
void outputZip(
  const SelectedMoves::Moveset& result,
  const string& sgfPath,
  const string& featureDir,
  const char* player,
  const char* title
);
// output all movesets in selected moves into one combined zip
void outputZip(
  const SelectedMoves& selectedMoves,
  const string& sgfPath,
  const string& featureDir,
  const char* player,
  const char* title
);

// takes a subset of the dataset games, feeds them into NN and prepares the results
class Worker {
 public:
  Worker() = default;
  Worker(
    SelectedMoves& allMoves_, // shared lookup structure for all workers
    LoadedModel& loadedModel_,
    const string& featureDir_,
    std::function<void(const string&)> progressCallback
  );

  void operator()(); // to be called as its own thread

  SelectedMoves* allMoves; // get trunks on all these

 private:

  LoadedModel* loadedModel;
  string featureDir;
  std::function<void(const string&)> progressCallback;
  unique_ptr<PrecomputeFeatures> extractor;

  void processGame(const string& sgfPath, SelectedMoves::Moveset& moveset);
  void processResults();

};

}

int MainCmds::extract_features(const vector<string>& args) {
  Parameters params = parseArgs(args);
  Logger logger(params.cfg.get(), false, true);
  logger.write(Version::getKataGoVersionForHelp());

  Board::initHash();
  ScoreValue::initTables();

  auto loadedModel = loadModel(params.modelFile);
  logger.write(strprintf("Loaded model %s.", params.modelFile.c_str()));

  Dataset dataset;
  dataset.load(params.listFile); // deliberately omit passing featureDir; we want to compute features, not load them
  logger.write("Find recent moves of all train/eval/test games using window size " + Global::intToString(params.windowSize) + "...");

  vector<RecentMoves> recentByGame = getRecentMovesOfEveryGame(dataset, params.windowSize, logger);
  SelectedMoves recentAll;
  for(auto& recent : recentByGame) {
    recentAll.merge(recent.sel);
  }
  size_t total = recentAll.bygame.size();
  logger.write(Global::intToString(total) + " games found. Extracting...");

  auto startTime = std::chrono::system_clock::now();
  size_t progress = 0;
  std::mutex progressMutex;
  auto progressCallback = [&] (const string& sgfPath) {
    std::lock_guard<std::mutex> lock(progressMutex);
    progress++;
    auto elapsedTime = std::chrono::system_clock::now() - startTime;
    auto remainingTime = elapsedTime * (total - progress) / progress;
    string remainingString = Global::longDurationToString(remainingTime);
    logger.write(strprintf("%d/%d (%s remaining): %s", progress, total, remainingString.c_str(), sgfPath.c_str()));
  };

  int threadCount = 1; // TODO: support more threads
  vector<Worker> workers;
  vector<std::thread> threads;
  for(int i = 0; i < threadCount; i++) {
    // size_t firstIndex = i * recentAll.bygame.size() / threadCount;
    // size_t lastIndex = (i+1) * recentAll.bygame.size() / threadCount;
    workers.push_back(Worker{
      recentAll, // TODO: divide jobs between workers
      // recentAll.bygame.begin() + firstIndex,
      // recentAll.bygame.begin() + lastIndex,  
      *loadedModel,
      params.featureDir,
      progressCallback
    });
    threads.emplace_back([&workers,i](){workers[i]();});
  }

  for(auto& thread : threads)
    thread.join();

  // all trunks are now precomputed; piece them back together into recent move sets and output ZIPs
  logger.write("Accumulating recent moves...");
  startTime = std::chrono::system_clock::now();
  total = recentByGame.size();
  progress = 0;
  for(RecentMoves& moves : recentByGame) {
    moves.sel.copyTrunkFrom(recentAll);
    assert(P_BLACK == moves.pla || P_WHITE == moves.pla);
    const char* player = P_BLACK == moves.pla ? "Black" : "White";
    outputZip(moves.sel, dataset.games[moves.game].sgfPath, params.featureDir, player, "Recent");
    progress++;
    auto elapsedTime = std::chrono::system_clock::now() - startTime;
    auto remainingTime = elapsedTime * (total - progress) / progress;
    string remainingString = Global::longDurationToString(remainingTime);
    logger.write(strprintf("%d/%d (%s remaining): %s", progress, total, remainingString.c_str(), dataset.games[moves.game].sgfPath.c_str()));
  }

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

string movesetToString(const SelectedMoves::Moveset& moveset) {
  std::ostringstream oss;
  const size_t N = moveset.moves.size();
  for(size_t i = 0; i < N; i++) {
    if(i > 0)
      oss << ", ";
    oss << moveset.moves[i].index;
  }
  return oss.str();
}

vector<RecentMoves> getRecentMovesOfEveryGame(const Dataset& dataset, int windowSize, Logger& logger) {
  vector<RecentMoves> recentByGame;
  for(size_t i = 0; i < dataset.games.size(); i++) {
    if(Dataset::Game::none != dataset.games[i].set) {
      logger.write(strprintf("Find recent moves of %s", dataset.games[i].sgfPath.c_str()));
      RecentMoves blackRecentMoves{i, P_BLACK, dataset.getRecentMoves(P_BLACK, i, windowSize)};
      // for(auto& kv : blackRecentMoves.sel.bygame)
        // logger.write(strprintf("\t(BLACK) %d moves in %s: %s", kv.second.moves.size(), kv.first.c_str(), movesetToString(kv.second).c_str()));
      RecentMoves whiteRecentMoves{i, P_WHITE, dataset.getRecentMoves(P_WHITE, i, windowSize)};
      // for(auto& kv : whiteRecentMoves.sel.bygame)
        // logger.write(strprintf("\t(WHITE) %d moves in %s: %s", kv.second.moves.size(), kv.first.c_str(), movesetToString(kv.second).c_str()));
      recentByGame.push_back(std::move(blackRecentMoves));
      recentByGame.push_back(std::move(whiteRecentMoves));
    }
  }
  return recentByGame;
}

void outputZip(
  const SelectedMoves::Moveset& result,
  const string& sgfPath,
  const string& featureDir,
  const char* player,
  const char* title
) {
  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  string featuresPath = strprintf("%s/%s_%s%s.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), player, title);
  result.writeToZip(featuresPath);
}

void outputZip(
  const SelectedMoves& selectedMoves,
  const string& sgfPath,
  const string& featureDir,
  const char* player,
  const char* title
) {
  SelectedMoves::Moveset combined;
  for(auto& kv : selectedMoves.bygame) {
    combined.moves.insert(combined.moves.end(), kv.second.moves.begin(), kv.second.moves.end());
  }
  outputZip(combined, sgfPath, featureDir, player, title);
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
  SelectedMoves& allMoves_,
  LoadedModel& loadedModel_,
  const string& featureDir_,
  std::function<void(const string&)> progressCallback_
)
: allMoves(&allMoves_),
  loadedModel(&loadedModel_),
  featureDir(featureDir_),
  progressCallback(progressCallback_),
  extractor(nullptr)
{}

void Worker::operator()() {
  extractor.reset(new PrecomputeFeatures(*loadedModel, maxMoves));
  for(auto& gm : allMoves->bygame) {
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
      throw StringError(strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  extractor->endGame(sgfPath);
}

void Worker::processResults() {
  extractor->evaluate();
  while(extractor->hasResult()) {
    PrecomputeFeatures::Result result = extractor->nextResult();
    SelectedMoves::Moveset& moveset = allMoves->bygame[result.sgfPath];
    PrecomputeFeatures::writeResultToMoveset(result, moveset);
    auto splitSet = moveset.splitBlackWhite();
    outputZip(splitSet.first, result.sgfPath, featureDir, "Black", "Trunk");
    outputZip(splitSet.second, result.sgfPath, featureDir, "White", "Trunk");
  }
}

}
