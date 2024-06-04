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
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <atomic>
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
using std::ref;
using Global::strprintf;

// params of this command
struct Parameters {
  unique_ptr<ConfigParser> cfg;
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  int windowSize; // Extract up to this many recent moves
  int batchSize; // Send this many moves to the GPU at once in a worker thread
  int batchThreads; // Number of concurrent workers feeding positions to GPU
  bool recompute; // Overwrite existing ZIPs, do not reuse them
  bool printRecentMoves; // Output information on which moves are recent moves for which game
};

// stores recent moves of one player in one game in a Dataset
struct RecentMoves {
  size_t game;
  Player pla;
  SelectedMoves sel;
};

Parameters parseArgs(const vector<string>& args);
// unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, int evaluatorThreads, int batchSize);
string movesetToString(const SelectedMoves::Moveset& moveset);
vector<RecentMoves> getRecentMovesOfEveryGame(const Dataset& dataset, int windowSize, const string& featureDir, bool recompute, Logger& logger);
void printRecentMoves(const vector<RecentMoves>& recentMoves, const Dataset& dataset);
string zipPath(const string& sgfPath, const string& featureDir, Player player, const char* title);
// output all movesets in selected moves into one combined zip
void outputZip(const SelectedMoves& selectedMoves, const string& path);

// takes a subset of the dataset games, feeds them into NN and prepares the results
class Worker {
 public:
  Worker() = default;
  explicit Worker(PrecomputeFeatures&& precompute_);

  static void setWork(SelectedMoves& work_); // assign shared workload for all workers
  void operator()(); // to be called as its own thread

  static Logger* logger; // shared logger for all workers
  static string featureDir; // shared configuration: directory for zip output
  static bool recompute; // shared configuration: set true to disregard existing zips
  static bool printResultsDebug; // summary of precompute results to stdout

 private:

  static SelectedMoves* work;
  static std::map<string, SelectedMoves::Moveset>::iterator workIterator; // points to next work item
  static std::mutex workMutex; // synchronizes access to workIterator
  static size_t progress; // shared counter for finished work
  static std::chrono::time_point<std::chrono::system_clock> startTime; // time when work was assigned
  static std::mutex reportMutex; // synchronizes access to logger

  PrecomputeFeatures precompute;

  bool fetchWork(const string*& sgfPath, SelectedMoves::Moveset*& moveset);
  void processGame(const string& sgfPath, SelectedMoves::Moveset& moveset);
  void processResults();
  void reportProgress(const string& sgfPath);

};

void combineAndDumpRecentMoves(
  RecentMoves* workBegin,       // workload for this thread as begin/end pair,
  RecentMoves* workEnd,         // specifies which moves are recent to which games
  const Dataset& dataset,       // games SGF path lookup
  const string& featureDir,     // where to put the zip files
  Logger& logger,               // progress report sink
  std::atomic<size_t>& counter, // shared counter to report common progress
  size_t total,                 // overall number of items, 100% progress
  std::chrono::time_point<std::chrono::system_clock> startTime // common start time of all workers
);

}

int MainCmds::extract_features(const vector<string>& args) {
  Parameters params = parseArgs(args);
  int workerThreads = 64; // number of CPU threads filling requests to GPU
  Logger logger(params.cfg.get(), false, true);
  logger.write(Version::getKataGoVersionForHelp());

  Board::initHash();
  ScoreValue::initTables();

  // auto loadedModel = loadModel(params.modelFile);
  // logger.write(strprintf("Loaded model %s.", params.modelFile.c_str()));
  auto evaluator = createEvaluator(params.modelFile, params.batchThreads, params.batchSize);

  Dataset dataset;
  dataset.load(params.listFile); // deliberately omit passing featureDir; we want to compute features, not load them
  logger.write("Find recent moves of all train/eval/test games using window size " + Global::intToString(params.windowSize) + "...");

  vector<RecentMoves> recentByGame = getRecentMovesOfEveryGame(dataset, params.windowSize, params.featureDir, params.recompute, logger);
  if(params.printRecentMoves)
    printRecentMoves(recentByGame, dataset);

  // combine exactly what needs to be precomputed for which game
  // and do it using concurrent GPU workers.
  size_t total = 0;
  {
    SelectedMoves recentAll;
    for(auto& recent : recentByGame)
      recentAll.merge(recent.sel);
    total = recentAll.bygame.size();
    logger.write(Global::intToString(total) + " games found. Extracting...");
  
    Worker::logger = &logger;
    Worker::featureDir = params.featureDir;
    Worker::recompute = params.recompute;
    Worker::printResultsDebug = params.printRecentMoves; // clumsy tie of unrelated debug print options
    Worker::setWork(recentAll);
  
    vector<Worker> workers;
    vector<std::thread> threads;
    for(int i = 0; i < workerThreads; i++) {
      PrecomputeFeatures precompute(*evaluator, params.batchSize);
      precompute.includeTrunk = false; // TODO: add parameter for this
      precompute.includePick = true; // TODO: add parameter for this
      workers.emplace_back(std::move(precompute));
    }
    for(int i = 0; i < workerThreads; i++)
      threads.emplace_back(ref(workers[i]));
    for(auto& thread : threads)
      thread.join();
  }

  // all picks are now available as precomputed pick ZIPs;
  // piece them back together into recent move sets and output recent ZIPs
  {
    constexpr int accumulateThreadCount = 16;
    logger.write(strprintf("Accumulating recent moves using %d threads...", accumulateThreadCount));
    auto startTime = std::chrono::system_clock::now();
    total = recentByGame.size();
    std::atomic<size_t> progress(0);
    vector<std::thread> threads;
    for(int i = 0; i < accumulateThreadCount; i++) {
      RecentMoves* workBegin = &recentByGame[total*i/accumulateThreadCount];
      RecentMoves* workEnd = &recentByGame[total*(i+1)/accumulateThreadCount];
      threads.emplace_back(&combineAndDumpRecentMoves, workBegin, workEnd, ref(dataset),
                           ref(params.featureDir), ref(logger), ref(progress), total, startTime);
    }
    for(auto& thread : threads)
      thread.join();
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

  TCLAP::ValueArg<string> listArg("l","list","CSV file listing all SGFs to be fed into the rating system.",true,"","FILE",cmd);
  TCLAP::ValueArg<string> featureDirArg("d","featuredir","Directory for move feature cache.",true,"","DIR",cmd);
  TCLAP::ValueArg<int> windowSizeArg("s","window-size","Extract up to this many recent moves.",false,1000,"SIZE",cmd);
  TCLAP::ValueArg<int> batchSizeArg("b","batch-size","Send this many moves to the GPU at once in a worker thread.",false,400,"SIZE",cmd);
  TCLAP::ValueArg<int> batchThreadsArg("t","batch-threads","Number of concurrent workers feeding positions to GPU.",false,4,"COUNT",cmd);
  TCLAP::SwitchArg recomputeArg("r","recompute","Overwrite existing ZIPs, do not reuse them.",cmd,false);
  TCLAP::SwitchArg printRecentMovesArg("p","print-recent-moves","Output information on which moves are recent moves for which game.",cmd,false);
  cmd.addOverrideConfigArg();
  cmd.parseArgs(args);

  params.modelFile = cmd.getModelFile();
  params.listFile = listArg.getValue();
  params.featureDir = featureDirArg.getValue();
  params.windowSize = windowSizeArg.getValue();
  params.batchSize = batchSizeArg.getValue();
  params.batchThreads = batchThreadsArg.getValue();
  params.recompute = recomputeArg.getValue();
  params.printRecentMoves = printRecentMovesArg.getValue();
  cmd.getConfig(*params.cfg);

  return params;
}

// unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file) {
//   return unique_ptr<LoadedModel, void(*)(LoadedModel*)>(NeuralNet::loadModelFile(file, ""), &NeuralNet::freeLoadedModel);
// }

unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, int evaluatorThreads, int batchSize) {
  assert(evaluatorThreads > 0);
  assert(batchSize > 0);
  const int maxConcurrentEvals = evaluatorThreads*2;
  constexpr int nnXLen = 19;
  constexpr int nnYLen = 19;
  vector<int> gpuIdxByServerThread(evaluatorThreads, -1);
  auto evaluator = make_unique<NNEvaluator>(
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

vector<RecentMoves> getRecentMovesOfEveryGame(const Dataset& dataset, int windowSize, const string& featureDir, bool recompute, Logger& logger) {
  vector<RecentMoves> recentByGame;
  for(size_t i = 0; i < dataset.games.size(); i++) {
    if(Dataset::Game::none != dataset.games[i].set) {
      if(!recompute) {
        string blackZipPath = zipPath(dataset.games[i].sgfPath, featureDir, P_BLACK, "Recent");
        string whiteZipPath = zipPath(dataset.games[i].sgfPath, featureDir, P_WHITE, "Recent");
        if(FileUtils::exists(blackZipPath) && FileUtils::exists(whiteZipPath))
          continue;
      }
      logger.write(strprintf("Find recent moves of %s", dataset.games[i].sgfPath.c_str()));
      RecentMoves blackRecentMoves{i, P_BLACK, dataset.getRecentMoves(P_BLACK, i, windowSize)};
      RecentMoves whiteRecentMoves{i, P_WHITE, dataset.getRecentMoves(P_WHITE, i, windowSize)};
      recentByGame.push_back(std::move(blackRecentMoves));
      recentByGame.push_back(std::move(whiteRecentMoves));
    }
  }
  return recentByGame;
}

void printRecentMoves(const vector<RecentMoves>& recentMoves, const Dataset& dataset) {
  for(const RecentMoves& rec : recentMoves) {
    const string& sgfPath = dataset.games[rec.game].sgfPath;
    std::cout << strprintf("%d recent moves to %s for %s:\n", rec.sel.size(), sgfPath.c_str(), PlayerIO::playerToString(rec.pla).c_str());
    for(const auto& kv : rec.sel.bygame) {
      const auto& moves = kv.second.moves;
      std::cout << strprintf("  %d from %s:", moves.size(), kv.first.c_str());
      for(const auto& move : moves)
        std::cout << ' ' << move.index;
      std::cout << "\n";
    }
  }
}

string zipPath(const string& sgfPath, const string& featureDir, Player player, const char* title) {
  string sgfPathWithoutExt = Global::chopSuffix(sgfPath, ".sgf");
  string playerString = PlayerIO::playerToString(player);
  return strprintf("%s/%s_%s%s.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), playerString.c_str(), title);
}

void outputZip(const SelectedMoves& selectedMoves, const string& path) {
  SelectedMoves::Moveset combined;
  for(auto& kv : selectedMoves.bygame) {
    combined.moves.insert(combined.moves.end(), kv.second.moves.begin(), kv.second.moves.end());
  }
  combined.writeToZip(path);
}

Logger* Worker::logger;
string Worker::featureDir;
bool Worker::recompute;
bool Worker::printResultsDebug;
SelectedMoves* Worker::work;
std::map<string, SelectedMoves::Moveset>::iterator Worker::workIterator;
std::mutex Worker::workMutex;
size_t Worker::progress;
std::chrono::time_point<std::chrono::system_clock> Worker::startTime;
std::mutex Worker::reportMutex;

Worker::Worker(PrecomputeFeatures&& precompute_)
: precompute(std::move(precompute_))
{}

void Worker::setWork(SelectedMoves& work_) {
  work = &work_;
  workIterator = work->bygame.begin();
  progress = 0;
  startTime = std::chrono::system_clock::now();
}

void Worker::operator()() {
  const string* sgfPath;
  SelectedMoves::Moveset* moveset;
  while(fetchWork(sgfPath, moveset)) {
    // try to get already computed data, if we are resuming and it is available
    string blackPath = zipPath(*sgfPath, featureDir, P_BLACK, "Pick");
    if(!recompute && FileUtils::exists(blackPath))
      moveset->merge(SelectedMoves::Moveset::readFromZip(blackPath, P_BLACK));
    string whitePath = zipPath(*sgfPath, featureDir, P_WHITE, "Pick");
    if(!recompute && FileUtils::exists(whitePath))
      moveset->merge(SelectedMoves::Moveset::readFromZip(whitePath, P_WHITE));

    // picks loaded from existing ZIPs are automatically excluded from processing
    if(moveset->hasAllPicks())
      moveset->releaseStorage(); // throw away and re-read the ZIP later to save memory
    else
      processGame(*sgfPath, *moveset);
  }
  processResults();
}

bool Worker::fetchWork(const string*& sgfPath, SelectedMoves::Moveset*& moveset) {
  if(work->bygame.end() == workIterator)
    return false;

  std::lock_guard<std::mutex> lock(workMutex);
  sgfPath = &workIterator->first;
  moveset = &workIterator->second;
  ++workIterator;
  return true;
}

void Worker::processGame(const string& sgfPath, SelectedMoves::Moveset& moveset) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);
  // moveset is always in ascending order; we calculate all moves which do not have pick data
  // auto needsPick = [](SelectedMoves::Move& m) { return nullptr == m.pick; };
  // auto moveIt = std::find_if(moveset.moves.begin(), moveset.moves.end(), needsPick);
  auto moveIt = moveset.moves.begin(); // TODO: only pick moves which are necessary for requested features
  auto moveEnd = moveset.moves.end();

  precompute.startGame(sgfPath);
  for(int turnIdx = 0; moveIt != moveEnd; turnIdx++) {
    assert(turnIdx < moves.size()); // moves beyond end of game should never occur in moveset
    Move move = moves[turnIdx];

    if(turnIdx == moveIt->index) {
      precompute.addBoard(board, history, move);
      // if(precompute.isFull()) {
      //   processResults();
      // }
      /*do*/ moveIt++;
      /*while(moveIt != moveEnd && !needsPick(*moveIt));*/
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  if(moveIt != moveEnd) { // add final board?
    assert(history.getCurrentTurnNumber() == moveIt->index);
    precompute.addBoard(board, history);
    assert(++moveIt == moveEnd);
  }
  precompute.endGame();
}

void printResultToStdout(const PrecomputeFeatures::Result& result, const SelectedMoves::Moveset& moveset) {
  if(result.rows.size() < 2) { // needs at least one board with move + final board
    std::cout << strprintf("Result %s: empty result\n", result.sgfPath.c_str());
  }
  else {
    int firstMove = moveset.moves.at(result.startIndex).index;
    int endIndex = result.startIndex + result.rows.size();
    int lastMove = moveset.moves.at(endIndex-2).index; // discard final board
    std::cout << strprintf("Result %s: move %d-%d\n", result.sgfPath.c_str(), firstMove, lastMove);
    // moveset.printSummary(std::cout);
  }
}

void Worker::processResults() {
  vector<PrecomputeFeatures::Result> results = precompute.evaluate();
  for(auto& result : results) {
    SelectedMoves::Moveset& moveset = work->bygame[result.sgfPath];
    if(printResultsDebug)
      printResultToStdout(result, moveset);
    PrecomputeFeatures::writeResultToMoveset(result, moveset);
    if(moveset.hasAllPicks()) {
      auto splitSet = moveset.splitBlackWhite();
      string blackPath = zipPath(result.sgfPath, featureDir, P_BLACK, "Pick");
      if(recompute || !FileUtils::exists(blackPath))
        splitSet.first.writeToZip(blackPath);
      string whitePath = zipPath(result.sgfPath, featureDir, P_WHITE, "Pick");
      if(recompute || !FileUtils::exists(whitePath))
        splitSet.second.writeToZip(whitePath);
      moveset.releaseStorage(); // keeping this in memory for every file would be too much
      reportProgress(result.sgfPath);
    }
  }
}

void Worker::reportProgress(const string& sgfPath) {
  std::lock_guard<std::mutex> lock(reportMutex);
  size_t p = ++progress;
  size_t total = work->bygame.size();
  auto elapsedTime = std::chrono::system_clock::now() - startTime;
  auto remainingTime = elapsedTime * (total - p) / p;
  string remainingString = Global::longDurationToString(remainingTime);
  logger->write(strprintf("%d/%d (%s remaining): %s", p, total, remainingString.c_str(), sgfPath.c_str()));
}

// thread worker function for a subset of recent moves
void combineAndDumpRecentMoves(
  RecentMoves* workBegin,       // workload for this thread as begin/end pair,
  RecentMoves* workEnd,         // specifies which moves are recent to which games
  const Dataset& dataset,       // games SGF path lookup
  const string& featureDir,     // where to put the zip files
  Logger& logger,               // progress report sink
  std::atomic<size_t>& counter, // shared counter to report common progress
  size_t total,                 // overall number of items, 100% progress
  std::chrono::time_point<std::chrono::system_clock> startTime // common start time of all workers
) {
  for(RecentMoves* moves = workBegin; moves != workEnd; ++moves) {
    // get relevant precomputations from disk
    SelectedMoves precomputed;
    for(auto& kv : moves->sel.bygame) {
      Player pla = kv.second.moves[0].pla; // players never play against themselves, therefore we can just pick color of first move
      string path = zipPath(kv.first, featureDir, pla, "Pick");
      precomputed.bygame[kv.first] = SelectedMoves::Moveset::readFromZip(path, moves->pla);
    }
    // adopt precomputated sets into recent move set
    moves->sel.copyFeaturesFrom(precomputed);
    string path = zipPath(dataset.games[moves->game].sgfPath, featureDir, moves->pla, "Recent");
    outputZip(moves->sel, path);
    size_t p = ++counter;
    auto elapsedTime = std::chrono::system_clock::now() - startTime;
    auto remainingTime = elapsedTime * (total - p) / p;
    string remainingString = Global::longDurationToString(remainingTime);
    logger.write(strprintf("%d/%d (%s remaining): %s (%s)",
      p, total, remainingString.c_str(),
      dataset.games[moves->game].sgfPath.c_str(),
      PlayerIO::playerToString(moves->pla).c_str()));
  }
}

}
