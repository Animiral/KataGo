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

  constexpr int maxMoves = 200; // capacity for moves

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

  // works on a subset of the dataset
  struct ProcessGamesWorker {
    ProcessGamesWorker() = default;
    ProcessGamesWorker(
      vector<Dataset::Game>::iterator begin_,
      vector<Dataset::Game>::iterator end_,
      LoadedModel* loadedModel_,
      const string& featureDir_,
      std::function<void(const string&)> progressCallback
    );

    vector<Dataset::Game>::iterator begin;
    vector<Dataset::Game>::iterator end;
    LoadedModel* loadedModel;
    string featureDir;
    std::function<void(const string&)> progressCallback;

    void operator()(); // to be called as its own thread
    void processGame(const Dataset::Game& game);
    void processResults();
    void outputZip(PrecomputeFeatures::Result result, const char* player);

    unique_ptr<PrecomputeFeatures> extractor;
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
  logger.write("Marking all train/eval/test games using window size " + Global::intToString(params.windowSize) + "...");
  dataset.markRecentGames(params.windowSize, &logger);
  size_t markedGames = std::count_if(dataset.games.begin(), dataset.games.end(), [](auto& g) { return Dataset::Game::batch == g.set; });
  logger.write(Global::intToString(markedGames) + " games marked. Extracting...");

  size_t progress = 0;
  std::mutex progressMutex;
  auto progressCallback = [&] (const string& sgfPath) {
    std::lock_guard<std::mutex> lock(progressMutex);
    progress++;
    logger.write(Global::strprintf("%d/%d: %s", progress, markedGames, sgfPath.c_str()));
  };

  int threadCount = 2;
  vector<ProcessGamesWorker> workers;
  vector<std::thread> threads;
  for(int i = 0; i < threadCount; i++) {
    size_t firstIndex = i * dataset.games.size() / threadCount;
    size_t lastIndex = (i+1) * dataset.games.size() / threadCount;
    workers.push_back({
      dataset.games.begin() + firstIndex,
      dataset.games.begin() + lastIndex,  
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

ProcessGamesWorker::ProcessGamesWorker(
  vector<Dataset::Game>::iterator begin_,
  vector<Dataset::Game>::iterator end_,
  LoadedModel* loadedModel_,
  const string& featureDir_,
  std::function<void(const string&)> progressCallback_
)
: begin(begin_),
  end(end_),
  loadedModel(loadedModel_),
  featureDir(featureDir_),
  progressCallback(progressCallback_),
  extractor(nullptr)
{}

void ProcessGamesWorker::operator()() {
  extractor.reset(new PrecomputeFeatures(*loadedModel, maxMoves));
  for(auto it = begin; it != end; ++it) {
    Dataset::Game& game = *it;
    if(Dataset::Game::batch == game.set) {
      processGame(game);
      progressCallback(game.sgfPath);
    }
  }
  processResults();
}

void ProcessGamesWorker::processGame(const Dataset::Game& game) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(game.sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];
    extractor->addBoard(board, history, move);
    if(extractor->count >= extractor->capacity) {
      processResults();
      extractor->flip();
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }
  extractor->endGame(game.sgfPath);
}

void ProcessGamesWorker::processResults() {
  extractor->evaluate();
  while(extractor->hasResult()) {
    PrecomputeFeatures::Result result = extractor->nextResult();
    PrecomputeFeatures::Result blackResult, whiteResult;
    std::tie(blackResult, whiteResult) = extractor->splitBlackWhite(result);
    outputZip(blackResult, "Black");
    outputZip(whiteResult, "White");
  }
}

void ProcessGamesWorker::outputZip(PrecomputeFeatures::Result result, const char* player)
{
  string sgfPathWithoutExt = Global::chopSuffix(result.sgfPath, ".sgf");
  string featuresPath = Global::strprintf("%s/%s_%sTrunk.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), player);
  PrecomputeFeatures::writeResultToZip(result, featuresPath);
}

}
