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
#include <zip.h>

using namespace std;

namespace
{

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

  Parameters parseArgs(const vector<string>& args);
  unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file);
  void evaluateTrunkToFile(const Dataset::Game& game, Player pla, PrecomputeFeatures& extractor, const string& featureDir);
}

int MainCmds::extract_features(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  Parameters params = parseArgs(args);
  Logger logger(params.cfg.get(), false, true);
  auto loadedModel = loadModel(params.modelFile);
  PrecomputeFeatures extractor(*loadedModel, maxMoves);

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
  for(size_t i = 0; i < dataset.games.size(); i++) {
    const Dataset::Game& game = dataset.games[i];
    if(Dataset::Game::batch == game.set) {
      evaluateTrunkToFile(game, C_BLACK, extractor, params.featureDir);
      evaluateTrunkToFile(game, C_WHITE, extractor, params.featureDir);
      logger.write(Global::strprintf("%d/%d: %s", progress, markedGames, game.sgfPath.c_str()));
      progress++;
    }
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

void evaluateTrunkToFile(const Dataset::Game& game, Player pla, PrecomputeFeatures& extractor, const string& featureDir) {
  extractor.clear();
  extractor.readFeaturesFromSgf(game.sgfPath, pla);
  extractor.evaluate();

  string color;
  if(C_BLACK == pla) color = "Black";
  if(C_WHITE == pla) color = "White";

  string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
  string featuresPath = Global::strprintf("%s/%s_%sTrunk.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), color.c_str());
  extractor.writeFeaturesToZip(featuresPath);
}


}
