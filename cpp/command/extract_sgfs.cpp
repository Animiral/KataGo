#include "../core/config_parser.h"
#include "../command/commandline.h"
#include "../strmodel/dataset.h"
#include "../strmodel/precompute.h"
#include "../main.h"

namespace
{

using namespace StrModel;

// params of this command
struct Parameters {
  string modelFile;
  vector<string> sgfs; // SGF file(s) to evaluate
  string outFile; // Output file for extracted features
  string playerName; // SGF player name to evaluate
  Selection selection; // features to extract
  int windowSize; // Extract up to this many recent moves
  int batchSize; // Send this many moves to the GPU at once (per evaluator thread)
  int batchThreads; // Number of concurrent evaluator threads feeding positions to GPU
};

// override the feature path structure with just one out path
class OneDatasetFile : public DatasetFiles {

public:

  explicit OneDatasetFile(const string& outFile);
  virtual void storeFeatures(const vector<BoardFeatures>& features, const string& path) const override;

private:

  string outFile;

};

Parameters parseArgs(const vector<string>& args, ConfigParser& cfg);
unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, int evaluatorThreads, int batchSize);

}

int MainCmds::extract_sgfs(const vector<string>& args) {
  ConfigParser cfg;
  Parameters params = parseArgs(args, cfg);
  Logger logger(&cfg, false, true);
  logger.write(Version::getKataGoVersionForHelp());

  Board::initHash();
  ScoreValue::initTables();

  logger.write(strprintf("Evaluating %d game records:", params.sgfs.size()));
  for(const string& sgfPath : params.sgfs)
    logger.write(strprintf("  - %s", sgfPath.c_str()));
  logger.write(strprintf("Model: %s", params.modelFile.c_str()));
  logger.write(strprintf("Output file: %s", params.outFile.c_str()));
  logger.write(strprintf("Extract trunk? %s", params.selection.trunk ? "true" : "false"));
  logger.write(strprintf("Extract head? %s", params.selection.head ? "true" : "false"));
  logger.write(strprintf("Extract pick? %s", params.selection.pick ? "true" : "false"));
  logger.write(strprintf("Window size: %d", params.windowSize));
  logger.write(strprintf("Batch size: %d", params.batchSize));
  logger.write(strprintf("Batch threads: %d", params.batchThreads));

  vector<Sgf*> sgfs;
  for(const string& sgfPath : params.sgfs)
    sgfs.push_back(Sgf::loadFile(sgfPath));
  OneDatasetFile datasetFiles(params.outFile);
  Dataset dataset(sgfs, datasetFiles);

  PlayerId playerId = -1;

  if(params.playerName.empty()) {
    playerId = dataset.findOmnipresentPlayer();
    params.playerName = dataset.players[playerId].name;
  }
  else {
    for(PlayerId i = 0; i < dataset.players.size(); i++)
      if(params.playerName == dataset.players[i].name)
        playerId = i;
  }
  if(playerId < 0)
    throw StringError("Player name not found in SGFs: " + params.playerName);

  logger.write(strprintf("Player: %s", params.playerName.c_str()));

  auto evaluator = createEvaluator(params.modelFile, params.batchThreads, params.batchSize);
  Precompute precompute(*evaluator);

  vector<BoardFeatures> allFeatures;
  GamesTurns gamesTurns = dataset.getRecentMoves(playerId, params.windowSize);
  for(auto it = gamesTurns.bygame.begin(); it != gamesTurns.bygame.end(); ++it) {
    GameId gameId = it->first;
    const vector<int>& turns = it->second;
    vector<BoardQuery> queries = Precompute::makeQuery(turns, params.selection);
    vector<BoardResult> results = precompute.evaluate(CompactSgf(sgfs[gameId]), queries);
    vector<BoardFeatures> features = Precompute::combine(results);
    features = Precompute::filter(features, turns);
    allFeatures.insert(allFeatures.end(), features.begin(), features.end());
  }

  // using our overriding OneDatasetFile, this stores allFeatures to params.outFile
  dataset.storeFeatures(allFeatures, 0, C_EMPTY, "");

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

OneDatasetFile::OneDatasetFile(const string& outFile_)
: DatasetFiles("."), outFile(outFile_) {}

void OneDatasetFile::storeFeatures(const vector<BoardFeatures>& features, const string& ) const {
  DatasetFiles::storeFeatures(features, outFile);
}

Parameters parseArgs(const vector<string>& args, ConfigParser& cfg) {
  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  cmd.addConfigFileArg("","analysis_example.cfg");
  cmd.addModelFileArg();
  cmd.setShortUsageArgLimit();

  TCLAP::UnlabeledMultiArg<string> sgfsArg("sgfs", "SGF file(s) to evaluate", true, "FILE", cmd);
  TCLAP::ValueArg<string> outFileArg("o","outfile","Output file for extracted features",false,"features.zip","OUTFILE",cmd);
  TCLAP::ValueArg<string> playerNameArg("p","playername","SGF player name to evaluate",false,"","PLAYERNAME",cmd);
  TCLAP::ValueArg<int> windowSizeArg("s","window-size","Extract up to this many recent moves.",false,1000,"SIZE",cmd);
  TCLAP::ValueArg<int> batchSizeArg("b","batch-size","Send this many moves to the GPU at once (per evaluator thread).",false,10,"SIZE",cmd);
  TCLAP::ValueArg<int> batchThreadsArg("t","batch-threads","Number of concurrent evaluator threads feeding positions to GPU.",false,4,"COUNT",cmd);
  TCLAP::SwitchArg withTrunkArg("T","with-trunk","Extract trunk features.",cmd,false);
  TCLAP::SwitchArg withPickArg("P","with-pick","Extract pick features.",cmd,false);
  TCLAP::SwitchArg withHeadArg("H","with-head","Extract head features.",cmd,false);
  cmd.addOverrideConfigArg();
  cmd.parseArgs(args);

  Parameters params;
  params.modelFile = cmd.getModelFile();
  params.sgfs = sgfsArg.getValue();
  params.outFile = outFileArg.getValue();
  params.playerName = playerNameArg.getValue();
  params.selection.trunk = withTrunkArg.getValue();
  params.selection.pick = withPickArg.getValue();
  params.selection.head = withHeadArg.getValue();
  params.windowSize = windowSizeArg.getValue();
  params.batchSize = batchSizeArg.getValue();
  params.batchThreads = batchThreadsArg.getValue();
  cmd.getConfig(cfg);

  if(!params.selection.trunk && !params.selection.pick && !params.selection.head) {
    throw StringError("No features selected for extraction.");
  }

  return params;
}

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

}
