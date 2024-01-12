#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../program/strengthmodel.h"
#include "../command/commandline.h"
#include "../main.h"
#include <iomanip>

using namespace std;

namespace {

// poor man's pre-C++20 format, https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
template<typename ... Args>
std::string customFormat( const std::string& format, Args ... args ) {
  int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
  if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
  auto size = static_cast<size_t>( size_s );
  std::unique_ptr<char[]> buf( new char[ size ] );
  std::snprintf( buf.get(), size, format.c_str(), args ... );
  return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void loadParams(ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
  params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS);
  perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
  //Set a default for conservativePass that differs from matches or selfplay
  if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
    params.conservativePass = true;
}

Dataset mockDataset() {
  return {};
}

FeaturesAndTargets mockFeaturesAndTargets() {
  FeaturesAndTargets fat;
  vector<MoveFeatures> mfs; // winProb, lead, movePolicy, maxPolicy, winrateLoss, pointsLoss
  mfs.push_back(MoveFeatures{.5f,  0.f, .99f, .99f,  0.f, 0.0f});
  mfs.push_back(MoveFeatures{.8f,  1.f, .91f, .94f,  5.f, 0.4f});
  mfs.push_back(MoveFeatures{.2f, -1.f, .20f, .30f, 10.f, 2.0f});
  fat.emplace_back(mfs, 2000);
  mfs.clear();
  mfs.push_back(MoveFeatures{.5f,  0.f, .03f, .70f, 20.f, 5.0f});
  mfs.push_back(MoveFeatures{.2f, -1.f, .01f, .50f, 30.f, 10.f});
  mfs.push_back(MoveFeatures{.9f,  3.f, .20f, .30f, 10.f, 2.0f});
  fat.emplace_back(mfs, 1000);
  return fat;
}

// this is what we give as input to the strength model for a single move
struct MoveFeatures {
  float winProb;
  float lead;
  float movePolicy;
  float maxPolicy;
  float winrateLoss;  // compared to previous move
  float pointsLoss;  // compared to previous move
};
}

int MainCmds::strength_training(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string listFile; // CSV file listing all SGFs with labels to be fed into the strength training
  string featureDir;
  string modelFile;
  string strengthModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;

  KataGoCommandLine cmd("Use labeled games to train the strength model from move features.");
  try {
    cmd.addConfigFileArg("","strength_analysis_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> strengthModelFileArg("","strengthmodel","Neural net strength model file.",true,"","STRENGTH_MODEL_FILE");
    cmd.add(strengthModelFileArg);
    cmd.setShortUsageArgLimit();
    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be trained on.",false,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",false,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    cmd.add(numAnalysisThreadsArg);
    cmd.parseArgs(args);

    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    modelFile = cmd.getModelFile();
    strengthModelFile = strengthModelFileArg.getValue();
    numAnalysisThreadsCmdlineSpecified = numAnalysisThreadsArg.isSet();
    numAnalysisThreadsCmdline = numAnalysisThreadsArg.getValue();
    // quitWithoutWaiting = quitWithoutWaitingArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  if(cfg.contains("numAnalysisThreads") && numAnalysisThreadsCmdlineSpecified)
    throw StringError("When specifying numAnalysisThreads in the config (" + cfg.getFileName() + "), it is redundant and disallowed to also specify it via -analysis-threads");

  const int numAnalysisThreads = numAnalysisThreadsCmdlineSpecified ? numAnalysisThreadsCmdline : cfg.getInt("numAnalysisThreads",1,16384);
  if(numAnalysisThreads <= 0 || numAnalysisThreads > 16384)
    throw StringError("Invalid value for numAnalysisThreads: " + Global::intToString(numAnalysisThreads));

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  // training config:
  float splitFraction = cfg.contains("trainingTestSplit") ? cfg.getFloat("trainingTestSplit", 0.f, 1.f) : .8f;
  int epochs = cfg.contains("trainingEpochs") ? cfg.getInt("trainingEpochs") : 100;
  size_t batchSize = cfg.contains("trainingBatchSize") ? cfg.getInt("trainingBatchSize") : 100;
  float weightPenalty = cfg.contains("trainingWeightPenalty") ? cfg.getFloat("trainingWeightPenalty") : 1e-5f;
  float learnrate = cfg.contains("trainingLearnrate") ? cfg.getFloat("trainingLearnrate") : 1e-3f;
  logger.write(customFormat("Training configuration: %.2f training/test split, %d epochs, %d batchsize, %f weight penalty, %f learnrate.",
    splitFraction, epochs, batchSize, weightPenalty, learnrate));

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write("Loaded model "+ modelFile);
  cmd.logOverrides(logger);

  logger.write("Strength Training starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  SearchParams searchParams;
  Player defaultPerspective;
  loadParams(cfg, searchParams, defaultPerspective, C_EMPTY);

  NNEvaluator* nnEval;
  // {
  //   Setup::initializeSession(cfg);
  //   const int maxConcurrentEvals = numAnalysisThreads * searchParams.numThreads * 2 + 16; // * 2 + 16 just to give plenty of headroom
  //   const int expectedConcurrentEvals = numAnalysisThreads * searchParams.numThreads;
  //   const bool defaultRequireExactNNLen = false;
  //   const int defaultMaxBatchSize = -1;
  //   const bool disableFP16 = false;
  //   const string expectedSha256 = "";
  //   nnEval = Setup::initializeNNEvaluator(
  //     modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
  //     NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
  //     Setup::SETUP_FOR_ANALYSIS
  //   );
  // }

  { // main training
    // Search search(searchParams, nnEval, &logger, "");
    StrengthModel strengthModel(strengthModelFile, nullptr /*search*/, featureDir);
    logger.write("Loaded strength model "+ strengthModelFile);
    Dataset dataset = strengthModel.loadDataset(listFile);
    // Dataset dataset = mockDataset();
    // dataset.resize(10);
    
    logger.write("Loaded dataset with " + Global::intToString(dataset.games().size()) + " games from " + listFile);
    FeaturesAndTargets featuresTargets = strengthModel.getFeaturesAndTargets(dataset);
    // FeaturesAndTargets featuresTargets = mockFeaturesAndTargets();

    size_t split = static_cast<size_t>(featuresTargets.size() * splitFraction);
    logger.write(customFormat("Training on set of size %d (%d training, %d test)",
      featuresTargets.size(), split, featuresTargets.size() - split));

    strengthModel.train(featuresTargets, split, epochs, batchSize, weightPenalty, learnrate);
  }

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
