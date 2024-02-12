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

void loadParams(ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
  params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS);
  perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
  //Set a default for conservativePass that differs from matches or selfplay
  if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
    params.conservativePass = true;
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
  Rand seedRand("deterministic training"); // use default c'tor for random training

  ConfigParser cfg;
  string listFile; // CSV file listing all SGFs with labels to be fed into the strength training
  string featureDir;
  string strengthModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;

  KataGoCommandLine cmd("Use labeled games to train the strength model from move features.");
  try {
    cmd.addConfigFileArg("","strength_analysis_example.cfg");
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
  float trainingFraction = cfg.contains("trainingFraction") ? cfg.getFloat("trainingFraction", 0.f, 1.f) : .8f;
  float validationFraction = cfg.contains("validationFraction") ? cfg.getFloat("validationFraction", 0.f, 1.f) : .8f;
  int epochs = cfg.contains("trainingEpochs") ? cfg.getInt("trainingEpochs") : 100;
  int steps = cfg.contains("trainingSteps") ? cfg.getInt("trainingSteps") : 100;
  size_t batchSize = cfg.contains("trainingBatchSize") ? cfg.getInt("trainingBatchSize") : 100;
  float weightPenalty = cfg.contains("trainingWeightPenalty") ? cfg.getFloat("trainingWeightPenalty") : 1e-5f;
  float learnrate = cfg.contains("trainingLearnrate") ? cfg.getFloat("trainingLearnrate") : 1e-3f;
  size_t windowSize = cfg.contains("recentMovesWindowSize") ? cfg.getInt("recentMovesWindowSize") : 1000;
  logger.write(Global::strprintf("Training configuration: %.2f trainingFraction, %.2f validationFraction, %d epochs, %d steps, %d batchsize, %f weight penalty, %f learnrate. %d recentMovesWindowSize",
    trainingFraction, validationFraction, epochs, steps, batchSize, weightPenalty, learnrate, windowSize));

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  cmd.logOverrides(logger);

  logger.write("Strength Training starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  { // main training
    Dataset dataset;
    dataset.load(listFile, featureDir);
    // dataset.resize(10);
    logger.write("Loaded dataset with " + Global::intToString(dataset.games.size()) + " games from " + listFile);
    // dataset.randomSplit(seedRand, trainingFraction, validationFraction); // disabled: this info is loaded from the input data file

    if(!strengthModelFile.empty())
      logger.write("Loading strength model " + strengthModelFile);
    StrengthModel strengthModel(strengthModelFile, &dataset, &seedRand);
    strengthModel.train(epochs, steps, batchSize, weightPenalty, learnrate, windowSize, seedRand);

    if(!strengthModelFile.empty())
      strengthModel.net.saveModelFile(strengthModelFile);
  }

  logger.write("All cleaned up, quitting");
  return 0;
}
