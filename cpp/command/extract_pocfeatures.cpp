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

namespace
{

  void loadParams(ConfigParser& config, SearchParams& params, Player& perspective, Player defaultPerspective) {
    params = Setup::loadSingleParams(config,Setup::SETUP_FOR_ANALYSIS);
    perspective = Setup::parseReportAnalysisWinrates(config,defaultPerspective);
    //Set a default for conservativePass that differs from matches or selfplay
    if(!config.contains("conservativePass") && !config.contains("conservativePass0"))
      params.conservativePass = true;
  }

}

int MainCmds::extract_pocfeatures(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir; // Directory for move feature cache
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be fed into the rating system.",true,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",true,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    cmd.add(numAnalysisThreadsArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    numAnalysisThreadsCmdlineSpecified = numAnalysisThreadsArg.isSet();
    numAnalysisThreadsCmdline = numAnalysisThreadsArg.getValue();

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

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write("Loaded model "+ modelFile);
  cmd.logOverrides(logger);

  logger.write("Analysis Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

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
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,maxConcurrentEvals,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,defaultMaxBatchSize,defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_ANALYSIS
    );
  }

  Search search(searchParams, nnEval, &logger, "");
  Dataset dataset;
  dataset.load(listFile); // deliberately omit passing featureDir; we want to compute features, not load them
  StrengthModel strengthModel("", &dataset);
  strengthModel.extractFeatures(featureDir, search, &logger);

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
