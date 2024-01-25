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

int MainCmds::strength_analysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string playerName; // Directory for move feature cache.
  string modelFile;
  string strengthModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;
  // bool quitWithoutWaiting;
  vector<string> sgfPaths;

  KataGoCommandLine cmd("Run strength analysis engine.");
  try {
    cmd.addConfigFileArg("","strength_analysis_example.cfg");
    TCLAP::ValueArg<string> playerNameArg("","player","Analyze the moves of the player with this name in the SGFs.",false,"","PLAYER_NAME");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> strengthModelFileArg("","strengthmodel","Neural net strength model file.",true,"","STRENGTH_MODEL_FILE");
    cmd.add(strengthModelFileArg);
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    cmd.add(numAnalysisThreadsArg);
    TCLAP::UnlabeledMultiArg<string> sgfFileArg("","Sgf file(s) to analyze",true,string());
    cmd.add(sgfFileArg);
    cmd.parseArgs(args);

    playerName = playerNameArg.getValue();
    modelFile = cmd.getModelFile();
    strengthModelFile = strengthModelFileArg.getValue();
    numAnalysisThreadsCmdlineSpecified = numAnalysisThreadsArg.isSet();
    numAnalysisThreadsCmdline = numAnalysisThreadsArg.getValue();
    sgfPaths = sgfFileArg.getValue();

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
  StrengthModel strengthModel(strengthModelFile);
  vector<Sgf*> sgfs = Sgf::loadFiles(sgfPaths);
  StrengthModel::Analysis analysis = strengthModel.analyze(sgfs, playerName, search);

  cout << "Avg win%% loss: "  << std::fixed << std::setprecision(3) << analysis.avgWRLoss << ", pt loss: " << analysis.avgPLoss << ".\n";
  cout << "Rating for " << playerName << ": " << std::fixed << std::setprecision(2) << analysis.rating << "\n";

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}
