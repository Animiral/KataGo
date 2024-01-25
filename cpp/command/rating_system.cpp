#include "../core/global.h"
#include "../core/config_parser.h"
#include "../program/setup.h"
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

}

int MainCmds::rating_system(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir;
  string outlistFile; // Rating system CSV output file
  string modelFile;
  string strengthModelFile;
  bool numAnalysisThreadsCmdlineSpecified;
  int numAnalysisThreadsCmdline;

  KataGoCommandLine cmd("Calculate all match outcome predictions and player ranks in a dataset using the strength model.");
  try {
    cmd.addConfigFileArg("","strength_analysis_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> strengthModelFileArg("","strengthmodel","Neural net strength model file.",true,"","STRENGTH_MODEL_FILE");
    cmd.add(strengthModelFileArg);
    cmd.setShortUsageArgLimit();
    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be fed into the rating system.",true,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",false,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    TCLAP::ValueArg<string> outlistArg("","outlist","Rating system CSV output file.",false,"","OUTLIST_FILE");
    cmd.add(outlistArg);
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numAnalysisThreadsArg("","analysis-threads","Analyze up to this many positions in parallel. Equivalent to numAnalysisThreads in the config.",false,0,"THREADS");
    cmd.add(numAnalysisThreadsArg);
    cmd.parseArgs(args);

    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    outlistFile = outlistArg.getValue();
    modelFile = cmd.getModelFile();
    strengthModelFile = strengthModelFileArg.getValue();
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

  logger.write("Rating System evaluation starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  Dataset dataset;
  dataset.load(listFile, featureDir);
  int set = Dataset::Game::training;
  StrengthModel strengthModel(strengthModelFile, &dataset);

  // Print all games for information
  for(size_t i = 0; i < dataset.games.size(); i++) {
    Dataset::Game& gm = dataset.games[i];
    if(gm.set != set && !(Dataset::Game::training == set && Dataset::Game::batch == gm.set))
      continue;

    string blackName = dataset.players[gm.black.player].name;
    string whiteName = dataset.players[gm.white.player].name;
    string winner = gm.score > .5 ? "B+":"W+";
    std::cout << blackName << " vs " << whiteName << ": " << winner << "\n";
  }
  
  StochasticPredictor predictor;
  size_t windowSize = 1000;
  StrengthModel::Evaluation eval = strengthModel.evaluate(predictor, Dataset::Game::training, windowSize);
  cout << Global::strprintf("Rating system sq.err=%f, successRate=%.3f, successLogp=%f\n", eval.sqerr, eval.rate, eval.logp);

  dataset.store(outlistFile);
  logger.write("All cleaned up, quitting");
  return 0;
}
