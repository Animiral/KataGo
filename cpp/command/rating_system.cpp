#include "../core/global.h"
#include "../core/config_parser.h"
#include "../program/setup.h"
#include "../strmodel/strengthmodel.h"
#include "../command/commandline.h"
#include "../main.h"
#include <iomanip>

using namespace std;

namespace {

  int parseSetMarker(const string& setMarker) {
    if("t" == setMarker || "T" == setMarker)
      return Dataset::Game::training;
    if("v" == setMarker || "V" == setMarker)
      return Dataset::Game::validation;
    if("e" == setMarker || "E" == setMarker)
      return Dataset::Game::test;
    if("b" == setMarker || "B" == setMarker)
      return Dataset::Game::batch;
    else
      return -1; // run on every game
  }

}

int MainCmds::rating_system(const vector<string>& args) {
  ConfigParser cfg;
  string listFile; // CSV file listing all SGFs to be fed into the rating system
  string featureDir;
  string outlistFile; // Rating system CSV output file
  string setMarker;
  string strengthModelFile;

  KataGoCommandLine cmd("Calculate all match outcome predictions and player ranks in a dataset using the strength model.");
  try {
    cmd.addConfigFileArg("","strength_analysis_example.cfg");
    TCLAP::ValueArg<string> strengthModelFileArg("","strengthmodel","Neural net strength model file.",false,"","STRENGTH_MODEL_FILE");
    cmd.add(strengthModelFileArg);
    cmd.setShortUsageArgLimit();
    TCLAP::ValueArg<string> listArg("","list","CSV file listing all SGFs to be fed into the rating system.",true,"","LIST_FILE");
    cmd.add(listArg);
    TCLAP::ValueArg<string> featureDirArg("","featuredir","Directory for move feature cache.",true,"","FEATURE_DIR");
    cmd.add(featureDirArg);
    TCLAP::ValueArg<string> outlistArg("","outlist","Rating system CSV output file.",false,"","OUTLIST_FILE");
    cmd.add(outlistArg);
    TCLAP::ValueArg<string> setArg("","set","Which set to rate: T/V/E/*.",false,"","SET");
    cmd.add(setArg);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    listFile = listArg.getValue();
    featureDir = featureDirArg.getValue();
    outlistFile = outlistArg.getValue();
    setMarker = setArg.getValue();
    strengthModelFile = strengthModelFileArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  const bool logToStderr = logger.isLoggingToStderr();

  logger.write("Loaded config "+ cfg.getFileName());
  cmd.logOverrides(logger);

  logger.write("Rating System evaluation starting...");
  logger.write(Version::getKataGoVersionForHelp());
  if(!logToStderr) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  Dataset dataset;
  dataset.load(listFile, featureDir);
  int set = parseSetMarker(setMarker);
  StrengthModel strengthModel(strengthModelFile, &dataset);

  unique_ptr<Predictor> predictor;
  if(strengthModelFile.empty()) {
    cout << "Using stochastic model.\n";
    predictor.reset(new StochasticPredictor());
  }
  else {
    cout << "Using strength model at " << strengthModelFile << ".\n";
    predictor.reset(new SmallPredictor(strengthModel.net));
  }
  size_t windowSize = 1000;
  StrengthModel::Evaluation eval = strengthModel.evaluate(*predictor, set, windowSize);
  cout << Global::strprintf("Evaluation on %d %c-games: mse=%f, successRate=%.3f, successLogp=%f\n",
    eval.count, "*-TVBE"[set+1], eval.mse, eval.rate, eval.logp);

  cout << Global::strprintf("Store evaluation in %s...\n", outlistFile.c_str());
  dataset.store(outlistFile);
  logger.write("All cleaned up, quitting");
  return 0;
}
