#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../strmodel/dataset.h"
#include "../strmodel/strengthmodel.h"
#include "../strmodel/precompute.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nneval.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <zip.h>

namespace
{

using namespace std;
using namespace StrModel;

unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, Logger& logger);
vector<BoardFeatures> readFromSgf(const string& sgfPath, const string& modelFile, int moveNumber, Selection selection);
vector<BoardFeatures> readFromZip(const string& zipPath);
void dumpTensor(string path, const FeatureVector& data);
void printSummary(const vector<BoardFeatures>& features, std::ostream& stream);

ConfigParser cfg;
Logger* theLogger = nullptr;

}

int MainCmds::position_tensor(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string modelFile;
  string sgfPath;
  string zipPath;
  int moveNumber;
  Selection selection;
  bool summary;

  KataGoCommandLine cmd("Precompute move features for all games in the dataset.");
  try {
    cmd.addConfigFileArg("","analysis_example.cfg");
    cmd.addModelFileArg();
    cmd.setShortUsageArgLimit();

    TCLAP::ValueArg<string> sgfArg("","sgf","SGF file with the game to extract.",false,"","SGF_FILE",cmd);
    TCLAP::ValueArg<string> zipArg("","zip","ZIP file with precomputed trunk to extract.",false,"","ZIP_FILE", cmd);
    TCLAP::ValueArg<int> moveNumberArg("","move-number","Extract the position or trunk at this move number/index, starting at 0.",false,0,"MOVE_NUMBER", cmd);
    TCLAP::SwitchArg withTrunkArg("T","with-trunk","Extract trunk features from SGF.",cmd,false);
    TCLAP::SwitchArg withPickArg("P","with-pick","Extract pick features from SGF.",cmd,false);
    TCLAP::SwitchArg withHeadArg("H","with-head","Extract head features from SGf.",cmd,false);
    TCLAP::SwitchArg summaryArg("s","summary","Print general info on all moves in the SGF or ZIP.",cmd,false);
    cmd.addOverrideConfigArg();
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    sgfPath = sgfArg.getValue();
    zipPath = zipArg.getValue();
    moveNumber = moveNumberArg.getValue();
    selection.trunk = withTrunkArg.getValue();
    selection.pick = withPickArg.getValue();
    selection.head = withHeadArg.getValue();
    summary = summaryArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  cfg.applyAlias("numSearchThreadsPerAnalysisThread", "numSearchThreads");

  const bool logToStdoutDefault = false;
  const bool logToStderrDefault = true;
  Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);
  theLogger = &logger;

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);

  logger.write("Loaded config "+ cfg.getFileName());
  logger.write(Version::getKataGoVersionForHelp());
  if(!logger.isLoggingToStderr())
    cerr << Version::getKataGoVersionForHelp() << endl;

  if(!sgfPath.empty()) {
    vector<BoardFeatures> moveset = readFromSgf(sgfPath, modelFile, moveNumber, selection);
    if(summary) {
      printSummary(moveset, std::cout);
    }
    else {
      if(selection.trunk) {
        string trunkPath = strprintf("%s_Trunk%d.txt", Global::chopSuffix(sgfPath, ".sgf").c_str(), moveNumber);
        dumpTensor(trunkPath, *moveset.at(moveNumber).trunk);
      }
      if(selection.pick) {
        string pickPath = strprintf("%s_Pick%d.txt", Global::chopSuffix(sgfPath, ".sgf").c_str(), moveNumber);
        dumpTensor(pickPath, *moveset.at(moveNumber).pick);
      }
      if(selection.head) {
        string headPath = strprintf("%s_Head%d.txt", Global::chopSuffix(sgfPath, ".sgf").c_str(), moveNumber);
        dumpTensor(headPath, *moveset.at(moveNumber).head);
      }
    }
  }

  if(!zipPath.empty()) {
    vector<BoardFeatures> moveset = readFromZip(zipPath);
    if(summary) {
      printSummary(moveset, std::cout);
    }
    else {
      if(selection.trunk) {
        string trunkPath = strprintf("%s_Trunk%d.txt", Global::chopSuffix(zipPath, ".zip").c_str(), moveNumber);
        dumpTensor(trunkPath, *moveset.at(moveNumber).trunk);
      }
      if(selection.pick) {
        string pickPath = strprintf("%s_Pick%d.txt", Global::chopSuffix(zipPath, ".zip").c_str(), moveNumber);
        dumpTensor(pickPath, *moveset.at(moveNumber).pick);
      }
      if(selection.head) {
        string headPath = strprintf("%s_Head%d.txt", Global::chopSuffix(zipPath, ".zip").c_str(), moveNumber);
        dumpTensor(headPath, *moveset.at(moveNumber).head);
      }
    }
  }

  ScoreValue::freeTables();
  logger.write("All cleaned up, quitting");
  return 0;
}

namespace {

unique_ptr<NNEvaluator> createEvaluator(const string& modelFile, Logger& logger) {
  constexpr int evaluatorThreads = 2;
  constexpr int maxConcurrentEvals = 4;
  constexpr int batchSize = 10;
  constexpr int nnXLen = 19;
  constexpr int nnYLen = 19;
  vector<int> gpuIdxByServerThread(evaluatorThreads, -1);
  auto evaluator = make_unique<NNEvaluator>(
    modelFile,
    modelFile,
    "", // expectedSha256
    &logger,
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

vector<BoardFeatures> readFromSgf(const string& sgfPath, const string& modelFile, int moveNumber, Selection selection) {
  auto evaluator = createEvaluator(modelFile, *theLogger);
  theLogger->write("Loaded model "+ modelFile);
  Precompute precompute(*evaluator);

  theLogger->write("Starting to extract features from " + sgfPath + "...");

  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  vector<int> turns(moves.size() - moveNumber);
  std::iota(turns.begin(), turns.end(), moveNumber); // query all moves from SGF

  vector<BoardQuery> query = Precompute::makeQuery(turns, selection);
  vector<BoardResult> results = precompute.evaluate(*sgf, query);
  vector<BoardFeatures> combined = Precompute::combine(results);

  return combined;
}

vector<BoardFeatures> readFromZip(const string& zipPath) {
  DatasetFiles files(".");
  theLogger->write("Starting to extract tensors from " + zipPath + "...");
  return files.loadFeatures(zipPath);
}

void dumpTensor(string path, const FeatureVector& data) {
  std::ofstream file(path);
  for(float f : data) {
    file << f << "\n";
  }
  file.close();
}

// condense vec to a single printable number, likely different from vecs (trunks) of other positions,
// but also close in value to very similar vecs (tolerant of float inaccuracies)
float vecChecksum(const vector<float>& vec) {
  float sum = 0.0f;
  float sos = 0.0f;
  float weight = 1.0f;
  float decay = 0.9999667797285222f; // = pow(0.01, (1/(vec.size()-1))) -> smallest weight is 0.01

  for (size_t i = 0; i < vec.size(); ++i) {
      sum += vec[i] * weight;
      sos += vec[i] * vec[i];
      weight *= decay;
  }

  return sum + std::sqrt(sos);
}

void printSummary(const vector<BoardFeatures>& features, std::ostream& stream) {
  for(const BoardFeatures& feat : features) {
    string trunkStr = feat.trunk ? strprintf("trunk %f", vecChecksum(*feat.trunk)) : "no trunk"s;
    string pickStr = feat.pick ? strprintf("pick %f", vecChecksum(*feat.pick)) : "no pick"s;
    string headStr = feat.head ? strprintf("head(wr %f, pt %f, p %f, maxp %f, wr- %f, pt- %f)",
      feat.head->at(0), feat.head->at(1), feat.head->at(2), feat.head->at(3), feat.head->at(4), feat.head->at(5)) : "no head"s;
    stream << strprintf("%d: pos %d, %s, %s, %s\n", feat.turn, feat.pos, trunkStr.c_str(), pickStr.c_str(), headStr.c_str());
  }
}

}
