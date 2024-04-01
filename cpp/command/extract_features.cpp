#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../strmodel/dataset.h"
#include "../command/commandline.h"
#include "../neuralnet/modelversion.h"
#include "../main.h"
#include <iomanip>
#include <memory>
#include <zip.h>

using namespace std;

namespace
{

  constexpr int maxBatchSize = 1000;
  constexpr int nnXLen = 19;
  constexpr int nnYLen = 19;
  constexpr int numTrunkFeatures = 384;  // strength model is limited to this size
  constexpr int gpuIdx = -1;

  struct ExtractFeatures {

    ExtractFeatures(LoadedModel& loadedModel, int cap);

    unique_ptr<ComputeHandle, void(*)(ComputeHandle*)> handle;
    int count; // first dimension size; rows entered
    int capacity; // first dimension size; max rows
    unique_ptr<InputBuffers, void(*)(InputBuffers*)> inputBuffers;
    unique_ptr<NumpyBuffer<float>> binaryInputNCHW; // game position tensors
    unique_ptr<NumpyBuffer<float>> globalInputNC; // global position tensors
    unique_ptr<NumpyBuffer<int>> movepos; // next move locations
    unique_ptr<NumpyBuffer<float>> trunkOutputNCHW; // NN state after last batch norm, without heads
    unique_ptr<NumpyBuffer<float>> pickNC; // trunk output at indicated location
    int numSpatialFeatures;
    int numGlobalFeatures;

    // extract input tensor and add it as new row
    void addBoard(Board& board, const BoardHistory& history, Move move);
    // run all added boards through the neural net
    void evaluate();
    // reset the extractor
    void clear();
    // write trunk & loc features in binary format
    void writeFeaturesToFile(const string& filePath);
    void writeFeaturesToBingz(const string& filePath);
    void writeRecentMovesNpz(const string& filePath);
    void writeTrunkNpz(const string& filePath);
    void writePickNpz(const string& filePath);

  };

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
  unique_ptr<ComputeHandle, void(*)(ComputeHandle*)> createComputeHandle(LoadedModel& loadedModel);
  unique_ptr<InputBuffers, void(*)(InputBuffers*)> createInputBuffers(LoadedModel& loadedModel);

  int readMovesIntoTensor(const string& sgfPath, Player pla, ExtractFeatures& extractor);
  void evaluateTrunkToFile(const Dataset::Game& game, Player pla, ExtractFeatures& extractor, const string& featureDir);
}

int MainCmds::extract_features(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  Parameters params = parseArgs(args);
  Logger logger(params.cfg.get(), false, true);
  auto loadedModel = loadModel(params.modelFile);
  ExtractFeatures extractor(*loadedModel, params.windowSize);

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

ExtractFeatures::ExtractFeatures(LoadedModel& loadedModel, int cap)
: handle(createComputeHandle(loadedModel)),
  inputBuffers(createInputBuffers(loadedModel))
{
  int modelVersion = NeuralNet::getModelVersion(&loadedModel);
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  capacity = cap;
  count = 0;

  binaryInputNCHW.reset(new NumpyBuffer<float>({cap, numSpatialFeatures, nnXLen, nnYLen}));
  globalInputNC.reset(new NumpyBuffer<float>({cap, numGlobalFeatures}));
  movepos.reset(new NumpyBuffer<int>({cap, 1}));
  trunkOutputNCHW.reset(new NumpyBuffer<float>({cap, numTrunkFeatures, nnXLen, nnYLen}));
  pickNC.reset(new NumpyBuffer<float>({cap, numTrunkFeatures}));
}

void ExtractFeatures::addBoard(Board& board, const BoardHistory& history, Move move) {
  assert(count < capacity);
  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  bool inputsUseNHWC = false;
  // write to row
  float* binaryInput = binaryInputNCHW->data + count * nnXLen * nnYLen * numSpatialFeatures;
  float* globalInput = globalInputNC->data + count * numGlobalFeatures;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInput, globalInput);
  movepos->data[count] = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  count++;
}

void ExtractFeatures::evaluate() {
  NeuralNet::getOutputTrunk(handle.get(), inputBuffers.get(), count, trunkOutputNCHW->data);
}

void ExtractFeatures::clear() {
  count = 0;
}

void ExtractFeatures::writeFeaturesToFile(const string& filePath) {
  string featureDir = FileUtils::dirname(filePath);
  if(!FileUtils::create_directories(featureDir))
    throw IOError("Failed to create directory " + featureDir);
  auto featureFile = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(filePath.c_str(), "wb"), &std::fclose);
  if(nullptr == featureFile)
    throw IOError("Failed to create feature file " + filePath);
  size_t writecount = std::fwrite(&Dataset::FEATURE_HEADER, 4, 1, featureFile.get());
  if(1 != writecount)
    throw IOError("Failed to write to feature file " + filePath);

  // feature file structure: 1 trunk output, 1 move pos (index), repeat.
  for(int i = 0; i < count; i++) {
    size_t trunkSize = nnXLen*nnYLen*numTrunkFeatures;
    writecount = std::fwrite(trunkOutputNCHW->data + i*trunkSize, sizeof(float), trunkSize, featureFile.get());
    if(trunkSize != writecount)
      throw IOError("Failed to write to feature file " + filePath);
    writecount = std::fwrite(&movepos->data[i], sizeof(int), 1, featureFile.get());
    if(1 != writecount)
      throw IOError("Failed to write to feature file " + filePath);
  }
  if(0 != std::fclose(featureFile.release()))
    throw IOError("Failed to write to feature file " + filePath);
}

void ExtractFeatures::writeFeaturesToBingz(const string& filePath) {
  int err;
  unique_ptr<zip_t, decltype(&zip_close)> archive{
    zip_open(filePath.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err),
    &zip_close
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw StringError(string("Error opening zip archive: ") + errstr);
  }

  // add trunk data
  {
    size_t trunkSize = nnXLen*nnYLen*numTrunkFeatures;
    unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
      zip_source_buffer(archive.get(), trunkOutputNCHW->data, count*trunkSize*sizeof(float), 0),
      &zip_source_free
    };
    if(!source)
      throw StringError(string("Error creating zip source: ") + zip_strerror(archive.get()));

    if(zip_add(archive.get(), "trunk.bin", source.get()) < 0)
      throw StringError(string("Error adding trunk.bin to zip archive: ") + zip_strerror(archive.get()));
    source.release(); // after zip_add, source is managed by libzip
  }

  // add movepos data
  {
    unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
      zip_source_buffer(archive.get(), movepos->data, count*sizeof(int), 0),
      &zip_source_free
    };
    if(!source)
      throw StringError(string("Error creating zip source: ") + zip_strerror(archive.get()));

    if(zip_add(archive.get(), "movepos.bin", source.get()) < 0)
      throw StringError(string("Error adding movepos.bin to zip archive: ") + zip_strerror(archive.get()));
    source.release(); // after zip_add, source is managed by libzip
  }

  zip_t* archivep = archive.release();
  if (zip_close(archivep) != 0)
    throw StringError(string("Error writing zip archive: ") + zip_strerror(archivep));
}

void ExtractFeatures::writeRecentMovesNpz(const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = binaryInputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW->dataIncludingHeader, numBytes);
  numBytes = movepos->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", movepos->dataIncludingHeader, numBytes);
  numBytes = globalInputNC->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("globalInputNC", globalInputNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void ExtractFeatures::writeTrunkNpz(const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = trunkOutputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("trunkOutputNCHW", trunkOutputNCHW->dataIncludingHeader, numBytes);
  numBytes = movepos->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", movepos->dataIncludingHeader, numBytes);
  zipFile.close();
}

void ExtractFeatures::writePickNpz(const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = pickNC->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("pickNC", pickNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

unique_ptr<LoadedModel, void(*)(LoadedModel*)> loadModel(string file) {
  return unique_ptr<LoadedModel, void(*)(LoadedModel*)>(NeuralNet::loadModelFile(file, ""), &NeuralNet::freeLoadedModel);
}

unique_ptr<ComputeHandle, void(*)(ComputeHandle*)> createComputeHandle(LoadedModel& loadedModel) {
  enabled_t useFP16Mode = enabled_t::False;
  enabled_t useNHWCMode = enabled_t::False;
  auto* computeContext = NeuralNet::createComputeContext(
    {gpuIdx}, nullptr, nnXLen, nnYLen,
    "", "", false,
    useFP16Mode, useNHWCMode, &loadedModel
  );

  bool requireExactNNLen = true;
  bool inputsUseNHWC = false;
  unique_ptr<ComputeHandle, void(*)(ComputeHandle*)> gpuHandle {
    NeuralNet::createComputeHandle(
      computeContext,
      &loadedModel,
      nullptr,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdx,
      0
    ),
    &NeuralNet::freeComputeHandle
  };

  return gpuHandle;
}

unique_ptr<InputBuffers, void(*)(InputBuffers*)> createInputBuffers(LoadedModel& loadedModel) {
  return unique_ptr<InputBuffers, void(*)(InputBuffers*)> {
    NeuralNet::createInputBuffers(&loadedModel, maxBatchSize, nnXLen, nnYLen),
    &NeuralNet::freeInputBuffers
  };
}

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

void readSingleMoveTensor(
  ExtractFeatures& extractor,
  NNEvaluator* nnEval,
  Board& board,
  const BoardHistory& history,
  Move move
) {
  extractor.addBoard(board, history, move);

  // NNResultBuf nnResultBuf;
  // // evaluate initial board once for initial prev-features
  // nnEval->evaluate(board, history, move.pla, nnInputParams, nnResultBuf, false, false);
  // assert(nnResultBuf.hasResult);
  // const NNOutput* nnout = nnResultBuf.result.get();
  // float* trunkOutputNCHW = rmt.trunkOutputNCHW->data + idx * nnXLen * nnYLen * numTrunkFeatures;
  // memcpy(trunkOutputNCHW, nnout->trunkData, nnXLen * nnYLen * numTrunkFeatures * sizeof(float));

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
}

int readMovesIntoTensor(const string& sgfPath, Player pla, ExtractFeatures& extractor) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(sgfPath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  int count = std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla; });
  int skip = max(0, extractor.count + count - extractor.capacity);
  count -= skip;

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];

    if(pla == move.pla && skip-- <= 0) {
      extractor.addBoard(board, history, move);
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }

  return count;
}

void evaluateTrunkToFile(const Dataset::Game& game, Player pla, ExtractFeatures& extractor, const string& featureDir) {
  extractor.clear();
  readMovesIntoTensor(game.sgfPath, pla, extractor);
  extractor.evaluate();

  string color;
  if(C_BLACK == pla) color = "Black";
  if(C_WHITE == pla) color = "White";

  string sgfPathWithoutExt = Global::chopSuffix(game.sgfPath, ".sgf");
  string featuresPath = Global::strprintf("%s/%s_%sTrunk.zip", featureDir.c_str(), sgfPathWithoutExt.c_str(), color.c_str());
  // extractor.writeFeaturesToFile(featuresPath);
  extractor.writeFeaturesToBingz(featuresPath);

  // string featuresPath = Global::strprintf("%s/%s_%sRecentMoves.npz", featureDir.c_str(), sgfPathWithoutExt.c_str(), color.c_str());
  // extractor.writeRecentMovesNpz(featuresPath);
  // string trunkPath = Global::strprintf("%s/%s_%sRecentMovesTrunk.npz", featureDir.c_str(), sgfPathWithoutExt.c_str(), color.c_str());
  // extractor.writeTrunkNpz(trunkPath);
  // string pickPath = Global::strprintf("%s/%s_%sRecentMovesPick.npz", featureDir.c_str(), sgfPathWithoutExt.c_str(), color.c_str());
  // extractor.writePickNpz(pickPath);
}


}
