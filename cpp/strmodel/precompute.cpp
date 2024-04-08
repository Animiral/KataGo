#include "precompute.h"
#include <vector>
#include <algorithm>
#include <zip.h>
#include "dataio/sgf.h"
#include "neuralnet/modelversion.h"

#include <iostream>

using std::unique_ptr;
using std::string;
using std::vector;
using namespace std::literals;

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel);
unique_ptr<InputBuffers, InputBuffersDeleter> createInputBuffers(LoadedModel& loadedModel);

}

PrecomputeFeatures::PrecomputeFeatures(LoadedModel& loadedModel, int cap)
: handle(createComputeHandle(loadedModel)),
  inputBuffers(createInputBuffers(loadedModel))
{
  int modelVersion = NeuralNet::getModelVersion(&loadedModel);
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  capacity = cap;
  allocateBuffers();
}

PrecomputeFeatures::PrecomputeFeatures(int cap)
: handle(nullptr, {}), inputBuffers(nullptr, {})
{
  capacity = cap;
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(NNModelVersion::defaultModelVersion); // 22 features
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(NNModelVersion::defaultModelVersion); // 19 features
  allocateBuffers();
}

void PrecomputeFeatures::addBoard(Board& board, const BoardHistory& history, Move move) {
  assert(count < capacity);
  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  bool inputsUseNHWC = false;
  // write to row
  float* binaryInput = NeuralNet::getSpatialBuffer(inputBuffers.get()) + count * nnXLen * nnYLen * numSpatialFeatures;
  float* globalInput = NeuralNet::getGlobalBuffer(inputBuffers.get()) + count * numGlobalFeatures;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInput, globalInput);
  movepos->data[count] = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  count++;
}

void PrecomputeFeatures::evaluate() {
  NeuralNet::getOutputTrunk(handle.get(), inputBuffers.get(), count, trunkOutputNCHW->data);
}

void PrecomputeFeatures::clear() {
  count = 0;
}

void PrecomputeFeatures::selectIndex(int index) {
  assert(index < count);

  movepos->data[0] = movepos->data[index];
  std::copy(
    trunkOutputNCHW->data + index*trunkSize,
    trunkOutputNCHW->data + (index+1)*trunkSize,
    trunkOutputNCHW->data
  );
  count = 1;
}

int PrecomputeFeatures::readFeaturesFromSgf(const string& filePath, Player pla) {
  auto sgf = std::unique_ptr<CompactSgf>(CompactSgf::loadFile(filePath));
  const auto& moves = sgf->moves;
  Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  Board board;
  BoardHistory history;
  Player initialPla;
  sgf->setupInitialBoardAndHist(rules, board, initialPla, history);

  int colorCount = std::count_if(moves.begin(), moves.end(), [pla](Move m) { return pla == m.pla; });
  int skip = std::max(0, count + colorCount - capacity);
  colorCount -= skip;

  for(int turnIdx = 0; turnIdx < moves.size(); turnIdx++) {
    Move move = moves[turnIdx];

    if((0 == pla || move.pla == pla) && skip-- <= 0) {
      addBoard(board, history, move);
    }

    // apply move
    bool suc = history.makeBoardMoveTolerant(board, move.loc, move.pla);
    if(!suc)
      throw StringError(Global::strprintf("Illegal move %s at %s", PlayerIO::playerToString(move.pla), Location::toString(move.loc, sgf->xSize, sgf->ySize)));
  }

  return colorCount;
}

void PrecomputeFeatures::readFeaturesFromZip(const std::string& filePath) {
  int err;
  unique_ptr<zip_t, decltype(&zip_close)> archive{
    zip_open(filePath.c_str(), ZIP_RDONLY, &err),
    &zip_close
  };
  if(!archive) {
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    string errstr = zip_error_strerror(&error);
    zip_error_fini(&error);
    throw StringError("Error opening zip archive: "s + errstr);
  }

  int countEntries = zip_get_num_entries(archive.get(), 0);
  if(2 != countEntries)
    throw StringError(Global::strprintf("Expected exactly two files in the archive, got %d.", countEntries));

  {
    string name = zip_get_name(archive.get(), 0, ZIP_FL_ENC_RAW);
    if("trunk.bin" != name)
        throw StringError("Name of file 0 in archive is unexpectedly not trunk.bin, but " + name);

    name = zip_get_name(archive.get(), 1, ZIP_FL_ENC_RAW);
    if("movepos.bin" != name)
        throw StringError("Name of file 0 in archive is unexpectedly not movepos.bin, but " + name);
  }

  zip_stat_t trunkStat, moveposStat;
  if(0 != zip_stat_index(archive.get(), 0, 0, &trunkStat))
    throw StringError("Error getting trunk.bin file information: "s + zip_strerror(archive.get()));
  uint64_t requiredCapacity = trunkStat.size / trunkSize / sizeof(float);
  if(trunkStat.size != requiredCapacity * trunkSize * sizeof(float))
    throw StringError("trunk.bin data should hold whole trunks, but has weird size: "s + Global::uint64ToString(trunkStat.size) + " bytes");
  if(0 != zip_stat_index(archive.get(), 1, 0, &moveposStat))
    throw StringError("Error getting movepos.bin file information: "s + zip_strerror(archive.get()));
  if(moveposStat.size != requiredCapacity * sizeof(int))
    throw StringError(Global::strprintf("movepos.bin data has %d bytes, but expected %s bytes", moveposStat.size, requiredCapacity * sizeof(int)));
  if(capacity < requiredCapacity) {
    capacity = requiredCapacity;
    allocateBuffers();
  }

  {
    unique_ptr<zip_file_t, decltype(&zip_fclose)> trunkFile{
      zip_fopen_index(archive.get(), 0, ZIP_RDONLY),
      &zip_fclose
    };
    if(!trunkFile)
      throw StringError("Error opening trunk.bin in zip archive: "s + zip_strerror(archive.get()));

    size_t read = zip_fread(trunkFile.get(), trunkOutputNCHW->data, trunkStat.size);
    if(trunkStat.size != read)
      throw StringError("Error reading zipped data of trunk.bin: "s + zip_strerror(archive.get()));
  }

  {
    unique_ptr<zip_file_t, decltype(&zip_fclose)> moveposFile{
      zip_fopen_index(archive.get(), 1, ZIP_RDONLY),
      &zip_fclose
    };
    if(!moveposFile)
      throw StringError("Error opening movepos.bin in zip archive: "s + zip_strerror(archive.get()));

    size_t read = zip_fread(moveposFile.get(), movepos->data, moveposStat.size);
    if(moveposStat.size != read)
      throw StringError("Error reading zipped data of movepos.bin: "s + zip_strerror(archive.get()));
  }

  count = requiredCapacity;
}

void PrecomputeFeatures::writeFeaturesToZip(const string& filePath) {
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
    throw StringError("Error opening zip archive: "s + errstr);
  }

  // add trunk data
  {
    unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
      zip_source_buffer(archive.get(), trunkOutputNCHW->data, count*trunkSize*sizeof(float), 0),
      &zip_source_free
    };
    if(!source)
      throw StringError("Error creating zip source: "s + zip_strerror(archive.get()));

    if(zip_add(archive.get(), "trunk.bin", source.get()) < 0)
      throw StringError("Error adding trunk.bin to zip archive: "s + zip_strerror(archive.get()));
    source.release(); // after zip_add, source is managed by libzip
  }

  // add movepos data
  {
    unique_ptr<zip_source_t, decltype(&zip_source_free)> source{
      zip_source_buffer(archive.get(), movepos->data, count*sizeof(int), 0),
      &zip_source_free
    };
    if(!source)
      throw StringError("Error creating zip source: "s + zip_strerror(archive.get()));

    if(zip_add(archive.get(), "movepos.bin", source.get()) < 0)
      throw StringError("Error adding movepos.bin to zip archive: "s + zip_strerror(archive.get()));
    source.release(); // after zip_add, source is managed by libzip
  }

  zip_t* archivep = archive.release();
  if (zip_close(archivep) != 0)
    throw StringError("Error writing zip archive: "s + zip_strerror(archivep));
}

void PrecomputeFeatures::writeInputsToNpz(const string& filePath) {
  auto binaryInputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{count, numSpatialFeatures, nnXLen, nnYLen});
  auto globalInputNC = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{count, numGlobalFeatures});
  std::copy_n(NeuralNet::getSpatialBuffer(inputBuffers.get()), count*nnXLen*nnYLen*numSpatialFeatures, binaryInputNCHW->data);
  std::copy_n(NeuralNet::getGlobalBuffer(inputBuffers.get()), count*numGlobalFeatures, globalInputNC->data);

  ZipFile zipFile(filePath);
  uint64_t numBytes = binaryInputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW->dataIncludingHeader, numBytes);
  numBytes = movepos->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", movepos->dataIncludingHeader, numBytes);
  numBytes = globalInputNC->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("globalInputNC", globalInputNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writeOutputsToNpz(const string& filePath) {
  ZipFile zipFile(filePath);
  uint64_t numBytes = trunkOutputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("trunkOutputNCHW", trunkOutputNCHW->dataIncludingHeader, numBytes);
  numBytes = movepos->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", movepos->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writePicksToNpz(const string& filePath) {
  NumpyBuffer<float> pickNC({count, numTrunkFeatures});

  for(int i = 0; i < count; i++) {
    int pos = movepos->data[i];
    if(pos >= 0 && pos < nnXLen * nnYLen) {
      for(int j = 0; j < numTrunkFeatures; j++) {
        pickNC.data[i*numTrunkFeatures + j] = trunkOutputNCHW->data[i*trunkSize + j*nnXLen*nnYLen + pos];
      }
    }
    else {
      std::fill(pickNC.data + i*numTrunkFeatures, pickNC.data + (i+1)*numTrunkFeatures, 0);
    }
  }

  ZipFile zipFile(filePath);
  uint64_t numBytes = pickNC.prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("pickNC", pickNC.dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::allocateBuffers() {
  movepos.reset(new NumpyBuffer<int>({capacity, 1}));
  trunkOutputNCHW.reset(new NumpyBuffer<float>({capacity, numTrunkFeatures, nnXLen, nnYLen}));
  count = 0;
}

void ComputeHandleDeleter::operator()(ComputeHandle* handle) noexcept {
  return NeuralNet::freeComputeHandle(handle);
}

void InputBuffersDeleter::operator()(InputBuffers* buffers) noexcept {
  return NeuralNet::freeInputBuffers(buffers);
}

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel) {
  enabled_t useFP16Mode = enabled_t::False;
  enabled_t useNHWCMode = enabled_t::False;
  auto* computeContext = NeuralNet::createComputeContext(
    {PrecomputeFeatures::gpuIdx}, nullptr, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen,
    "", "", false,
    useFP16Mode, useNHWCMode, &loadedModel
  );

  bool requireExactNNLen = true;
  bool inputsUseNHWC = false;
  unique_ptr<ComputeHandle, ComputeHandleDeleter> gpuHandle {
    NeuralNet::createComputeHandle(
      computeContext,
      &loadedModel,
      nullptr,
      PrecomputeFeatures::maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      PrecomputeFeatures::gpuIdx,
      0
    ),
    ComputeHandleDeleter()
  };

  return gpuHandle;
}

unique_ptr<InputBuffers, InputBuffersDeleter> createInputBuffers(LoadedModel& loadedModel) {
  return unique_ptr<InputBuffers, InputBuffersDeleter> {
    NeuralNet::createInputBuffers(&loadedModel, PrecomputeFeatures::maxBatchSize, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen),
    InputBuffersDeleter()
  };
}

}
