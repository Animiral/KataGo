#include "precompute.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <zip.h>
#include "dataio/sgf.h"
#include "dataio/numpywrite.h"
#include "neuralnet/modelversion.h"

#include <iostream>

using std::unique_ptr;
using std::string;
using std::vector;
using namespace std::literals;

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity);

}

PrecomputeFeatures::PrecomputeFeatures(LoadedModel& loadedModel, int cap)
: handle(createComputeHandle(loadedModel, cap)),
  inputBuffers(nullptr, {})
{
  int modelVersion = NeuralNet::getModelVersion(&loadedModel);
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  capacity = carrySize = cap;
  inputBuffers.reset(NeuralNet::createInputBuffers(&loadedModel, capacity, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen));
  allocateBuffers();
}

PrecomputeFeatures::PrecomputeFeatures(int cap)
: handle(nullptr, {}), inputBuffers(nullptr, {})
{
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(NNModelVersion::defaultModelVersion); // 22 features
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(NNModelVersion::defaultModelVersion); // 19 features
  capacity = carrySize = cap;
  allocateBuffers();
}

void PrecomputeFeatures::addBoard(Board& board, const BoardHistory& history, Move move) {
  if(count >= capacity)
    throw StringError("Precompute capacity exhausted: cannot add another board to this batch.");

  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;
  bool inputsUseNHWC = false;
  // write to row
  float* binaryInput = NeuralNet::getSpatialBuffer(inputBuffers.get()) + count * nnXLen * nnYLen * numSpatialFeatures;
  float* globalInput = NeuralNet::getGlobalBuffer(inputBuffers.get()) + count * numGlobalFeatures;
  NNInputs::fillRowV7(board, history, move.pla, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, binaryInput, globalInput);
  size_t resultIndex = carrySize + count;
  movepos[resultIndex] = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  plas[resultIndex] = move.pla;
  count++;
}

void PrecomputeFeatures::endGame(const std::string& sgfPath) {
  int moves = carrySize + count - resultTip;
  Result result {
    sgfPath,
    moves,
    trunk.data() + resultTip * trunkSize,
    movepos.data() + resultTip,
    plas.data() + resultTip
  };
  results.push(result);
  resultTip += moves;
}

void PrecomputeFeatures::evaluate() {
  float* trunkResultArea = trunk.data() + carrySize*trunkSize;
  NeuralNet::getOutputTrunk(handle.get(), inputBuffers.get(), count, trunkResultArea);
}

PrecomputeFeatures::Result PrecomputeFeatures::nextResult() {
  Result result = results.front();
  results.pop();
  return result;
}

bool PrecomputeFeatures::hasResult() {
  return !results.empty();
}

void PrecomputeFeatures::flip() {
  if(hasResult())
    throw StringError("Precomputed results are unused and would be discarded on flip.");

  // move buffers contents from output area to carry area
  size_t carryCount = carrySize + count - resultTip;
  if(carryCount >= carrySize)
    throw StringError("Precompute capacity too small: partial game results would be discarded.");

  size_t newResultTip = carrySize - carryCount;
  std::copy_n(trunk.data() + resultTip*trunkSize, carryCount*trunkSize, trunk.data() + newResultTip*trunkSize);
  std::copy_n(movepos.data() + resultTip, carryCount, movepos.data() + newResultTip);
  std::copy_n(plas.data() + resultTip, carryCount, plas.data() + newResultTip);
  resultTip = newResultTip;
  count = 0;
}

void PrecomputeFeatures::selectIndex(int index) {
  assert(index < count);

  size_t bufferIndex = carrySize + index;
  std::copy_n(trunk.data() + bufferIndex*trunkSize, trunkSize, trunk.data() + carrySize*trunkSize);
  movepos[carrySize] = movepos[bufferIndex];
  plas[carrySize] = plas[bufferIndex];
  count = 1;
}

std::pair<PrecomputeFeatures::Result, PrecomputeFeatures::Result>
PrecomputeFeatures::splitBlackWhite(Result result) {
  // temp space for swapping things around
  vector<float> whiteTrunk(result.moves * trunkSize);
  vector<int> whiteMovepos(result.moves);

  // move white data to temp space
  int target = 0; // index to fill next
  for(int i = 0; i < result.moves; i++) {
    if(P_WHITE == result.player[i]) {
      std::copy_n(result.trunk + i*trunkSize, trunkSize, whiteTrunk.begin() + target*trunkSize);
      whiteMovepos[target] = result.movepos[i];
      target++;
    }
  }
  int whiteCount = target;

  // move black data to start of result
  target = 0; // index to fill next
  for(int i = 0; i < result.moves; i++) {
    if(P_BLACK == result.player[i]) {
      std::copy_n(result.trunk + i*trunkSize, trunkSize, result.trunk + target*trunkSize);
      result.movepos[target] = result.movepos[i];
      target++;
    }
  }

  assert(whiteCount + target == result.moves); // sanity check

  // append white temp data to fill result
  std::copy_n(whiteTrunk.begin(), whiteCount*trunkSize, result.trunk + target*trunkSize);
  std::copy_n(whiteMovepos.begin(), whiteCount, result.movepos + target);

  return {
    Result { // black
      result.sgfPath,
      target,
      result.trunk,
      result.movepos,
      nullptr // no player data
    },
    Result { // white
      result.sgfPath,
      whiteCount,
      result.trunk + target*trunkSize,
      result.movepos + target,
      nullptr // no player data
    }
  };
}

void PrecomputeFeatures::writeResultToMoveset(Result result, SelectedMoves::Moveset& moveset) {
  assert(result.moves == moveset.moves.size());

  for(size_t i = 0; i < result.moves; i++) {
    float* trunkBegin = result.trunk + i*trunkSize;
    float* trunkEnd = result.trunk + (i+1)*trunkSize;
    moveset.moves[i].trunk.reset(new vector<float>(trunkBegin, trunkEnd));
    moveset.moves[i].pos = result.movepos[i];
  }
}

void PrecomputeFeatures::writeResultToZip(Result result, const string& filePath) {
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
      zip_source_buffer(archive.get(), result.trunk, result.moves*trunkSize*sizeof(float), 0),
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
      zip_source_buffer(archive.get(), result.movepos, result.moves*sizeof(int), 0),
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
  auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{count});
  std::copy_n(movepos.begin() + carrySize, count, moveposN->data);

  ZipFile zipFile(filePath);
  uint64_t numBytes = binaryInputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW->dataIncludingHeader, numBytes);
  numBytes = moveposN->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
  numBytes = globalInputNC->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("globalInputNC", globalInputNC->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writeOutputsToNpz(const string& filePath) {
  auto trunkOutputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{count, numTrunkFeatures, nnXLen, nnYLen});
  std::copy_n(trunk.begin() + carrySize*trunkSize, count*trunkSize, trunkOutputNCHW->data);
  auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{count});
  std::copy_n(movepos.begin() + carrySize, count, moveposN->data);

  ZipFile zipFile(filePath);
  uint64_t numBytes = trunkOutputNCHW->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("trunkOutputNCHW", trunkOutputNCHW->dataIncludingHeader, numBytes);
  numBytes = moveposN->prepareHeaderWithNumRows(count);
  zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
  zipFile.close();
}

void PrecomputeFeatures::writePicksToNpz(const string& filePath) {
  NumpyBuffer<float> pickNC({count, numTrunkFeatures});

  for(int i = 0; i < count; i++) {
    int pos = movepos[carrySize+i];
    if(pos >= 0 && pos < nnXLen * nnYLen) {
      for(int j = 0; j < numTrunkFeatures; j++) {
        pickNC.data[i*numTrunkFeatures + j] = trunk[(carrySize+i)*trunkSize + j*nnXLen*nnYLen + pos];
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
  // The following buffers are part "carry area", part "result area", sized for carrySize/capacity moves.
  // Results are evaluated into result area and picked up from there via nextResult().
  // Any leftover evaluated moves that have not been finalized into a result (because there are more moves in the game)
  // move to the carry area before overwriting the result area and later pieced together into a whole result.
  trunk.resize((carrySize + capacity) * trunkSize);
  movepos.resize(carrySize + capacity);
  plas.resize(carrySize + capacity);
  resultTip = carrySize; // results start at beginning of result area
  count = 0;
}

void ComputeHandleDeleter::operator()(ComputeHandle* handle) noexcept {
  return NeuralNet::freeComputeHandle(handle);
}

void InputBuffersDeleter::operator()(InputBuffers* buffers) noexcept {
  return NeuralNet::freeInputBuffers(buffers);
}

namespace {

unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity) {
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
      capacity,
      requireExactNNLen,
      inputsUseNHWC,
      PrecomputeFeatures::gpuIdx,
      0
    ),
    ComputeHandleDeleter()
  };

  return gpuHandle;
}

}
