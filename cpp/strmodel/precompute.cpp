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

// namespace {

// unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity);

// }

PrecomputeFeatures::PrecomputeFeatures(NNEvaluator& nnEvaluator, int cap)
: PrecomputeFeatures(cap)
{
  evaluator = &nnEvaluator;
  // handle = createComputeHandle(loadedModel, cap);
  // int modelVersion = NeuralNet::getModelVersion(&loadedModel);
  // numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  // numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  // inputBuffers.reset(NeuralNet::createInputBuffers(&loadedModel, capacity, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen));
  // allocateBuffers();
}

PrecomputeFeatures::PrecomputeFeatures(int cap)
: // handle(nullptr, {}), inputBuffers(nullptr, {}),
  includeTrunk(false),
  includePick(true),
  count(0),
  capacity(cap),
  resultTip(0)
{
  numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(NNModelVersion::defaultModelVersion); // 22 features
  numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(NNModelVersion::defaultModelVersion); // 19 features
  // allocateBuffers();
}

void PrecomputeFeatures::startGame(const std::string& sgfPath) {
  nextResult.sgfPath = sgfPath;
  nextResult.startIndex = 0;
  // nextResult.trunk = trunk.data() + count * trunkSize;
  // nextResult.movepos.clear();
  // nextResult.pick.clear();
  // nextResult.player.clear();
  nextResult.rows.clear();
}

void PrecomputeFeatures::addBoard(Board& board, const BoardHistory& history, Move move) {
  int rowPos = NNPos::locToPos(move.loc, board.x_size, nnXLen, nnYLen);
  addBoardImpl(board, history, rowPos, move.pla, true);
}

void PrecomputeFeatures::addBoard(Board& board, const BoardHistory& history) {
  addBoardImpl(board, history, 0, history.presumedNextMovePla, false);
}

void PrecomputeFeatures::endGame() {
  results.push_back(nextResult);
  resultTip = count;
  nextResult = {};
}

bool PrecomputeFeatures::isFull() const {
  return count >= capacity;
}

std::vector<PrecomputeFeatures::Result> PrecomputeFeatures::evaluate() {
  if(count > 0) {
    // if there is an open game, it becomes a partial result; the process resembles endGame(), then startGame()
    if(count > resultTip) {
      results.push_back(nextResult);
      nextResult.startIndex += nextResult.rows.size(); // continue from intermediate state
      nextResult.rows.clear();
    }
  }
  count = resultTip = 0;
  return move(results);
}

// std::vector<PrecomputeFeatures::Result> PrecomputeFeatures::evaluateTrunks() {
//   if(count > 0) {
//     NeuralNet::getOutputTrunk(handle.get(), inputBuffers.get(), count, trunk.data());
//     // if there is an open game, it becomes a partial result; the process resembles endGame(), then startGame()
//     if(count > resultTip) {
//       nextResult.endIndex = nextResult.startIndex + count - resultTip;
//       results.push_back(nextResult);
//       nextResult.startIndex = nextResult.endIndex;
//       nextResult.trunk = trunk.data();
//       nextResult.movepos = movepos.data();
//       nextResult.pick = pick.data();
//       nextResult.player = plas.data();
//     }
//   }
//   count = resultTip = 0;
//   for(Result& result : results)
//     result.pick = nullptr; // no picks when evaluating trunks
//   return move(results);
// }

// std::vector<PrecomputeFeatures::Result> PrecomputeFeatures::evaluatePicks() {
//   if(count > 0) {
//     NeuralNet::getOutputPick(handle.get(), inputBuffers.get(), count, pick.data());
//     // if there is an open game, it becomes a partial result; the process resembles endGame(), then startGame()
//     if(count > resultTip) {
//       nextResult.endIndex = nextResult.startIndex + count - resultTip;
//       results.push_back(nextResult);
//       nextResult.startIndex = nextResult.endIndex;
//       nextResult.trunk = trunk.data();
//       nextResult.movepos = movepos.data();
//       nextResult.pick = pick.data();
//       nextResult.player = plas.data();
//     }
//   }
//   count = resultTip = 0;
//   for(Result& result : results)
//     result.trunk = nullptr; // no trunks when evaluating picks
//   return move(results);
// }

void PrecomputeFeatures::writeResultToMoveset(Result result, SelectedMoves::Moveset& moveset) {
  size_t count = result.rows.size();
  assert(result.startIndex + count - 1 <= moveset.moves.size()); // max one result per move + final board

  for(size_t i = 0; i < count; i++) {
    SelectedMoves::Move& move = moveset.moves.at(result.startIndex+i);
    ResultRow& row = result.rows.at(i);
    assert(move.index == row.index);
    move.pos = row.pos;
    if(!row.trunk.empty())
      move.trunk.reset(new TrunkOutput(row.trunk));
    if(!row.pick.empty())
      move.pick.reset(new PickOutput(row.pick));

    if(i < count-1) { // POC features only work for boards showing the move outcome
      ResultRow& nextRow = result.rows.at(i+1);
      if(nextRow.index != row.index+1) // we cannot determine poc features when missing next move
        continue;
      assert(6 <= numPocFeatures);
      vector<float> poc(numPocFeatures);
      poc[0] = P_WHITE == move.pla ? nextRow.whiteWinProb : nextRow.whiteLossProb; // post-move winProb
      poc[1] = P_WHITE == move.pla ? nextRow.whiteLead : -nextRow.whiteLead; // lead
      poc[2] = row.movePolicy; // movePolicy
      poc[3] = row.maxPolicy; // maxPolicy
      poc[4] = P_WHITE == move.pla // winrateLoss
                       ? row.whiteWinProb - nextRow.whiteWinProb
                       : row.whiteLossProb - nextRow.whiteLossProb;
      poc[5] = P_WHITE == move.pla // pointsLoss
                       ? row.whiteLead - nextRow.whiteLead
                       : -(row.whiteLead - nextRow.whiteLead);
      move.poc = std::make_shared<vector<float>>(poc);
    }
  }
}

// void PrecomputeFeatures::writeInputsToNpz(const string& filePath) {
//   int rows = count;
//   auto binaryInputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numSpatialFeatures, nnXLen, nnYLen});
//   auto globalInputNC = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numGlobalFeatures});
//   std::copy_n(NeuralNet::getSpatialBuffer(inputBuffers.get()), rows*nnXLen*nnYLen*numSpatialFeatures, binaryInputNCHW->data);
//   std::copy_n(NeuralNet::getGlobalBuffer(inputBuffers.get()), rows*numGlobalFeatures, globalInputNC->data);
//   auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows});
//   std::copy_n(movepos.begin(), rows, moveposN->data);

//   ZipFile zipFile(filePath);
//   uint64_t numBytes = binaryInputNCHW->prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW->dataIncludingHeader, numBytes);
//   numBytes = moveposN->prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
//   numBytes = globalInputNC->prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("globalInputNC", globalInputNC->dataIncludingHeader, numBytes);
//   zipFile.close();
// }

// void PrecomputeFeatures::writeOutputsToNpz(const string& filePath) {
//   int rows = count;
//   auto trunkOutputNCHW = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows, numTrunkFeatures, nnXLen, nnYLen});
//   std::copy_n(trunk.begin(), rows*trunkSize, trunkOutputNCHW->data);
//   auto moveposN = std::make_unique<NumpyBuffer<float>>(vector<int64_t>{rows});
//   std::copy_n(movepos.begin(), rows, moveposN->data);

//   ZipFile zipFile(filePath);
//   uint64_t numBytes = trunkOutputNCHW->prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("trunkOutputNCHW", trunkOutputNCHW->dataIncludingHeader, numBytes);
//   numBytes = moveposN->prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("movepos", moveposN->dataIncludingHeader, numBytes);
//   zipFile.close();
// }

// void PrecomputeFeatures::writePicksToNpz(const string& filePath) {
//   int rows = count;
//   NumpyBuffer<float> pickNC({rows, numTrunkFeatures});

//   for(int i = 0; i < rows; i++) {
//     int pos = movepos[i];
//     if(pos >= 0 && pos < nnXLen * nnYLen) {
//       for(int j = 0; j < numTrunkFeatures; j++) {
//         pickNC.data[i*numTrunkFeatures + j] = trunk[i*trunkSize + j*nnXLen*nnYLen + pos];
//       }
//     }
//     else {
//       std::fill(pickNC.data + i*numTrunkFeatures, pickNC.data + (i+1)*numTrunkFeatures, 0);
//     }
//   }

//   ZipFile zipFile(filePath);
//   uint64_t numBytes = pickNC.prepareHeaderWithNumRows(rows);
//   zipFile.writeBuffer("pickNC", pickNC.dataIncludingHeader, numBytes);
//   zipFile.close();
// }

// void PrecomputeFeatures::allocateBuffers() {
//   trunk.resize(capacity * trunkSize);
//   movepos.resize(capacity);
//   pick.resize(capacity * numTrunkFeatures);
//   plas.resize(capacity);
//   count = resultTip = 0;
// }

void PrecomputeFeatures::addBoardImpl(Board& board, const BoardHistory& history, int rowPos, Player pla, bool maybeTrunkPick) {
  assert(evaluator);

  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = 0;
  nnInputParams.policyOptimism = 0;

  NNResultBuf buf;
  buf.rowPos = rowPos;
  if(maybeTrunkPick) {
    buf.includeTrunk = includeTrunk;
    buf.includePick = includePick;
  }
  evaluator->evaluate(board, history, pla, nnInputParams, buf, false, false);
  assert(buf.hasResult);
  NNOutput& nnout = *buf.result;

  // interpret NN result
  ResultRow row;
  row.index = history.getCurrentTurnNumber();
  row.pos = rowPos;
  row.whiteWinProb = nnout.whiteWinProb;
  row.whiteLossProb = nnout.whiteLossProb;
  row.expectedScore = nnout.whiteScoreMean;
  row.whiteLead = nnout.whiteLead;
  row.movePolicy = nnout.policyProbs[buf.rowPos];
  row.maxPolicy = *std::max_element(std::begin(nnout.policyProbs), std::end(nnout.policyProbs));
  if(nnout.trunk) 
    row.trunk = vector<float>(nnout.trunk, nnout.trunk + numTrunkFeatures * nnXLen * nnYLen); // trunk features at move location
  if(nnout.pick)
    row.pick = vector<float>(nnout.pick, nnout.pick + numTrunkFeatures); // trunk features at move location
  nextResult.rows.push_back(row);

  count++;
}

// void ComputeHandleDeleter::operator()(ComputeHandle* handle) noexcept {
//   return NeuralNet::freeComputeHandle(handle);
// }

// void InputBuffersDeleter::operator()(InputBuffers* buffers) noexcept {
//   return NeuralNet::freeInputBuffers(buffers);
// }

// namespace {

// unique_ptr<ComputeHandle, ComputeHandleDeleter> createComputeHandle(LoadedModel& loadedModel, int capacity) {
//   enabled_t useFP16Mode = enabled_t::False;
//   enabled_t useNHWCMode = enabled_t::False;
//   auto* computeContext = NeuralNet::createComputeContext(
//     {PrecomputeFeatures::gpuIdx}, nullptr, PrecomputeFeatures::nnXLen, PrecomputeFeatures::nnYLen,
//     "", "", false,
//     useFP16Mode, useNHWCMode, &loadedModel
//   );

//   bool requireExactNNLen = true;
//   bool inputsUseNHWC = false;
//   unique_ptr<ComputeHandle, ComputeHandleDeleter> gpuHandle {
//     NeuralNet::createComputeHandle(
//       computeContext,
//       &loadedModel,
//       nullptr,
//       capacity,
//       requireExactNNLen,
//       inputsUseNHWC,
//       PrecomputeFeatures::gpuIdx,
//       0
//     ),
//     ComputeHandleDeleter()
//   };

//   return gpuHandle;
// }

// }
