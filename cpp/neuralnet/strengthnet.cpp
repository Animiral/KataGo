#include "strengthnet.h"
#include "core/global.h"
#include "core/rand.h"
#include "core/using.h"
#include "cudaerrorcheck.h"
#include <cstdio>
#include <iomanip>

// functions to call kernels
void forwardFull(float* inx, float* outx, float* weights, float* biases, int N, int in_ch, int out_ch);
void forwardRelu(float* inx, int N, int ch);
void forwardAggregate(float* outputx, float* softx, float* aggregx, int N, int ch);
void backwardLoss(float* target, float* aggregx, float* outgrads);
void backwardAggregate(float* ingrads, float* outputx, int N, float* outgrads);
void backwardTanh(float* ingrads, float* outputx, int N, float* outgrads);
void backwardFull(float* ingrads, float* inx, float* weights, float* biases, int in_ch, int out_ch, int N, float learnrate, float* outgrads);
void backwardRelu(float* ingrads, float* inx, int ch, int N, float* outgrads);

namespace {
  void dumpDeviceArray(string name, float* data, int N, int ch) {
    vector<float> hostdata(N*ch);
    CUDA_ERR("dumpDeviceArray", cudaMemcpy(hostdata.data(), data, N * ch * sizeof(float), cudaMemcpyDeviceToHost));
    cerr << "=== DUMP " << name << "\n";
    for(int c = 0; c < ch; c++) {
      for(int n = 0; n < N; n++) {
        cerr << std::setw(6) << std::fixed << std::setprecision(2) << hostdata[n * ch + c] << " ";
      }
      cerr << "\n";
    }
    cerr << "===\n";
  }
}

StrengthNet::StrengthNet() {
  CUDA_ERR("StrengthNet", cudaMalloc(&inputx, maxN * in_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&hiddenx, maxN * hidden_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&outputx, maxN * out_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&softx, maxN * out_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&aggregx, sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&back1x, maxN * std::max({in_ch, hidden_ch, out_ch}) * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&back2x, maxN * std::max({in_ch, hidden_ch, out_ch}) * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&Wh, in_ch * hidden_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&bh, hidden_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&Wo, hidden_ch * out_ch * sizeof(float)));
  CUDA_ERR("StrengthNet", cudaMalloc(&bo, out_ch * sizeof(float)));
}

StrengthNet::~StrengthNet() {
  cudaFree(inputx);
  cudaFree(hiddenx);
  cudaFree(outputx);
  cudaFree(softx);
  cudaFree(aggregx);
  cudaFree(back1x);
  cudaFree(back2x);
  cudaFree(Wh);
  cudaFree(bh);
  cudaFree(Wo);
  cudaFree(bo);
}

void StrengthNet::randomInit(Rand& rand) {
  // init all parameters uniformly in (-d^(-2), +d^(-2)), where d is the input dimension
  size_t bufferSize = in_ch*hidden_ch + hidden_ch*out_ch;
  vector<float> buffer(bufferSize);

  // Wh
  double high = 1. / std::sqrt(in_ch);
  double low = -high;
  for(int i = 0; i < hidden_ch; i++)
    for(int j = 0; j < in_ch; j++)
      buffer[i * in_ch + j] = static_cast<float>(rand.nextDouble(low, high));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(Wh, buffer.data(), in_ch*hidden_ch * sizeof(float), cudaMemcpyHostToDevice));

  // bh
  for(int j = 0; j < hidden_ch; j++)
    buffer[j] = static_cast<float>(rand.nextDouble(low, high));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(bh, buffer.data(), hidden_ch * sizeof(float), cudaMemcpyHostToDevice));

  // Wo
  high = 1. / std::sqrt(hidden_ch);
  low = -high;
  for(int i = 0; i < out_ch; i++)
    for(int j = 0; j < hidden_ch; j++)
      buffer[i * hidden_ch + j] = static_cast<float>(rand.nextDouble(low, high));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(Wo, buffer.data(), hidden_ch*out_ch * sizeof(float), cudaMemcpyHostToDevice));

  // bo
  for(int j = 0; j < out_ch; j++)
    buffer[j] = static_cast<float>(rand.nextDouble(low, high));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(bo, buffer.data(), out_ch * sizeof(float), cudaMemcpyHostToDevice));
}

const uint32_t StrengthNet::STRNET_HEADER = 0x57237;

bool StrengthNet::loadModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "rb"), &std::fclose);
  if(nullptr == file)
    return false;
  uint32_t header; // must match
  size_t readheader = std::fread(&header, 4, 1, file.get());
  if(1 != readheader || STRNET_HEADER != header)
    throw IOError(path + " is not a strength model file.");

  size_t bufferSize = in_ch*hidden_ch + hidden_ch*out_ch;
  vector<float> buffer(bufferSize);
  size_t readWh = std::fread(buffer.data(), sizeof(float), in_ch*hidden_ch, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(Wh, buffer.data(), in_ch*hidden_ch * sizeof(float), cudaMemcpyHostToDevice));
  size_t readbh = std::fread(buffer.data(), sizeof(float), hidden_ch, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(bh, buffer.data(), hidden_ch * sizeof(float), cudaMemcpyHostToDevice));
  size_t readWo = std::fread(buffer.data(), sizeof(float), hidden_ch*out_ch, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(Wo, buffer.data(), hidden_ch*out_ch * sizeof(float), cudaMemcpyHostToDevice));
  size_t readbo = std::fread(buffer.data(), sizeof(float), out_ch, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(bo, buffer.data(), out_ch * sizeof(float), cudaMemcpyHostToDevice));
  if(in_ch*hidden_ch + hidden_ch + hidden_ch*out_ch + out_ch != readWh + readbh + readWo + readbo)
    throw IOError("Failed to read complete strength model from " + path);

  std::fclose(file.release());
  return true;
}

void StrengthNet::saveModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "wb"), &std::fclose);
  if(nullptr == file)
    throw IOError("Could not save strength model to " + path);
	size_t wroteheader = std::fwrite(&STRNET_HEADER, 4, 1, file.get());

  size_t bufferSize = in_ch*hidden_ch + hidden_ch*out_ch;
  vector<float> buffer(bufferSize);
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), Wh, in_ch*hidden_ch * sizeof(float), cudaMemcpyDeviceToHost));
	size_t wroteWh = std::fwrite(buffer.data(), sizeof(float), in_ch*hidden_ch, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), bh, hidden_ch * sizeof(float), cudaMemcpyDeviceToHost));
	size_t wrotebh = std::fwrite(buffer.data(), sizeof(float), hidden_ch, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), Wo, hidden_ch*out_ch * sizeof(float), cudaMemcpyDeviceToHost));
	size_t wroteWo = std::fwrite(buffer.data(), sizeof(float), hidden_ch*out_ch, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), bo, out_ch * sizeof(float), cudaMemcpyDeviceToHost));
	size_t wrotebo = std::fwrite(buffer.data(), sizeof(float), out_ch, file.get());
  if(1 + in_ch*hidden_ch + hidden_ch + hidden_ch*out_ch + out_ch != wroteheader + wroteWh + wrotebh + wroteWo + wrotebo)
    throw IOError("Failed to save complete strength model to " + path);

  if(0 != std::fclose(file.release()))
    throw IOError("Failed to save complete strength model to " + path);
}

StrengthNet::Output StrengthNet::forward(const StrengthNet::Input& input) {
  assert(input.size() <= maxN);
	assert(reinterpret_cast<const float*>(&input[0]) + in_ch == reinterpret_cast<const float*>(&input[1])); // crude packing/alignment safety check
  N = input.size();
	const float* inputPtr = reinterpret_cast<const float*>(input.data());
  CUDA_ERR("StrengthNet.forward", cudaMemcpy(inputx, inputPtr, N * in_ch * sizeof(float), cudaMemcpyHostToDevice));

	forwardFull(inputx, hiddenx, Wh, bh, N, in_ch, hidden_ch);
	forwardRelu(hiddenx, N, hidden_ch);
	forwardFull(hiddenx, outputx, Wo, bo, N, hidden_ch, out_ch);
	forwardAggregate(outputx, softx, aggregx, N, out_ch);

	StrengthNet::Output result;
  CUDA_ERR("StrengthNet.forward", cudaMemcpy(&result, aggregx, sizeof(StrengthNet::Output), cudaMemcpyDeviceToHost));
  return result;
}

void StrengthNet::backward(Output target, float learnrate) {
  CUDA_ERR("StrengthNet.backward", cudaMemcpy(back1x, &target, sizeof(StrengthNet::Output), cudaMemcpyHostToDevice));
  // dumpDeviceArray("inputx", inputx, N, in_ch);
  // dumpDeviceArray("hiddenx", hiddenx, N, hidden_ch);
  // dumpDeviceArray("outputx", outputx, N, out_ch);
  // dumpDeviceArray("softx", softx, N, out_ch);
  // dumpDeviceArray("aggregx", aggregx, 1, 1);
  backwardLoss(back1x, aggregx, back2x);
  dumpDeviceArray("dLoss/dAggreg", back2x, 1, 1);

  backwardAggregate(back2x, softx, N, back1x);
  backwardTanh(back1x, outputx, N, back2x);
  dumpDeviceArray("dLoss/dOutput", back2x, N, out_ch);
  backwardFull(back2x, hiddenx, Wo, bo, hidden_ch, out_ch, N, learnrate, back1x);
  dumpDeviceArray("dLoss/dHidden", back1x, N, hidden_ch);
  backwardRelu(back1x, hiddenx, hidden_ch, N, back2x);
  dumpDeviceArray("dLoss/dHidden+ReLu", back2x, N, hidden_ch);
  backwardFull(back2x, inputx, Wh, bh, in_ch, hidden_ch, N, learnrate, back1x);
  // dumpDeviceArray("Wo", Wo, hidden_ch, out_ch);
  // dumpDeviceArray("bo", bo, 1, out_ch);
  // dumpDeviceArray("Wh", Wh, in_ch, hidden_ch);
  // dumpDeviceArray("bh", bh, 1, hidden_ch);
}
