#include "strengthnet.h"
#include "core/global.h"
#include "core/rand.h"
#include "core/using.h"
#include "cudaerrorcheck.h"
#include <cstdio>
#include <iomanip>

Tensor::Tensor(uint xdim, uint ydim)
: data(nullptr), dims{xdim, ydim}, isOwner(true) {
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor(dims)", cudaMalloc(&data, n * sizeof(float)));
}

Tensor::Tensor(const Tensor& rhs)
: data(rhs.data), dims(rhs.dims), isOwner(false)
{}

Tensor::Tensor(Tensor&& rhs) noexcept
: data(rhs.data), dims(rhs.dims), isOwner(true) {
  rhs.data = nullptr;
}

Tensor& Tensor::operator=(const Tensor& rhs) {
  size_t n = dims.x * dims.y;
  assert(rhs.dims.x * dims.y == n);
  if(isOwner)
    cudaFree(data);
  data = rhs.data;
  isOwner = false;
  return *this;
}

Tensor& Tensor::operator=(Tensor&& rhs) noexcept {
  data = rhs.data;
  dims = rhs.dims;
  isOwner = rhs.isOwner;
  rhs.data = nullptr;
  rhs.isOwner = false;
  return *this;
}

Tensor::~Tensor() noexcept {
  if(isOwner)
    cudaFree(data);
  data = nullptr;
}

Tensor::operator std::vector<float>() const {
  size_t n = dims.x * dims.y;
  std::vector<float> result(n);
  CUDA_ERR("Tensor::operator std::vector<float>()", cudaMemcpy(result.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

void Tensor::randomInit(Rand& rand) {
  size_t n = dims.x;
  size_t m = dims.y;

  // init all parameters uniformly in (-n^(-2), +n^(-2))
  vector<float> buffer(n*m);
  double high = 1. / std::sqrt(n);
  double low = -high;
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++)
      buffer[i * n + j] = static_cast<float>(rand.nextDouble(low, high));
  CUDA_ERR("Tensor.randomInit", cudaMemcpy(data, buffer.data(), n * m * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor Tensor::clone() const {
  Tensor copy(dims.x, dims.y);
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor::clone()", cudaMemcpy(copy.data, data, n * sizeof(float), cudaMemcpyDeviceToDevice));
  return copy;
}

void Tensor::assignFrom(const Tensor& rhs) {
  assert(dims.x == rhs.dims.x);
  assert(dims.y == rhs.dims.y);
  CUDA_ERR("Tensor::assignFrom()", cudaMemcpy(data, rhs.data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice));
}

float Tensor::variance() const {
  size_t n = dims.x * dims.y;
  assert(n > 1); // 1 number has no variance
  vector<float> buffer(n);
  CUDA_ERR("Tensor.variance", cudaMemcpy(buffer.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));

  // est. average
  vector<float> mubuffer = buffer;
  float mu = 0;
  for(size_t s = 1; s < n; s*=2)
    for(size_t i = 0; i+s < n; i+=2*s)
      mubuffer[i] += mubuffer[i+s];
  mu = mubuffer[0] / n;

  // est. variance
  for(size_t i = 0; i < n; i++)
    buffer[i] = (buffer[i] - mu) * (buffer[i] - mu);
  for(size_t s = 1; s < n; s*=2)
    for(size_t i = 0; i+s < n; i+=2*s)
      buffer[i] += buffer[i+s];
  return buffer[0] / (n - 1);
}

void Tensor::print(std::ostream& stream, const std::string& name) {
  vector<float> hostdata(dims.x * dims.y);
  CUDA_ERR("Tensor::print", cudaMemcpy(hostdata.data(), data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  stream << "=== DUMP " << name << "\n";
  for(int c = 0; c < dims.y; c++) {
    for(int i = 0; i < dims.x; i++) {
      stream << std::setw(6) << std::fixed << std::setprecision(2) << hostdata[i * dims.y + c] << " ";
    }
    stream << "\n";
  }
  stream << "===\n";
}

StrengthNet::StrengthNet()
: x(maxN, in_ch), h(maxN, hidden_ch), r(maxN, 1), a(maxN, 1), y(1, 1),
  h_grad(maxN, hidden_ch), hr_grad(maxN, hidden_ch), hz_grad(maxN, hidden_ch),
  r_grad(maxN, 1), z_grad(maxN, 1), y_grad(1, 1),
  W1(in_ch+1, hidden_ch), W2r(hidden_ch+1, 1), W2z(hidden_ch+1, 1),
  W1_grad(in_ch, hidden_ch), W2r_grad(hidden_ch, 1), W2z_grad(hidden_ch, 1)
{
}

void StrengthNet::randomInit(Rand& rand) {
  // init all parameters uniformly in (-d^(-2), +d^(-2)), where d is the input dimension
  size_t bufferSize = (in_ch+1)*hidden_ch + (hidden_ch+1)*out_ch;
  vector<float> buffer(bufferSize);

  // W1
  double d = 1. / std::sqrt(in_ch);
  for(int i = 0; i < W1.dims.y; i++)
    for(int j = 0; j < W1.dims.x; j++)
      buffer[i * W1.dims.x + j] = static_cast<float>(rand.nextDouble(-d, d));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(W1.data, buffer.data(), W1.dims.x*W1.dims.y * sizeof(float), cudaMemcpyHostToDevice));

  // W2r
  d = 1. / std::sqrt(hidden_ch);
  for(int i = 0; i < W2r.dims.y; i++)
    for(int j = 0; j < W2r.dims.x; j++)
      buffer[i * W2r.dims.x + j] = static_cast<float>(rand.nextDouble(-d, d));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(W2r.data, buffer.data(), W2r.dims.x*W2r.dims.y * sizeof(float), cudaMemcpyHostToDevice));

  // W2z
  d = 1. / std::sqrt(hidden_ch);
  for(int i = 0; i < W2z.dims.y; i++)
    for(int j = 0; j < W2z.dims.x; j++)
      buffer[i * W2z.dims.x + j] = static_cast<float>(rand.nextDouble(-d, d));
  CUDA_ERR("StrengthNet.randomInit", cudaMemcpy(W2z.data, buffer.data(), W2z.dims.x*W2z.dims.y * sizeof(float), cudaMemcpyHostToDevice));
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

  size_t bufferSize = (in_ch+1)*hidden_ch + (hidden_ch+1)*out_ch;
  vector<float> buffer(bufferSize);
  size_t readW1 = std::fread(buffer.data(), sizeof(float), W1.dims.x*W1.dims.y, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W1.data, buffer.data(), W1.dims.x*W1.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  size_t readW2r = std::fread(buffer.data(), sizeof(float), W2r.dims.x*W2r.dims.y, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W2r.data, buffer.data(), W2r.dims.x*W2r.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  size_t readW2z = std::fread(buffer.data(), sizeof(float), W2z.dims.x*W2z.dims.y, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W2z.data, buffer.data(), W2z.dims.x*W2z.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  if(W1.dims.x*W1.dims.y + W2r.dims.x*W2r.dims.y + W2z.dims.x*W2z.dims.y != readW1 + readW2r + readW2z)
    throw IOError("Failed to read complete strength model from " + path);

  std::fclose(file.release());
  return true;
}

void StrengthNet::saveModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "wb"), &std::fclose);
  if(nullptr == file)
    throw IOError("Could not save strength model to " + path);
	size_t wroteheader = std::fwrite(&STRNET_HEADER, 4, 1, file.get());

  size_t bufferSize = (in_ch+1)*hidden_ch + (hidden_ch+1)*out_ch;
  vector<float> buffer(bufferSize);
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W1.data, W1.dims.x*W1.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
	size_t wroteW1 = std::fwrite(buffer.data(), sizeof(float), W1.dims.x*W1.dims.y, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W2r.data, W2r.dims.x*W2r.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteW2r = std::fwrite(buffer.data(), sizeof(float), W2r.dims.x*W2r.dims.y, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W2z.data, W2z.dims.x*W2z.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteW2z = std::fwrite(buffer.data(), sizeof(float), W2z.dims.x*W2z.dims.y, file.get());
  if(1 + W1.dims.x*W1.dims.y + W2r.dims.x*W2r.dims.y + W2z.dims.x*W2z.dims.y != wroteheader + wroteW1 + wroteW2r + wroteW2z)
    throw IOError("Failed to save complete strength model to " + path);

  if(0 != std::fclose(file.release()))
    throw IOError("Failed to save complete strength model to " + path);
}

void StrengthNet::setInput(const std::vector<MoveFeatures>& features) {
  N = features.size();
  assert(N <= maxN);
  assert(6 == in_ch); // this function must be adapted if in_ch changes

  vector<float> rawfeatures(in_ch*N);
  for(size_t i = 0; i < N; i++) {
    rawfeatures[in_ch*i+0] = features[i].winProb - .5f;
    rawfeatures[in_ch*i+1] = features[i].lead * .1f;
    rawfeatures[in_ch*i+2] = features[i].movePolicy - .5f;
    rawfeatures[in_ch*i+3] = features[i].maxPolicy - .5f;
    rawfeatures[in_ch*i+4] = features[i].winrateLoss;
    rawfeatures[in_ch*i+5] = features[i].pointsLoss * .1f;
  }

  CUDA_ERR("StrengthNet::setInput", cudaMemcpy(x.data, rawfeatures.data(), in_ch * N * sizeof(float), cudaMemcpyHostToDevice));
  x.dims.x = h.dims.x = r.dims.x = a.dims.x = N;
  h_grad.dims.x = hr_grad.dims.x = hz_grad.dims.x = r_grad.dims.x = z_grad.dims.x = N;
}

float StrengthNet::getOutput() const {
  float output;
  CUDA_ERR("StrengthNet::getOutput", cudaMemcpy(&output, y.data, sizeof(float), cudaMemcpyDeviceToHost));
  return output * 500.f + 1500.f;
}
