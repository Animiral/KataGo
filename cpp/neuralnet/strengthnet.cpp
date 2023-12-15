#include "strengthnet.h"
#include "core/global.h"
#include "core/rand.h"
#include "core/using.h"
#include "cudaerrorcheck.h"
#include <cstdio>
#include <iomanip>

Tensor::Tensor(uint xdim, uint ydim, uint zdim)
: data(nullptr), dims{xdim, ydim, zdim}, viewDims(dims), transposed(false), isOwner(true) {
  size_t n = dims.x * dims.y * dims.z;
  CUDA_ERR("Tensor(dims)", cudaMalloc(&data, n * sizeof(float)));
}

Tensor::Tensor(const Tensor& rhs)
: data(rhs.data), dims(rhs.dims), viewDims(rhs.viewDims), transposed(rhs.transposed), isOwner(false)
{}

Tensor::Tensor(Tensor&& rhs) noexcept
: data(rhs.data), dims(rhs.dims), viewDims(rhs.viewDims), transposed(rhs.transposed), isOwner(true) {
  rhs.data = nullptr;
}

Tensor::~Tensor() noexcept {
  if(isOwner)
    cudaFree(data);
  data = nullptr;
}

Tensor::operator std::vector<float>() const {
  size_t n = dims.x * dims.y * dims.z;
  std::vector<float> result(n);
  CUDA_ERR("Tensor::operator std::vector<float>()", cudaMemcpy(result.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

void Tensor::randomInit(Rand& rand) {
  size_t n = dims.x;
  size_t m = dims.y;
  size_t o = dims.z;

  // init all parameters uniformly in (-n^(-2), +n^(-2))
  vector<float> buffer(n*m*o);
  double d = 1. / std::sqrt(n);
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++)
      for(size_t k = 0; k < o; k++)
        buffer[i * n * o + j * o + k] = static_cast<float>(rand.nextDouble(-d, d));
  CUDA_ERR("Tensor.randomInit", cudaMemcpy(data, buffer.data(), n * m * o * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor Tensor::clone() const {
  Tensor copy(dims.x, dims.y, dims.z);
  copy.viewDims = viewDims;
  size_t n = dims.x * dims.y * dims.z;
  CUDA_ERR("Tensor::clone()", cudaMemcpy(copy.data, data, n * sizeof(float), cudaMemcpyDeviceToDevice));
  return copy;
}

void Tensor::assignFrom(const Tensor& rhs) {
  assert(dims.x == rhs.dims.x);
  assert(dims.y == rhs.dims.y);
  assert(dims.z == rhs.dims.z);
  CUDA_ERR("Tensor::assignFrom()", cudaMemcpy(data, rhs.data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Tensor::reshape(uint xdim, uint ydim, uint zdim) {
  assert(xdim * ydim * zdim == dims.x * dims.y * dims.z);
  dims.x = xdim;
  dims.y = ydim;
  dims.z = zdim;
}

void Tensor::broadcast(uint xdim, uint ydim, uint zdim) {
  assert(1 == dims.x || xdim == dims.x);
  assert(1 == dims.y || ydim == dims.y);
  assert(1 == dims.z || zdim == dims.z);
  viewDims = {xdim, ydim, zdim};
}

void Tensor::transpose() {
  std::swap(dims.x, dims.y);
  std::swap(viewDims.x, viewDims.y);
  transposed = !transposed;
}

void Tensor::print(std::ostream& stream, const std::string& name) const {
  vector<float> hostdata(dims.x * dims.y);
  CUDA_ERR("Tensor::print", cudaMemcpy(hostdata.data(), data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  stream << "=== DUMP " << name << "\n";
  for(int c = 0; c < dims.y; c++) {
    for(int i = 0; i < dims.x; i++) {
      stream << std::setw(6) << std::fixed << std::setprecision(4) << hostdata[i * dims.y + c] << " ";
    }
    stream << "\n";
  }
  stream << "===\n";
}

float Tensor::mean(std::initializer_list<Tensor> ts) {
  float sum = 0;
  size_t N = 0;
  for(const Tensor& t : ts) {
    auto data = static_cast<vector<float>>(t);
    for(float f : data)
      sum += f;
    N += data.size();
  }
  return sum / N;
}

float Tensor::variance(std::initializer_list<Tensor> ts) {
  float mu = mean(ts);
  float sumSq = 0;
  size_t N = 0;
  for(const Tensor& t : ts) {
    auto data = static_cast<vector<float>>(t);
    for(float f : data)
      sumSq += (f-mu) * (f-mu);
    N += data.size();
  }
  return sumSq / (N - 1);
}

StrengthNet::StrengthNet()
: batchSize(maxBatchSize),
  x(maxN, in_ch), h(maxN, hidden_ch), /* r(maxN, 1), a(maxN, 1),*/ y(1, 1),
  h_grad(maxN, hidden_ch), // hr_grad(maxN, hidden_ch), hz_grad(maxN, hidden_ch),
  /* r_grad(maxN, 1), z_grad(maxN, 1), */ y_grad(1, 1), tgt(1, 1),
  W(in_ch, hidden_ch), b(1, hidden_ch), // W1(in_ch, hidden_ch), W2r(hidden_ch+1, 1), W2z(hidden_ch+1, 1),
  W_grad(in_ch, hidden_ch, maxBatchSize), b_grad(1, hidden_ch, maxBatchSize) // W1_grad(in_ch, hidden_ch, maxBatchSize), W2r_grad(hidden_ch+1, 1, maxBatchSize), W2z_grad(hidden_ch+1, 1, maxBatchSize)
{
}

void StrengthNet::randomInit(Rand& rand) {
  W.randomInit(rand);
  b.randomInit(rand);
}

const uint32_t StrengthNet::STRNET_HEADER = 0x57237;

namespace {
void tensorFromFile(Tensor& tensor, FILE* file) {
  size_t bufferSize = tensor.dims.x * tensor.dims.y * tensor.dims.z;
  vector<float> buffer(bufferSize);
  size_t readSize = std::fread(buffer.data(), sizeof(float), bufferSize, file);
  if(bufferSize != readSize)
    throw IOError("Failed to read strength model from file.");
  CUDA_ERR("tensorFromFile", cudaMemcpy(tensor.data, buffer.data(), bufferSize * sizeof(float), cudaMemcpyHostToDevice));
}
void tensorToFile(Tensor& tensor, FILE* file) {
  size_t bufferSize = tensor.dims.x * tensor.dims.y * tensor.dims.z;
  vector<float> buffer(bufferSize);
  CUDA_ERR("tensorToFile", cudaMemcpy(buffer.data(), tensor.data, bufferSize * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteSize = std::fwrite(buffer.data(), sizeof(float), bufferSize, file);
  if(bufferSize != wroteSize)
    throw IOError("Failed to save strength model to file.");
}
}

bool StrengthNet::loadModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "rb"), &std::fclose);
  if(nullptr == file)
    return false;
  uint32_t header; // must match
  size_t readheader = std::fread(&header, 4, 1, file.get());
  if(1 != readheader || STRNET_HEADER != header)
    throw IOError(path + " is not a strength model file.");
  tensorFromFile(W, file.get());
  tensorFromFile(b, file.get());
  // tensorFromFile(W1, file.get());
  // tensorFromFile(W2r, file.get());
  // tensorFromFile(W2z, file.get());
  std::fclose(file.release());
  return true;
}

void StrengthNet::saveModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "wb"), &std::fclose);
  if(nullptr == file)
    throw IOError("Could not save strength model to " + path);
	size_t wroteheader = std::fwrite(&STRNET_HEADER, 4, 1, file.get());
  if(1 != wroteheader)
    throw IOError("Failed to save complete strength model to " + path);
  tensorToFile(W, file.get());
  tensorToFile(b, file.get());
  // tensorToFile(W1, file.get());
  // tensorToFile(W2r, file.get());
  // tensorToFile(W2z, file.get());
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
  x.dims.x = x.viewDims.x = N;
  h.dims.x = h.viewDims.x = N;
  // r.dims.x = r.viewDims.x = N;
  // a.dims.x = a.viewDims.x = N;
  h_grad.dims.x = h_grad.viewDims.x = N;
  // hr_grad.dims.x = hr_grad.viewDims.x = N;
  // hz_grad.dims.x = hz_grad.viewDims.x = N;
  // r_grad.dims.x = r_grad.viewDims.x = N;
  // z_grad.dims.x = z_grad.viewDims.x = N;
}

void StrengthNet::setBatchSize(size_t batchSize_) noexcept {
  assert(batchSize_ <= maxBatchSize);
  batchSize = batchSize_;
  W_grad.dims.z = b_grad.dims.z = batchSize_;  // parameter update gradients
  // W1_grad.dims.z = W2r_grad.dims.z = W2z_grad.dims.z = batchSize_;  // parameter update gradients
}

float StrengthNet::getOutput() const {
  float output;
  CUDA_ERR("StrengthNet::getOutput", cudaMemcpy(&output, y.data, sizeof(float), cudaMemcpyDeviceToHost));
  return output * 500.f + 1500.f;
}

void StrengthNet::printWeights(std::ostream& stream, const std::string& name) const {
  stream << "* W *\n";   W.print(stream, name);
  stream << "* b *\n";   b.print(stream, name);
  // stream << "* W1 *\n";   W1.print(stream, name);
  // stream << "* W2r *\n";  W2r.print(stream, name);
  // stream << "* W2z *\n";  W2z.print(stream, name);
}

void StrengthNet::printState(std::ostream& stream, const std::string& name) const {
  stream << "* x *\n";  x.print(stream, name);
  stream << "* h *\n";  h.print(stream, name);
  // stream << "* r *\n";  r.print(stream, name);
  // stream << "* a *\n";  a.print(stream, name);
  stream << "* y *\n";  y.print(stream, name);
}

void StrengthNet::printGrads(std::ostream& stream, const std::string& name) const {
  stream << "* h_grad *\n";  h_grad.print(stream, name);
  // stream << "* hr_grad *\n";  hr_grad.print(stream, name);
  // stream << "* hz_grad *\n";  hz_grad.print(stream, name);
  // stream << "* r_grad *\n";  r_grad.print(stream, name);
  // stream << "* z_grad *\n";  z_grad.print(stream, name);
  stream << "* y_grad *\n";  y_grad.print(stream, name);
  stream << "* W_grad *\n";  W_grad.print(stream, name);
  stream << "* b_grad *\n";  b_grad.print(stream, name);
  // stream << "* W1_grad *\n";  W1_grad.print(stream, name);   // only prints first grad (z==0)!
  // stream << "* W2r_grad *\n";  W2r_grad.print(stream, name); // only prints first grad (z==0)!
  // stream << "* W2z_grad *\n";  W2z_grad.print(stream, name); // only prints first grad (z==0)!
}

float StrengthNet::thetaVar() const {
  return Tensor::variance({W, b});
}

float StrengthNet::gradsVar() const {
  return Tensor::variance({W_grad, b_grad});
}
