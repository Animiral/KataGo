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
: data(rhs.data), dims(rhs.dims), viewDims(rhs.viewDims), transposed(false), isOwner(false)
{}

Tensor::Tensor(Tensor&& rhs) noexcept
: data(rhs.data), dims(rhs.dims), viewDims(rhs.viewDims), transposed(false), isOwner(true) {
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

float Tensor::variance() const {
  size_t n = dims.x * dims.y * dims.z;
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
  size_t readW = std::fread(buffer.data(), sizeof(float), W.dims.x*W.dims.y, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W.data, buffer.data(), W.dims.x*W.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  size_t readb = std::fread(buffer.data(), sizeof(float), b.dims.x*b.dims.y, file.get());
  CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(b.data, buffer.data(), b.dims.x*b.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  // size_t readW1 = std::fread(buffer.data(), sizeof(float), W1.dims.x*W1.dims.y, file.get());
  // CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W1.data, buffer.data(), W1.dims.x*W1.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  // size_t readW2r = std::fread(buffer.data(), sizeof(float), W2r.dims.x*W2r.dims.y, file.get());
  // CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W2r.data, buffer.data(), W2r.dims.x*W2r.dims.y * sizeof(float), cudaMemcpyHostToDevice));
  // size_t readW2z = std::fread(buffer.data(), sizeof(float), W2z.dims.x*W2z.dims.y, file.get());
  // CUDA_ERR("StrengthNet.loadModelFile", cudaMemcpy(W2z.data, buffer.data(), W2z.dims.x*W2z.dims.y * sizeof(float), cudaMemcpyHostToDevice));

  // if(W1.dims.x*W1.dims.y + W2r.dims.x*W2r.dims.y + W2z.dims.x*W2z.dims.y != readW1 + readW2r + readW2z)
  if(W.dims.x*W.dims.y + b.dims.x*b.dims.y != readW + readb)
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

  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W.data, W.dims.x*W.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteW = std::fwrite(buffer.data(), sizeof(float), W.dims.x*W.dims.y, file.get());
  CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), b.data, b.dims.x*b.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteb = std::fwrite(buffer.data(), sizeof(float), b.dims.x*b.dims.y, file.get());

  // CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W1.data, W1.dims.x*W1.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
	// size_t wroteW1 = std::fwrite(buffer.data(), sizeof(float), W1.dims.x*W1.dims.y, file.get());
  // CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W2r.data, W2r.dims.x*W2r.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  // size_t wroteW2r = std::fwrite(buffer.data(), sizeof(float), W2r.dims.x*W2r.dims.y, file.get());
  // CUDA_ERR("StrengthNet.saveModelFile", cudaMemcpy(buffer.data(), W2z.data, W2z.dims.x*W2z.dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  // size_t wroteW2z = std::fwrite(buffer.data(), sizeof(float), W2z.dims.x*W2z.dims.y, file.get());

  // if(1 + W1.dims.x*W1.dims.y + W2r.dims.x*W2r.dims.y + W2z.dims.x*W2z.dims.y != wroteheader + wroteW1 + wroteW2r + wroteW2z)
  if(1 + W.dims.x*W.dims.y + b.dims.x*b.dims.y != wroteheader + wroteW + wroteb)
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
  x.dims.x = h.dims.x = /*r.dims.x = a.dims.x =*/ N;
  h_grad.dims.x = /*hr_grad.dims.x = hz_grad.dims.x = r_grad.dims.x = z_grad.dims.x =*/ N;
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
