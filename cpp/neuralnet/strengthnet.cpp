#include "strengthnet.h"
#include "core/global.h"
#include "core/rand.h"
#include "core/using.h"
#include "cudaerrorcheck.h"
#include <cstdio>
#include <iomanip>

Tensor::Tensor(uint xdim, uint ydim)
: data(nullptr), dims{xdim, ydim, 1}, viewDims(dims), transposed(false), isOwner(true) {
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor(xdim, ydim)", cudaMalloc(&data, n * sizeof(float)));
}

Tensor::Tensor(uint xdim, uint ydim, uint batchSize, const uint zs[])
: data(nullptr), dims{xdim, ydim, batchSize}, viewDims(dims), transposed(false), isOwner(true) {
  assert(batchSize <= MAX_BATCHSIZE);
  std::copy(zs, zs + batchSize, &zoffset[0]);
  std::fill(&zoffset[batchSize], &zoffset[MAX_BATCHSIZE], xdim);
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor(xdim, ydim, batchSize, zs)", cudaMalloc(&data, n * sizeof(float)));
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
  double d = 1. / std::sqrt(n);
  for(size_t i = 0; i < m; i++)
    for(size_t j = 0; j < n; j++)
      buffer[i + j * m] = static_cast<float>(rand.nextDouble(-d, d));
  CUDA_ERR("Tensor.randomInit", cudaMemcpy(data, buffer.data(), n * m * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor Tensor::clone() const {
  Tensor copy(dims.x, dims.y, dims.z, zoffset);
  copy.viewDims = viewDims;
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor::clone()", cudaMemcpy(copy.data, data, n * sizeof(float), cudaMemcpyDeviceToDevice));
  return copy;
}

void Tensor::assignFrom(const Tensor& rhs) {
  assert(dims.x == rhs.dims.x);
  assert(dims.y == rhs.dims.y);
  assert(dims.z == rhs.dims.z);
  CUDA_ERR("Tensor::assignFrom()", cudaMemcpy(data, rhs.data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice));
}

// void Tensor::reshape(uint xdim, uint ydim, uint batchSize) {
//   assert(xdim * ydim * batchSize == dims.x * dims.y * dims.z);
//   dims.x = xdim;
//   dims.y = ydim;
//   dims.z = batchSize;
// }

void Tensor::broadcast(uint xdim, uint ydim, uint batchSize) {
  assert(1 == dims.x || xdim == dims.x);
  assert(1 == dims.y || ydim == dims.y);
  assert(1 == dims.z || batchSize == dims.z);
  viewDims = {xdim, ydim, batchSize};
}

void Tensor::transpose() {
  std::swap(dims.x, dims.y);
  std::swap(viewDims.x, viewDims.y);
  transposed = !transposed;
}

void Tensor::print(std::ostream& stream, const std::string& name, bool humanReadable) const {
  vector<float> hostdata(dims.x * dims.y);
  CUDA_ERR("Tensor::print", cudaMemcpy(hostdata.data(), data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToHost));
  if(humanReadable)
    stream << "=== DUMP " << name << "\n";
  for(int c = 0; c < dims.y; c++) {
    for(int i = 0; i < dims.x; i++) {
      if(humanReadable)
        stream << std::setw(6) << std::fixed << std::setprecision(4);
      stream << hostdata[i * dims.y + c] << (humanReadable ? " " : ",");
    }
    stream << "\n";
  }
  if(humanReadable)
    stream << "===";
  stream << "\n";
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
: N(0), batchSize(1), zoffset{0},
x(nullptr), h(nullptr), y(nullptr),
// r(nullptr), // a(nullptr),
h_grad(nullptr), y_grad(nullptr), tgt(nullptr),
// hr_grad(nullptr), // hz_grad(nullptr), // r_grad(nullptr), // z_grad(nullptr),
W(in_ch, hidden_ch), b(1, hidden_ch),
W_grad(nullptr), b_grad(nullptr)
// W1_grad(nullptr), // W2r_grad(nullptr), // W2z_grad(nullptr),
{
}

StrengthNet::~StrengthNet()
{
  freeTensors();
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
  static_assert(6 == in_ch, "this function must be adapted if in_ch changes");
  freeTensors();

  N = features.size();
  batchSize = 1;
  allocateTensors();

  vector<float> rawfeatures(in_ch*N);
  for(size_t i = 0; i < N; i++) {
    rawfeatures[in_ch*i+0] = features[i].winProb - .5f;
    rawfeatures[in_ch*i+1] = features[i].lead * .1f;
    rawfeatures[in_ch*i+2] = features[i].movePolicy - .5f;
    rawfeatures[in_ch*i+3] = features[i].maxPolicy - .5f;
    rawfeatures[in_ch*i+4] = features[i].winrateLoss;
    rawfeatures[in_ch*i+5] = features[i].pointsLoss * .1f;
  }

  CUDA_ERR("StrengthNet::setInput", cudaMemcpy(x->data, rawfeatures.data(), in_ch * N * sizeof(float), cudaMemcpyHostToDevice));
}

float StrengthNet::getOutput() const {
  float output;
  CUDA_ERR("StrengthNet::getOutput", cudaMemcpy(&output, y->data, sizeof(float), cudaMemcpyDeviceToHost));
  return output * 500.f + 1500.f;
}

void StrengthNet::printWeights(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* W *\n";   W.print(stream, name, humanReadable);
  stream << "* b *\n";   b.print(stream, name, humanReadable);
  // stream << "* W1 *\n";   W1->print(stream, name, humanReadable);
  // stream << "* W2r *\n";  W2r->print(stream, name, humanReadable);
  // stream << "* W2z *\n";  W2z->print(stream, name, humanReadable);
}

void StrengthNet::printState(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* x *\n";  x->print(stream, name, humanReadable);
  stream << "* h *\n";  h->print(stream, name, humanReadable);
  // stream << "* r *\n";  r->print(stream, name, humanReadable);
  // stream << "* a *\n";  a->print(stream, name, humanReadable);
  stream << "* y *\n";  y->print(stream, name, humanReadable);
}

void StrengthNet::printGrads(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* h_grad *\n";  h_grad->print(stream, name, humanReadable);
  // stream << "* hr_grad *\n";  hr_grad->print(stream, name, humanReadable);
  // stream << "* hz_grad *\n";  hz_grad->print(stream, name, humanReadable);
  // stream << "* r_grad *\n";  r_grad->print(stream, name, humanReadable);
  // stream << "* z_grad *\n";  z_grad->print(stream, name, humanReadable);
  stream << "* y_grad *\n";  y_grad->print(stream, name, humanReadable);
  stream << "* W_grad *\n";  W_grad->print(stream, name, humanReadable);
  stream << "* b_grad *\n";  b_grad->print(stream, name, humanReadable);
  // stream << "* W1_grad *\n";  W1_grad->print(stream, name, humanReadable);   // only prints first grad (z==0)!
  // stream << "* W2r_grad *\n";  W2r_grad->print(stream, name, humanReadable); // only prints first grad (z==0)!
  // stream << "* W2z_grad *\n";  W2z_grad->print(stream, name, humanReadable); // only prints first grad (z==0)!
}

float StrengthNet::thetaVar() const {
  return Tensor::variance({W, b});
}

float StrengthNet::gradsVar() const {
  return Tensor::variance({*W_grad, *b_grad});
}

namespace {
  // generate zoffsets for a batch where every input has exactly `step` elements
  vector<uint> iota(size_t batchSize, uint step = 1) {
    vector<uint> buffer(Tensor::MAX_BATCHSIZE);
    for(size_t i = 0; i < batchSize; i++)
      buffer[i] = step * (i+1);
    for(size_t i = batchSize; i < buffer.size(); i++)
      buffer[i] = step * batchSize;
    return buffer;
  }
}

void StrengthNet::allocateTensors() {
  x = new Tensor(N, in_ch, batchSize, zoffset);
  h = new Tensor(N, hidden_ch, batchSize, zoffset);
  // r = new Tensor(N, 1, batchSize, zoffset);
  // a = new Tensor(N, 1, batchSize, zoffset);
  vector<uint> yoffsets = iota(batchSize);
  y = new Tensor(batchSize, 1, batchSize, yoffsets.data());

  h_grad = new Tensor(N, hidden_ch, batchSize, zoffset);
  // hr_grad = new Tensor(N, hidden_ch, batchSize, zoffset);
  // hz_grad = new Tensor(N, hidden_ch, batchSize, zoffset);
  // r_grad = new Tensor(N, 1, batchSize, zoffset);
  // z_grad = new Tensor(N, 1, batchSize, zoffset);
  y_grad = new Tensor(batchSize, 1, batchSize, yoffsets.data());
  tgt = new Tensor(batchSize, 1, batchSize, yoffsets.data());

  vector<uint> Woffsets = iota(batchSize, in_ch*hidden_ch);
  W_grad = new Tensor(in_ch*batchSize, hidden_ch, batchSize, Woffsets.data());
  // W1_grad = new Tensor(in_ch*batchSize, hidden_ch, batchSize, Woffsets.data())
  vector<uint> boffsets = iota(batchSize, hidden_ch);
  b_grad = new Tensor(batchSize, hidden_ch, batchSize, Woffsets.data());
  // vector<uint> W2offsets = iota(batchSize, hidden_ch);
  // W2r_grad = new Tensor(hidden_ch*batch_size, 1, batchSize, W2offsets);
  // W2z_grad = new Tensor(hidden_ch*batch_size, 1, batchSize, W2offsets);
}

void StrengthNet::freeTensors() noexcept {
  delete x; x = nullptr;
  delete h; h = nullptr;
  // delete r; r = nullptr;
  // delete a; a = nullptr;
  delete y; y = nullptr;

  delete h_grad; h_grad = nullptr;
  // delete hr_grad; hr_grad = nullptr;
  // delete hz_grad; hz_grad = nullptr;
  // delete r_grad; r_grad = nullptr;
  // delete z_grad; z_grad = nullptr;
  delete y_grad; y_grad = nullptr;
  delete tgt; tgt = nullptr;

  delete W_grad; W_grad = nullptr;
  delete b_grad; b_grad = nullptr;
  // delete W1_grad; W1_grad = nullptr;
  // delete W2r_grad; W2r_grad = nullptr;
  // delete W2z_grad; W2z_grad = nullptr;
}
