#include "strengthnet.h"
#include "core/global.h"
#include "core/rand.h"
#include "core/using.h"
#include "cudaerrorcheck.h"
#include <cstdio>
#include <iomanip>

Tensor::Tensor(uint xdim, uint ydim)
: data(nullptr), zoffset(nullptr), dims{xdim, ydim, 1}, viewDims(dims), transposed(false), isOwner(true) {
  size_t n = dims.x * dims.y;
  CUDA_ERR("Tensor(xdim, ydim)", cudaMalloc(&data, n * sizeof(float)));
  CUDA_ERR("Tensor(xdim, ydim)", cudaMalloc(&zoffset, 2 * sizeof(uint)));
  uint zs[] = {0, xdim};
  CUDA_ERR("Tensor(xdim, ydim)", cudaMemcpy(zoffset, zs, 2 * sizeof(uint), cudaMemcpyHostToDevice));
}

Tensor::Tensor(uint xdim, uint ydim, const vector<uint>& zs)
: data(nullptr), zoffset(nullptr), dims{xdim, ydim, static_cast<uint>(zs.size()-1)}, viewDims(dims),
transposed(false), isOwner(true) {
  assert(zs.size() >= 2);
  assert(0 == zs.front());
  assert(xdim == zs.back());
  CUDA_ERR("Tensor(xdim, ydim, zs)", cudaMalloc(&data, xdim * ydim * sizeof(float)));
  CUDA_ERR("Tensor(xdim, ydim, zs)", cudaMalloc(&zoffset, zs.size() * sizeof(uint)));
  CUDA_ERR("Tensor(xdim, ydim, zs)", cudaMemcpy(zoffset, zs.data(), zs.size() * sizeof(uint), cudaMemcpyHostToDevice));
}

Tensor::Tensor(const Tensor& rhs)
: data(rhs.data), zoffset(rhs.zoffset), dims(rhs.dims), viewDims(rhs.viewDims),
transposed(rhs.transposed), isOwner(false)
{}

Tensor::Tensor(Tensor&& rhs) noexcept
: data(rhs.data), zoffset(rhs.zoffset), dims(rhs.dims), viewDims(rhs.viewDims),
transposed(rhs.transposed), isOwner(true) {
  rhs.data = nullptr;
  rhs.zoffset = nullptr;
}

Tensor::~Tensor() noexcept {
  if(isOwner) {
    cudaFree(data);
    cudaFree(zoffset);
  }
  data = nullptr;
  zoffset = nullptr;
}

Tensor::operator vector<float>() const {
  size_t n = dims.x * dims.y;
  std::vector<float> result(n);
  CUDA_ERR("Tensor::operator std::vector<float>()", cudaMemcpy(result.data(), data, n * sizeof(float), cudaMemcpyDeviceToHost));
  return result;
}

void Tensor::randomInit(Rand& rand) {
  assert(1 == dims.z); // weights and biases cannot use batch features
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
  Tensor copy;
  CUDA_ERR("Tensor::clone()", cudaMalloc(&copy.data, dims.x * dims.y * sizeof(float)));
  CUDA_ERR("Tensor::clone()", cudaMemcpy(copy.data, data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_ERR("Tensor::clone()", cudaMalloc(&copy.zoffset, (dims.z + 1) * sizeof(uint)));
  CUDA_ERR("Tensor::clone()", cudaMemcpy(copy.zoffset, zoffset, (dims.z + 1) * sizeof(uint), cudaMemcpyDeviceToDevice));
  copy.dims = dims;
  copy.viewDims = viewDims;
  copy.transposed = transposed;
  copy.isOwner = true;
  return copy;
}

void Tensor::assignFrom(const Tensor& rhs) {
  assert(dims.x == rhs.dims.x);
  assert(dims.y == rhs.dims.y);
  assert(dims.z == rhs.dims.z);
  CUDA_ERR("Tensor::assignFrom()", cudaMemcpy(data, rhs.data, dims.x * dims.y * sizeof(float), cudaMemcpyDeviceToDevice));
  CUDA_ERR("Tensor::assignFrom()", cudaMemcpy(zoffset, rhs.zoffset, (dims.z + 1) * sizeof(uint), cudaMemcpyDeviceToDevice));
}

// void Tensor::reshape(uint xdim, uint ydim, uint batchSize) {
//   assert(xdim * ydim * batchSize == dims.x * dims.y * dims.z);
//   dims.x = xdim;
//   dims.y = ydim;
//   dims.z = batchSize;
// }

void Tensor::broadcast(uint xdim, uint ydim, uint batchSize) {
  assert(dims.z == dims.x || xdim == dims.x);
  assert(1 == dims.y || ydim == dims.y);
  assert(1 == dims.z || batchSize == dims.z);
  viewDims = {xdim, ydim, batchSize};
}

void Tensor::transpose() {
  std::swap(dims.x, dims.y);
  std::swap(viewDims.x, viewDims.y);
  transposed = !transposed;
}

void Tensor::cat() {
  viewDims.z = 1;
}

void Tensor::uncat() {
  viewDims.z = dims.z;
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
: N(0), zoffset{0, 0},
x(nullptr), h(nullptr), r(nullptr), a(nullptr), ra(nullptr), y(nullptr),
h_grad(nullptr), hr_grad(nullptr), hz_grad(nullptr),
r_grad(nullptr), z_grad(nullptr), a_grad(nullptr), y_grad(nullptr), tgt(nullptr),
W1(in_ch, hidden_ch), b1(1, hidden_ch), W2r(hidden_ch, 1), b2r(1, 1), W2z(hidden_ch, 1), b2z(1, 1),
W1_grad(nullptr), b1_grad(nullptr), W2r_grad(nullptr), b2r_grad(nullptr), W2z_grad(nullptr), b2z_grad(nullptr)
{
}

StrengthNet::~StrengthNet()
{
  freeTensors();
}

void StrengthNet::randomInit(Rand& rand) {
  W1.randomInit(rand);
  b1.randomInit(rand);
  W2r.randomInit(rand);
  b2r.randomInit(rand);
  W2z.randomInit(rand);
  b2z.randomInit(rand);
}

const uint32_t StrengthNet::STRNET_HEADER = 0x57237;

namespace {
void tensorFromFile(Tensor& tensor, FILE* file) {
  size_t bufferSize = tensor.dims.x * tensor.dims.y;
  vector<float> buffer(bufferSize);
  size_t readSize = std::fread(buffer.data(), sizeof(float), bufferSize, file);
  if(bufferSize != readSize)
    throw IOError("Failed to read strength model from file.");
  CUDA_ERR("tensorFromFile", cudaMemcpy(tensor.data, buffer.data(), bufferSize * sizeof(float), cudaMemcpyHostToDevice));
}
void tensorToFile(Tensor& tensor, FILE* file) {
  size_t bufferSize = tensor.dims.x * tensor.dims.y;
  vector<float> buffer(bufferSize);
  CUDA_ERR("tensorToFile", cudaMemcpy(buffer.data(), tensor.data, bufferSize * sizeof(float), cudaMemcpyDeviceToHost));
  size_t wroteSize = std::fwrite(buffer.data(), sizeof(float), bufferSize, file);
  if(bufferSize != wroteSize)
    throw IOError("Failed to save strength model to file.");
}
}

void StrengthNet::loadModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "rb"), &std::fclose);
  if(nullptr == file)
    throw IOError(path + " is not a strength model file.");
  uint32_t header; // must match
  size_t readheader = std::fread(&header, 4, 1, file.get());
  if(1 != readheader || STRNET_HEADER != header)
    throw IOError(path + " is not a strength model file.");
  tensorFromFile(W1, file.get());
  tensorFromFile(b1, file.get());
  tensorFromFile(W2r, file.get());
  tensorFromFile(b2r, file.get());
  tensorFromFile(W2r, file.get());
  tensorFromFile(b2z, file.get());
  std::fclose(file.release());
}

void StrengthNet::saveModelFile(const std::string& path) {
  auto file = std::unique_ptr<std::FILE, decltype(&std::fclose)>(std::fopen(path.c_str(), "wb"), &std::fclose);
  if(nullptr == file)
    throw IOError("Could not save strength model to " + path);
	size_t wroteheader = std::fwrite(&STRNET_HEADER, 4, 1, file.get());
  if(1 != wroteheader)
    throw IOError("Failed to save complete strength model to " + path);
  tensorToFile(W1, file.get());
  tensorToFile(b1, file.get());
  tensorToFile(W2r, file.get());
  tensorToFile(b2r, file.get());
  tensorToFile(W2z, file.get());
  tensorToFile(b2z, file.get());
  if(0 != std::fclose(file.release()))
    throw IOError("Failed to save complete strength model to " + path);
}

void StrengthNet::setInput(const vector<vector<MoveFeatures>>& features) {
  static_assert(6 == in_ch, "this function must be adapted if in_ch changes");
  assert(features.size() > 0); // must have at least one input
  freeTensors();

  N = 0;
  zoffset.resize(features.size() + 1);
  zoffset[0] = 0;
  for(size_t i = 0; i < features.size(); i++) {
    N += features[i].size();
    zoffset[i+1] = N;
  }
  allocateTensors();

  vector<float> rawfeatures(in_ch*N);
  for(size_t j = 0; j < features.size(); j++) {
  for(size_t i = 0; i < features[j].size(); i++) {
    uint zoff = zoffset[j];
    rawfeatures[in_ch*(i+zoff)+0] = features[j][i].winProb - .5f;
    rawfeatures[in_ch*(i+zoff)+1] = features[j][i].lead * .1f;
    rawfeatures[in_ch*(i+zoff)+2] = features[j][i].movePolicy - .5f;
    rawfeatures[in_ch*(i+zoff)+3] = features[j][i].maxPolicy - .5f;
    rawfeatures[in_ch*(i+zoff)+4] = features[j][i].winrateLoss;
    rawfeatures[in_ch*(i+zoff)+5] = features[j][i].pointsLoss * .1f;
  }
  }

  CUDA_ERR("StrengthNet::setInput", cudaMemcpy(x->data, rawfeatures.data(), in_ch * N * sizeof(float), cudaMemcpyHostToDevice));
}

vector<float> StrengthNet::getOutput() const {
  assert(zoffset.size() > 0);
  uint batchSize = zoffset.size()-1;
  vector<float> output(batchSize);
  CUDA_ERR("StrengthNet::getOutput", cudaMemcpy(output.data(), y->data, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for(float& o : output)
    o = o * 500.f + 1500.f;

  return output;
}

void StrengthNet::setTarget(const Output& targets) {
  assert(tgt); // tensors must be allocated by previous setInput()
  assert(tgt->dims.z == targets.size());

  vector<float> scaledTargets = targets;
  for(float& f : scaledTargets)
    f = (f - 1500.f) / 500.f;
  cudaMemcpy(tgt->data, scaledTargets.data(), scaledTargets.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void StrengthNet::printWeights(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* W1 *\n";   W1.print(stream, name, humanReadable);
  stream << "* b1 *\n";   b1.print(stream, name, humanReadable);
  stream << "* W2r *\n";  W2r.print(stream, name, humanReadable);
  stream << "* b2r *\n";  b2r.print(stream, name, humanReadable);
  stream << "* W2z *\n";  W2z.print(stream, name, humanReadable);
  stream << "* b2z *\n";  b2z.print(stream, name, humanReadable);
}

void StrengthNet::printState(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* x *\n";  x->print(stream, name, humanReadable);
  stream << "* h *\n";  h->print(stream, name, humanReadable);
  stream << "* r *\n";  r->print(stream, name, humanReadable);
  stream << "* a *\n";  a->print(stream, name, humanReadable);
  stream << "* ra *\n"; ra->print(stream, name, humanReadable);
  stream << "* y *\n";  y->print(stream, name, humanReadable);
}

void StrengthNet::printGrads(std::ostream& stream, const std::string& name, bool humanReadable) const {
  stream << "* h_grad *\n";    h_grad->print(stream, name, humanReadable);
  stream << "* hr_grad *\n";   hr_grad->print(stream, name, humanReadable);
  stream << "* hz_grad *\n";   hz_grad->print(stream, name, humanReadable);
  stream << "* r_grad *\n";    r_grad->print(stream, name, humanReadable);
  stream << "* z_grad *\n";    z_grad->print(stream, name, humanReadable);
  stream << "* a_grad *\n";    a_grad->print(stream, name, humanReadable);
  stream << "* y_grad *\n";    y_grad->print(stream, name, humanReadable);
  stream << "* W1_grad *\n";   W1_grad->print(stream, name, humanReadable);
  stream << "* b1_grad *\n";   b1_grad->print(stream, name, humanReadable);
  stream << "* W2r_grad *\n";  W2r_grad->print(stream, name, humanReadable);
  stream << "* b2r_grad *\n";  b2r_grad->print(stream, name, humanReadable);
  stream << "* W2z_grad *\n";  W2z_grad->print(stream, name, humanReadable);
  stream << "* b2z_grad *\n";  b2z_grad->print(stream, name, humanReadable);
}

float StrengthNet::thetaVar() const {
  return Tensor::variance({W1, b1, W2r, b2r, W2z, b2z});
}

float StrengthNet::gradsVar() const {
  return Tensor::variance({*W1_grad, *b1_grad, *W2r_grad, *b2r_grad, *W2z_grad, *b2z_grad});
}

namespace {
  // generate zoffsets for a batch where every input has exactly `step` elements
  vector<uint> iota(size_t batchSize, uint step = 1) {
    vector<uint> buffer(batchSize+1);
    for(size_t i = 0; i < batchSize+1; i++)
      buffer[i] = step * i;
    return buffer;
  }
}

void StrengthNet::allocateTensors() {
  uint batchSize = zoffset.size() - 1;
  x = new Tensor(N, in_ch, zoffset);
  h = new Tensor(N, hidden_ch, zoffset);
  r = new Tensor(N, 1, zoffset);
  a = new Tensor(N, 1, zoffset);
  ra = new Tensor(N, 1, zoffset);
  vector<uint> yoffsets = iota(batchSize);
  y = new Tensor(batchSize, 1, yoffsets);

  h_grad = new Tensor(N, hidden_ch, zoffset);
  hr_grad = new Tensor(N, hidden_ch, zoffset);
  hz_grad = new Tensor(N, hidden_ch, zoffset);
  r_grad = new Tensor(N, 1, zoffset);
  z_grad = new Tensor(N, 1, zoffset);
  a_grad = new Tensor(N, 1, zoffset);
  y_grad = new Tensor(batchSize, 1, yoffsets);
  tgt = new Tensor(batchSize, 1, yoffsets);

  W1_grad = new Tensor(in_ch, hidden_ch);
  b1_grad = new Tensor(1, hidden_ch);
  W2r_grad = new Tensor(hidden_ch, 1);
  b2r_grad = new Tensor(1, 1);
  W2z_grad = new Tensor(hidden_ch, 1);
  b2z_grad = new Tensor(1, 1);
}

void StrengthNet::freeTensors() noexcept {
  delete x; x = nullptr;
  delete h; h = nullptr;
  delete r; r = nullptr;
  delete a; a = nullptr;
  delete ra; ra = nullptr;
  delete y; y = nullptr;

  delete h_grad; h_grad = nullptr;
  delete hr_grad; hr_grad = nullptr;
  delete hz_grad; hz_grad = nullptr;
  delete r_grad; r_grad = nullptr;
  delete z_grad; z_grad = nullptr;
  delete a_grad; a_grad = nullptr;
  delete y_grad; y_grad = nullptr;
  delete tgt; tgt = nullptr;

  delete W1_grad; W1_grad = nullptr;
  delete b1_grad; b1_grad = nullptr;
  delete W2r_grad; W2r_grad = nullptr;
  delete b2r_grad; b2r_grad = nullptr;
  delete W2z_grad; W2z_grad = nullptr;
  delete b2z_grad; b2z_grad = nullptr;
}
