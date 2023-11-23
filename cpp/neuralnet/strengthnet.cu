#include "strengthnet.h"

// todo: parallelize
__global__ void maxelem(float* x, float* maxx, int N, int ch, int row) {
  if(N <= 0)
    return;
  float m = x[row];
  for(int i = 1; i < N; i++) {
    if(x[i] > m)
      m = x[i*ch+row];
  }
  *maxx = m;
}

// todo: parallelize
__global__ void sumelem(float* x, float* sumx, int N, int ch, int row) {
  if(N <= 0)
    return;
  float s = x[row];
  for(int i = 1; i < N; i++) {
    s += x[i*ch+row];
  }
  *sumx = s;
}

__device__ float relu(float x) {
    if(x > 0)
        return x;
    else
        return 0.f;
}

__global__ void forwardFullKernel(float* inx, float* outx, float* weights, float* biases, int in_ch, int out_ch) {
  for (int i = 0; i < out_ch; i++) {
    float h = 0.0f;
    for (int j = 0; j < in_ch; j++) {
      h += inx[threadIdx.x * in_ch + j] * weights[i * in_ch + j];
    }
    h += biases[i];
    outx[threadIdx.x * out_ch + i] = h;
  }
}

void forwardFull(float* inx, float* outx, float* weights, float* biases, int N, int in_ch, int out_ch) {
    forwardFullKernel<<<1, N>>>(inx, outx, weights, biases, in_ch, out_ch);
}

__global__ void forwardReluKernel(float* inx, int ch) {
  for (int i = 0; i < ch; i++) {
      inx[threadIdx.x * ch + i] = relu(inx[threadIdx.x * ch + i]);
  }
}

void forwardRelu(float* inx, int N, int ch) {
  forwardReluKernel<<<1, N>>>(inx, ch);
}

__global__ void forwardTanhKernel(float* softx, int ch, int row) {
  softx[threadIdx.x * ch + row] = 10.f * tanhf(softx[threadIdx.x * ch + row]);
}

// modify channel [row] of outputx by x = exp(x) for attention softmax, normalized by maxx
__global__ void forwardExpKernel(float* softx, int ch, int row, float* maxx) {
  softx[threadIdx.x * ch + row] = expf(softx[threadIdx.x * ch + row] - *maxx);
}

__global__ void forwardScaleKernel(float* softx, int ch, int row, float* sigma) {
  softx[threadIdx.x * ch + row] = softx[threadIdx.x * ch + row] / *sigma;
}

__global__ void forwardAggregateKernel(float* softx, float* aggregx, int N, int ch) {
  float sum = 0.f;
  for(int i = 0; i < N; i++) {
    sum += softx[i * ch] * softx[i * ch + 1];
  }
  *aggregx = sum;
}

// output rating = sum_i[rating_i * softmax_i(attention_1..N)] = sum_i [ rating_i * exp(attention_i) / sum_j(exp(attention_j)) ]
void forwardAggregate(float* outputx, float* softx, float* aggregx, int N, int ch) {
  cudaMemcpy(softx, outputx, N * ch * sizeof(float), cudaMemcpyDeviceToDevice);
  forwardTanhKernel<<<1, N>>>(softx, ch, 1);
  maxelem<<<1, 1>>>(softx, aggregx, N, ch, 1);
  forwardExpKernel<<<1, N>>>(softx, ch, 1, aggregx);
  sumelem<<<1, 1>>>(softx, aggregx, N, ch, 1);
  forwardScaleKernel<<<1, N>>>(softx, ch, 1, aggregx);
  forwardAggregateKernel<<<1, 1>>>(softx, aggregx, N, ch);
}

__global__ void backwardLossKernel(float* target, float* aggregx, float* outgrads) {
  // currently, we use a fixed batch size of 1, so no parallelism here
  // loss is MSE, so d Loss / d aggreg = 2*(aggreg - target)
  outgrads[0] = 2*(aggregx[0] - target[0]);
}

void backwardLoss(float* target, float* aggregx, float* outgrads) {
  backwardLossKernel<<<1, 1>>>(target, aggregx, outgrads);
}

__global__ void backwardAggregateKernel(float* ingrads, float* softx, int N, int ch, float* outgrads) {
  // currently, we do not compute the sum of exp-attention in parallel, so just do everything single-threaded here
  // currently, we use a fixed batch size of 1, so we only have ingrads[0]
  int i = threadIdx.x;
  float a_i = softx[i * ch + 1];
  outgrads[i * ch] = ingrads[0] * a_i;  // d Loss / d rating_i

  float grad_ij = 0.f;
  for(int j = 0; j < N; j++) {
    float r_j = softx[j * ch];
    float a_j = softx[j * ch + 1];
    float delta_ij = i == j ? 1.f : 0.f;
    grad_ij += r_j * a_i * (delta_ij - a_j);
  }
  outgrads[i * ch + 1] = ingrads[0] * grad_ij;  // d Loss / d attention_i
}

__global__ void backwardTanhKernel(float* ingrads, float* outputx, float* outgrads) {
  float cosa = cosf(outputx[threadIdx.x*2 + 1]);
  outgrads[threadIdx.x*2 + 1] = ingrads[threadIdx.x*2 + 1] * 10.f / (cosa*cosa);
}

void backwardAggregate(float* ingrads, float* softx, int N, float* outgrads) {
  backwardAggregateKernel<<<1, N>>>(ingrads, softx, N, 2, outgrads);
}

void backwardTanh(float* ingrads, float* outputx, int N, float* outgrads) {
  backwardTanhKernel<<<1, N>>>(ingrads, outputx, outgrads);
}

// update weights & biases
__global__ void backwardFullParamsKernel(float* ingrads, float* inx, float* weights, float* biases, int in_ch, int out_ch, int N, float learnrate) {
  for(int n = 0; n < N; n++) {
    for (int j = 0; j < out_ch; j++) {
      float grad_nj = ingrads[n * out_ch + j];
      for (int i = 0; i < in_ch; i++) {
        float in_n = inx[n * in_ch + i];
        weights[j * in_ch + i] -= grad_nj * in_n * learnrate;
      }
      biases[j] -= grad_nj * learnrate;
    }
  }
}

__global__ void backwardFullKernel(float* ingrads, float* weights, float* biases, int in_ch, int out_ch, float* outgrads) {
  for (int i = 0; i < in_ch; i++) {
    outgrads[threadIdx.x * in_ch + i] = 0;
    for (int j = 0; j < out_ch; j++) {
      float grad_ij = ingrads[threadIdx.x * out_ch + j];
      outgrads[threadIdx.x * in_ch + i] += grad_ij * weights[j * in_ch + i];
    }
  }
}

void backwardFull(float* ingrads, float* inx, float* weights, float* biases, int in_ch, int out_ch, int N, float learnrate, float* outgrads) {
  backwardFullKernel<<<1, N>>>(ingrads, weights, biases, in_ch, out_ch, outgrads);
  backwardFullParamsKernel<<<1, 1>>>(ingrads, inx, weights, biases, in_ch, out_ch, N, learnrate);
}

__global__ void backwardReluKernel(float* ingrads, float* inx, float* outgrads) {
  if(ingrads[threadIdx.x] > 0)
    outgrads[threadIdx.x] = ingrads[threadIdx.x];
  else
    outgrads[threadIdx.x] = 0;
}

void backwardRelu(float* ingrads, float* inx, int ch, int N, float* outgrads) {
  backwardReluKernel<<<1, ch*N>>>(ingrads, inx, outgrads);
}
