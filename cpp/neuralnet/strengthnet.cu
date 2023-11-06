#include "strengthnet.h"

__device__ float relu(float x) {
    if(x > 0)
        return x;
    else
        return 0.f;
}

__device__ float d_relu(float x) {
    if(x > 0)
        return 1.f;
    else
        return 0.f;
}

__global__ void forwardFullKernel(float* inx, float* outx, float* weights, float* biases, int in_ch, int out_ch) {
  for (int i = 0; i < out_ch; ++i) {
    float h = 0.0f;
    for (int j = 0; j < in_ch; ++j) {
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
  for (int i = 0; i < ch; ++i) {
      inx[threadIdx.x * ch + i] = relu(inx[threadIdx.x * ch + i]);
  }
}

void forwardRelu(float* inx, int N, int ch) {
  forwardReluKernel<<<1, N>>>(inx, ch);
}

__global__ void forwardAggregateKernel(float* outputx, float* aggregx, int N, int ch) {
  float sum = 0.f;
  float scale = 0.f;

  for (int i = 0; i < N; ++i) {
    float s = expf(outputx[i * ch + 1]);
    scale += s;
    sum += outputx[i * ch] * s;
  }

  *aggregx = sum/scale;
}

void forwardAggregate(float* outputx, float* aggregx, int N, int ch) {
  forwardAggregateKernel<<<1, 1>>>(outputx, aggregx, N, ch);
}

__global__ void backwardOutputKernel(float* output, float* target, float* weights, float* hidden, float* biases) {
  // int idx = threadIdx.x;
  // float error = target[0] - output[idx];
  // float gradient = error * sigmoid_derivative(output[idx]);
  // // Update weights and biases
  // for (int i = 0; i < HIDDEN_SIZE; ++i) {
  //     weights[i] += LEARNING_RATE * gradient * hidden[i];
  // }
  // biases[0] += LEARNING_RATE * gradient;
}

void backwardOutput(float* output, int N, float* target, float* weights, float* hidden, float* biases) {
  backwardOutputKernel<<<1, N>>>(output, target, weights, hidden, biases);
}

__global__ void backwardHiddenKernel(float* hidden, float* output_gradients, float* weights, float* input, float* biases) {
  // int idx = threadIdx.x;
  // float error = 0.0f;
  // for (int i = 0; i < OUTPUT_SIZE; ++i) {
  //     error += output_gradients[i] * weights[i * HIDDEN_SIZE + idx];
  // }
  // float gradient = error * sigmoid_derivative(hidden[idx]);
  // // Update weights and biases
  // for (int i = 0; i < INPUT_SIZE; ++i) {
  //     weights[idx * INPUT_SIZE + i] += LEARNING_RATE * gradient * input[i];
  // }
  // biases[idx] += LEARNING_RATE * gradient;
}

void backwardHidden(float* hidden, int N, float* output_gradients, float* weights, float* input, float* biases) {
  backwardHiddenKernel<<<1, N>>>(hidden, output_gradients, weights, input, biases);
}
