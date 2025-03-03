__global__ void add(const float* x, const float* y, float* z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        z[i] = x[i] + y[i];
    }
}

__global__ void sub(const float* x, const float* y, float* z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        z[i] = x[i] - y[i];
    }
}

__global__ void mul(const float* x, const float* y, float* z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        z[i] = x[i] * y[i];
    }
}

__global__ void div(const float* x, const float* y, float* z, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        z[i] = (y[i] != 0.0f) ? (x[i] / y[i]) : 0.0f;
    }
}