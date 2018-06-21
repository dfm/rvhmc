#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "kepler.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void KeplerCudaKernel(const int maxiter, const float tol, const int size, const T* M, const T* e, T* E) {
  int n;
  T e_, M_, E0, E_, s, c, g, gp;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
    e_ = ldg(e + i);
    M_ = ldg(M + i);
    if (fabs(e_) < tol) {
      E[i] = M_;
    } else {
      E0 = M_;
      E_ = E0;
      for (n = 0; n < maxiter; ++n) {
        sincos(E0, &s, &c);
        g = E0 - e_ * s - M_;
        gp = 1.0 - e_ * c;
        E_ = E0 - g / gp;
        if (fabs((E_ - E0) / E_) <= tol) {
          E[i] = E_;
          n = maxiter;
        }
        E0 = E_;
      }
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void KeplerFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, int maxiter, float tol, int size, const T* M, const T* e, T* E) {
  CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
  KeplerCudaKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(maxiter, tol, size, M, e, E);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct KeplerFunctor<GPUDevice, float>;
template struct KeplerFunctor<GPUDevice, double>;

#endif  // GOOGLE_CUDA
