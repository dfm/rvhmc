#ifndef _KEPLER_H_
#define _KEPLER_H_

#include <Eigen/Core>
#include <cmath>

template <typename T>
inline T kepler (const T& M, const T& e, int maxiter, float tol) {
  T E0 = M, E = M;
  if (std::abs(e) < tol) return E;
  for (int i = 0; i < maxiter; ++i) {
    T g = E0 - e * sin(E0) - M, gp = 1.0 - e * cos(E0);
    E = E0 - g / gp;
    if (std::abs((E - E0) / E) <= T(tol)) {
      return E;
    }
    E0 = E;
  }

  // If we get here, we didn't converge, but return the best estimate.
  return E;
}

template <typename Device, typename T>
struct KeplerFunctor {
  void operator()(const Device& d, int maxiter, float tol, int size, const T* M, const T* e, T* E);
};

#if GOOGLE_CUDA
template <typename T>
struct KeplerFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, int maxiter, float tol, int size, const T* M, const T* e, T* E);
};
#endif

#endif
