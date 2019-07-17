#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

const unsigned N = (1 << 26);
const float XVAL = rand() % 1000000;
const float YVAL = rand() % 1000000;
const float AVAL = rand() % 1000000;

static bool saxpy_verify(const std::vector<float> y) {
  float err = 0.0;
  for (size_t i = 0; i < N; ++i)
    err = err + fabs(y[i] - (AVAL * XVAL + YVAL));

  std::cout << "Errors: " << err << std::endl;
  return err == 0.0;
}

static bool saxpy_verify(const float *y) {
  float err = 0.0;
  for (size_t i = 0; i < N; ++i)
    err = err + fabs(y[i] - (AVAL * XVAL + YVAL));

  std::cout << "Errors: " << err << std::endl;
  return err == 0.0;
}

class saxpy_timer {
 public:
  saxpy_timer() { reset(); }
  void reset() { t0_ = std::chrono::high_resolution_clock::now(); }
  double elapsed(bool reset_timer = false) {
    std::chrono::high_resolution_clock::time_point t =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t - t0_);
    if (reset_timer)
      reset();
    return time_span.count();
  }
  double elapsed_msec(bool reset_timer = false) {
    return elapsed(reset_timer) * 1000;
  }

 private:
  std::chrono::high_resolution_clock::time_point t0_;
};
