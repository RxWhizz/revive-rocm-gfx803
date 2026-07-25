// Standalone HIP SGEMM microbenchmark for gfx803/Polaris.
// Row-major C[M,N] = A[M,K] * B[K,N], FP32 only.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK_HIP(expr)                                                          \
  do {                                                                          \
    hipError_t _err = (expr);                                                   \
    if (_err != hipSuccess) {                                                   \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "      \
                << hipGetErrorString(_err) << std::endl;                        \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

struct Shape {
  int m = 4096;
  int n = 4096;
  int k = 4096;
};

struct Result {
  std::string variant;
  std::string gpu_name;
  std::string arch;
  int device = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  double avg_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double gflops = 0.0;
  bool valid = false;
};

__global__ __launch_bounds__(256, 2)
void sgemm_conservative_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ c,
                               int m, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int p = 0; p < k; ++p) {
    acc += a[row * k + p] * b[p * n + col];
  }
  c[row * n + col] = acc;
}

template <int TILE>
__global__ __launch_bounds__(256, 2)
void sgemm_lds_kernel(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ c,
                      int m, int n, int k) {
  __shared__ float as[TILE][TILE + 1];
  __shared__ float bs[TILE][TILE + 1];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * TILE + tx;
  int row = blockIdx.y * TILE + ty;

  float acc = 0.0f;
  for (int tile = 0; tile < k; tile += TILE) {
    int a_col = tile + tx;
    int b_row = tile + ty;
    as[ty][tx] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    bs[ty][tx] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int p = 0; p < TILE; ++p) {
      acc += as[ty][p] * bs[p][tx];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

__global__ __launch_bounds__(256, 2)
void sgemm_vec4_kernel(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c,
                       int m, int n, int k) {
  int vec_col = blockIdx.x * blockDim.x + threadIdx.x;
  int col = vec_col * 4;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) return;

  float4 acc;
  acc.x = 0.0f;
  acc.y = 0.0f;
  acc.z = 0.0f;
  acc.w = 0.0f;

  bool aligned_vec = ((n & 3) == 0) && (col + 3 < n);
  for (int p = 0; p < k; ++p) {
    float av = a[row * k + p];
    const float* bp = b + p * n + col;
    float4 bv;
    if (aligned_vec) {
      bv = *reinterpret_cast<const float4*>(bp);
    } else {
      bv.x = (col + 0 < n) ? bp[0] : 0.0f;
      bv.y = (col + 1 < n) ? bp[1] : 0.0f;
      bv.z = (col + 2 < n) ? bp[2] : 0.0f;
      bv.w = (col + 3 < n) ? bp[3] : 0.0f;
    }
    acc.x += av * bv.x;
    acc.y += av * bv.y;
    acc.z += av * bv.z;
    acc.w += av * bv.w;
  }

  float* cp = c + row * n + col;
  if (col + 0 < n) cp[0] = acc.x;
  if (col + 1 < n) cp[1] = acc.y;
  if (col + 2 < n) cp[2] = acc.z;
  if (col + 3 < n) cp[3] = acc.w;
}

Shape parse_shape(const std::string& s) {
  Shape shape;
  std::string tmp = s;
  std::replace(tmp.begin(), tmp.end(), 'X', 'x');
  std::replace(tmp.begin(), tmp.end(), ',', 'x');
  std::vector<int> parts;
  std::stringstream ss(tmp);
  std::string item;
  while (std::getline(ss, item, 'x')) {
    if (!item.empty()) parts.push_back(std::stoi(item));
  }
  if (parts.size() == 1) {
    shape.m = shape.n = shape.k = parts[0];
  } else if (parts.size() == 3) {
    shape.m = parts[0];
    shape.n = parts[1];
    shape.k = parts[2];
  } else {
    std::cerr << "Invalid shape: " << s << " (use N or MxNxK)" << std::endl;
    std::exit(2);
  }
  return shape;
}

void fill_matrix(std::vector<float>& v, int stride) {
  for (size_t i = 0; i < v.size(); ++i) {
    int x = static_cast<int>((i * 17 + stride * 31) % 127);
    v[i] = (static_cast<float>(x) - 63.0f) / 64.0f;
  }
}

bool validate_sample(const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& c, const Shape& shape) {
  int rows = std::min(shape.m, 16);
  int cols = std::min(shape.n, 16);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      double ref = 0.0;
      for (int p = 0; p < shape.k; ++p) {
        ref += static_cast<double>(a[row * shape.k + p]) *
               static_cast<double>(b[p * shape.n + col]);
      }
      double got = static_cast<double>(c[row * shape.n + col]);
      double tol = 1e-2 + 1e-3 * std::abs(ref);
      if (std::abs(got - ref) > tol) {
        std::cerr << "Mismatch at (" << row << "," << col << "): got "
                  << got << " ref " << ref << " tol " << tol << std::endl;
        return false;
      }
    }
  }
  return true;
}

void launch_variant(const std::string& variant, const float* d_a, const float* d_b,
                    float* d_c, const Shape& shape) {
  if (variant == "conservative") {
    dim3 block(16, 16);
    dim3 grid((shape.n + block.x - 1) / block.x,
              (shape.m + block.y - 1) / block.y);
    hipLaunchKernelGGL(sgemm_conservative_kernel, grid, block, 0, 0,
                       d_a, d_b, d_c, shape.m, shape.n, shape.k);
  } else if (variant == "lds") {
    constexpr int tile = 16;
    dim3 block(tile, tile);
    dim3 grid((shape.n + tile - 1) / tile, (shape.m + tile - 1) / tile);
    hipLaunchKernelGGL((sgemm_lds_kernel<tile>), grid, block, 0, 0,
                       d_a, d_b, d_c, shape.m, shape.n, shape.k);
  } else if (variant == "vec4") {
    dim3 block(16, 16);
    int n_vec = (shape.n + 3) / 4;
    dim3 grid((n_vec + block.x - 1) / block.x,
              (shape.m + block.y - 1) / block.y);
    hipLaunchKernelGGL(sgemm_vec4_kernel, grid, block, 0, 0,
                       d_a, d_b, d_c, shape.m, shape.n, shape.k);
  } else {
    std::cerr << "Unknown variant: " << variant << std::endl;
    std::exit(2);
  }
  CHECK_HIP(hipGetLastError());
}

Result run_one(const std::string& variant, const Shape& shape, int repeats,
               int warmup, bool validate) {
  int device = 0;
  CHECK_HIP(hipGetDevice(&device));
  hipDeviceProp_t prop;
  CHECK_HIP(hipGetDeviceProperties(&prop, device));

  size_t a_elems = static_cast<size_t>(shape.m) * shape.k;
  size_t b_elems = static_cast<size_t>(shape.k) * shape.n;
  size_t c_elems = static_cast<size_t>(shape.m) * shape.n;
  std::vector<float> h_a(a_elems), h_b(b_elems), h_c(c_elems);
  fill_matrix(h_a, shape.k);
  fill_matrix(h_b, shape.n);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CHECK_HIP(hipMalloc(&d_a, a_elems * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_b, b_elems * sizeof(float)));
  CHECK_HIP(hipMalloc(&d_c, c_elems * sizeof(float)));
  CHECK_HIP(hipMemcpy(d_a, h_a.data(), a_elems * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_b, h_b.data(), b_elems * sizeof(float), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemset(d_c, 0, c_elems * sizeof(float)));

  for (int i = 0; i < warmup; ++i) {
    launch_variant(variant, d_a, d_b, d_c, shape);
  }
  CHECK_HIP(hipDeviceSynchronize());

  std::vector<float> times_ms;
  times_ms.reserve(repeats);
  for (int i = 0; i < repeats; ++i) {
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipEventRecord(start, 0));
    launch_variant(variant, d_a, d_b, d_c, shape);
    CHECK_HIP(hipEventRecord(stop, 0));
    CHECK_HIP(hipEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&ms, start, stop));
    times_ms.push_back(ms);
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
  }

  CHECK_HIP(hipMemcpy(h_c.data(), d_c, c_elems * sizeof(float), hipMemcpyDeviceToHost));
  bool ok = !validate || validate_sample(h_a, h_b, h_c, shape);

  CHECK_HIP(hipFree(d_a));
  CHECK_HIP(hipFree(d_b));
  CHECK_HIP(hipFree(d_c));

  double sum = 0.0;
  for (float ms : times_ms) sum += ms;
  double avg_ms = sum / static_cast<double>(times_ms.size());
  auto mm = std::minmax_element(times_ms.begin(), times_ms.end());
  double gflops = (2.0 * shape.m * shape.n * shape.k) / (avg_ms / 1000.0) / 1e9;

  Result r;
  r.variant = variant;
  r.gpu_name = prop.name;
  r.arch = prop.gcnArchName;
  r.device = device;
  r.m = shape.m;
  r.n = shape.n;
  r.k = shape.k;
  r.avg_ms = avg_ms;
  r.min_ms = *mm.first;
  r.max_ms = *mm.second;
  r.gflops = gflops;
  r.valid = ok;
  return r;
}

void print_csv_header(std::ostream& os) {
  os << "variant,gpu_device,gpu_name,arch,m,n,k,avg_ms,min_ms,max_ms,gflops,tflops,valid\n";
}

void print_csv_row(std::ostream& os, const Result& r) {
  os << r.variant << "," << r.device << ",\"" << r.gpu_name << "\"," << r.arch << ","
     << r.m << "," << r.n << "," << r.k << ","
     << std::fixed << std::setprecision(4) << r.avg_ms << ","
     << r.min_ms << "," << r.max_ms << ","
     << std::setprecision(2) << r.gflops << ","
     << std::setprecision(4) << (r.gflops / 1000.0) << ","
     << (r.valid ? "true" : "false") << "\n";
}

int main(int argc, char** argv) {
  std::vector<Shape> shapes;
  std::vector<std::string> variants = {"conservative", "lds", "vec4"};
  int repeats = 20;
  int warmup = 5;
  bool validate = true;
  std::string csv_path;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--shape" && i + 1 < argc) {
      shapes.push_back(parse_shape(argv[++i]));
    } else if (arg == "--variant" && i + 1 < argc) {
      variants.clear();
      std::stringstream ss(argv[++i]);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) variants.push_back(item);
      }
    } else if (arg == "--repeats" && i + 1 < argc) {
      repeats = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    } else if (arg == "--csv" && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (arg == "--no-validate") {
      validate = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--shape N|MxNxK] [--variant conservative,lds,vec4]"
                << " [--repeats N] [--warmup N] [--csv path] [--no-validate]\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 2;
    }
  }
  if (shapes.empty()) {
    shapes = {parse_shape("512"), parse_shape("1024"), parse_shape("2048"), parse_shape("4096")};
  }

  std::vector<Result> results;
  for (const auto& shape : shapes) {
    for (const auto& variant : variants) {
      Result r = run_one(variant, shape, repeats, warmup, validate);
      results.push_back(r);
      print_csv_row(std::cout, r);
    }
  }

  if (!csv_path.empty()) {
    std::ofstream f(csv_path);
    print_csv_header(f);
    for (const auto& r : results) print_csv_row(f, r);
    std::cerr << "Wrote " << csv_path << std::endl;
  }

  bool ok = true;
  for (const auto& r : results) ok = ok && r.valid;
  return ok ? 0 : 1;
}
