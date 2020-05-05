#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> matmul_cuda_forward(
    torch::Tensor a,
    torch::Tensor b,
    int mode);

std::vector<torch::Tensor> matmul_cuda_backward(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor grad_output,
        torch::Tensor part,
        int mode);

std::vector<torch::Tensor> banded_cuda_forward(
    torch::Tensor a,
    int a_lu,
    int a_lb,
    torch::Tensor b,
    int b_lu,
    int b_lb,
    int mode);

std::vector<torch::Tensor> banded_cuda_backward(
        torch::Tensor a,
        int a_lu,
        int a_lb,
        torch::Tensor b,
        int b_lu,
        int b_lb,
        torch::Tensor grad_output,
        torch::Tensor part,
        int mode);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> matmul_forward(
    torch::Tensor a,
    torch::Tensor b,
    int mode) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  return matmul_cuda_forward(a, b, mode);
}


std::vector<torch::Tensor> matmul_backward(
        torch::Tensor a,
        torch::Tensor b,
        torch::Tensor grad_output,
        torch::Tensor part,
        int mode) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(part);
  return matmul_cuda_backward(a, b, grad_output, part, mode);
}


std::vector<torch::Tensor> banded_forward(
    torch::Tensor a,
    int a_lu,
    int a_lb,
    torch::Tensor b,
    int b_lu,
    int b_lb,
    int mode) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);

  return banded_cuda_forward(a, a_lu, a_lb, b, b_lu, b_lb, mode);
}

std::vector<torch::Tensor> banded_backward(
        torch::Tensor a,
        int a_lu,
        int a_lb,
        torch::Tensor b,
        int b_lu,
        int b_lb,
        torch::Tensor grad_output,
        torch::Tensor part,
        int mode) {
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(grad_output);
  CHECK_INPUT(part);
  return banded_cuda_backward(a, a_lu, a_lb,
                              b, b_lu, b_lb,
                              grad_output, part, mode);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &matmul_forward, "Log-Matmul forward (CUDA)");
  m.def("backward", &matmul_backward, "Log-Matmul backward (CUDA)");
  m.def("forward_band", &banded_forward, "Banded Log-Matmul forward (CUDA)");
  m.def("backward_band", &banded_backward, "Banded Log-Matmul backward (CUDA)");
}
