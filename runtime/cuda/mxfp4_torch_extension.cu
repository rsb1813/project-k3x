// native MXFP4 바이트를 복호화 버퍼 없이 직접 행렬-벡터 곱으로 계산합니다.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

__device__ __forceinline__ float decode_e2m1(std::uint8_t code) {
    float magnitude = 0.0F;
    switch (code & 0x07U) {
        case 1: magnitude = 0.5F; break;
        case 2: magnitude = 1.0F; break;
        case 3: magnitude = 1.5F; break;
        case 4: magnitude = 2.0F; break;
        case 5: magnitude = 3.0F; break;
        case 6: magnitude = 4.0F; break;
        case 7: magnitude = 6.0F; break;
        default: break;
    }
    return (code & 0x08U) != 0 ? -magnitude : magnitude;
}

__global__ void mxfp4_matvec_kernel(
    const float* input,
    const std::uint8_t* packed,
    const std::uint8_t* scales,
    float* output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    float sum = 0.0F;
    const auto row_offset = row * columns;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < columns;
         column += static_cast<std::int64_t>(blockDim.x)) {
        const auto logical = row_offset + column;
        const auto pair = packed[logical / 2];
        const auto code = static_cast<std::uint8_t>(
            logical % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent = static_cast<int>(scales[logical / 32]) - 127;
        sum = fmaf(input[column], ldexpf(decode_e2m1(code), exponent), sum);
    }
    extern __shared__ float partial[];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = partial[0];
    }
}

struct MatrixDescriptor {
    const std::uint8_t* packed;
    const std::uint8_t* scales;
};

__global__ void mxfp4_expert_grid_kernel(
    const float* inputs,
    const MatrixDescriptor* descriptors,
    float* outputs,
    std::int64_t rows,
    std::int64_t columns,
    std::int64_t expert_count,
    bool shared_input,
    bool row_major_output) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    const auto expert = static_cast<std::int64_t>(blockIdx.y);
    if (row >= rows || expert >= expert_count) {
        return;
    }
    const auto& matrix = descriptors[expert];
    const auto* input = inputs + (shared_input ? 0 : expert * columns);
    float sum = 0.0F;
    const auto row_offset = row * columns;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < columns;
         column += static_cast<std::int64_t>(blockDim.x)) {
        const auto logical = row_offset + column;
        const auto pair = matrix.packed[logical / 2];
        const auto code = static_cast<std::uint8_t>(
            logical % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent = static_cast<int>(matrix.scales[logical / 32]) - 127;
        sum = fmaf(input[column], ldexpf(decode_e2m1(code), exponent), sum);
    }
    extern __shared__ float partial[];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const auto output_index = row_major_output
            ? row * expert_count + expert
            : expert * rows + row;
        outputs[output_index] = partial[0];
    }
}

__global__ void situ_kernel(
    const float* gate,
    const float* up,
    float* output,
    std::int64_t count) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (index >= count) {
        return;
    }
    const auto sigmoid = 1.0F / (1.0F + expf(-gate[index]));
    const auto bounded_gate = 4.0F * tanhf(gate[index] / 4.0F) * sigmoid;
    const auto bounded_up = 25.0F * tanhf(up[index] / 25.0F);
    output[index] = bounded_gate * bounded_up;
}

__global__ void ordered_mix_kernel(
    const float* expert_outputs,
    const float* contributions,
    float* output,
    std::int64_t rows,
    std::int64_t expert_count) {
    const auto row = static_cast<std::int64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0F;
    for (std::int64_t expert = 0; expert < expert_count; ++expert) {
        sum = fmaf(
            contributions[expert],
            expert_outputs[row * expert_count + expert],
            sum);
    }
    output[row] = sum;
}

torch::Tensor mxfp4_matvec(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t rows,
    std::int64_t columns) {
    TORCH_CHECK(input.is_cuda() && packed.is_cuda() && scales.is_cuda());
    TORCH_CHECK(input.device() == packed.device() && input.device() == scales.device());
    TORCH_CHECK(input.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.is_contiguous() && packed.is_contiguous() && scales.is_contiguous());
    TORCH_CHECK(rows > 0 && columns > 0 && columns % 32 == 0);
    TORCH_CHECK(input.numel() == columns);
    TORCH_CHECK(packed.numel() == rows * columns / 2);
    TORCH_CHECK(scales.numel() == rows * columns / 32);

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({rows}, input.options());
    const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<static_cast<unsigned>(rows), threads,
                           threads * sizeof(float), stream>>>(
        input.data_ptr<float>(),
        packed.data_ptr<std::uint8_t>(),
        scales.data_ptr<std::uint8_t>(),
        output.data_ptr<float>(),
        rows,
        columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor mxfp4_expert_batch(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& gate_packed,
    const std::vector<torch::Tensor>& gate_scales,
    const std::vector<torch::Tensor>& up_packed,
    const std::vector<torch::Tensor>& up_scales,
    const std::vector<torch::Tensor>& down_packed,
    const std::vector<torch::Tensor>& down_scales,
    const torch::Tensor& contributions,
    std::int64_t latent_size,
    std::int64_t intermediate_size) {
    const auto expert_count = static_cast<std::int64_t>(gate_packed.size());
    TORCH_CHECK(input.is_cuda() && contributions.is_cuda());
    TORCH_CHECK(input.device() == contributions.device());
    TORCH_CHECK(input.scalar_type() == torch::kFloat32);
    TORCH_CHECK(contributions.scalar_type() == torch::kFloat32);
    TORCH_CHECK(input.is_contiguous() && contributions.is_contiguous());
    TORCH_CHECK(latent_size > 0 && intermediate_size > 0);
    TORCH_CHECK(latent_size % 32 == 0 && intermediate_size % 32 == 0);
    TORCH_CHECK(input.numel() == latent_size);
    TORCH_CHECK(expert_count > 0 && expert_count <= 65535);
    TORCH_CHECK(contributions.numel() == expert_count);
    TORCH_CHECK(
        gate_scales.size() == gate_packed.size()
        && up_packed.size() == gate_packed.size()
        && up_scales.size() == gate_packed.size()
        && down_packed.size() == gate_packed.size()
        && down_scales.size() == gate_packed.size());

    std::vector<std::int64_t> host_descriptors;
    host_descriptors.reserve(static_cast<std::size_t>(expert_count) * 6);
    const auto append_role = [&](const std::vector<torch::Tensor>& packed_values,
                                 const std::vector<torch::Tensor>& scale_values,
                                 std::int64_t rows,
                                 std::int64_t columns) {
        for (std::int64_t index = 0; index < expert_count; ++index) {
            const auto& packed = packed_values[static_cast<std::size_t>(index)];
            const auto& scales = scale_values[static_cast<std::size_t>(index)];
            TORCH_CHECK(packed.is_cuda() && scales.is_cuda());
            TORCH_CHECK(packed.device() == input.device());
            TORCH_CHECK(scales.device() == input.device());
            TORCH_CHECK(packed.scalar_type() == torch::kUInt8);
            TORCH_CHECK(scales.scalar_type() == torch::kUInt8);
            TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous());
            TORCH_CHECK(packed.numel() == rows * columns / 2);
            TORCH_CHECK(scales.numel() == rows * columns / 32);
            host_descriptors.push_back(reinterpret_cast<std::intptr_t>(
                packed.data_ptr<std::uint8_t>()));
            host_descriptors.push_back(reinterpret_cast<std::intptr_t>(
                scales.data_ptr<std::uint8_t>()));
        }
    };
    append_role(gate_packed, gate_scales, intermediate_size, latent_size);
    append_role(up_packed, up_scales, intermediate_size, latent_size);
    append_role(down_packed, down_scales, latent_size, intermediate_size);

    const c10::cuda::CUDAGuard guard(input.device());
    const auto descriptor_host = torch::from_blob(
        host_descriptors.data(),
        {3, expert_count, 2},
        torch::TensorOptions().dtype(torch::kInt64)).clone();
    const auto descriptors = descriptor_host.to(input.device());
    auto gate = torch::empty(
        {expert_count, intermediate_size}, input.options());
    auto up = torch::empty_like(gate);
    auto activated = torch::empty_like(gate);
    auto down = torch::empty(
        {latent_size, expert_count}, input.options());
    auto output = torch::empty({latent_size}, input.options());
    const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    constexpr unsigned threads = 256;
    const auto* descriptor_base = reinterpret_cast<const MatrixDescriptor*>(
        descriptors.data_ptr<std::int64_t>());
    const auto shared_bytes = threads * sizeof(float);
    const dim3 first_grid(
        static_cast<unsigned>(intermediate_size),
        static_cast<unsigned>(expert_count));
    mxfp4_expert_grid_kernel<<<first_grid, threads, shared_bytes, stream>>>(
        input.data_ptr<float>(), descriptor_base, gate.data_ptr<float>(),
        intermediate_size, latent_size, expert_count, true, false);
    mxfp4_expert_grid_kernel<<<first_grid, threads, shared_bytes, stream>>>(
        input.data_ptr<float>(), descriptor_base + expert_count,
        up.data_ptr<float>(), intermediate_size, latent_size, expert_count,
        true, false);
    const auto activated_values = expert_count * intermediate_size;
    situ_kernel<<<static_cast<unsigned>((activated_values + threads - 1) / threads),
                  threads, 0, stream>>>(
        gate.data_ptr<float>(), up.data_ptr<float>(), activated.data_ptr<float>(),
        activated_values);
    const dim3 down_grid(
        static_cast<unsigned>(latent_size),
        static_cast<unsigned>(expert_count));
    mxfp4_expert_grid_kernel<<<down_grid, threads, shared_bytes, stream>>>(
        activated.data_ptr<float>(), descriptor_base + 2 * expert_count,
        down.data_ptr<float>(), latent_size, intermediate_size, expert_count,
        false, true);
    ordered_mix_kernel<<<static_cast<unsigned>((latent_size + threads - 1) / threads),
                         threads, 0, stream>>>(
        down.data_ptr<float>(), contributions.data_ptr<float>(),
        output.data_ptr<float>(), latent_size, expert_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mxfp4_matvec", &mxfp4_matvec, "K3X native MXFP4 matvec");
    module.def(
        "mxfp4_expert_batch", &mxfp4_expert_batch,
        "K3X native MXFP4 expert-major batch");
}
