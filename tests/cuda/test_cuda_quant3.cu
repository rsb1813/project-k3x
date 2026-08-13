// group-wise 3-bit packed CUDA matvec를 portable oracle과 비교합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include <array>
#include <cmath>

int main() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 32;
    std::array<float, cols> input{};
    input[0] = 1.0F;
    input[1] = -2.0F;
    input[6] = 0.5F;

    std::array<std::byte, rows * 12> packed{};
    packed[0] = std::byte{0x88};
    packed[1] = std::byte{0xc6};
    packed[2] = std::byte{0x7a};
    packed[12] = std::byte{0x46};
    packed[13] = std::byte{0x44};
    packed[14] = std::byte{0x44};
    const std::array<std::byte, rows * 2> scales{
        std::byte{0x80}, std::byte{0x3f},
        std::byte{0x00}, std::byte{0x40},
    };

    const auto decoded = k3x::decode_groupwise_3bit(
        packed, scales, rows * cols, 32);
    if (!decoded) return 1;
    std::array<float, rows> expected{};
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < cols; ++column) {
            expected[row] += input[column] * decoded.value()[row * cols + column];
        }
    }

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 2;
    const auto actual = backend.value()->quant3_matvec(
        input, {701, packed, scales, rows, cols, 32},
        1, k3x::ProfilePhase::decode);
    if (!actual || actual.value().size() != rows) return 3;
    for (std::size_t row = 0; row < rows; ++row) {
        if (std::abs(actual.value()[row] - expected[row]) > 1.0e-5F) return 4;
    }
    const auto stats = backend.value()->runtime_stats();
    if (stats.weight_h2d_bytes != packed.size() + scales.size() ||
        stats.activation_h2d_bytes != input.size() * sizeof(float) ||
        stats.device_to_host_bytes != rows * sizeof(float)) return 5;
    return 0;
}
