// C++ RMSNorm, SiTU, MXFP4 literal 결과를 독립 검증합니다.
#include "k3x/ops.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    std::array<float, 2> input{3.0F, 4.0F}, weight{1.0F, 2.0F}, output{};
    k3x::rms_norm(output, input, weight, 0.0F);
    const float scale = std::sqrt(12.5F);
    if (std::abs(output[0] - 3.0F / scale) > 1e-6F ||
        std::abs(output[1] - 8.0F / scale) > 1e-6F) return 1;
    std::array<float, 2> gate{-1.0F, 2.0F}, up{3.0F, 4.0F}, activated{};
    k3x::situ_glu(activated, gate, up, 1.0F, 1.0F);
    const auto expected0 = std::tanh(-1.0F) * (1.0F / (1.0F + std::exp(1.0F))) * std::tanh(3.0F);
    if (std::abs(activated[0] - expected0) > 1e-6F) return 2;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    std::array<std::byte, 1> scales{std::byte{127}};
    const auto decoded = k3x::decode_mxfp4(packed, scales, 1, 32, 32);
    if (!decoded || decoded.value()[0] != 0.0F || decoded.value()[1] != 0.5F) return 3;
    std::array<std::byte, 12> quant3_packed{};
    quant3_packed[0] = std::byte{0x88};
    quant3_packed[1] = std::byte{0xc6};
    quant3_packed[2] = std::byte{0x7a};
    std::array<std::byte, 2> quant3_scales{
        std::byte{0x80}, std::byte{0x3f}};
    const auto quant3 = k3x::decode_groupwise_3bit(
        quant3_packed, quant3_scales, 8, 32);
    if (!quant3 || quant3.value().size() != 8 ||
        quant3.value()[0] != -3.0F || quant3.value()[1] != -2.0F ||
        quant3.value()[6] != 3.0F || quant3.value()[7] != 0.0F) return 4;
    quant3_packed[0] = std::byte{0x8f};
    if (k3x::decode_groupwise_3bit(
            quant3_packed, quant3_scales, 8, 32)) return 5;
    std::array<std::byte, 128> quant8_codes{};
    quant8_codes[0] = std::byte{0x81};
    quant8_codes[1] = std::byte{0x7f};
    const auto quant8 = k3x::decode_groupwise_8bit(
        quant8_codes, quant3_scales, 2, 128);
    if (!quant8 || quant8.value().size() != 2 ||
        quant8.value()[0] != -127.0F || quant8.value()[1] != 127.0F) return 6;
    return 0;
}
