// Python writer가 만든 K3X artifact를 C++ reader로 검증합니다.
#include "k3x/reader.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto reader = k3x::Reader::open(std::filesystem::path(argv[1]));
    if (!reader) {
        std::cerr << k3x::error_code_name(reader.error()) << '\n';
        return 1;
    }
    if (reader.value().tensors().empty()) return 3;
    const auto payload = reader.value().read_tensor(reader.value().tensors().front().tensor_id);
    if (!payload || payload.value().empty()) return 4;
    return 0;
}
