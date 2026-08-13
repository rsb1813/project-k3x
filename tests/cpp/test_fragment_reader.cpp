// 여러 K3X fragment를 하나의 논리 Reader로 여는 경계를 검증합니다.
#include "k3x/reader.hpp"

#include <array>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::array paths{
        std::filesystem::path(argv[1]), std::filesystem::path(argv[2])};
    auto reader = k3x::Reader::open_fragments(paths);
    if (!reader || reader.value().tensors().size() != 2) return 3;
    const auto& records = reader.value().tensors();
    const auto first = reader.value().read_tensor(records[0].tensor_id);
    const auto second = reader.value().read_tensor(records[1].tensor_id);
    if (!first || !second || first.value().size() != records[0].data_length ||
        second.value().size() != records[1].data_length) {
        return 4;
    }
    const std::array requests{
        k3x::ExtentRequest{records[1].data_offset, records[1].data_length},
        k3x::ExtentRequest{records[0].data_offset, records[0].data_length},
    };
    const auto batch = reader.value().read_extents(requests);
    if (!batch || batch.value().size() != 2 ||
        batch.value()[0] != second.value() || batch.value()[1] != first.value()) {
        return 5;
    }
    return 0;
}
