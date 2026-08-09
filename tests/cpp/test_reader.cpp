// Python writer가 만든 K3X artifact를 C++ reader로 검증합니다.
#include "k3x/reader.hpp"

#include <array>
#include <filesystem>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    const auto path = std::filesystem::path(argv[1]);
    if (argc == 3 && std::string_view(argv[2]) == "io-uring") {
        k3x::ReaderOptions options;
        options.io_engine = k3x::L2IoEngine::io_uring;
        options.queue_depth = 8;
        const auto uring_reader = k3x::Reader::open(path, options);
        if (!uring_reader) {
            std::cerr << k3x::error_code_name(uring_reader.error()) << '\n';
            return 12;
        }
        const auto uring_payload = uring_reader.value().read_tensor(
            uring_reader.value().tensors().front().tensor_id);
        if (!uring_payload || uring_payload.value().empty() ||
            uring_reader.value().options().io_engine !=
                k3x::L2IoEngine::io_uring) {
            return 13;
        }
        return 0;
    }
    const auto reader = k3x::Reader::open(path);
    if (!reader) {
        std::cerr << k3x::error_code_name(reader.error()) << '\n';
        return 1;
    }
    if (reader.value().tensors().empty()) return 3;
    if (argc == 3) {
        if (std::string_view(argv[2]) != "persistent") return 10;
#ifdef __linux__
        const auto moved = path.string() + ".moved";
        std::filesystem::rename(path, moved);
        const auto persistent = reader.value().read_tensor(
            reader.value().tensors().front().tensor_id);
        std::filesystem::rename(moved, path);
        if (!persistent || persistent.value().empty()) return 11;
#endif
        return 0;
    }
    const auto payload = reader.value().read_tensor(reader.value().tensors().front().tensor_id);
    if (!payload || payload.value().empty()) return 4;
    if (reader.value().counters().calls != 1 ||
        reader.value().counters().batch_submissions != 1) {
        return 5;
    }

    const auto& records = reader.value().tensors();
    if (records.size() < 2) return 6;
    const std::array requests{
        k3x::ExtentRequest{records[1].data_offset, records[1].data_length},
        k3x::ExtentRequest{records[0].data_offset, records[0].data_length},
    };
    const auto batch = reader.value().read_extents(requests);
    if (!batch || batch.value().size() != requests.size() ||
        batch.value()[0].size() != requests[0].length ||
        batch.value()[1].size() != requests[1].length ||
        reader.value().counters().calls != 3 ||
        reader.value().counters().batch_submissions != 2 ||
        reader.value().counters().completions != 3) {
        return 7;
    }

    const auto before = reader.value().counters();
    const std::array invalid{
        k3x::ExtentRequest{reader.value().superblock().file_length, 1}};
    const auto rejected = reader.value().read_extents(invalid);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_extent ||
        reader.value().counters().calls != before.calls ||
        reader.value().counters().batch_submissions != before.batch_submissions ||
        reader.value().counters().failures != before.failures) {
        return 8;
    }

    const std::span<const k3x::ExtentRequest> empty;
    const auto empty_result = reader.value().read_extents(empty);
    if (!empty_result || !empty_result.value().empty() ||
        reader.value().counters().batch_submissions != before.batch_submissions) {
        return 9;
    }
    return 0;
}
