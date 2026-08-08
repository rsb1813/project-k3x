// K3X 메타데이터와 tensor extent를 경계 검증 후 읽습니다.
#pragma once

#include "k3x/format.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace k3x {
enum class VerifyMode { checksums, metadata_only };
struct ReadCounters { std::uint64_t calls{}, requested_bytes{}, completed_bytes{}; };

class Reader {
public:
    static Result<Reader> open(const std::filesystem::path& path,
                               VerifyMode mode = VerifyMode::checksums);
    Result<std::vector<std::byte>> read_tensor(std::uint64_t tensor_id) const;
    Result<std::vector<std::byte>> read_auxiliary(std::uint64_t tensor_id) const;
    const std::vector<TensorRecord>& tensors() const { return tensors_; }
    const Superblock& superblock() const { return superblock_; }
    const std::array<std::byte, model_config_bytes>& model_config() const { return model_config_; }
    const ReadCounters& counters() const { return counters_; }
private:
    Result<std::vector<std::byte>> read_extent(std::uint64_t offset, std::uint64_t length) const;
    std::filesystem::path path_;
    Superblock superblock_;
    std::vector<TensorRecord> tensors_;
    std::array<std::byte, model_config_bytes> model_config_{};
    mutable ReadCounters counters_{};
};
}
