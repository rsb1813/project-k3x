// K3X v1을 aliasing 없이 little-endian으로 해석하고 검증합니다.
#include "k3x/reader.hpp"

#include "k3x/checksums.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <unordered_set>

namespace k3x {
namespace {
template <typename T>
T little(std::span<const std::byte> bytes, std::size_t offset) {
    T value{};
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(std::to_integer<std::uint8_t>(bytes[offset + index])) << (index * 8U);
    }
    return value;
}

bool range_valid(std::uint64_t offset, std::uint64_t length, std::uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

bool all_zero(std::span<const std::byte> value) {
    return std::all_of(value.begin(), value.end(), [](std::byte item) {
        return item == std::byte{0};
    });
}

std::optional<std::uint64_t> directory_count(std::span<const std::byte> directory,
                                             const char (&tag)[5],
                                             std::uint32_t record_size) {
    if (directory.size() < 16 || std::memcmp(directory.data(), tag, 4) ||
        little<std::uint32_t>(directory, 4) != record_size) {
        return std::nullopt;
    }
    const auto count = little<std::uint64_t>(directory, 8);
    if (count > (directory.size() - 16) / record_size ||
        directory.size() != 16 + count * record_size) {
        return std::nullopt;
    }
    return count;
}

Result<std::vector<std::byte>> read_raw(const std::filesystem::path& path,
                                        std::uint64_t offset, std::uint64_t length) {
    if (length > std::numeric_limits<std::size_t>::max()) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::invalid_extent);
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return Result<std::vector<std::byte>>::failure(ErrorCode::io_error);
    stream.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> data(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (stream.gcount() != static_cast<std::streamsize>(data.size())) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::truncated_file);
    }
    return Result<std::vector<std::byte>>::success(std::move(data));
}

TensorRecord decode_tensor(std::span<const std::byte> value) {
    TensorRecord result{};
    result.tensor_id = little<std::uint64_t>(value, 0);
    result.dtype = little<std::uint16_t>(value, 12);
    result.quantization = little<std::uint16_t>(value, 14);
    result.rank = little<std::uint8_t>(value, 16);
    result.layer_id = static_cast<std::int32_t>(little<std::uint32_t>(value, 20));
    result.expert_id = static_cast<std::int32_t>(little<std::uint32_t>(value, 24));
    for (std::size_t index = 0; index < 4; ++index) result.dimensions[index] = little<std::uint64_t>(value, 32 + index * 8);
    result.data_offset = little<std::uint64_t>(value, 64);
    result.data_length = little<std::uint64_t>(value, 72);
    result.logical_length = little<std::uint64_t>(value, 80);
    result.auxiliary_offset = little<std::uint64_t>(value, 88);
    result.auxiliary_length = little<std::uint64_t>(value, 96);
    result.data_crc32c = little<std::uint32_t>(value, 104);
    result.auxiliary_crc32c = little<std::uint32_t>(value, 108);
    return result;
}
}

std::uint64_t fnv1a64(const char* value) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    while (*value) {
        hash ^= static_cast<unsigned char>(*value++);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

Result<Reader> Reader::open(const std::filesystem::path& path, VerifyMode mode) {
    std::error_code filesystem_error;
    const auto file_length = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || file_length < superblock_bytes) {
        return Result<Reader>::failure(ErrorCode::truncated_file);
    }
    auto raw_block = read_raw(path, 0, superblock_bytes);
    if (!raw_block) return Result<Reader>::failure(raw_block.error());
    const auto block = std::span(raw_block.value());
    constexpr std::array<char, 8> magic{'K','3','X','C','H','K','P','T'};
    if (std::memcmp(block.data(), magic.data(), magic.size()) != 0) {
        return Result<Reader>::failure(ErrorCode::bad_magic);
    }
    if (little<std::uint16_t>(block, 8) != 1 || little<std::uint16_t>(block, 10) != 0 ||
        little<std::uint32_t>(block, 12) != superblock_bytes ||
        little<std::uint32_t>(block, 16) != 4096) {
        return Result<Reader>::failure(ErrorCode::unsupported_version);
    }
    if (crc32c(block.first(4092)) != little<std::uint32_t>(block, 4092)) {
        return Result<Reader>::failure(ErrorCode::superblock_crc_mismatch);
    }
    if (little<std::uint64_t>(block, 24) != 0) {
        return Result<Reader>::failure(ErrorCode::unsupported_required_feature);
    }
    if (!all_zero(block.subspan(232, 4092 - 232))) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    Reader reader;
    reader.path_ = path;
    reader.superblock_.state = little<std::uint32_t>(block, 20);
    reader.superblock_.required_features = little<std::uint64_t>(block, 24);
    std::uint64_t* values[] = {&reader.superblock_.tensor_directory_offset,
                      &reader.superblock_.tensor_directory_length,
                      &reader.superblock_.layer_directory_offset,
                      &reader.superblock_.layer_directory_length,
                      &reader.superblock_.expert_directory_offset,
                      &reader.superblock_.expert_directory_length,
                      &reader.superblock_.model_config_offset,
                      &reader.superblock_.model_config_length,
                      &reader.superblock_.payload_offset,
                      &reader.superblock_.file_length};
    for (std::size_t index = 0; index < 10; ++index) *values[index] = little<std::uint64_t>(block, 88 + index * 8);
    std::copy(block.begin() + 168, block.begin() + 200, reader.superblock_.directory_sha256.begin());
    std::copy(block.begin() + 200, block.begin() + 232, reader.superblock_.root_sha256.begin());
    if (reader.superblock_.state != 1 || reader.superblock_.file_length != file_length) {
        return Result<Reader>::failure(ErrorCode::truncated_file);
    }
    for (const auto [offset, length] : {
             std::pair{reader.superblock_.tensor_directory_offset, reader.superblock_.tensor_directory_length},
             std::pair{reader.superblock_.layer_directory_offset, reader.superblock_.layer_directory_length},
             std::pair{reader.superblock_.expert_directory_offset, reader.superblock_.expert_directory_length},
             std::pair{reader.superblock_.model_config_offset, reader.superblock_.model_config_length}}) {
        if (offset % 4096 || !range_valid(offset, length, file_length)) {
            return Result<Reader>::failure(ErrorCode::invalid_extent);
        }
    }
    auto tensor_directory = read_raw(path, reader.superblock_.tensor_directory_offset,
                                     reader.superblock_.tensor_directory_length);
    auto layer_directory = read_raw(path, reader.superblock_.layer_directory_offset,
                                    reader.superblock_.layer_directory_length);
    auto expert_directory = read_raw(path, reader.superblock_.expert_directory_offset,
                                     reader.superblock_.expert_directory_length);
    auto config = read_raw(path, reader.superblock_.model_config_offset,
                           reader.superblock_.model_config_length);
    if (!tensor_directory || !layer_directory || !expert_directory || !config ||
        config.value().size() != model_config_bytes) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    const auto directory = std::span(tensor_directory.value());
    const auto tensor_count = directory_count(directory, "TENS", tensor_record_bytes);
    const auto layer_bytes = std::span(layer_directory.value());
    const auto layer_count = directory_count(layer_bytes, "LAYR", layer_record_bytes);
    const auto expert_bytes = std::span(expert_directory.value());
    const auto expert_count = directory_count(expert_bytes, "EXPT", expert_record_bytes);
    if (!tensor_count || !layer_count || !expert_count ||
        little<std::uint32_t>(config.value(), 8) != *layer_count ||
        !all_zero(std::span(config.value()).subspan(112)) ||
        (little<std::uint32_t>(config.value(), 108) & ~3U)) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{
        {0, superblock_bytes},
        {reader.superblock_.tensor_directory_offset,
         reader.superblock_.tensor_directory_offset + reader.superblock_.tensor_directory_length},
        {reader.superblock_.layer_directory_offset,
         reader.superblock_.layer_directory_offset + reader.superblock_.layer_directory_length},
        {reader.superblock_.expert_directory_offset,
         reader.superblock_.expert_directory_offset + reader.superblock_.expert_directory_length},
        {reader.superblock_.model_config_offset,
         reader.superblock_.model_config_offset + reader.superblock_.model_config_length},
    };
    std::unordered_set<std::uint64_t> tensor_ids;
    for (std::size_t index = 0; index < *tensor_count; ++index) {
        const auto raw_record = directory.subspan(16 + index * tensor_record_bytes, tensor_record_bytes);
        auto record = decode_tensor(raw_record);
        if (record.rank > 4 || (record.dtype != 1 && record.dtype != 2) ||
            record.quantization > 1 || little<std::uint32_t>(raw_record, 8) != 0 ||
            little<std::uint8_t>(raw_record, 17) != 0 ||
            little<std::uint16_t>(raw_record, 18) != 0 ||
            little<std::uint32_t>(raw_record, 28) != 0 ||
            !all_zero(raw_record.subspan(112)) || !tensor_ids.insert(record.tensor_id).second) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (std::size_t dimension = record.rank; dimension < 4; ++dimension) {
            if (record.dimensions[dimension]) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
        const bool plain = record.dtype == 1 && record.quantization == 0 &&
                           record.auxiliary_length == 0 && record.auxiliary_offset == 0 &&
                           record.auxiliary_crc32c == 0;
        const bool mxfp4 = record.dtype == 2 && record.quantization == 1 &&
                           record.auxiliary_length != 0;
        if (!record.data_length || !record.logical_length || (!plain && !mxfp4)) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (const auto [offset, length] : {std::pair{record.data_offset, record.data_length},
                                           std::pair{record.auxiliary_offset, record.auxiliary_length}}) {
            if (!length) {
                if (offset) return Result<Reader>::failure(ErrorCode::invalid_extent);
            } else {
                if (offset % 4096 || !range_valid(offset, length, file_length)) {
                    return Result<Reader>::failure(ErrorCode::invalid_extent);
                }
                ranges.emplace_back(offset, offset + length);
            }
        }
        reader.tensors_.push_back(record);
    }
    const auto configured_experts = little<std::uint32_t>(config.value(), 48);
    for (const auto& tensor : reader.tensors_) {
        if (tensor.layer_id < -1 || tensor.layer_id >= static_cast<std::int32_t>(*layer_count) ||
            tensor.expert_id < -1 || tensor.expert_id >= static_cast<std::int32_t>(configured_experts) ||
            (tensor.expert_id >= 0 && tensor.layer_id < 0)) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
    }
    for (std::size_t index = 0; index < *layer_count; ++index) {
        const auto record = layer_bytes.subspan(16 + index * layer_record_bytes, layer_record_bytes);
        const auto layer_index = little<std::uint32_t>(record, 0);
        const auto attention_kind = little<std::uint16_t>(record, 4);
        const auto ffn_kind = little<std::uint16_t>(record, 6);
        const auto first_tensor = little<std::uint32_t>(record, 8);
        const auto count = little<std::uint32_t>(record, 12);
        const auto first_expert = little<std::uint32_t>(record, 16);
        const auto experts = little<std::uint32_t>(record, 20);
        if (layer_index != index || (attention_kind != 1 && attention_kind != 2) ||
            (ffn_kind != 1 && ffn_kind != 2) || little<std::uint32_t>(record, 28) != 0 ||
            !all_zero(record.subspan(32)) || first_tensor > *tensor_count ||
            count > *tensor_count - first_tensor || first_expert > *expert_count ||
            experts > *expert_count - first_expert) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (std::size_t tensor = first_tensor; tensor < first_tensor + count; ++tensor) {
            if (reader.tensors_[tensor].layer_id != static_cast<std::int32_t>(index)) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    std::unordered_set<std::uint32_t> physical_orders;
    for (std::size_t index = 0; index < *expert_count; ++index) {
        const auto record = expert_bytes.subspan(16 + index * expert_record_bytes, expert_record_bytes);
        const auto layer = little<std::uint32_t>(record, 0);
        const auto expert = little<std::uint32_t>(record, 4);
        const auto physical_order = little<std::uint32_t>(record, 8);
        if (layer >= *layer_count || expert >= configured_experts ||
            physical_order >= *expert_count || !physical_orders.insert(physical_order).second ||
            little<std::uint32_t>(record, 12) != 0 ||
            !all_zero(record.subspan(48))) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (const auto offset : {16U, 24U, 32U}) {
            const auto id = little<std::uint64_t>(record, offset);
            const auto tensor = std::find_if(reader.tensors_.begin(), reader.tensors_.end(),
                                             [id](const auto& item) { return item.tensor_id == id; });
            if (tensor == reader.tensors_.end() || tensor->layer_id != static_cast<std::int32_t>(layer) ||
                tensor->expert_id != static_cast<std::int32_t>(expert) || tensor->quantization != 1) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    for (std::size_t layer = 0; layer < *layer_count; ++layer) {
        const auto record = layer_bytes.subspan(16 + layer * layer_record_bytes, layer_record_bytes);
        const auto first = little<std::uint32_t>(record, 16);
        const auto count = little<std::uint32_t>(record, 20);
        for (std::size_t expert = first; expert < first + count; ++expert) {
            if (little<std::uint32_t>(expert_bytes, 16 + expert * expert_record_bytes) != layer) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first < ranges[index - 1].second) {
            return Result<Reader>::failure(ErrorCode::invalid_extent);
        }
    }
    std::copy(config.value().begin(), config.value().end(), reader.model_config_.begin());
    if (mode == VerifyMode::checksums) {
        for (const auto& record : reader.tensors_) {
            auto data = read_raw(path, record.data_offset, record.data_length);
            if (!data || crc32c(data.value()) != record.data_crc32c) {
                return Result<Reader>::failure(ErrorCode::data_crc_mismatch);
            }
            if (record.auxiliary_length) {
                auto auxiliary = read_raw(path, record.auxiliary_offset, record.auxiliary_length);
                if (!auxiliary || crc32c(auxiliary.value()) != record.auxiliary_crc32c) {
                    return Result<Reader>::failure(ErrorCode::auxiliary_crc_mismatch);
                }
            }
        }
        std::vector<std::byte> directories;
        for (const auto* item : {&tensor_directory.value(), &layer_directory.value(),
                                 &expert_directory.value(), &config.value()}) {
            directories.insert(directories.end(), item->begin(), item->end());
        }
        if (sha256(directories) != reader.superblock_.directory_sha256) {
            return Result<Reader>::failure(ErrorCode::directory_sha256_mismatch);
        }
        std::ifstream root_stream(path, std::ios::binary);
        if (!root_stream) return Result<Reader>::failure(ErrorCode::io_error);
        Sha256Hasher root_hasher;
        std::vector<std::byte> root_chunk(1024 * 1024);
        std::uint64_t position = 0;
        while (position < file_length) {
            const auto requested = static_cast<std::size_t>(
                std::min<std::uint64_t>(root_chunk.size(), file_length - position));
            root_stream.read(reinterpret_cast<char*>(root_chunk.data()), requested);
            if (root_stream.gcount() != static_cast<std::streamsize>(requested)) {
                return Result<Reader>::failure(ErrorCode::truncated_file);
            }
            for (const auto [start, end] : {std::pair{200ULL, 232ULL},
                                                  std::pair{4092ULL, 4096ULL}}) {
                const auto left = std::max(start, position);
                const auto right = std::min(end, position + requested);
                if (left < right) {
                    std::fill(root_chunk.begin() + (left - position),
                              root_chunk.begin() + (right - position), std::byte{0});
                }
            }
            root_hasher.update(std::span(root_chunk).first(requested));
            position += requested;
        }
        if (root_hasher.finish() != reader.superblock_.root_sha256) {
            return Result<Reader>::failure(ErrorCode::root_sha256_mismatch);
        }
    }
    return Result<Reader>::success(std::move(reader));
}

Result<std::vector<std::byte>> Reader::read_extent(std::uint64_t offset, std::uint64_t length) const {
    ++counters_.calls;
    counters_.requested_bytes += length;
    auto result = read_raw(path_, offset, length);
    if (result) counters_.completed_bytes += result.value().size();
    return result;
}

Result<std::vector<std::byte>> Reader::read_tensor(std::uint64_t tensor_id) const {
    const auto item = std::find_if(tensors_.begin(), tensors_.end(),
                                   [tensor_id](const auto& value) { return value.tensor_id == tensor_id; });
    if (item == tensors_.end()) return Result<std::vector<std::byte>>::failure(ErrorCode::tensor_not_found);
    return read_extent(item->data_offset, item->data_length);
}

Result<std::vector<std::byte>> Reader::read_auxiliary(std::uint64_t tensor_id) const {
    const auto item = std::find_if(tensors_.begin(), tensors_.end(),
                                   [tensor_id](const auto& value) { return value.tensor_id == tensor_id; });
    if (item == tensors_.end()) return Result<std::vector<std::byte>>::failure(ErrorCode::tensor_not_found);
    return read_extent(item->auxiliary_offset, item->auxiliary_length);
}
}
