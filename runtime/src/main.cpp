// K3X synthetic runtime을 실행하고 bounded JSON metrics를 기록합니다.
#include "k3x/model.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    std::filesystem::path model_path, output_path;
    std::string prompt_text, mode = "incremental";
    std::size_t count = 0;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--model") model_path = value;
        else if (key == "--prompt-ids") prompt_text = value;
        else if (key == "--generate") count = std::stoull(value);
        else if (key == "--mode") mode = value;
        else if (key == "--json") output_path = value;
        else { std::cerr << "unknown argument: " << key << '\n'; return 2; }
    }
    std::vector<std::uint32_t> prompt;
    std::stringstream parser(prompt_text);
    std::string item;
    while (std::getline(parser, item, ',')) prompt.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    auto reader = k3x::Reader::open(model_path, k3x::VerifyMode::metadata_only);
    if (!reader) { std::cerr << reader.message() << '\n'; return 3; }
    auto result = k3x::generate_greedy(reader.value(), prompt, count, mode == "incremental");
    if (!result) { std::cerr << result.message() << '\n'; return 4; }
    std::ofstream output(output_path);
    if (!output) return 5;
    output << "{\"decode_nanoseconds\":" << result.value().decode_nanoseconds
           << ",\"prefill_nanoseconds\":" << result.value().prefill_nanoseconds
           << ",\"read_bytes\":" << reader.value().counters().completed_bytes
           << ",\"read_calls\":" << reader.value().counters().calls
           << ",\"per_layer_nanoseconds\":[";
    for (std::size_t index = 0; index < result.value().per_layer_nanoseconds.size(); ++index) {
        if (index) output << ',';
        output << result.value().per_layer_nanoseconds[index];
    }
    output << "],\"token_ids\":[";
    for (std::size_t index = 0; index < result.value().token_ids.size(); ++index) {
        if (index) output << ',';
        output << result.value().token_ids[index];
    }
    output << "]}\n";
    return 0;
}
