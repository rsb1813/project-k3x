// 공식 두 레이어를 모델 순서대로 실행하고 독립 KDA 상태를 유지합니다.
#include "k3x/official_two_layer.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace k3x {

Result<OfficialTwoLayerResult> official_two_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    std::span<const OfficialTwoLayerWeights> layers,
    std::span<const OfficialKdaState> initial_states,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta) {
    if (inputs.size() != 2 || layers.size() != 2 ||
        initial_states.size() != 2 || layers[0].layer_id != 1 ||
        layers[1].layer_id != 2) {
        return Result<OfficialTwoLayerResult>::failure(
            ErrorCode::invalid_state);
    }

    std::array<OfficialKdaState, 2> states{
        initial_states[0], initial_states[1]};
    std::vector<OfficialTwoLayerStepResult> steps;
    steps.reserve(4);
    std::array<std::vector<float>, 2> outputs;
    for (std::size_t position = 0; position < inputs.size(); ++position) {
        OfficialLayerInput current = inputs[position];
        for (std::size_t layer_index = 0; layer_index < layers.size();
             ++layer_index) {
            auto result = official_layer_cpu(
                std::span<const OfficialLayerInput>(&current, 1),
                layers[layer_index].weights,
                states[layer_index],
                config,
                top_k,
                situ_beta,
                situ_linear_beta);
            if (!result) {
                return Result<OfficialTwoLayerResult>::failure(
                    result.error(), result.message());
            }
            if (result.value().steps.size() != 1) {
                return Result<OfficialTwoLayerResult>::failure(
                    ErrorCode::invalid_state);
            }
            states[layer_index] = std::move(result.value().kda.state);
            current.hidden_input = result.value().steps[0].moe.output;
            steps.push_back({
                position,
                layers[layer_index].layer_id,
                std::move(result.value().steps[0]),
            });
        }
        outputs[position] = std::move(current.hidden_input);
    }
    return Result<OfficialTwoLayerResult>::success({
        std::move(states), std::move(steps), std::move(outputs)});
}

}  // namespace k3x
