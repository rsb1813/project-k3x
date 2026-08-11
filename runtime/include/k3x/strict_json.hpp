// 공식 fixture manifest를 위한 중복 키와 비유한 숫자를 거부하는 JSON 파서를 제공합니다.
#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace k3x::strict_json {

struct Value {
    using Array = std::vector<Value>;
    using Object = std::map<std::string, Value>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
};

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    std::optional<Value> parse() {
        auto result = parse_value();
        whitespace();
        return result && position_ == text_.size() ? result : std::nullopt;
    }

private:
    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) {
            ++position_;
        }
    }

    bool consume(char expected) {
        whitespace();
        if (position_ == text_.size() || text_[position_] != expected) return false;
        ++position_;
        return true;
    }

    std::optional<std::string> parse_string() {
        if (!consume('"')) return std::nullopt;
        std::string output;
        while (position_ < text_.size()) {
            const auto current = static_cast<unsigned char>(text_[position_++]);
            if (current == '"') return output;
            if (current < 0x20 || current >= 0x80) return std::nullopt;
            if (current != '\\') {
                output.push_back(static_cast<char>(current));
                continue;
            }
            if (position_ == text_.size()) return std::nullopt;
            const auto escaped = text_[position_++];
            if (escaped == '"' || escaped == '\\' || escaped == '/') {
                output.push_back(escaped);
            } else if (escaped == 'b') {
                output.push_back('\b');
            } else if (escaped == 'f') {
                output.push_back('\f');
            } else if (escaped == 'n') {
                output.push_back('\n');
            } else if (escaped == 'r') {
                output.push_back('\r');
            } else if (escaped == 't') {
                output.push_back('\t');
            } else {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }

    std::optional<Value> parse_number() {
        whitespace();
        const auto start = position_;
        if (position_ < text_.size() && text_[position_] == '-') ++position_;
        if (position_ == text_.size()) return std::nullopt;
        if (text_[position_] == '0') {
            ++position_;
        } else {
            if (text_[position_] < '1' || text_[position_] > '9') return std::nullopt;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const auto fraction = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == fraction) return std::nullopt;
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            const auto exponent = position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
            if (position_ == exponent) return std::nullopt;
        }
        double number{};
        const auto result = std::from_chars(
            text_.data() + start, text_.data() + position_, number);
        return result.ec == std::errc{} && std::isfinite(number)
            ? std::optional(Value{number})
            : std::nullopt;
    }

    std::optional<Value> parse_array() {
        if (!consume('[')) return std::nullopt;
        Value::Array output;
        whitespace();
        if (consume(']')) return Value{std::move(output)};
        while (true) {
            auto item = parse_value();
            if (!item) return std::nullopt;
            output.push_back(std::move(*item));
            if (consume(']')) return Value{std::move(output)};
            if (!consume(',')) return std::nullopt;
        }
    }

    std::optional<Value> parse_object() {
        if (!consume('{')) return std::nullopt;
        Value::Object output;
        whitespace();
        if (consume('}')) return Value{std::move(output)};
        while (true) {
            auto key = parse_string();
            if (!key || !consume(':')) return std::nullopt;
            auto item = parse_value();
            if (!item || !output.emplace(std::move(*key), std::move(*item)).second) {
                return std::nullopt;
            }
            if (consume('}')) return Value{std::move(output)};
            if (!consume(',')) return std::nullopt;
        }
    }

    std::optional<Value> parse_value() {
        whitespace();
        if (position_ == text_.size()) return std::nullopt;
        if (text_[position_] == '{') return parse_object();
        if (text_[position_] == '[') return parse_array();
        if (text_[position_] == '"') {
            auto value = parse_string();
            return value ? std::optional(Value{std::move(*value)}) : std::nullopt;
        }
        for (const auto& literal : {
                 std::pair{"true", Value{true}},
                 std::pair{"false", Value{false}},
                 std::pair{"null", Value{nullptr}}}) {
            const std::string_view name = literal.first;
            if (text_.substr(position_, name.size()) == name) {
                position_ += name.size();
                return literal.second;
            }
        }
        return parse_number();
    }

    std::string_view text_;
    std::size_t position_{};
};

inline const Value* member(const Value& value, std::string_view name) {
    const auto* object = std::get_if<Value::Object>(&value.value);
    if (!object) return nullptr;
    const auto found = object->find(std::string(name));
    return found == object->end() ? nullptr : &found->second;
}

inline const std::string* string(const Value* value) {
    return value ? std::get_if<std::string>(&value->value) : nullptr;
}

inline const Value::Array* array(const Value* value) {
    return value ? std::get_if<Value::Array>(&value->value) : nullptr;
}

inline const Value::Object* object(const Value* value) {
    return value ? std::get_if<Value::Object>(&value->value) : nullptr;
}

inline const double* number(const Value* value) {
    return value ? std::get_if<double>(&value->value) : nullptr;
}

}  // namespace k3x::strict_json
