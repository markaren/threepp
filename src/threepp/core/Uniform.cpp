
#include "threepp/core/Uniform.hpp"

#include <utility>

using namespace threepp;

Uniform::Uniform(std::optional<UniformValue> value, std::optional<bool> needsUpdate)
    : needsUpdate(needsUpdate), value_(std::move(value)) {}

bool Uniform::hasValue() const {

    return value_.has_value();
}

UniformValue& Uniform::value() {

    return *value_;
}

void Uniform::setValue(UniformValue value) {

    this->value_ = std::move(value);
}
