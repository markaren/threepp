
#ifndef THREEPP_OPTIMIMIZER_HPP
#define THREEPP_OPTIMIMIZER_HPP

#include "Problem.hpp"

class Optimizer {

public:
    [[nodiscard]] virtual size_t size() const = 0;

    [[nodiscard]] virtual const Candidate& getCandiateAt(size_t index) const = 0;

    virtual void init(const Problem& problem) = 0;

    virtual const Candidate& step(const Problem& problem) = 0;

    virtual ~Optimizer() = default;
};

#endif//THREEPP_OPTIMIMIZER_HPP
