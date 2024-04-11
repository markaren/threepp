
#ifndef THREEPP_OPTIMIMIZER_HPP
#define THREEPP_OPTIMIMIZER_HPP

#include "Problem.hpp"

class Optimizer {

public:

    virtual void init(const Problem& problem) = 0;

    virtual const Candidate& step(const Problem& problem) = 0;

    virtual ~Optimizer() = default;

};

#endif//THREEPP_OPTIMIMIZER_HPP
