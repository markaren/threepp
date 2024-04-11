//
// Created by Lars Ivar Hatledal on 11.04.2024.
//

#ifndef THREEPP_PROBLEM_HPP
#define THREEPP_PROBLEM_HPP

#include <cassert>
#include <utility>
#include <vector>
#include <cmath>
#include <random>

class Range {
    float lower;
    float upper;

public:
    Range(float lower, float upper)
        : lower(lower), upper(upper) {}

    [[nodiscard]] float normalize(float value) const {
        return (value - lower) / (upper - lower);
    }

    [[nodiscard]] float denormalize(float value) const {
        return (value) * (upper - lower) / 1 + lower;
    }
};

class Candidate {
public:
    Candidate(const std::vector<float>& candidate, float cost)
        : candidate_(candidate), cost_(cost) {}

    float& operator[](size_t index) {

        return candidate_[index];
    }

    float operator[](size_t index) const {

        return candidate_[index];
    }

    [[nodiscard]] size_t size() const {

        return candidate_.size();
    }

    [[nodiscard]] float cost() const {

        return cost_;
    }

    [[nodiscard]] std::vector<float> data() const {

        return candidate_;
    }

    bool operator<(const Candidate& other) const {

        return cost_ < other.cost_;
    }

private:
    float cost_;
    std::vector<float> candidate_;
};

class Problem {

public:
    Problem(int dimensions, Range bounds)
        : Problem(dimensions, std::vector<Range>(dimensions, bounds)) {}

    Problem(int dimensions, std::vector<Range> bounds)
        : dimensions_(dimensions),
          bounds_(std::move(bounds)) {}

    [[nodiscard]] int dimensions() const {

        return dimensions_;
    }

    [[nodiscard]] std::vector<float> denormalize(const std::vector<float>& candidate) const {

        assert(candidate.size() == dimensions_);

        std::vector<float> denormalized(dimensions_);
        for (auto i = 0; i < dimensions_; i++) {
            float value = candidate[i];
            denormalized[i] = bounds_[i].denormalize(value);
        }

        return denormalized;
    }

    [[nodiscard]] float eval(const std::vector<float>& candidate) const {

        return evaluate(denormalize(candidate));
    }

    [[nodiscard]] Candidate generateCandidate() const {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0, 1.0);

        auto candidate = std::vector<float>(dimensions_, dis(gen));
        auto fitness = eval(candidate);

        return {candidate, fitness};
    }

    virtual ~Problem() = default;

protected:
    [[nodiscard]] virtual float evaluate(const std::vector<float>& candidate) const = 0;

private:
    int dimensions_;
    std::vector<Range> bounds_;
};

#endif//THREEPP_PROBLEM_HPP
