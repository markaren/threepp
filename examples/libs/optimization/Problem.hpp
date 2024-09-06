
#ifndef THREEPP_PROBLEM_HPP
#define THREEPP_PROBLEM_HPP

#include <cassert>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

class Constraint {

public:
    Constraint(float lower, float upper)
        : lower_(lower), upper_(upper) {}

    [[nodiscard]] float lower() const {
        return lower_;
    }
    [[nodiscard]] float upper() const {
        return upper_;
    }

    [[nodiscard]] float normalize(float value) const {
        return (value - lower_) / (upper_ - lower_);
    }

    [[nodiscard]] float denormalize(float value) const {
        return (value) * (upper_ - lower_) / 1 + lower_;
    }

private:
    float lower_;
    float upper_;
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

    void set(const std::vector<float>& candidate, float cost) {
        candidate_ = candidate;
        cost_ = cost;
    }

    void setCost(float cost) {
        cost_ = cost;
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
    Problem(int dimensions, const Constraint& bounds)
        : Problem(dimensions, std::vector<Constraint>(dimensions, bounds)) {}

    Problem(int dimensions, std::vector<Constraint> bounds)
        : dimensions_(dimensions),
          bounds_(std::move(bounds)) {}

    [[nodiscard]] int dimensions() const {

        return dimensions_;
    }

    [[nodiscard]] std::vector<float> normalize(const std::vector<float>& candidate) const {

        assert(candidate.size() == dimensions_);

        std::vector<float> normalized(dimensions_);
        for (auto i = 0; i < dimensions_; i++) {
            float value = candidate[i];
            normalized[i] = bounds_[i].normalize(value);
        }

        return normalized;
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

        auto candidate = std::vector<float>(dimensions_);
        for (auto& value : candidate) {
            value = dis(gen);
        }

        return {candidate, eval(candidate)};
    }

    [[nodiscard]] virtual std::vector<std::vector<float>> solutions() const = 0;

    virtual ~Problem() = default;

protected:
    [[nodiscard]] virtual float evaluate(const std::vector<float>& candidate) const = 0;

private:
    int dimensions_;
    std::vector<Constraint> bounds_;
};

#endif//THREEPP_PROBLEM_HPP
