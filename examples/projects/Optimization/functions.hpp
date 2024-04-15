
#ifndef THREEPP_FUNCTIONS_HPP
#define THREEPP_FUNCTIONS_HPP

#include "optimization/Problem.hpp"

#include <numbers>

struct Ackleys: Problem {

    Ackleys(): Problem(2, {-5, 5}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return -20 * std::exp(-0.2 * std::sqrt(0.5 * (x * x + z * z))) - std::exp(0.5 * (std::cos(2 * std::numbers::pi * x) + std::cos(2 * std::numbers::pi * z))) + 20 + std::exp(1);
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{0, 0}};
    }
};

struct Rosenbruck: Problem {

    Rosenbruck(): Problem(2, {{-2, 2}, {-1, 3}}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        float a = 1, b = 100;
        return ((a - x) * (a - x)) + b * ((z - (x * x)) * (z - (x * x)));
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{1, 1}};
    }
};

struct HolderTable: Problem {

    HolderTable(): Problem(2, {-10, 10}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return -std::abs(std::sin(x) * std::cos(z) * std::exp(std::abs(1 - std::sqrt(x * x + z * z) / std::numbers::pi)));
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{8.055, 9.66}, {-8.055, 9.66}, {8.055, -9.66}, {-8.055, -9.66}};
    }
};

struct Rastrigin: Problem {

    Rastrigin(): Problem(2, {-5.12, 5.12}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return 20 + x * x - 10 * std::cos(2 * std::numbers::pi * x) + z * z - 10 * std::cos(2 * std::numbers::pi * z);
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{0, 0}};
    }
};

struct Himmelblaus: Problem {

    Himmelblaus(): Problem(2, {-5, 5}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return std::pow(x * x + z - 11, 2) + std::pow(x + z * z - 7, 2);
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{3, 2}, {-2.805118, 3.131312}, {-3.779310, -3.283186}, {3.584428, -1.848126}};
    }
};

struct Beale: Problem {

    Beale(): Problem(2, {-4.5, 4.5}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return std::pow(1.5 - x + x * z, 2) + std::pow(2.25 - x + x * z * z, 2) + std::pow(2.625 - x + x * z * z * z, 2);
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{3, 0.5}};
    }
};

struct ThreeHumpCamel: Problem {

    ThreeHumpCamel(): Problem(2, {-5, 5}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return 2 * x * x - 1.05 * x * x * x * x + x * x * x * x * x * x / 6 + x * z + z * z;
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{0, 0}};
    }
};

struct Booth: Problem {

    Booth(): Problem(2, {-10, 10}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return (x + 2 * z - 7) * (x + 2 * z - 7) + (2 * x + z - 5) * (2 * x + z - 5);
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{1, 3}};
    }
};

struct EggHolder: Problem {

    EggHolder(): Problem(2, {-512, 512}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return -(z + 47) * std::sin(std::sqrt(std::abs(z + x / 2 + 47))) - x * std::sin(std::sqrt(std::abs(x - (z + 47))));
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{512, 404.2319}};
    }
};

struct GoldsteinPrice: Problem {

    GoldsteinPrice(): Problem(2, {-2, 2}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return (1 + (x + z + 1) * (x + z + 1) * (19 - 14 * x + 3 * x * x - 14 * z + 6 * x * z + 3 * z * z)) *
               (30 + (2 * x - 3 * z) * (2 * x - 3 * z) * (18 - 32 * x + 12 * x * x + 48 * z - 36 * x * z + 27 * z * z));
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{0, -1}};
    }
};

struct SphereFunction: Problem {

    SphereFunction(): Problem(2, {-5.12, 5.12}) {}

    [[nodiscard]] float evaluate(const std::vector<float>& candidate) const override {

        float x = candidate[0];
        float z = candidate[1];

        return x * x + z * z;
    }

    [[nodiscard]] std::vector<std::vector<float>> solutions() const override {
        return {{0, 0}};
    }
};

#endif//THREEPP_FUNCTIONS_HPP
