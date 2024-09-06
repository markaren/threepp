
#ifndef THREEPP_DIFFERENTIALEVOLUTION_HPP
#define THREEPP_DIFFERENTIALEVOLUTION_HPP

#include "Optimizer.hpp"

class DifferentialEvolution: public Optimizer {
public:
    explicit DifferentialEvolution(size_t pop_size = 100, float mut_factor = 0.2, float cross_prob = 0.9)
        : populationSize(pop_size), differentialWeight(mut_factor), crossoverProbability(cross_prob) {}

    [[nodiscard]] size_t size() const override {
        return populationSize;
    }

    [[nodiscard]] const Candidate& getCandiateAt(size_t index) const override {
        return population.at(index);
    }

    void init(const Problem& problem) override {
        population.clear();
        for (int i = 0; i < populationSize; i++) {
            population.emplace_back(problem.generateCandidate());
        }
    }

    const Candidate& step(const Problem& problem) override {

        std::pair<int, float> best{0, std::numeric_limits<float>::max()};
        for (int i = 0; i < populationSize; ++i) {
            Candidate& candidate = population[i];
            const auto mutatedData = mutate(i);
            const auto mutatedCost = problem.eval(mutatedData);
            if (mutatedCost < candidate.cost()) {
                candidate.set(mutatedData, mutatedCost);
            }
            if (candidate.cost() < best.second) {
                best = {i, candidate.cost()};
            }
        }

        return population[best.first];
    }

private:
    size_t populationSize;
    float differentialWeight;
    float crossoverProbability;
    std::vector<Candidate> population;

    std::vector<float> mutate(int target_index) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, static_cast<int>(populationSize) - 1);
        static std::uniform_real_distribution<float> randDis{0.0, 1.0};

        int r1, r2, r3;
        do {
            r1 = dis(gen);
        } while (r1 == target_index);
        do {
            r2 = dis(gen);
        } while (r2 == target_index || r2 == r1);
        do {
            r3 = dis(gen);
        } while (r3 == target_index || r3 == r1 || r3 == r2);

        auto mutatedIndividual = population[target_index].data();

        const float randVal = randDis(gen);
        if (randVal < crossoverProbability) {

            for (int i = 0; i < mutatedIndividual.size(); i++) {
                mutatedIndividual[i] += differentialWeight * (population[r1][i] - population[r2][i]) + differentialWeight * (population[r3][i] - population[target_index][i]);
                mutatedIndividual[i] = std::clamp(mutatedIndividual[i], 0.f, 1.f);
            }
        }

        return mutatedIndividual;
    }
};

#endif//THREEPP_DIFFERENTIALEVOLUTION_HPP
