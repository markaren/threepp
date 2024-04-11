
#ifndef THREEPP_DIFFERENTIALEVOLUTION_HPP
#define THREEPP_DIFFERENTIALEVOLUTION_HPP

#include "Optimizer.hpp"

class DifferentialEvolution: public Optimizer {
public:
    explicit DifferentialEvolution(size_t pop_size = 100, float mut_factor = 0.2, float cross_prob = 0.9)
        : populationSize(pop_size), differentialWeight(mut_factor), crossoverProbability(cross_prob) {}

    void init(const Problem& problem) override {
        population.clear();
        for (int i = 0; i < populationSize; i++) {
            population.emplace_back(problem.generateCandidate());
        }
    }

    const std::vector<Candidate>& getPopulation() const {
        return population;
    }

    const Candidate& step(const Problem& problem) override {

        for (int i = 0; i < populationSize; ++i) {
            Candidate& candidate = population[i];
            const auto mutated_data = mutate(i);
            const auto cost = problem.eval(mutated_data);
            if (cost < candidate.cost()) {
                candidate = {mutated_data, cost};
            }
        }

        // Find the best individual
        int best_index = 0;
        for (auto i = 1; i < populationSize; i++) {
            if (population[i].cost() < population[best_index].cost()) {
                best_index = i;
            }
        }

        return population[best_index];
    }

private:
    size_t populationSize;
    float differentialWeight;
    float crossoverProbability;
    std::vector<Candidate> population;

    std::vector<float> mutate(int target_index) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, populationSize - 1);

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

        auto mutated_individual = population[target_index].data();

        std::uniform_real_distribution<float> rand_dis(0.0, 1.0);
        const float rand_val = rand_dis(gen);
        if (rand_val < crossoverProbability) {

            for (int i = 0; i < mutated_individual.size(); i++) {
                mutated_individual[i] += differentialWeight * (population[r1][i] - population[r2][i]) + differentialWeight * (population[r3][i] - population[target_index][i]);
                mutated_individual[i] = std::clamp(mutated_individual[i], 0.f, 1.f);
            }
        }

        return mutated_individual;
    }
};

#endif//THREEPP_DIFFERENTIALEVOLUTION_HPP
