
#ifndef THREEPP_PARTICLESWARMOPTIMIZATION_HPP
#define THREEPP_PARTICLESWARMOPTIMIZATION_HPP

#include "Optimizer.hpp"

class Particle {
public:
    Candidate position;
    Candidate localBestPosition;

    explicit Particle(const Candidate& candidate)
        : position(candidate), localBestPosition(candidate), velocity(candidate.size()) {

        for (auto& v : velocity) {
            v = rand() * 2 - 1;
        }
    }

    void update(float omega, float c1, float c2, const std::vector<float>& globalBest, const Problem& problem) {

        for (int i = 0; i < position.size(); i++) {
            float r1 = rand();
            float r2 = rand();

            velocity[i] = omega * velocity[i] + c1 * r1 * (localBestPosition[i] - position[i]) + c2 * r2 * (globalBest[i] - position[i]);
            velocity[i] = std::clamp(velocity[i], velocityBounds_.lower(), velocityBounds_.upper());

            position[i] += velocity[i];
            position[i] = std::clamp(position[i], 0.0f, 1.0f);
        }

        float cost = problem.eval(position.data());
        position.setCost(cost);
        if (cost < localBestPosition.cost()) {
            localBestPosition = position;
        }
    }

private:
    std::vector<float> velocity;
    Constraint velocityBounds_{-0.5, 0.5};

    static float rand() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> randDis{0, 1};

        return randDis(gen);
    }
};

class ParticleSwarmOptimization: public Optimizer {

public:
    explicit ParticleSwarmOptimization(size_t populationSize = 30, float omega = 0.9, float c1 = 1.41, float c2 = 1.41)
        : populationSize_(populationSize), omega_(omega), c1_(c1), c2_(c2) {}

    [[nodiscard]] size_t size() const override {

        return populationSize_;
    }

    [[nodiscard]] const Candidate& getCandiateAt(size_t index) const override {

        return particles.at(index).localBestPosition;
    }

    void init(const Problem& problem) override {
        particles.clear();

        for (int i = 0; i < populationSize_; i++) {

            particles.emplace_back(problem.generateCandidate());
        }
    }

    const Candidate& step(const Problem& problem) override {

        auto* best = &getBest();

        for (int i = 0; i < populationSize_; i++) {
            Particle& particle = particles[i];

            particle.update(omega_, c1_, c2_, best->position.data(), problem);
            if (particle.localBestPosition.cost() < best->localBestPosition.cost()) {
                best = &particle;
            }
        }

        return best->localBestPosition;
    }

private:
    size_t populationSize_;
    float omega_;
    float c1_;
    float c2_;

    std::vector<Particle> particles;

    [[nodiscard]] const Particle& getBest() const {
        std::pair<int, float> best{0, std::numeric_limits<float>::max()};
        for (int i = 0; i < populationSize_; i++) {
            if (particles[i].localBestPosition.cost() < best.second) {
                best = {i, particles[i].localBestPosition.cost()};
            }
        }

        return particles[best.first];
    }
};

#endif//THREEPP_PARTICLESWARMOPTIMIZATION_HPP
