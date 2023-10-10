
#ifndef THREEPP_FPSCOUNTER_HPP
#define THREEPP_FPSCOUNTER_HPP

struct FPSCounter {

    int fps = 0;

    void update(float currentTime) {
        nbFrames++;
        if (currentTime - lastTime >= 1) {
            fps = nbFrames;
            nbFrames = 0;
            lastTime += 1;
        }
    }

private:
    int nbFrames = 0;
    float lastTime = 0;
};

#endif//THREEPP_FPSCOUNTER_HPP
