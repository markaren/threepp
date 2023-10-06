
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <glad/glad.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace threepp;


int main() {

    Canvas canvas("OpenGL demo");
    auto size = canvas.size();
    GLRenderer renderer{size};
    renderer.setClearColor(Color::aliceblue);

    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto scene = Scene::create();

    auto ball = Mesh::create(SphereGeometry::create(), MeshBasicMaterial::create({{"color", Color::blue}}));
    scene->add(ball);

//    std::mutex m;
//    bool stop = false;
//    std::condition_variable cv;
    std::string windowTitle = "OpenCV";
    namedWindow(windowTitle, WINDOW_AUTOSIZE);
    Mat image(size.height, size.width, CV_8UC3);
//    std::atomic_bool ready = false;
//    std::thread cvThread([&] {
//        std::unique_lock<std::mutex> lck(m);
//        cv.wait(lck, [&] {
//            return ready.load();
//        });
//        while (!stop) {
//
//            imshow(windowTitle, image);
//            waitKey(1);
//        }
//    });

    OrbitControls controls{*camera, canvas};

    canvas.animate([&] {
        renderer.render(*scene, *camera);

//        std::lock_guard<std::mutex> lck(m);
        glReadPixels(0, 0, size.width, size.height, GL_RGB, GL_UNSIGNED_BYTE, image.data);
        flip(image, image, 0);
        cvtColor(image, image, cv::COLOR_BGR2RGB);
        imshow(windowTitle, image);
        waitKey(1);
//        ready = true;
    });

//    stop = true;
//    if (cvThread.joinable()) cvThread.join();
}
