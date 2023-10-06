
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

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

    OrbitControls controls{*camera, canvas};

    auto ball = Mesh::create(SphereGeometry::create(), MeshBasicMaterial::create({{"color", Color::blue}}));
    scene->add(ball);

    std::string windowTitle = "OpenCV";
    namedWindow(windowTitle, WINDOW_AUTOSIZE);
    Mat image(size.height, size.width, CV_8UC3);

    canvas.animate([&] {
        renderer.render(*scene, *camera);

        renderer.readPixels({0,0}, size, RGBFormat, image.data);

        flip(image, image, 0);
        cvtColor(image, image, cv::COLOR_BGR2RGB);
        imshow(windowTitle, image);
        if (waitKey(1) == 'q') {
            canvas.close();
        }

    });

}
