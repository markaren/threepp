
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace threepp;

namespace {

    Mat colorDetection(Mat& image, cv::Scalar lower_hsv_range, cv::Scalar upper_hsv_range) {
        flip(image, image, 0);
        cvtColor(image, image, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(image, lower_hsv_range, upper_hsv_range, mask);

        std::vector<cv::Point> locations;
        cv::findNonZero(mask, locations);

        if (!locations.empty()) {

            cv::Point mean_point;
            for (const auto& point : locations) {
                mean_point += point;
            }
            mean_point.x /= locations.size();
            mean_point.y /= locations.size();

            int crossSize = 10;
            cv::Scalar crossColor(0, 255, 0);
            cv::line(mask, cv::Point(mean_point.x - crossSize / 2, mean_point.y),
                     cv::Point(mean_point.x + crossSize / 2, mean_point.y),
                     crossColor, 2, 8, 0);
            cv::line(mask, cv::Point(mean_point.x, mean_point.y - crossSize / 2),
                     cv::Point(mean_point.x, mean_point.y + crossSize / 2),
                     crossColor, 2, 8, 0);

        }
        return mask;
    }

}// namespace

int main() {

    Canvas canvas("OpenGL demo", {{"aa", 4}, {"resizable", false}});
    auto size = canvas.size();
    GLRenderer renderer{size};

    auto camera = PerspectiveCamera::create(70, canvas.aspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto scene = Scene::create();

    auto light = HemisphereLight::create();
    scene->add(light);

    auto grid = GridHelper::create(100, 100, 100);
    grid->position.y = -1;
    scene->add(grid);

    auto ball = Mesh::create(SphereGeometry::create(), MeshPhongMaterial::create({{"color", Color::green}}));
    scene->add(ball);

    OrbitControls controls{*camera, canvas};

    std::string windowTitle = "OpenCV";
    namedWindow(windowTitle, WINDOW_AUTOSIZE);
    Mat image(size.height, size.width, CV_8UC3);

    cv::Scalar lower_hsv_range(60, 120, 70);
    cv::Scalar upper_hsv_range(120, 255, 255);

    Clock clock;
    float A = 2;
    float f = 0.1f;
    canvas.animate([&] {
        auto dt = clock.getDelta();
        ball->position.x = A * std::sin(2 * math::PI * clock.elapsedTime * f);

        renderer.render(*scene, *camera);
        renderer.readPixels({0, 0}, size, Format::RGB, image.data);

        Mat mask = colorDetection(image, lower_hsv_range, upper_hsv_range);
        imshow(windowTitle, mask);
        if (waitKey(1) == 'q') {
            canvas.close();
        }
    });
}
