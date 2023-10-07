
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <opencv2/opencv.hpp>

#include <sstream>

using namespace cv;
using namespace threepp;

namespace {

    std::optional<cv::Point> colorDetection(Mat& image, cv::Scalar lower_hsv_range, cv::Scalar upper_hsv_range, Mat& mask) {
        flip(image, image, 0);
        cvtColor(image, image, cv::COLOR_BGR2HSV);

        cv::inRange(image, lower_hsv_range, upper_hsv_range, mask);

        std::vector<cv::Point> locations;
        cv::findNonZero(mask, locations);

        if (!locations.empty()) {

            cv::Point meanPoint;
            for (const auto& point : locations) {
                meanPoint += point;
            }
            meanPoint.x /= static_cast<int>(locations.size());
            meanPoint.y /= static_cast<int>(locations.size());

            int crossSize = 10;
            cv::Scalar crossColor(0, 255, 0);
            cv::line(mask, cv::Point(meanPoint.x - crossSize / 2, meanPoint.y),
                     cv::Point(meanPoint.x + crossSize / 2, meanPoint.y),
                     crossColor, 2, 8, 0);
            cv::line(mask, cv::Point(meanPoint.x, meanPoint.y - crossSize / 2),
                     cv::Point(meanPoint.x, meanPoint.y + crossSize / 2),
                     crossColor, 2, 8, 0);

            return meanPoint;
        }

        return std::nullopt;
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

    Mat mask;
    cv::Scalar lower_hsv_range(60, 120, 70);
    cv::Scalar upper_hsv_range(120, 255, 255);

    TextRenderer textRenderer;
    auto& handle1 = textRenderer.createHandle("+");
    handle1.verticalAlignment = TextHandle::VerticalAlignment::CENTER;
    handle1.horizontalAlignment = TextHandle::HorizontalAlignment::CENTER;
    handle1.scale = 2;
    auto& handle2 = textRenderer.createHandle();

    Clock clock;
    float A = 2;
    float f = 0.1f;

    canvas.animate([&] {
        auto dt = clock.getDelta();
        ball->position.x = A * std::sin(2 * math::PI * clock.elapsedTime * f);

        renderer.render(*scene, *camera);
        renderer.readPixels({0, 0}, size, Format::RGB, image.data);

        auto pos = colorDetection(image, lower_hsv_range, upper_hsv_range, mask);
        imshow(windowTitle, mask);
        if (waitKey(1) == 'q') {
            canvas.close();
        }

        if (pos) {
            std::stringstream ss;
            ss << "Detected ball at pos (" << pos->x << ", " << pos->y << ")";

            handle2.setText(ss.str());
            handle1.setPosition(pos->x, pos->y);

            renderer.resetState();
            textRenderer.render();
        }
    });
}
