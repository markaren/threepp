
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"
#include "utility/PID.hpp"

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

    auto scene = Scene::create();

    auto light = HemisphereLight::create();
    scene->add(light);

    auto grid = GridHelper::create(100, 100, 100);
    grid->position.y = -1;
    scene->add(grid);

    auto ball = Mesh::create(SphereGeometry::create(), MeshPhongMaterial::create({{"color", Color::green}}));
    scene->add(ball);

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

    // simulate a pan-tilt mechanism with camera attached
    Object3D yRot;
    yRot.position.z = 5;
    Object3D zRot;
    zRot.position.y = 1;
    PerspectiveCamera camera(70, canvas.aspect(), 0.1f, 1000);

    yRot.add(zRot);
    zRot.add(camera);
    scene->add(yRot);

    PID pid1(0.1, 0.001, 0.0001);// horizontal
    PID pid2(0.1, 0.001, 0.0001);// vertical

    pid1.setWindupGuard(10.f);
    pid2.setWindupGuard(10.f);

    float maxRotationSpeed = 2;// deg

    Clock clock;
    canvas.animate([&] {
        auto dt = clock.getDelta();
        ball->position.x = 1 * std::sin(math::TWO_PI * clock.elapsedTime * 0.1f);
        ball->position.y = 2 * std::sin(math::TWO_PI * clock.elapsedTime * 0.5f);

        renderer.render(*scene, camera);
        renderer.readPixels({0, 0}, size, Format::RGB, image.data);

        auto pos = colorDetection(image, lower_hsv_range, upper_hsv_range, mask);
        imshow(windowTitle, mask);
        if (waitKey(1) == 'q') {
            canvas.close();
        }

        if (pos) {

            // want (0,0) to be center
            int xZeroed = pos->x - size.width / 2;
            int yZeroed = pos->y - size.height / 2;

            auto gain1 = pid1.regulate(0, static_cast<float>(xZeroed), dt);
            auto gain2 = pid2.regulate(0, static_cast<float>(yZeroed), dt);

            yRot.rotation.y += gain1 * math::degToRad(maxRotationSpeed);
            zRot.rotation.x += gain2 * math::degToRad(maxRotationSpeed);

            std::stringstream ss;
            ss << "Detected ball at pos (" << xZeroed << ", " << yZeroed << ")";

            handle2.setText(ss.str());
            handle1.setPosition(pos->x, pos->y);

            renderer.resetState();
            textRenderer.render();
        }
    });
}
