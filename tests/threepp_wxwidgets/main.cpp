#include "threepp/threepp.hpp"

using namespace threepp;

#include <wx/glcanvas.h>
#include <wx/wx.h>

namespace {

    auto createBox() {

        const auto boxGeometry = BoxGeometry::create();
        const auto boxMaterial = MeshBasicMaterial::create();
        boxMaterial->color.setRGB(1, 0, 0);
        boxMaterial->transparent = true;
        boxMaterial->opacity = 0.1f;
        auto box = Mesh::create(boxGeometry, boxMaterial);

        auto wiredBox = LineSegments::create(WireframeGeometry::create(*boxGeometry));
        wiredBox->material()->as<LineBasicMaterial>()->depthTest = false;
        wiredBox->material()->as<LineBasicMaterial>()->color = Color::gray;
        box->add(wiredBox);

        return box;
    }

    auto createSphere() {

        const auto sphereGeometry = SphereGeometry::create(0.5f);
        const auto sphereMaterial = MeshBasicMaterial::create();
        sphereMaterial->color.setHex(0x00ff00);
        sphereMaterial->wireframe = true;
        auto sphere = Mesh::create(sphereGeometry, sphereMaterial);
        sphere->position.setX(-1);

        return sphere;
    }

    auto createPlane() {

        const auto planeGeometry = PlaneGeometry::create(5, 5);
        const auto planeMaterial = MeshBasicMaterial::create();
        planeMaterial->color.setHex(Color::yellow);
        planeMaterial->transparent = true;
        planeMaterial->opacity = 0.5f;
        planeMaterial->side = Side::Double;
        auto plane = Mesh::create(planeGeometry, planeMaterial);
        plane->position.setZ(-2);

        return plane;
    }

}// namespace


class ThreeppContext {

public:
    explicit ThreeppContext(PeripheralsEventSource &evt)
        : renderer(evt.size()), camera(PerspectiveCamera(75, evt.size().aspect())), orbitControls(camera, evt) {

        scene.background = Color::aliceblue;
        camera.position.z = 5;

        auto box = createBox();
        scene.add(box);

        auto sphere = createSphere();
        box->add(sphere);

        auto plane = createPlane();
        auto planeMaterial = plane->material()->as<MeshBasicMaterial>();
        scene.add(plane);
    }

    void loop() {
        static Clock clock;
        float dt = clock.getDelta();

        scene.children[0]->rotation.y += 1.f * dt;

        renderer.render(scene, camera);
    }

    void onWindowResize(WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    }

    void OnButtonClick(wxCommandEvent &) {

        scene.children[1]->material()->as<MaterialWithColor>()->color.randomize();
    }

private:
    GLRenderer renderer;
    Scene scene;
    PerspectiveCamera camera;
    OrbitControls orbitControls;
};

class OpenGLCanvas : public wxGLCanvas, public PeripheralsEventSource {
public:
    OpenGLCanvas(wxFrame *parent, const wxGLAttributes &canvasAttrs);

    void init();

    void OnPaint(wxPaintEvent &event);
    void OnSize(wxSizeEvent &event);
    void OnMouseMove(wxMouseEvent &event);
    void OnMousePress(wxMouseEvent &event);
    void OnMouseRelease(wxMouseEvent &event);
    void OnMouseWheel(wxMouseEvent &event);
    void OnInternalIdle() override;

    [[nodiscard]] WindowSize size() const override;

private:
    std::unique_ptr<ThreeppContext> threeppContext;
    std::unique_ptr<wxGLContext> openGLContext;
};


OpenGLCanvas::OpenGLCanvas(wxFrame *parent, const wxGLAttributes &canvasAttrs)
    : wxGLCanvas(parent, canvasAttrs){}

void OpenGLCanvas::init() {
    wxGLContextAttrs ctxAttrs;
    ctxAttrs.PlatformDefaults().CoreProfile().OGLVersion(3, 3).EndList();
    openGLContext = std::make_unique<wxGLContext>(this, nullptr, &ctxAttrs);

    Bind(wxEVT_PAINT, &OpenGLCanvas::OnPaint, this);
    Bind(wxEVT_SIZE, &OpenGLCanvas::OnSize, this);

    Bind(wxEVT_MOTION, &OpenGLCanvas::OnMouseMove, this);
    Bind(wxEVT_LEFT_DOWN, &OpenGLCanvas::OnMousePress, this);
    Bind(wxEVT_RIGHT_DOWN, &OpenGLCanvas::OnMousePress, this);
    Bind(wxEVT_LEFT_UP, &OpenGLCanvas::OnMouseRelease, this);
    Bind(wxEVT_RIGHT_UP, &OpenGLCanvas::OnMouseRelease, this);
    Bind(wxEVT_MOUSEWHEEL, &OpenGLCanvas::OnMouseWheel, this);

    SetCurrent(*openGLContext);

    threeppContext = std::make_unique<ThreeppContext>(*this);

    //yes, raw pointer
    auto button = new wxButton(this, wxID_ANY, "Click Me", wxPoint(10, 10), wxSize(150, 30));
    button->Bind(wxEVT_BUTTON, &ThreeppContext::OnButtonClick, threeppContext.get());
}

void OpenGLCanvas::OnPaint(wxPaintEvent &WXUNUSED(event)) {
    wxPaintDC dc(this);
    threeppContext->loop();
    SwapBuffers();
}

void OpenGLCanvas::OnInternalIdle() {
    wxWindow::OnInternalIdle();
    Refresh(false);
}

WindowSize OpenGLCanvas::size() const {
    auto viewPortSize = GetSize() * GetContentScaleFactor();
    return {viewPortSize.x, viewPortSize.y};
}

void OpenGLCanvas::OnSize(wxSizeEvent &event) {
    auto viewPortSize = event.GetSize() * GetContentScaleFactor();
    WindowSize size{viewPortSize.x, viewPortSize.y};
    threeppContext->onWindowResize(size);

    event.Skip();
}

void OpenGLCanvas::OnMouseMove(wxMouseEvent &event) {
    wxPoint pos = event.GetPosition();
    Vector2 mousePos(static_cast<float>(pos.x), static_cast<float>(pos.y));
    onMouseMoveEvent(mousePos);

    event.Skip();
}

void OpenGLCanvas::OnMousePress(wxMouseEvent &event) {
    int buttonFlag = event.GetButton();
    wxPoint pos = event.GetPosition();
    int button = 0;
    if (wxMOUSE_BTN_LEFT == buttonFlag) {
        button = 0;
    } else if (wxMOUSE_BTN_RIGHT == buttonFlag) {
        button = 1;
    }
    Vector2 p{pos.x, pos.y};
    onMousePressedEvent(button, p, PeripheralsEventSource::MouseAction::PRESS);

    event.Skip();
}

void OpenGLCanvas::OnMouseRelease(wxMouseEvent &event) {
    int buttonFlag = event.GetButton();
    wxPoint pos = event.GetPosition();
    int button = 0;
    if (wxMOUSE_BTN_LEFT == buttonFlag) {
        button = 0;
    } else if (wxMOUSE_BTN_RIGHT == buttonFlag) {
        button = 1;
    }
    Vector2 p{pos.x, pos.y};
    onMousePressedEvent(button, p, PeripheralsEventSource::MouseAction::RELEASE);

    event.Skip();
}

void OpenGLCanvas::OnMouseWheel(wxMouseEvent &event) {
    int direction = event.GetWheelRotation() / 120;// 1 or -1
    int xoffset = 0;
    int yoffset = direction;

    onMouseWheelEvent({static_cast<float>(xoffset), static_cast<float>(yoffset)});

    event.Skip();
}

class MyFrame : public wxFrame {
public:
    explicit MyFrame(const wxString &title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
        auto sizer = std::make_unique<wxBoxSizer>(wxVERTICAL);// deleted by window

        wxGLAttributes vAttrs;
        vAttrs.PlatformDefaults().Defaults().EndList();

        if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
            openGLCanvas = std::make_unique<OpenGLCanvas>(this, vAttrs);
            openGLCanvas->SetMinSize(FromDIP(wxSize(640, 480)));
            sizer->Add(openGLCanvas.get(), 1, wxEXPAND);
        }

        SetSizerAndFit(sizer.release());
    }

    bool Show(bool show) override {
        auto status = wxFrame::Show(show);
        openGLCanvas->init();
        return status;
    }

private:
    std::unique_ptr<OpenGLCanvas> openGLCanvas;
};


class MyApp : public wxApp {
public:
    bool OnInit() override {
        if (!wxApp::OnInit()) {
            return false;
        }

        // yes, raw pointer
        auto frame = new MyFrame("Hello ThreePP + wxWidgets");
        frame->Show(true);

        return true;
    }
};


int main(int argc, char **argv) {
    // yes, raw pointer
    auto app = new MyApp();

    // Initialize wxWidgets
    wxApp::SetInstance(app);
    wxEntryStart(argc, argv);

    // Run the application
    int exitCode = wxEntry(argc, argv);

    // Cleanup
    wxEntryCleanup();
    return exitCode;
}
