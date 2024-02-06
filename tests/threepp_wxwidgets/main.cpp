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

class MyFrame;

class MyApp : public wxApp {
public:
    bool OnInit() wxOVERRIDE;

private:
    std::unique_ptr<MyFrame> frame;
};

class OpenGLCanvas;

class MyFrame : public wxFrame {
public:
    explicit MyFrame(const wxString &title);

private:
    std::unique_ptr<OpenGLCanvas> openGLCanvas;
};

class OpenGLCanvas : public wxGLCanvas, public PeripheralsEventSource {
public:
    OpenGLCanvas(MyFrame *parent, const wxGLAttributes &canvasAttrs);

    void setup();

    void OnPaint(wxPaintEvent &event);
    void OnSize(wxSizeEvent &event);
    void OnMouseMove(wxMouseEvent &event);
    void OnMousePress(wxMouseEvent &event);
    void OnMouseRelease(wxMouseEvent &event);
    void OnMouseWheel(wxMouseEvent &event);

    [[nodiscard]] virtual WindowSize size() const override;

private:
    std::unique_ptr<wxGLContext> openGLContext;

    //////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<GLRenderer> renderer;
    std::shared_ptr<Scene> scene;
    std::shared_ptr<PerspectiveCamera> camera;
    std::unique_ptr<OrbitControls> orbitControls;
    //////////////////////////////////////////////////////////////////////////////
};


wxIMPLEMENT_APP(MyApp);

bool MyApp::OnInit() {
    if (!wxApp::OnInit()) {
        return false;
    }

    frame = std::make_unique<MyFrame>("Hello ThreePP + wxWidgets");
    frame->Show(true);

    return true;
}


MyFrame::MyFrame(const wxString &title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL); // deleted by window

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        openGLCanvas = std::make_unique<OpenGLCanvas>(this, vAttrs);
        openGLCanvas->SetMinSize(FromDIP(wxSize(640, 480)));
        sizer->Add(openGLCanvas.get(), 1, wxEXPAND);
        openGLCanvas->setup();
    }

    SetSizerAndFit(sizer);
}

OpenGLCanvas::OpenGLCanvas(MyFrame *parent, const wxGLAttributes &canvasAttrs)
    : wxGLCanvas(parent, canvasAttrs) {

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
}

void OpenGLCanvas::setup() {
    const auto viewPortSize = GetSize() * GetContentScaleFactor();
    WindowSize size(viewPortSize.x, viewPortSize.y);
    renderer = std::make_shared<GLRenderer>(size);

    scene = Scene::create();
    scene->background = Color::aliceblue;
    camera = PerspectiveCamera::create(75, size.aspect(), 0.1f, 1000);
    camera->position.z = 5;

    auto box = createBox();
    scene->add(box);

    auto sphere = createSphere();
    box->add(sphere);

    auto plane = createPlane();
    auto planeMaterial = plane->material()->as<MeshBasicMaterial>();
    scene->add(plane);

    orbitControls = std::make_unique<OrbitControls>(*camera, *this);
}

void OpenGLCanvas::OnPaint(wxPaintEvent &WXUNUSED(event)) {

    renderer->render(*scene, *camera);

    SwapBuffers();
}

WindowSize OpenGLCanvas::size() const {
    auto viewPortSize = GetSize() * GetContentScaleFactor();
    return {viewPortSize.x, viewPortSize.y};
}

void OpenGLCanvas::OnSize(wxSizeEvent &event) {

    auto viewPortSize = event.GetSize() * GetContentScaleFactor();
    WindowSize size{viewPortSize.x, viewPortSize.y};

    camera->aspect = size.aspect();
    camera->updateProjectionMatrix();
    renderer->setSize(size);

    event.Skip();
}

void OpenGLCanvas::OnMouseMove(wxMouseEvent &event) {
    wxPoint pos = event.GetPosition();
    Vector2 mousePos(static_cast<float>(pos.x), static_cast<float>(pos.y));
    onMouseMoveEvent(mousePos);
    Refresh(false);
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
    Refresh(false);
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
    Refresh(false);
    event.Skip();
}

void OpenGLCanvas::OnMouseWheel(wxMouseEvent &event) {
    int direction = event.GetWheelRotation() / 120;// 1 or -1
    int xoffset = 0;
    int yoffset = direction;

    // call the PeripheralsEventSource's member function
    onMouseWheelEvent({static_cast<float>(xoffset), static_cast<float>(yoffset)});

    Refresh(false);
    event.Skip();
}
