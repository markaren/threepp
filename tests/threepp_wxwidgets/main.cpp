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


class MyApp : public wxApp {
public:
    MyApp() {}
    bool OnInit() wxOVERRIDE;
};

class OpenGLCanvas;

class MyFrame : public wxFrame {
public:
    MyFrame(const wxString &title);

private:
    OpenGLCanvas *openGLCanvas{nullptr};
};

class OpenGLCanvas : public wxGLCanvas {
public:
    OpenGLCanvas(MyFrame *parent, const wxGLAttributes &canvasAttrs);

    void OnPaint(wxPaintEvent &event);
    void OnSize(wxSizeEvent &event);

    wxColour triangleColor{wxColour(255, 128, 51)};

private:
    std::unique_ptr<wxGLContext> openGLContext;

    //////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<GLRenderer> renderer;
    std::shared_ptr<Scene> scene;
    std::shared_ptr<PerspectiveCamera> camera;

    //////////////////////////////////////////////////////////////////////////////
};

wxIMPLEMENT_APP(MyApp);

bool MyApp::OnInit() {
    if (!wxApp::OnInit())
        return false;

    MyFrame *frame = new MyFrame("Hello ThreePP + wxWidgets");
    frame->Show(true);

    return true;
}

MyFrame::MyFrame(const wxString &title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        openGLCanvas = new OpenGLCanvas(this, vAttrs);
        openGLCanvas->SetMinSize(FromDIP(wxSize(640, 480)));
        sizer->Add(openGLCanvas, 1, wxEXPAND);
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

    SetCurrent(*openGLContext);

    //////////////////////////////////////////////////////////////////////////////////////

    auto viewPortSize = GetSize() * GetContentScaleFactor();
    WindowSize size{viewPortSize.x, viewPortSize.y};
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
}


void OpenGLCanvas::OnPaint(wxPaintEvent &WXUNUSED(event)) {
    wxPaintDC dc(this);
    SetCurrent(*openGLContext);

    renderer->clear();
    renderer->render(*scene, *camera);

    SwapBuffers();
}

void OpenGLCanvas::OnSize(wxSizeEvent &event) {


    auto viewPortSize = event.GetSize() * GetContentScaleFactor();
    glViewport(0, 0, viewPortSize.x, viewPortSize.y);

    WindowSize size{viewPortSize.x, viewPortSize.y};

    camera->aspect = size.aspect();
    camera->updateProjectionMatrix();
    renderer->setSize(size);

    event.Skip();
}
