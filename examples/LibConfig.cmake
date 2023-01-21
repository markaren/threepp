find_package(imgui CONFIG)
find_package(Bullet CONFIG)

if (BULLET_FOUND AND NOT TARGET Bullet::Bullet)
    add_library(Bullet::Bullet INTERFACE IMPORTED)
    if("${BULLET_INCLUDE_DIRS}" MATCHES ".*/bullet$")
        target_include_directories(Bullet::Bullet INTERFACE "${BULLET_INCLUDE_DIRS}/..")
    endif()
    target_link_libraries(Bullet::Bullet INTERFACE ${BULLET_LIBRARIES})
endif ()

if (imgui_FOUND)

    if(EXISTS "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        set(imgui_bindings "${CMAKE_BINARY_DIR}/_deps/imgui_glfw")
        add_library(imgui_glfw "${imgui_bindings}/imgui_impl_glfw.cpp" "${imgui_bindings}/imgui_impl_opengl3.cpp")
        target_link_libraries(imgui_glfw PUBLIC glad::glad glfw::glfw imgui::imgui)
        target_include_directories(imgui_glfw PUBLIC "${imgui_bindings}")

        set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_LINK_LIBRARIES imgui_glfw)
    endif()

    set_property(TARGET imgui::imgui APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS HAS_IMGUI)

endif ()

if (NOT TARGET "matplotlib::matplotlib")

    find_package(Python3 COMPONENTS Interpreter Development)
    if (Python3_FOUND)
        find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
        if (MATPLOTLIB_CPP_INCLUDE_DIRS)
            add_library(matplotlib::matplotlib INTERFACE IMPORTED)
            target_compile_definitions(matplotlib::matplotlib INTERFACE WITHOUT_NUMPY HAS_MATPLOTLIB)
            target_link_libraries(matplotlib::matplotlib INTERFACE Python3::Python)
            target_include_directories(matplotlib::matplotlib INTERFACE "${MATPLOTLIB_CPP_INCLUDE_DIRS}")
            set(matplotlib_FOUND TRUE)
        else()
            set(matplotlib_FOUND FALSE)
        endif()
    endif()

endif()


include(FetchContent)
FetchContent_Declare(
        glText
        GIT_REPOSITORY git@github.com:vallentin/glText.git
        GIT_TAG 8200fa70e32acec0a3cd777d404f41ee0c203ca4
)
FetchContent_Populate(glText)
add_library(gltext INTERFACE)
add_library(gltext::gltext ALIAS gltext)
target_include_directories(gltext INTERFACE "${gltext_SOURCE_DIR}")
