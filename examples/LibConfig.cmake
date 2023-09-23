find_package(imgui CONFIG)
find_package(Bullet CONFIG)
find_package(CURL CONFIG QUIET)
find_package(assimp CONFIG QUIET)
find_package(pugixml CONFIG QUIET)
find_package(nlohmann_json CONFIG QUIET)

if (BULLET_FOUND AND NOT TARGET Bullet::Bullet)
    add_library(Bullet::Bullet INTERFACE IMPORTED)
    if("${BULLET_INCLUDE_DIRS}" MATCHES ".*/bullet$")
        target_include_directories(Bullet::Bullet INTERFACE "${BULLET_INCLUDE_DIRS}/..")
    endif()
    target_link_libraries(Bullet::Bullet INTERFACE ${BULLET_LIBRARIES})
endif ()

if (imgui_FOUND)

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
