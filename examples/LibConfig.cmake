find_package(imgui CONFIG)
find_package(Bullet CONFIG)
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
