
if (NOT AGX_DIR)
    set(AGX_DIR $ENV{AGX_DIR})
endif ()

if (AGX_DIR)

    set(AGX_INCLUDE_DIRS ${AGX_DIR}/include)
    if (NOT WIN32)
        set(AGX_LIBRARY_DIR ${AGX_DIR}/lib)
    else ()
        set(AGX_LIBRARY_DIR ${AGX_DIR}/lib/x64)
    endif ()

    IF (CMAKE_BUILD_TYPE MATCHES Debug)
        set(AGX_LIBRARIES agxSabred agxModeld agxCored agxPhysicsd)
    else ()
        set(AGX_LIBRARIES agxSabre agxModel agxCore agxPhysics)
    endif ()

    message(${AGX_INCLUDE_DIRS})
    message(${AGX_LIBRARY_DIR})


    add_executable(agx_test agx_test.cpp)
    target_include_directories(agx_test PUBLIC ${AGX_INCLUDE_DIRS})
    target_link_directories(agx_test PUBLIC ${AGX_LIBRARY_DIR})
    target_link_libraries(agx_test PUBLIC threepp ${AGX_LIBRARIES})

else()

    message("AGX_DIR not found")

endif ()
