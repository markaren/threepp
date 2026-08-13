add_example(NAME "ardupilot_sitl" SOURCES main.cpp LINK_IMGUI LINK_PHYSX)

# add_example bails without creating the target under Emscripten (no sockets
# there anyway) and when PhysX is missing, hence the guard. POSIX sockets need
# no extra library.
if (TARGET ardupilot_sitl AND WIN32)
    target_link_libraries(ardupilot_sitl PRIVATE ws2_32)
endif ()
