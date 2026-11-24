add_example(NAME "PelicanBike" SOURCES main.cpp LINK_IMGUI)
# the noise and texture workshop of the watermill (headers only)
target_include_directories(PelicanBike PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../Watermill)
