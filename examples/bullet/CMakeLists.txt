
add_executable(hello_bullet hello_bullet.cpp BulletEngine.cpp)
target_link_libraries(hello_bullet PUBLIC threepp PRIVATE CONAN_PKG::bullet3)
