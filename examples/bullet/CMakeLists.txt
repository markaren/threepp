
add_library(bullet_integration BulletEngine.cpp)
target_link_libraries(bullet_integration PUBLIC threepp Bullet::Bullet)

add_executable(hello_bullet hello_bullet.cpp)
target_link_libraries(hello_bullet PRIVATE bullet_integration)
