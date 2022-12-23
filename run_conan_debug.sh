#!/usr/bin/env bash

conan install . -s build_type=Debug -if=cmake-build-debug -o with_bullet=True -o with_assimp=True -o with_imgui=True -o with_kine=True -b missing
read -p "Press any key to continue "
