#!/usr/bin/env bash

conan install . -s build_type=Release -if cmake-build-release -o with_bullet=True -o with_assimp=True -o with_imgui=True -b missing
read -p "Press any key to continue "
