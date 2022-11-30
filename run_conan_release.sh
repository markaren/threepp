#!/usr/bin/env bash

conan install . -s build_type=Release -if cmake-build-release -o with_bullet=True -o with_assimp=True -b missing
