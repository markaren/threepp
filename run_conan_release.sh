#!/usr/bin/env bash

conan install . -s build_type=Release -if cmake-build-release -o with_bullet=True -b missing
