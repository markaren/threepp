# threepp vcpkg test

This is a standalone demo project that uses `threepp` as a dependency.

In order to update `threepp`, replace the baseline value in [vcpkg-configuration.json](vcpkg-configuration.json).
The baseline should point to a [commit](https://github.com/Ecos-platform/vcpkg-registry/commits/main) 
from the [custom vcpkg registry](https://github.com/Ecos-platform/vcpkg-registry) that hosts the port.

In the future, it could be added to the main vcpkg registry.

As `vcpkg` is used, you need to tell CMake about it in order for dependency resolution to work: 

`-DCMAKE_TOOLCHAIN_FILE=[path to vcpkg]/scripts/buildsystems/vcpkg.cmake`

###### Building under MinGW

Under MinGW you'll need to specify the vcpkg triplet:
```shell
-DVCPKG_TARGET_TRIPLET=x64-mingw-[static|dynamic]  # choose either `static` or `dynamic`.
-DVCPKG_HOST_TRIPLET=x64-mingw-[static|dynamic]    # <-- needed only if MSVC cannot be found. 
```
