
set(publicHeaders
        "threepp/math/matrix3.hpp"
        "threepp/math/matrix4.hpp"
        "threepp/math/vector2.hpp"
        "threepp/math/vector3.hpp"
        "threepp/math/triangle.hpp"
        )

set(publicHeadersFull)
foreach (header IN LISTS publicHeaders)
    list(APPEND publicHeadersFull "${publicHeaderDir}/${header}")
endforeach ()

set(sources
        )

set(privateHeaders
        )


#add_library(math ${sources} ${privateHeaders} ${publicHeadersFull})
