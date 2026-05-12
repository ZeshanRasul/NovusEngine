#pragma once

#include <glm/glm.hpp>
#include <assimp/matrix4x4.h>

namespace Tools
{
    inline glm::mat4 convertAiToGLM(const aiMatrix4x4& m)
    {
        // aiMatrix4x4 is row-major; glm is column-major
        return glm::mat4(
            m.a1, m.b1, m.c1, m.d1,
            m.a2, m.b2, m.c2, m.d2,
            m.a3, m.b3, m.c3, m.d3,
            m.a4, m.b4, m.c4, m.d4
        );
    }
}
