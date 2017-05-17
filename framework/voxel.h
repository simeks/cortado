#pragma once

#include <stdint.h>

namespace voxel
{
    enum Type : uint8_t
    {
        Type_Unknown = 0,
        Type_Float,
        Type_Float2,
        Type_Float3,
        Type_Float4,
        Type_Double,
        Type_Double2,
        Type_Double3,
        Type_Double4,
        Type_UChar,
        Type_UChar2,
        Type_UChar3,
        Type_UChar4
    };
    
    /// Returns the total size in bytes of the specified type
    size_t size(uint8_t type);
    
    /// Returns the number of components of the specified type
    int num_components(uint8_t type);

    /// Returns the base type of a type, i.e. Float3 -> Float
    uint8_t base_type(uint8_t type);

}
