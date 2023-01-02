#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <algorithm>
#include <cuda_runtime_api.h>

/// @brief Rounds up a number to the next number divisible by the given denominator. If the number is already divisible by the denominator it remains the same
/// @param val number to round up
/// @param roundup_denominator has to be positive
/// @tparam Integer has to be integer
template <typename Integer>
__host__ __device__ Integer roundup_next_multiple(const Integer val,
                                                  int32_t roundup_denominator)
{
    static_assert(std::is_integral<Integer>::value, "Expected an integer");
    assert(roundup_denominator > 0);

    const Integer remainder = val % roundup_denominator;

    if (remainder == 0)
    {
        return val;
    }

    if (val > 0)
    {
        // for value 11 and denomintor 4 remainder is 3 so 11 - 3 + 4 = 8 + 4 = 12
        return val - remainder + roundup_denominator;
    }
    else
    {
        // remainder is negative is this case, i.e. for value -11 and denominator 4 remainder is -3,
        // so -11 - (-3) = -11 + 3 = -8
        return val - remainder;
    }
}