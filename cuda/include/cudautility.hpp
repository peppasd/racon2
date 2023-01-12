#pragma once

#include <cassert>
#include <string>
#include <memory>
#include <cuda_runtime_api.h>

std::size_t find_largest_contiguous_device_memory_section();



/// align
/// Alignment of memory chunks in cudapoa. Must be a power of two
/// \tparam IntType type of data to align
/// \tparam boundary Boundary to align to (NOTE: must be power of 2)
/// \param value Input value that is to be aligned
/// \return Value aligned to boundary
template <typename IntType, int32_t boundary>
__host__ __device__ __forceinline__
    IntType
    align(const IntType& value)
{
    static_assert((boundary & (boundary - 1)) == 0, "Boundary for align must be power of 2");
    return (value + boundary - 1) & ~(boundary - 1);
}



class scoped_device_switch
{
public:
    /// \brief Constructor
    ///
    /// \param device_id ID of CUDA device to switch to while class is in scope
    explicit scoped_device_switch(int32_t device_id)
    {
        cudaGetDevice(&device_id_before_);
        cudaSetDevice(device_id);
    }

    /// \brief Destructor switches back to original device ID
    ~scoped_device_switch()
    {
        cudaSetDevice(device_id_before_);
    }

    scoped_device_switch()                            = delete;
    scoped_device_switch(scoped_device_switch const&) = delete;
    scoped_device_switch& operator=(scoped_device_switch const&) = delete;

private:
    int32_t device_id_before_;
};