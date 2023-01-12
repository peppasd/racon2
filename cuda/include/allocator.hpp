#pragma once

#include <memory>
#include <vector>


#include "cudautility.hpp"
#include "device_preallocated_allocator.cuh"
#include <cuda_runtime_api.h>

/// \brief Allocator that allocates device memory using cudaMalloc/cudaFree
template <typename T>
class CudaMallocAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// \brief default constructor
    /// \param default_stream if a call to allocate() does not specify any streams this stream will be used instead, ignored in this allocator
    explicit CudaMallocAllocator(cudaStream_t default_stream = 0)
    {
        static_cast<void>(default_stream);
    }

    /// \brief copy constructor
    /// \param rhs input allocator
    CudaMallocAllocator(const CudaMallocAllocator& rhs) = default;

    /// \brief copy constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    template <typename U>
    CudaMallocAllocator(const CudaMallocAllocator<U>& rhs)
    {
        static_cast<void>(rhs);
    }

    /// \brief copy assignment operator
    /// \param rhs input allocator
    /// \return reference to this object
    CudaMallocAllocator& operator=(const CudaMallocAllocator& rhs) = default;

    /// \brief copy assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    /// \return reference to this object
    template <typename U>
    CudaMallocAllocator& operator=(const CudaMallocAllocator<U>& rhs)
    {
        return *this;
    }

    /// \brief move constructor
    /// \param rhs input allocator
    CudaMallocAllocator(CudaMallocAllocator&& rhs) = default;

    /// \brief move constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    template <typename U>
    CudaMallocAllocator(CudaMallocAllocator<U>&& rhs)
    {
    }

    /// \brief move assignment operator
    /// \param rhs input allocator
    /// \return reference to this object
    CudaMallocAllocator& operator=(CudaMallocAllocator&& rhs) = default;

    /// \brief move assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    /// \return reference to this object
    template <typename U>
    CudaMallocAllocator& operator=(CudaMallocAllocator<U>&& rhs)
    {
        return *this;
    }

    /// \brief destructor
    ~CudaMallocAllocator() = default;

    /// \brief asynchronously allocates a device array with enough space for n elements of value_type
    /// \param n number of elements to allocate the array for
    /// \param streams CUDA streams to be associated with this allocation, ignored in this allocator
    /// \return pointer to allocated memory
    /// \throw device_memory_allocation_exception if allocation was not successful
    pointer allocate(std::size_t n,
                     const std::vector<cudaStream_t>& streams = {})
    {
        static_cast<void>(streams);
        void* ptr       = nullptr;
        cudaError_t err = cudaMalloc(&ptr, n * sizeof(T));
        
        return static_cast<pointer>(ptr);
    }

    /// \brief Asynchronously dealllocates allocated array
    /// \param p pointer to the array to deallocate
    /// \param n number of elements the array was allocated for
    void deallocate(pointer p, std::size_t n)
    {
        static_cast<void>(n);
        cudaFree(p);
    }

    /// \brief Get the size of the largest free memory block
    /// \return returns the size in bytes
    int64_t get_size_of_largest_free_memory_block() const
    {
        return find_largest_contiguous_device_memory_section();
    }
};

/// \brief A simple caching allocator for device memory allocations
/// \tparam T
/// \tparam MemoryResource resource that does actual allocation, e.g. cudautils::details::DevicePreallocatedAllocator
template <typename T, typename MemoryResource>
class CachingDeviceAllocator
{
public:
    /// type of elements of allocated array
    using value_type = T;

    /// pointer to elements of allocated array
    using pointer = T*;

    /// \brief Default constructor
    /// Constructs an invalid CachingDeviceAllocator to allow default-construction of containers.
    /// A container using this allocator needs obtain a non-default constructed CachingDeviceAllocator object before performing any allocations.
    /// This can be achieved through through container assignment for example.
    CachingDeviceAllocator()
        : default_stream_(0)
    {
    }

    /// \brief Constructor
    /// \param max_cached_bytes max bytes used by memory resource
    /// \param default_stream if a call to allocate() does not specify any streams this stream will be used instead
    explicit CachingDeviceAllocator(size_t max_cached_bytes,
                                    cudaStream_t default_stream = 0)
        : memory_resource_(std::make_shared<MemoryResource>(max_cached_bytes))
        , default_stream_(default_stream)
    {
    }

    /// \brief copy constructor
    /// \param rhs input allocator
    CachingDeviceAllocator(const CachingDeviceAllocator& rhs) = default;

    /// \brief copy constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(const CachingDeviceAllocator<U, MemoryResource>& rhs)
        : memory_resource_(rhs.memory_resource())
        , default_stream_(rhs.default_stream())
    {
    }

    /// \brief copy assignment operator
    /// \param rhs input allocator
    /// \return reference to this object
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator& rhs) = default;

    /// \brief copy assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    /// \return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(const CachingDeviceAllocator<U, MemoryResource>& rhs)
    {
        memory_resource_ = rhs.memory_resource();
        default_stream_  = rhs.default_stream();
        return *this;
    }

    /// \brief move constructor
    /// \param rhs input allocator
    CachingDeviceAllocator(CachingDeviceAllocator&& rhs) = default;

    /// \brief move constructor from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    template <typename U>
    CachingDeviceAllocator(CachingDeviceAllocator<U, MemoryResource>&& rhs)
        : memory_resource_(rhs.memory_resource())
        , default_stream_(rhs.default_stream())
    {
    }

    /// \brief move assignment operator
    /// \param rhs input allocator
    /// \return reference to this object
    CachingDeviceAllocator& operator=(CachingDeviceAllocator&& rhs) = default;

    /// \brief move assignement operator from an allocator with another value_type
    /// Internal state of allocator does not acutally depend on value_type so this is possible
    /// \param rhs input allocator
    /// \tparam U Type of rhs::value_type
    /// \return reference to this object
    template <typename U>
    CachingDeviceAllocator& operator=(CachingDeviceAllocator<U, MemoryResource>&& rhs)
    {
        memory_resource_ = rhs.memory_resource();
        default_stream_  = rhs.default_stream();
        return *this;
    }

    /// \brief destructor
    ~CachingDeviceAllocator() = default;

    /// \brief asynchronously allocates a device array with enough space for n elements of value_type
    /// \param n number of elements to allocate the array for
    /// \param streams on deallocation this memory block is guaranteed to live at least until all previously scheduled work in these streams has finished, if no streams are specified default_stream from constructor are used, if no default_stream was specified in constructor default stream is used
    /// \return pointer to allocated memory
    /// \throw device_memory_allocation_exception if allocation was not successful
    pointer allocate(std::size_t n,
                     const std::vector<cudaStream_t>& streams = {})
    {

        void* ptr       = nullptr;
        cudaError_t err = memory_resource_->DeviceAllocate(&ptr,
                                                           n * sizeof(T),
                                                           streams.empty() ? std::vector<cudaStream_t>(1, default_stream_) : streams); // if no streams have been specified use default_stream_
        if (err == cudaErrorMemoryAllocation)
        {
            // throw device_memory_allocation_exception();
            abort();
        }
        return static_cast<pointer>(ptr);
    }

    /// \brief Asynchronously deallocates allocated array, may call cudaStreamSynchronize() on associated streams or may defer that call to a later point
    /// \param p pointer to the array to deallocate
    /// \param n number of elements the array was allocated for
    void deallocate(pointer p, std::size_t n)
    {
        static_cast<void>(n);
        if (!memory_resource_)
        {
            std::abort();
        }
        // deallocate should not throw execeptions which is why GW_CU_CHECK_ERR is not used.
        memory_resource_->DeviceFree(p);
    }

    /// \brief Get the size of the largest free memory block
    /// \return returns the size in bytes
    int64_t get_size_of_largest_free_memory_block() const
    {
        return memory_resource_->get_size_of_largest_free_memory_block();
    }

    /// \brief returns a shared pointer to memory_resource
    /// \return a shared pointer to memory_resource
    std::shared_ptr<MemoryResource> memory_resource() const { return memory_resource_; }

    /// \brief returns default stream
    /// \return default stream
    cudaStream_t default_stream() const { return default_stream_; }

private:
    std::shared_ptr<MemoryResource> memory_resource_;
    cudaStream_t default_stream_;
};


using DefaultDeviceAllocator = CudaMallocAllocator<char>;

/// Gets the size of the largest free memory block in the allocator
///
/// \return returns the size in bytes
inline int64_t get_size_of_largest_free_memory_block(DefaultDeviceAllocator const& allocator)
{
    return allocator.get_size_of_largest_free_memory_block();
}

/// Constructs a DefaultDeviceAllocator
///
/// This function provides a way to construct a valid DefaultDeviceAllocator
/// for all possible DefaultDeviceAllocators.
/// Use this function to obtain a DefaultDeviceAllocator object.
/// This function is needed, since construction of CachingDeviceAllocator
/// requires a max_caching_size argument to obtain a valid allocator.
/// Default constuction of CachingDeviceAllocator yields an dummy object
/// which cannot allocate memory.
/// \param max_cached_bytes max bytes used by memory resource used by CachingDeviceAllocator (default: 2GiB, unused for CudaMallocAllocator)
/// \param default_stream if a call to allocate() does not specify any streams this stream will be used instead (unused for CudaMallocAllocator)
inline DefaultDeviceAllocator create_default_device_allocator(std::size_t max_caching_size = 2ull * 1024 * 1024 * 1024,
                                                              cudaStream_t default_stream  = 0)
{
    static_cast<void>(max_caching_size);
    static_cast<void>(default_stream);
    return DefaultDeviceAllocator();
}