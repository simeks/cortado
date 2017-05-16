#include <cuda_runtime.h>

#include <framework/gpu_volume.h>
#include <framework/helper_cuda.h>
#include <framework/profiler.h>
#include <framework/volume.h>

texture<float, 3, cudaReadModeElementType> volume_in;
surface<void, 3> volume_out;

__global__ void copy_kernel(dim3 volume_size)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= volume_size.x ||
        y >= volume_size.y ||
        z >= volume_size.z)
    {
        return;
    }

    float v = tex3D(volume_in, x, y, z);
    surf3Dwrite(v, volume_out, x*sizeof(float), y, z);
}

namespace sample
{
    void copy_volume(const Volume& in, Volume& out)
    {
        PROFILE_SCOPE("copy_volume");

        GpuVolume gpu_in, gpu_out;

        {
            PROFILE_SCOPE("alloc_and_upload");

            // Allocate memory and upload to GPU
            gpu_in = in.upload();
            gpu_out = gpu::allocate_volume(
                in.voxel_type(),
                in.size(),
                gpu::Flag_BindAsSurface);
        }

        // Bind arrays as textures and surfaces
        checkCudaErrors(cudaBindTextureToArray(volume_in, gpu_in.ptr, gpu_in.format_desc));
        checkCudaErrors(cudaBindSurfaceToArray(volume_out, gpu_out.ptr));

        // Parameters
        dim3 block_size(8, 8, 1);
        dim3 grid_size(in.size().width/8, in.size().height / 8, in.size().depth);

        // Launch kernel
        dim3 volume_size = { uint32_t(in.size().width), 
            uint32_t(in.size().height), uint32_t(in.size().depth) };
        copy_kernel<<<grid_size, block_size>>>(volume_size);
        getLastCudaError("");
        checkCudaErrors(cudaDeviceSynchronize());

        // Download results
        out.download(gpu_out);

        // Cleanup
        gpu::release_volume(gpu_in);
        gpu::release_volume(gpu_out);
    }
}
