#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CALC_HASH_KERNEL_NAME \
    "_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj"
#define REORDER_KERNEL_NAME \
    "_Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej"
#define DENSITY_KERNEL_NAME \
    "_Z19KernelFluidDensityD17PxgFixedSizeArrayIfES0_15DensityDebugGPUPiS_IK6float4ES_IKjES7_S7_S7_"
#define TEXTURE_PARTICLE_COUNT 64u
#define CELL_TEXTURE_COUNT 524288u
#define SORTED_POSITION_BYTES (TEXTURE_PARTICLE_COUNT * 4u * sizeof(uint32_t))
#define PARTICLE_HASH_BYTES (TEXTURE_PARTICLE_COUNT * sizeof(uint32_t))
#define DENSITY_CONSTANT_COPY_BYTES 1024u
#define EXPECTED_DENSITY_BITS 0x40000000u
#define EXPECTED_HASH (1u + (2u << 6u) + (3u << 12u))

typedef struct cuda_api {
    HMODULE library;
    CUresult(CUDAAPI *cuInit)(unsigned int);
    CUresult(CUDAAPI *cuDeviceGet)(CUdevice *, int);
    CUresult(CUDAAPI *cuDeviceGetName)(char *, int, CUdevice);
    CUresult(CUDAAPI *cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
    CUresult(CUDAAPI *cuCtxDestroy_v2)(CUcontext);
    CUresult(CUDAAPI *cuCtxSynchronize)(void);
    CUresult(CUDAAPI *cuModuleLoadData)(CUmodule *, const void *);
    CUresult(CUDAAPI *cuModuleUnload)(CUmodule);
    CUresult(CUDAAPI *cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult(CUDAAPI *cuModuleGetGlobal_v2)(CUdeviceptr *, size_t *, CUmodule, const char *);
    CUresult(CUDAAPI *cuModuleGetTexRef)(CUtexref *, CUmodule, const char *);
    CUresult(CUDAAPI *cuMemAlloc_v2)(CUdeviceptr *, size_t);
    CUresult(CUDAAPI *cuMemFree_v2)(CUdeviceptr);
    CUresult(CUDAAPI *cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    CUresult(CUDAAPI *cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
    CUresult(CUDAAPI *cuMemsetD32_v2)(CUdeviceptr, unsigned int, size_t);
    CUresult(CUDAAPI *cuLaunchKernel)(CUfunction,
                                      unsigned int,
                                      unsigned int,
                                      unsigned int,
                                      unsigned int,
                                      unsigned int,
                                      unsigned int,
                                      unsigned int,
                                      CUstream,
                                      void **,
                                      void **);
    CUresult(CUDAAPI *cuTexRefSetAddress_v2)(size_t *, CUtexref, CUdeviceptr, size_t);
    CUresult(CUDAAPI *cuTexRefSetFormat)(CUtexref, CUarray_format, int);
    CUresult(CUDAAPI *cuTexRefSetFlags)(CUtexref, unsigned int);
    CUresult(CUDAAPI *cuTexRefSetFilterMode)(CUtexref, CUfilter_mode);
    CUresult(CUDAAPI *cuTexRefSetAddressMode)(CUtexref, int, CUaddress_mode);
    CUresult(CUDAAPI *cuGetErrorName)(CUresult, const char **);
    CUresult(CUDAAPI *cuGetErrorString)(CUresult, const char **);
} cuda_api;

static bool load_function(HMODULE library, const char *name, void *output, size_t output_size)
{
    FARPROC raw = GetProcAddress(library, name);
    if (raw == NULL || output_size != sizeof(raw)) {
        fprintf(stderr, "missing CUDA export: %s\n", name);
        return false;
    }
    memcpy(output, &raw, output_size);
    return true;
}

#define LOAD_API(api, name)                                                             \
    do {                                                                                 \
        if (!load_function((api)->library, #name, &(api)->name, sizeof((api)->name))) { \
            return false;                                                                \
        }                                                                                \
    } while (0)

static bool load_cuda(cuda_api *api)
{
    memset(api, 0, sizeof(*api));
    api->library = LoadLibraryW(L"nvcuda.dll");
    if (api->library == NULL) {
        fprintf(stderr, "failed to load nvcuda.dll: %lu\n", (unsigned long)GetLastError());
        return false;
    }
    LOAD_API(api, cuInit);
    LOAD_API(api, cuDeviceGet);
    LOAD_API(api, cuDeviceGetName);
    LOAD_API(api, cuCtxCreate_v2);
    LOAD_API(api, cuCtxDestroy_v2);
    LOAD_API(api, cuCtxSynchronize);
    LOAD_API(api, cuModuleLoadData);
    LOAD_API(api, cuModuleUnload);
    LOAD_API(api, cuModuleGetFunction);
    LOAD_API(api, cuModuleGetGlobal_v2);
    LOAD_API(api, cuModuleGetTexRef);
    LOAD_API(api, cuMemAlloc_v2);
    LOAD_API(api, cuMemFree_v2);
    LOAD_API(api, cuMemcpyHtoD_v2);
    LOAD_API(api, cuMemcpyDtoH_v2);
    LOAD_API(api, cuMemsetD32_v2);
    LOAD_API(api, cuLaunchKernel);
    LOAD_API(api, cuTexRefSetAddress_v2);
    LOAD_API(api, cuTexRefSetFormat);
    LOAD_API(api, cuTexRefSetFlags);
    LOAD_API(api, cuTexRefSetFilterMode);
    LOAD_API(api, cuTexRefSetAddressMode);
    LOAD_API(api, cuGetErrorName);
    LOAD_API(api, cuGetErrorString);
    return true;
}

static void report_cuda_error(cuda_api *api, const char *operation, CUresult result)
{
    const char *name = NULL;
    const char *description = NULL;
    api->cuGetErrorName(result, &name);
    api->cuGetErrorString(result, &description);
    fprintf(stderr,
            "%s failed: %d (%s: %s)\n",
            operation,
            (int)result,
            name != NULL ? name : "unknown",
            description != NULL ? description : "no description");
}

#define CUDA_TRY(api, expression)                                \
    do {                                                         \
        CUresult result__ = (expression);                        \
        if (result__ != CUDA_SUCCESS) {                          \
            report_cuda_error((api), #expression, result__);    \
            goto cleanup;                                       \
        }                                                        \
    } while (0)

static char *read_text_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    long length;
    char *result;
    if (file == NULL) {
        fprintf(stderr, "failed to open PTX file: %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    result = (char *)malloc((size_t)length + 1u);
    if (result == NULL || fread(result, 1, (size_t)length, file) != (size_t)length) {
        free(result);
        fclose(file);
        return NULL;
    }
    result[length] = '\0';
    fclose(file);
    return result;
}

static bool pointer32(CUdeviceptr pointer, uint32_t *output)
{
    if (pointer > UINT32_MAX) {
        fprintf(stderr, "CUDA allocation is outside the 32-bit PTX address space\n");
        return false;
    }
    *output = (uint32_t)pointer;
    return true;
}

static void free_device(cuda_api *api, CUdeviceptr *pointer)
{
    if (*pointer != 0) {
        api->cuMemFree_v2(*pointer);
        *pointer = 0;
    }
}

int main(int argc, char **argv)
{
    cuda_api api;
    char *ptx = NULL;
    char *density_ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUmodule density_module = NULL;
    CUfunction calc_hash_function = NULL;
    CUfunction reorder_function = NULL;
    CUfunction density_function = NULL;
    CUtexref sorted_positions_texture = NULL;
    CUtexref particle_hash_texture = NULL;
    CUtexref cell_start_texture = NULL;
    CUtexref cell_end_texture = NULL;
    CUdeviceptr density_constant_symbol = 0;
    size_t density_constant_bytes = 0;
    CUdeviceptr output_hash_device = 0;
    CUdeviceptr output_index_device = 0;
    CUdeviceptr positions_device = 0;
    CUdeviceptr velocities_device = 0;
    CUdeviceptr search_starts_device = 0;
    CUdeviceptr search_counts_device = 0;
    CUdeviceptr parameters_device = 0;
    CUdeviceptr reordered_positions_device = 0;
    CUdeviceptr reordered_velocities_device = 0;
    CUdeviceptr output_cell_starts_device = 0;
    CUdeviceptr output_cell_ends_device = 0;
    CUdeviceptr mapping_device = 0;
    CUdeviceptr density_device = 0;
    CUdeviceptr remapped_density_device = 0;
    uint32_t output_hash = 0xdeadbeefu;
    uint32_t output_index = 0xdeadbeefu;
    const uint32_t position_bits[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0u,
    };
    const uint32_t velocity_bits[4] = {
        0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u,
    };
    uint32_t reordered_position_bits[4] = {0};
    uint32_t reordered_velocity_bits[4] = {0};
    uint32_t output_cell_start = 0xdeadbeefu;
    uint32_t output_cell_end = 0xdeadbeefu;
    const uint32_t search_starts[2] = {0u, 1u};
    const uint32_t search_counts[1] = {1u};
    const uint32_t mapping[1] = {0u};
    uint32_t parameter_words[40] = {0};
    uint32_t density_constant_words[
        DENSITY_CONSTANT_COPY_BYTES / sizeof(uint32_t)] = {0};
    uint32_t density_bits = 0xdeadbeefu;
    uint32_t remapped_density_bits = 0xdeadbeefu;
    uint32_t output_hash_arg[2] = {0};
    uint32_t output_index_arg[2] = {0};
    uint32_t positions_arg[2] = {0};
    uint32_t velocities_arg[2] = {0};
    uint32_t search_starts_arg[2] = {0};
    uint32_t search_counts_arg[2] = {0};
    uint32_t parameters_arg[2] = {0};
    uint32_t reordered_positions_arg[2] = {0};
    uint32_t reordered_velocities_arg[2] = {0};
    uint32_t output_cell_starts_arg[2] = {0};
    uint32_t output_cell_ends_arg[2] = {0};
    uint32_t mapping_arg[2] = {0};
    uint32_t density_arg[2] = {0};
    uint32_t remapped_density_arg[2] = {0};
    uint32_t density_debug[6] = {0};
    uint32_t density_debug_pointer = 0;
    uint32_t particle_count = 1u;
    const size_t cell_output_count = CELL_TEXTURE_COUNT;
    char device_name[256] = {0};
    size_t sorted_positions_texture_offset = ~(size_t)0;
    size_t particle_hash_texture_offset = ~(size_t)0;
    size_t cell_start_texture_offset = ~(size_t)0;
    size_t cell_end_texture_offset = ~(size_t)0;
    FILE *output = NULL;
    int exit_code = 1;
    bool pass = false;

    if (argc != 4) {
        fprintf(
            stderr,
            "usage: cuda-physx-fixture <module-0013.ptx> <module-0010.ptx> "
            "<output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) {
        return 1;
    }
    ptx = read_text_file(argv[1]);
    if (ptx == NULL) {
        goto cleanup;
    }
    density_ptx = read_text_file(argv[2]);
    if (density_ptx == NULL) {
        goto cleanup;
    }

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    CUDA_TRY(&api, api.cuModuleLoadData(&module, ptx));
    CUDA_TRY(&api, api.cuModuleLoadData(&density_module, density_ptx));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(
                 &calc_hash_function, module, CALC_HASH_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(
                 &reorder_function, module, REORDER_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(
                 &density_function, density_module, DENSITY_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(
                 &density_constant_symbol,
                 &density_constant_bytes,
                 density_module,
                 "parameters_cmem"));
    if (density_constant_bytes < DENSITY_CONSTANT_COPY_BYTES) {
        fprintf(stderr, "density constant symbol is too small\n");
        goto cleanup;
    }
    CUDA_TRY(&api,
             api.cuModuleGetTexRef(
                 &sorted_positions_texture,
                 density_module,
                 "inputSortedPositionsTex"));
    CUDA_TRY(&api,
             api.cuModuleGetTexRef(
                 &particle_hash_texture,
                 density_module,
                 "inputParticleHashKeyTex"));
    CUDA_TRY(&api,
             api.cuModuleGetTexRef(
                 &cell_start_texture,
                 density_module,
                 "inputCellStartTex"));
    CUDA_TRY(&api,
             api.cuModuleGetTexRef(
                 &cell_end_texture,
                 density_module,
                 "inputCellEndTex"));

    CUDA_TRY(&api, api.cuMemAlloc_v2(&output_hash_device, PARTICLE_HASH_BYTES));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&output_index_device, sizeof(output_index)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&positions_device, sizeof(position_bits)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&velocities_device, sizeof(velocity_bits)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&search_starts_device, sizeof(search_starts)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&search_counts_device, sizeof(search_counts)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&parameters_device, sizeof(parameter_words)));
    CUDA_TRY(&api,
             api.cuMemAlloc_v2(
                 &reordered_positions_device, SORTED_POSITION_BYTES));
    CUDA_TRY(&api,
             api.cuMemAlloc_v2(
                 &reordered_velocities_device, sizeof(reordered_velocity_bits)));
    CUDA_TRY(&api,
             api.cuMemAlloc_v2(
                 &output_cell_starts_device,
                 cell_output_count * sizeof(uint32_t)));
    CUDA_TRY(&api,
             api.cuMemAlloc_v2(
                 &output_cell_ends_device,
                 cell_output_count * sizeof(uint32_t)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&mapping_device, sizeof(mapping)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&density_device, sizeof(density_bits)));
    CUDA_TRY(&api,
             api.cuMemAlloc_v2(
                 &remapped_density_device, sizeof(remapped_density_bits)));

    if (!pointer32(output_hash_device, &output_hash_arg[0]) ||
        !pointer32(output_index_device, &output_index_arg[0]) ||
        !pointer32(positions_device, &positions_arg[0]) ||
        !pointer32(velocities_device, &velocities_arg[0]) ||
        !pointer32(search_starts_device, &search_starts_arg[0]) ||
        !pointer32(search_counts_device, &search_counts_arg[0]) ||
        !pointer32(parameters_device, &parameters_arg[0]) ||
        !pointer32(reordered_positions_device, &reordered_positions_arg[0]) ||
        !pointer32(reordered_velocities_device, &reordered_velocities_arg[0]) ||
        !pointer32(output_cell_starts_device, &output_cell_starts_arg[0]) ||
        !pointer32(output_cell_ends_device, &output_cell_ends_arg[0]) ||
        !pointer32(mapping_device, &mapping_arg[0]) ||
        !pointer32(density_device, &density_arg[0]) ||
        !pointer32(remapped_density_device, &remapped_density_arg[0])) {
        goto cleanup;
    }
    output_hash_arg[1] = 1u;
    output_index_arg[1] = 1u;
    positions_arg[1] = 1u;
    velocities_arg[1] = 1u;
    search_starts_arg[1] = 1u;
    search_counts_arg[1] = 1u;
    parameters_arg[1] = 1u;
    reordered_positions_arg[1] = 1u;
    reordered_velocities_arg[1] = 1u;
    output_cell_starts_arg[1] = (uint32_t)cell_output_count;
    output_cell_ends_arg[1] = (uint32_t)cell_output_count;
    mapping_arg[1] = 1u;
    density_arg[1] = 1u;
    remapped_density_arg[1] = 1u;
    parameter_words[1] = 0x3f800000u;
    density_constant_words[13] = 0x40000000u;
    density_constant_words[18] = 0x3f800000u;
    density_constant_words[21] = 0x3f800000u;

    CUDA_TRY(&api,
             api.cuMemsetD32_v2(
                 output_hash_device, 0xffffffffu, TEXTURE_PARTICLE_COUNT));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(output_index_device, &output_index, sizeof(output_index)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(positions_device, position_bits, sizeof(position_bits)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(velocities_device, velocity_bits, sizeof(velocity_bits)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(search_starts_device, search_starts, sizeof(search_starts)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(search_counts_device, search_counts, sizeof(search_counts)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(parameters_device, parameter_words, sizeof(parameter_words)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(mapping_device, mapping, sizeof(mapping)));
    CUDA_TRY(&api,
             api.cuMemsetD32_v2(
                 reordered_positions_device,
                 0u,
                 SORTED_POSITION_BYTES / sizeof(uint32_t)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 density_constant_symbol,
                 density_constant_words,
                 DENSITY_CONSTANT_COPY_BYTES));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 density_device, &density_bits, sizeof(density_bits)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 remapped_density_device,
                 &remapped_density_bits,
                 sizeof(remapped_density_bits)));
    CUDA_TRY(&api,
             api.cuMemsetD32_v2(
                 output_cell_starts_device, 0xffffffffu, cell_output_count));
    CUDA_TRY(&api,
             api.cuMemsetD32_v2(
                 output_cell_ends_device, 0xffffffffu, cell_output_count));

    CUDA_TRY(&api,
             api.cuTexRefSetFormat(
                 sorted_positions_texture, CU_AD_FORMAT_FLOAT, 4));
    CUDA_TRY(&api, api.cuTexRefSetFlags(sorted_positions_texture, 0));
    CUDA_TRY(&api,
             api.cuTexRefSetFilterMode(
                 sorted_positions_texture, CU_TR_FILTER_MODE_POINT));
    CUDA_TRY(&api,
             api.cuTexRefSetAddressMode(
                 sorted_positions_texture, 0, CU_TR_ADDRESS_MODE_CLAMP));
    CUDA_TRY(&api,
             api.cuTexRefSetAddress_v2(
                 &sorted_positions_texture_offset,
                 sorted_positions_texture,
                 reordered_positions_device,
                 SORTED_POSITION_BYTES));
    CUDA_TRY(&api,
             api.cuTexRefSetFormat(
                 particle_hash_texture, CU_AD_FORMAT_UNSIGNED_INT32, 1));
    CUDA_TRY(&api,
             api.cuTexRefSetFlags(
                 particle_hash_texture, CU_TRSF_READ_AS_INTEGER));
    CUDA_TRY(&api,
             api.cuTexRefSetFilterMode(
                 particle_hash_texture, CU_TR_FILTER_MODE_POINT));
    CUDA_TRY(&api,
             api.cuTexRefSetAddressMode(
                 particle_hash_texture, 0, CU_TR_ADDRESS_MODE_CLAMP));
    CUDA_TRY(&api,
             api.cuTexRefSetAddress_v2(
                 &particle_hash_texture_offset,
                 particle_hash_texture,
                 output_hash_device,
                 PARTICLE_HASH_BYTES));
    CUDA_TRY(&api,
             api.cuTexRefSetFormat(
                 cell_start_texture, CU_AD_FORMAT_UNSIGNED_INT32, 1));
    CUDA_TRY(&api,
             api.cuTexRefSetFlags(
                 cell_start_texture, CU_TRSF_READ_AS_INTEGER));
    CUDA_TRY(&api,
             api.cuTexRefSetFilterMode(
                 cell_start_texture, CU_TR_FILTER_MODE_POINT));
    CUDA_TRY(&api,
             api.cuTexRefSetAddressMode(
                 cell_start_texture, 0, CU_TR_ADDRESS_MODE_CLAMP));
    CUDA_TRY(&api,
             api.cuTexRefSetAddress_v2(
                 &cell_start_texture_offset,
                 cell_start_texture,
                 output_cell_starts_device,
                 cell_output_count * sizeof(uint32_t)));
    CUDA_TRY(&api,
             api.cuTexRefSetFormat(
                 cell_end_texture, CU_AD_FORMAT_UNSIGNED_INT32, 1));
    CUDA_TRY(&api,
             api.cuTexRefSetFlags(
                 cell_end_texture, CU_TRSF_READ_AS_INTEGER));
    CUDA_TRY(&api,
             api.cuTexRefSetFilterMode(
                 cell_end_texture, CU_TR_FILTER_MODE_POINT));
    CUDA_TRY(&api,
             api.cuTexRefSetAddressMode(
                 cell_end_texture, 0, CU_TR_ADDRESS_MODE_CLAMP));
    CUDA_TRY(&api,
             api.cuTexRefSetAddress_v2(
                 &cell_end_texture_offset,
                 cell_end_texture,
                 output_cell_ends_device,
                 cell_output_count * sizeof(uint32_t)));
    if (sorted_positions_texture_offset != 0u ||
        particle_hash_texture_offset != 0u ||
        cell_start_texture_offset != 0u ||
        cell_end_texture_offset != 0u) {
        fprintf(stderr, "density texture binding returned a nonzero byte offset\n");
        goto cleanup;
    }

    {
        void *calc_parameters[] = {
            output_hash_arg,
            output_index_arg,
            positions_arg,
            search_starts_arg,
            search_counts_arg,
            parameters_arg,
            &particle_count,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(calc_hash_function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          calc_parameters,
                                          NULL));
    }
    {
        void *reorder_parameters[] = {
            reordered_positions_arg,
            reordered_velocities_arg,
            output_cell_starts_arg,
            output_cell_ends_arg,
            positions_arg,
            velocities_arg,
            output_hash_arg,
            output_index_arg,
            mapping_arg,
            &particle_count,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(reorder_function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          260,
                                          NULL,
                                          reorder_parameters,
                                          NULL));
    }
    {
        void *density_parameters[] = {
            density_arg,
            remapped_density_arg,
            density_debug,
            &density_debug_pointer,
            reordered_positions_arg,
            output_hash_arg,
            mapping_arg,
            output_cell_starts_arg,
            output_cell_ends_arg,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(density_function,
                                          1, 1, 1,
                                          192, 1, 1,
                                          0,
                                          NULL,
                                          density_parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(&output_hash, output_hash_device, sizeof(output_hash)));
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(&output_index, output_index_device, sizeof(output_index)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 reordered_position_bits,
                 reordered_positions_device,
                 sizeof(reordered_position_bits)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 reordered_velocity_bits,
                 reordered_velocities_device,
                 sizeof(reordered_velocity_bits)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 &output_cell_start,
                 output_cell_starts_device +
                     (CUdeviceptr)EXPECTED_HASH * sizeof(uint32_t),
                 sizeof(output_cell_start)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 &output_cell_end,
                 output_cell_ends_device +
                     (CUdeviceptr)EXPECTED_HASH * sizeof(uint32_t),
                 sizeof(output_cell_end)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 &density_bits, density_device, sizeof(density_bits)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 &remapped_density_bits,
                 remapped_density_device,
                 sizeof(remapped_density_bits)));

    pass =
        output_hash == EXPECTED_HASH &&
        output_index == 0u &&
        memcmp(reordered_position_bits, position_bits, sizeof(position_bits)) == 0 &&
        memcmp(reordered_velocity_bits, velocity_bits, sizeof(velocity_bits)) == 0 &&
        output_cell_start == 0u &&
        output_cell_end == 1u &&
        density_bits == EXPECTED_DENSITY_BITS &&
        remapped_density_bits == EXPECTED_DENSITY_BITS;

    output = fopen(argv[3], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[3]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 3,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernels\": [\"KernelFluidCalcHashPacked\", "
                "\"KernelReorderPosVelAndFindCellStartPacked\", "
                "\"KernelFluidDensity\"],\n"
            "  \"position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"velocity_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"cell_divisor_bits\": \"0x3f800000\",\n"
            "  \"density_constants\": {\"radius_term\": \"0x3f800000\", "
                "\"distance_scale\": \"0x3f800000\", "
                "\"normalization\": \"0x40000000\"},\n"
            "  \"hash\": \"0x%08x\",\n"
            "  \"index\": %u,\n"
            "  \"reordered_position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"reordered_velocity_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"cell_start\": %u,\n"
            "  \"cell_end\": %u,\n"
            "  \"density_bits\": \"0x%08x\",\n"
            "  \"remapped_density_bits\": \"0x%08x\",\n"
            "  \"pass\": %s\n"
            "}\n",
            device_name,
            position_bits[0], position_bits[1], position_bits[2], position_bits[3],
            velocity_bits[0], velocity_bits[1], velocity_bits[2], velocity_bits[3],
            output_hash,
            output_index,
            reordered_position_bits[0], reordered_position_bits[1],
            reordered_position_bits[2], reordered_position_bits[3],
            reordered_velocity_bits[0], reordered_velocity_bits[1],
            reordered_velocity_bits[2], reordered_velocity_bits[3],
            output_cell_start,
            output_cell_end,
            density_bits,
            remapped_density_bits,
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    exit_code = pass ? 0 : 1;
    printf("%s  %s: hash=0x%08x index=%u cells=[%u,%u] density=0x%08x\n",
           pass ? "PASS" : "FAIL",
           device_name,
           output_hash,
           output_index,
           output_cell_start,
           output_cell_end,
           density_bits);

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    if (density_module != NULL) {
        api.cuModuleUnload(density_module);
    }
    if (module != NULL) {
        api.cuModuleUnload(module);
    }
    free_device(&api, &remapped_density_device);
    free_device(&api, &density_device);
    free_device(&api, &mapping_device);
    free_device(&api, &output_cell_ends_device);
    free_device(&api, &output_cell_starts_device);
    free_device(&api, &reordered_velocities_device);
    free_device(&api, &reordered_positions_device);
    free_device(&api, &parameters_device);
    free_device(&api, &search_counts_device);
    free_device(&api, &search_starts_device);
    free_device(&api, &velocities_device);
    free_device(&api, &positions_device);
    free_device(&api, &output_index_device);
    free_device(&api, &output_hash_device);
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    free(ptx);
    free(density_ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return exit_code;
}
