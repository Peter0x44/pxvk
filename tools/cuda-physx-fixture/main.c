#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_NAME \
    "_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj"
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
    CUresult(CUDAAPI *cuMemAlloc_v2)(CUdeviceptr *, size_t);
    CUresult(CUDAAPI *cuMemFree_v2)(CUdeviceptr);
    CUresult(CUDAAPI *cuMemcpyHtoD_v2)(CUdeviceptr, const void *, size_t);
    CUresult(CUDAAPI *cuMemcpyDtoH_v2)(void *, CUdeviceptr, size_t);
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
    LOAD_API(api, cuMemAlloc_v2);
    LOAD_API(api, cuMemFree_v2);
    LOAD_API(api, cuMemcpyHtoD_v2);
    LOAD_API(api, cuMemcpyDtoH_v2);
    LOAD_API(api, cuLaunchKernel);
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
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr output_hash_device = 0;
    CUdeviceptr output_index_device = 0;
    CUdeviceptr positions_device = 0;
    CUdeviceptr cell_starts_device = 0;
    CUdeviceptr cell_counts_device = 0;
    CUdeviceptr parameters_device = 0;
    uint32_t output_hash = 0xdeadbeefu;
    uint32_t output_index = 0xdeadbeefu;
    const float positions[4] = {1.0f, 2.0f, 3.0f, 0.0f};
    const uint32_t cell_starts[2] = {0u, 1u};
    const uint32_t cell_counts[1] = {1u};
    uint32_t parameter_words[40] = {0};
    uint32_t output_hash_arg[2] = {0};
    uint32_t output_index_arg[2] = {0};
    uint32_t positions_arg[2] = {0};
    uint32_t cell_starts_arg[2] = {0};
    uint32_t cell_counts_arg[2] = {0};
    uint32_t parameters_arg[2] = {0};
    uint32_t particle_count = 1u;
    char device_name[256] = {0};
    FILE *output = NULL;
    int exit_code = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-physx-fixture <module.ptx> <output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) {
        return 1;
    }
    ptx = read_text_file(argv[1]);
    if (ptx == NULL) {
        goto cleanup;
    }

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    CUDA_TRY(&api, api.cuModuleLoadData(&module, ptx));
    CUDA_TRY(&api, api.cuModuleGetFunction(&function, module, KERNEL_NAME));

    CUDA_TRY(&api, api.cuMemAlloc_v2(&output_hash_device, sizeof(output_hash)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&output_index_device, sizeof(output_index)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&positions_device, sizeof(positions)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&cell_starts_device, sizeof(cell_starts)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&cell_counts_device, sizeof(cell_counts)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&parameters_device, sizeof(parameter_words)));

    if (!pointer32(output_hash_device, &output_hash_arg[0]) ||
        !pointer32(output_index_device, &output_index_arg[0]) ||
        !pointer32(positions_device, &positions_arg[0]) ||
        !pointer32(cell_starts_device, &cell_starts_arg[0]) ||
        !pointer32(cell_counts_device, &cell_counts_arg[0]) ||
        !pointer32(parameters_device, &parameters_arg[0])) {
        goto cleanup;
    }
    output_hash_arg[1] = 1u;
    output_index_arg[1] = 1u;
    positions_arg[1] = 1u;
    cell_starts_arg[1] = 1u;
    cell_counts_arg[1] = 1u;
    parameters_arg[1] = 1u;
    parameter_words[1] = 0x3f800000u;

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(output_hash_device, &output_hash, sizeof(output_hash)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(output_index_device, &output_index, sizeof(output_index)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(positions_device, positions, sizeof(positions)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(cell_starts_device, cell_starts, sizeof(cell_starts)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(cell_counts_device, cell_counts, sizeof(cell_counts)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(parameters_device, parameter_words, sizeof(parameter_words)));

    {
        void *kernel_parameters[] = {
            output_hash_arg,
            output_index_arg,
            positions_arg,
            cell_starts_arg,
            cell_counts_arg,
            parameters_arg,
            &particle_count,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          kernel_parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(&output_hash, output_hash_device, sizeof(output_hash)));
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(&output_index, output_index_device, sizeof(output_index)));

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"KernelFluidCalcHashPacked\",\n"
            "  \"position_bits\": [\"0x3f800000\", \"0x40000000\", \"0x40400000\"],\n"
            "  \"cell_divisor_bits\": \"0x3f800000\",\n"
            "  \"hash\": \"0x%08x\",\n"
            "  \"index\": %u,\n"
            "  \"expected_hash\": \"0x%08x\",\n"
            "  \"pass\": %s\n"
            "}\n",
            device_name,
            output_hash,
            output_index,
            EXPECTED_HASH,
            output_hash == EXPECTED_HASH && output_index == 0u ? "true" : "false");
    fclose(output);
    output = NULL;
    exit_code = output_hash == EXPECTED_HASH && output_index == 0u ? 0 : 1;
    printf("%s  %s: hash=0x%08x index=%u\n",
           exit_code == 0 ? "PASS" : "FAIL",
           device_name,
           output_hash,
           output_index);

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    if (module != NULL) {
        api.cuModuleUnload(module);
    }
    free_device(&api, &parameters_device);
    free_device(&api, &cell_counts_device);
    free_device(&api, &cell_starts_device);
    free_device(&api, &positions_device);
    free_device(&api, &output_index_device);
    free_device(&api, &output_hash_device);
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    free(ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return exit_code;
}
