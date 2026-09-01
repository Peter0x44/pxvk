#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_NAME "_Z26KernelFluidSimpleParticleD17PxgFixedSizeArrayI12PxcVectorPadES_IKS0_E11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj"
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 96u
#define OUTPUT_OFFSET 8u
#define INPUT_OFFSET 16u
#define START_INDICES_OFFSET 24u
#define FLUID_SIZES_OFFSET 30u
#define PARAMETERS_OFFSET 40u

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

#define CUDA_TRY(api, expression)                             \
    do {                                                      \
        CUresult result__ = (expression);                     \
        if (result__ != CUDA_SUCCESS) {                       \
            report_cuda_error((api), #expression, result__); \
            goto cleanup;                                    \
        }                                                     \
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

static bool heap_pointer(CUdeviceptr heap, size_t offset, uint32_t *output)
{
    return pointer32(heap + offset * sizeof(uint32_t), output);
}

static void initialize_range(uint32_t *heap,
                             size_t offset,
                             const uint32_t *values,
                             size_t words)
{
    heap[offset - 1u] = GUARD_BEFORE;
    memcpy(heap + offset, values, words * sizeof(uint32_t));
    heap[offset + words] = GUARD_AFTER;
}

static bool range_guards_match(const uint32_t *heap, size_t offset, size_t words)
{
    return heap[offset - 1u] == GUARD_BEFORE && heap[offset + words] == GUARD_AFTER;
}

static bool words_equal(const uint32_t *heap,
                        size_t offset,
                        const uint32_t *expected,
                        size_t words)
{
    return memcmp(heap + offset, expected, words * sizeof(uint32_t)) == 0;
}

static void write_words(FILE *output, const uint32_t *values, size_t words)
{
    size_t index;
    fputc('[', output);
    for (index = 0u; index < words; ++index) {
        fprintf(output, "%s\"0x%08x\"", index == 0u ? "" : ", ", values[index]);
    }
    fputc(']', output);
}

int main(int argc, char **argv)
{
    static const uint32_t sentinel_output[4] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu,
    };
    static const uint32_t input[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0x11223344u,
    };
    static const uint32_t start_indices[2] = {0u, 1u};
    static const uint32_t fluid_sizes[1] = {1u};
    static const uint32_t expected_output[4] = {
        0x3fe00000u, 0x00000000u, 0x40440000u, 0x00000000u,
    };
    uint32_t parameters[40] = {0};
    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t output_array[2];
    uint32_t input_array[2];
    uint32_t starts_array[2];
    uint32_t sizes_array[2];
    uint32_t parameters_array[2];
    uint32_t particle_count = 1u;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool guards_intact = true;
    bool pass = false;
    int exit_code = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-fluid-simple-particle-fixture <particle.ptx> <output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) return 1;
    ptx = read_text_file(argv[1]);
    if (ptx == NULL) goto cleanup;

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    CUDA_TRY(&api, api.cuModuleLoadData(&module, ptx));
    CUDA_TRY(&api, api.cuModuleGetFunction(&function, module, KERNEL_NAME));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&heap_device, sizeof(heap)));

    if (!heap_pointer(heap_device, OUTPUT_OFFSET, &output_array[0]) ||
        !heap_pointer(heap_device, INPUT_OFFSET, &input_array[0]) ||
        !heap_pointer(heap_device, START_INDICES_OFFSET, &starts_array[0]) ||
        !heap_pointer(heap_device, FLUID_SIZES_OFFSET, &sizes_array[0]) ||
        !heap_pointer(heap_device, PARAMETERS_OFFSET, &parameters_array[0])) {
        goto cleanup;
    }
    output_array[1] = 1u;
    input_array[1] = 1u;
    starts_array[1] = 2u;
    sizes_array[1] = 1u;
    parameters_array[1] = 1u;

    parameters[0] = 0x3f000000u;
    parameters[4] = 0x40000000u;
    parameters[5] = 0xc0800000u;
    parameters[6] = 0x3f800000u;
    parameters[9] = 0x3e800000u;
    parameters[16] = 0x41200000u;

    initialize_range(heap, OUTPUT_OFFSET, sentinel_output, 4u);
    initialize_range(heap, INPUT_OFFSET, input, 4u);
    initialize_range(heap, START_INDICES_OFFSET, start_indices, 2u);
    initialize_range(heap, FLUID_SIZES_OFFSET, fluid_sizes, 1u);
    initialize_range(heap, PARAMETERS_OFFSET, parameters, 40u);

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    {
        void *launch_parameters[] = {
            output_array,
            input_array,
            starts_array,
            sizes_array,
            parameters_array,
            &particle_count,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          256, 1, 1,
                                          0,
                                          NULL,
                                          launch_parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact = range_guards_match(heap, OUTPUT_OFFSET, 4u) &&
                    range_guards_match(heap, INPUT_OFFSET, 4u) &&
                    range_guards_match(heap, START_INDICES_OFFSET, 2u) &&
                    range_guards_match(heap, FLUID_SIZES_OFFSET, 1u) &&
                    range_guards_match(heap, PARAMETERS_OFFSET, 40u);
    pass = guards_intact &&
           words_equal(heap, OUTPUT_OFFSET, expected_output, 4u) &&
           words_equal(heap, INPUT_OFFSET, input, 4u) &&
           words_equal(heap, START_INDICES_OFFSET, start_indices, 2u) &&
           words_equal(heap, FLUID_SIZES_OFFSET, fluid_sizes, 1u) &&
           words_equal(heap, PARAMETERS_OFFSET, parameters, 40u);

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"KernelFluidSimpleParticle\",\n"
            "  \"particle_count\": 1,\n"
            "  \"fluid_count\": 1,\n"
            "  \"static_shared_bytes\": 10756,\n"
            "  \"time_step_bits\": \"0x%08x\",\n"
            "  \"damping_bits\": \"0x%08x\",\n"
            "  \"input_bits\": ",
            device_name,
            parameters[0],
            parameters[9]);
    write_words(output, input, 4u);
    fprintf(output, ",\n  \"expected_output_bits\": ");
    write_words(output, expected_output, 4u);
    fprintf(output, ",\n  \"output_bits\": ");
    write_words(output, heap + OUTPUT_OFFSET, 4u);
    fprintf(output,
            ",\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;

    printf("%s  %s: output=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[OUTPUT_OFFSET],
           heap[OUTPUT_OFFSET + 1u],
           heap[OUTPUT_OFFSET + 2u],
           heap[OUTPUT_OFFSET + 3u]);
    exit_code = pass ? 0 : 1;

cleanup:
    if (output != NULL) fclose(output);
    if (module != NULL) api.cuModuleUnload(module);
    if (heap_device != 0) api.cuMemFree_v2(heap_device);
    if (context != NULL) api.cuCtxDestroy_v2(context);
    free(ptx);
    if (api.library != NULL) FreeLibrary(api.library);
    return exit_code;
}
