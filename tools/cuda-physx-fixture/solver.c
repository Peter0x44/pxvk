#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOLVER_KERNEL_NAME "_Z26KernelDeformableUberSolverILj512EEvjbPhPi"
#define SOLVER_CONSTANT_NAME "solverGpuData"
#define SOLVER_CONSTANT_BYTES 5888u
#define SOLVER_DEFAULT_DYNAMIC_SHARED_BYTES 7072u
#define SOLVER_SAMPLE_CLOTH_DYNAMIC_SHARED_BYTES 42240u
#define SOLVER_DEFAULT_BLOCK_X 64u
#define SOLVER_SAMPLE_CLOTH_BLOCK_X 512u
#define HEAP_WORDS 256u
#define A_OFFSET 32u
#define B_OFFSET 48u
#define C_OFFSET 64u
#define POSITION_OFFSET 80u
#define FLAGS_OFFSET 96u
#define PACKED_OFFSET 112u
#define OUTPUT_OFFSET 128u
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u

static const uint32_t EXPECTED_OUTPUT[16] = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0x00000400u,
    0x41f00000u, 0x41f80000u, 0x42000000u, 0xabcdef01u,
    0xc1100000u, 0xc1100000u, 0xc1100000u, 0x00000000u,
    0x41a00000u, 0x41a80000u, 0x41b00000u, 0x00000000u,
};
static const uint32_t EXPECTED_A[4] = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0x00000000u,
};
static const uint32_t EXPECTED_POSITION[4] = {
    0xc1100000u, 0xc1100000u, 0xc1100000u, 0x00000000u,
};

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
    LOAD_API(api, cuModuleGetGlobal_v2);
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

static bool range_guards_match(const uint32_t *heap, size_t offset, size_t words)
{
    return heap[offset - 1u] == GUARD_BEFORE && heap[offset + words] == GUARD_AFTER;
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

int main(int argc, char **argv)
{
    static const uint32_t a_input[4] = {
        0x41200000u, 0x41300000u, 0x41400000u, 0u,
    };
    static const uint32_t b_input[4] = {
        0x41a00000u, 0x41a80000u, 0x41b00000u, 0u,
    };
    static const uint32_t c_input[4] = {
        0x41f00000u, 0x41f80000u, 0x42000000u, 0u,
    };
    static const uint32_t position_input[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
    };
    static const uint32_t flags_input[1] = {0x00000400u};
    static const uint32_t packed_input[1] = {0xabcdef01u};
    static const uint32_t output_input[16] = {0};

    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr constants_device = 0;
    size_t constants_bytes = 0;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t constants[SOLVER_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t a_pointer = 0;
    uint32_t b_pointer = 0;
    uint32_t c_pointer = 0;
    uint32_t position_pointer = 0;
    uint32_t flags_pointer = 0;
    uint32_t packed_pointer = 0;
    uint32_t output_pointer = 0;
    uint32_t iteration_count = 0;
    uint8_t write_velocity = 0;
    uint32_t error_pointer = 0;
    char device_name[256] = {0};
    const char *ptx_path = NULL;
    const char *output_path = NULL;
    unsigned int block_x = SOLVER_DEFAULT_BLOCK_X;
    unsigned int dynamic_shared_bytes =
        SOLVER_DEFAULT_DYNAMIC_SHARED_BYTES;
    FILE *output = NULL;
    bool guards_intact = false;
    bool pass = false;
    int exit_code = 1;

    if (argc == 3) {
        ptx_path = argv[1];
        output_path = argv[2];
    } else if (
        argc == 4 &&
        strcmp(argv[1], "--sample-cloth-shape") == 0) {
        block_x = SOLVER_SAMPLE_CLOTH_BLOCK_X;
        dynamic_shared_bytes =
            SOLVER_SAMPLE_CLOTH_DYNAMIC_SHARED_BYTES;
        ptx_path = argv[2];
        output_path = argv[3];
    } else {
        fprintf(
            stderr,
            "usage: cuda-deformable-solver-fixture "
            "[--sample-cloth-shape] <solver.ptx> <output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) {
        return 1;
    }
    ptx = read_text_file(ptx_path);
    if (ptx == NULL) {
        goto cleanup;
    }

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    CUDA_TRY(&api, api.cuModuleLoadData(&module, ptx));
    CUDA_TRY(&api, api.cuModuleGetFunction(&function, module, SOLVER_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(
                 &constants_device, &constants_bytes, module, SOLVER_CONSTANT_NAME));
    if (constants_bytes < SOLVER_CONSTANT_BYTES) {
        fprintf(stderr, "%s symbol is too small\n", SOLVER_CONSTANT_NAME);
        goto cleanup;
    }

    initialize_range(heap, A_OFFSET, a_input, 4u);
    initialize_range(heap, B_OFFSET, b_input, 4u);
    initialize_range(heap, C_OFFSET, c_input, 4u);
    initialize_range(heap, POSITION_OFFSET, position_input, 4u);
    initialize_range(heap, FLAGS_OFFSET, flags_input, 1u);
    initialize_range(heap, PACKED_OFFSET, packed_input, 1u);
    initialize_range(heap, OUTPUT_OFFSET, output_input, 16u);

    CUDA_TRY(&api, api.cuMemAlloc_v2(&heap_device, sizeof(heap)));
    if (!pointer32(heap_device + A_OFFSET * sizeof(uint32_t), &a_pointer) ||
        !pointer32(heap_device + B_OFFSET * sizeof(uint32_t), &b_pointer) ||
        !pointer32(heap_device + C_OFFSET * sizeof(uint32_t), &c_pointer) ||
        !pointer32(heap_device + POSITION_OFFSET * sizeof(uint32_t), &position_pointer) ||
        !pointer32(heap_device + FLAGS_OFFSET * sizeof(uint32_t), &flags_pointer) ||
        !pointer32(heap_device + PACKED_OFFSET * sizeof(uint32_t), &packed_pointer) ||
        !pointer32(heap_device + OUTPUT_OFFSET * sizeof(uint32_t), &output_pointer)) {
        goto cleanup;
    }

    constants[1] = 1u;
    constants[8] = 0x3f800000u;
    constants[9] = 0u;
    constants[14] = a_pointer;
    constants[15] = position_pointer;
    constants[16] = b_pointer;
    constants[17] = c_pointer;
    constants[18] = c_pointer;
    constants[21] = flags_pointer;
    constants[22] = packed_pointer;

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 constants_device, constants, SOLVER_CONSTANT_BYTES));
    {
        void *parameters[] = {
            &iteration_count, &write_velocity, &output_pointer, &error_pointer,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          block_x, 1, 1,
                                          dynamic_shared_bytes,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact =
        range_guards_match(heap, A_OFFSET, 4u) &&
        range_guards_match(heap, B_OFFSET, 4u) &&
        range_guards_match(heap, C_OFFSET, 4u) &&
        range_guards_match(heap, POSITION_OFFSET, 4u) &&
        range_guards_match(heap, FLAGS_OFFSET, 1u) &&
        range_guards_match(heap, PACKED_OFFSET, 1u) &&
        range_guards_match(heap, OUTPUT_OFFSET, 16u);
    pass = guards_intact &&
           memcmp(heap + A_OFFSET, EXPECTED_A, sizeof(EXPECTED_A)) == 0 &&
           memcmp(heap + POSITION_OFFSET,
                  EXPECTED_POSITION,
                  sizeof(EXPECTED_POSITION)) == 0 &&
           heap[FLAGS_OFFSET] == flags_input[0] &&
           heap[PACKED_OFFSET] == packed_input[0] &&
           memcmp(heap + OUTPUT_OFFSET,
                  EXPECTED_OUTPUT,
                  sizeof(EXPECTED_OUTPUT)) == 0;

    output = fopen(output_path, "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", output_path);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"KernelDeformableUberSolver<512>\",\n"
            "  \"particle_count\": 1,\n"
            "  \"iteration_count\": 0,\n"
            "  \"local_size_x\": %u,\n"
            "  \"dynamic_shared_bytes\": %u,\n"
            "  \"output_bits\": [",
            device_name,
            block_x,
            dynamic_shared_bytes);
    {
        size_t index;
        for (index = 0; index < 16u; ++index) {
            fprintf(output,
                    "%s\"0x%08x\"",
                    index == 0u ? "" : ", ",
                    heap[OUTPUT_OFFSET + index]);
        }
    }
    fprintf(output,
            "],\n"
            "  \"position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            heap[POSITION_OFFSET],
            heap[POSITION_OFFSET + 1u],
            heap[POSITION_OFFSET + 2u],
            heap[POSITION_OFFSET + 3u],
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    printf("%s  %s: output[0..3]=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[OUTPUT_OFFSET],
           heap[OUTPUT_OFFSET + 1u],
           heap[OUTPUT_OFFSET + 2u],
           heap[OUTPUT_OFFSET + 3u]);
    exit_code = pass ? 0 : 1;

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    if (module != NULL) {
        api.cuModuleUnload(module);
    }
    if (heap_device != 0) {
        api.cuMemFree_v2(heap_device);
    }
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    free(ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return exit_code;
}
