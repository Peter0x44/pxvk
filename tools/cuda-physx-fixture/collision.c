#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLLISION_KERNEL_NAME "_Z29KernelDeformableUberCollisionPi"
#define INSTANCE_CONSTANT_NAME "gDeformableCollisionInstanceData"
#define INSTANCE_CONSTANT_BYTES 17920u
#define BATCH_CONSTANT_NAME "gDeformableCollisionBatchedData"
#define BATCH_CONSTANT_BYTES 72u
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 192u

#define BATCH_OFFSET 8u
#define INPUT_POSITION_OFFSET 20u
#define INPUT_MOTION_OFFSET 28u
#define OUTPUT_8_OFFSET 36u
#define OUTPUT_12_OFFSET 44u
#define OUTPUT_20_OFFSET 52u
#define FLAGS_OFFSET 56u
#define OUTPUT_POSITION_OFFSET 60u
#define OUTPUT_MOTION_OFFSET 68u
#define COLLISION_40_OFFSET 76u
#define COLLISION_44_OFFSET 84u
#define SCALAR_48_OFFSET 92u
#define SCALAR_52_OFFSET 96u
#define SCALAR_56_OFFSET 100u
#define SCALAR_60_OFFSET 104u
#define COLLISION_64_OFFSET 108u
#define ACTIVE_OFFSET 116u

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
    return heap[offset - 1u] == GUARD_BEFORE &&
           heap[offset + words] == GUARD_AFTER;
}

static bool vector_equals(const uint32_t *heap, size_t offset, const uint32_t *expected)
{
    return memcmp(heap + offset, expected, 4u * sizeof(uint32_t)) == 0;
}

int main(int argc, char **argv)
{
    static const uint32_t input_position[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u,
    };
    static const uint32_t input_motion[4] = {
        0x40a00000u, 0x40c00000u, 0x40e00000u, 0x41000000u,
    };
    static const uint32_t sentinel_vector[4] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu,
    };
    static const uint32_t zero_vector[4] = {0u, 0u, 0u, 0u};
    static const uint32_t batch_record[2] = {0x00010000u, 0u};
    static const uint32_t zero_scalar[1] = {0u};
    static const uint32_t sentinel_scalar[1] = {0xfeedfaceu};
    static const uint32_t active_scalar[1] = {1u};

    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr instance_device = 0;
    CUdeviceptr batch_constant_device = 0;
    size_t instance_bytes = 0;
    size_t batch_constant_bytes = 0;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t instance_constants[INSTANCE_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t batch_constants[BATCH_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t pointers[17] = {0};
    uint32_t error_pointer = 0;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool guards_intact = false;
    bool pass = false;
    int exit_code = 1;
    size_t index;
    const size_t offsets[17] = {
        BATCH_OFFSET,
        INPUT_POSITION_OFFSET,
        INPUT_MOTION_OFFSET,
        OUTPUT_8_OFFSET,
        OUTPUT_12_OFFSET,
        OUTPUT_20_OFFSET,
        FLAGS_OFFSET,
        OUTPUT_POSITION_OFFSET,
        OUTPUT_MOTION_OFFSET,
        COLLISION_40_OFFSET,
        COLLISION_44_OFFSET,
        SCALAR_48_OFFSET,
        SCALAR_52_OFFSET,
        SCALAR_56_OFFSET,
        SCALAR_60_OFFSET,
        COLLISION_64_OFFSET,
        ACTIVE_OFFSET,
    };

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-deformable-collision-fixture <collision.ptx> <output.json>\n");
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
    CUDA_TRY(&api, api.cuModuleGetFunction(&function, module, COLLISION_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(
                 &instance_device, &instance_bytes, module, INSTANCE_CONSTANT_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(
                 &batch_constant_device,
                 &batch_constant_bytes,
                 module,
                 BATCH_CONSTANT_NAME));
    if (instance_bytes < INSTANCE_CONSTANT_BYTES ||
        batch_constant_bytes < BATCH_CONSTANT_BYTES) {
        fprintf(stderr, "collision constant symbol is too small\n");
        goto cleanup;
    }

    initialize_range(heap, BATCH_OFFSET, batch_record, 2u);
    initialize_range(heap, INPUT_POSITION_OFFSET, input_position, 4u);
    initialize_range(heap, INPUT_MOTION_OFFSET, input_motion, 4u);
    initialize_range(heap, OUTPUT_8_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, OUTPUT_12_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, OUTPUT_20_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, FLAGS_OFFSET, zero_scalar, 1u);
    initialize_range(heap, OUTPUT_POSITION_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, OUTPUT_MOTION_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_40_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_44_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, SCALAR_48_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, SCALAR_52_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, SCALAR_56_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, SCALAR_60_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_64_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, ACTIVE_OFFSET, active_scalar, 1u);

    CUDA_TRY(&api, api.cuMemAlloc_v2(&heap_device, sizeof(heap)));
    for (index = 0u; index < 17u; ++index) {
        if (!pointer32(
                heap_device + offsets[index] * sizeof(uint32_t),
                &pointers[index])) {
            goto cleanup;
        }
    }

    instance_constants[0] = pointers[1];
    instance_constants[1] = pointers[2];
    instance_constants[2] = pointers[3];
    instance_constants[3] = pointers[4];
    instance_constants[5] = pointers[5];
    instance_constants[6] = pointers[6];
    instance_constants[8] = pointers[7];
    instance_constants[9] = pointers[8];
    instance_constants[10] = pointers[9];
    instance_constants[11] = pointers[10];
    instance_constants[12] = pointers[11];
    instance_constants[13] = pointers[12];
    instance_constants[14] = pointers[13];
    instance_constants[15] = pointers[14];
    instance_constants[16] = pointers[15];
    instance_constants[17] = pointers[16];
    instance_constants[28] = 0u;
    instance_constants[33] = 0u;
    instance_constants[34] = 0u;
    batch_constants[17] = pointers[0];

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 instance_device, instance_constants, INSTANCE_CONSTANT_BYTES));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 batch_constant_device, batch_constants, BATCH_CONSTANT_BYTES));
    {
        void *parameters[] = {&error_pointer};
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact =
        range_guards_match(heap, BATCH_OFFSET, 2u) &&
        range_guards_match(heap, INPUT_POSITION_OFFSET, 4u) &&
        range_guards_match(heap, INPUT_MOTION_OFFSET, 4u) &&
        range_guards_match(heap, OUTPUT_8_OFFSET, 4u) &&
        range_guards_match(heap, OUTPUT_12_OFFSET, 4u) &&
        range_guards_match(heap, OUTPUT_20_OFFSET, 1u) &&
        range_guards_match(heap, FLAGS_OFFSET, 1u) &&
        range_guards_match(heap, OUTPUT_POSITION_OFFSET, 4u) &&
        range_guards_match(heap, OUTPUT_MOTION_OFFSET, 4u) &&
        range_guards_match(heap, COLLISION_40_OFFSET, 4u) &&
        range_guards_match(heap, COLLISION_44_OFFSET, 4u) &&
        range_guards_match(heap, SCALAR_48_OFFSET, 1u) &&
        range_guards_match(heap, SCALAR_52_OFFSET, 1u) &&
        range_guards_match(heap, SCALAR_56_OFFSET, 1u) &&
        range_guards_match(heap, SCALAR_60_OFFSET, 1u) &&
        range_guards_match(heap, COLLISION_64_OFFSET, 4u) &&
        range_guards_match(heap, ACTIVE_OFFSET, 1u);
    pass = guards_intact &&
           vector_equals(heap, INPUT_POSITION_OFFSET, input_position) &&
           vector_equals(heap, INPUT_MOTION_OFFSET, input_motion) &&
           vector_equals(heap, OUTPUT_8_OFFSET, zero_vector) &&
           vector_equals(heap, OUTPUT_12_OFFSET, zero_vector) &&
           heap[OUTPUT_20_OFFSET] == 0xbf800000u &&
           heap[FLAGS_OFFSET] == 0u &&
           vector_equals(heap, OUTPUT_POSITION_OFFSET, input_position) &&
           vector_equals(heap, OUTPUT_MOTION_OFFSET, input_motion) &&
           vector_equals(heap, COLLISION_40_OFFSET, zero_vector) &&
           vector_equals(heap, COLLISION_44_OFFSET, zero_vector) &&
           heap[SCALAR_48_OFFSET] == 0xbf800000u &&
           heap[SCALAR_52_OFFSET] == 0u &&
           heap[SCALAR_56_OFFSET] == 0x0000ffffu &&
           heap[SCALAR_60_OFFSET] == 0u &&
           vector_equals(heap, COLLISION_64_OFFSET, zero_vector) &&
           heap[ACTIVE_OFFSET] == 0u;

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"KernelDeformableUberCollision\",\n"
            "  \"particle_count\": 1,\n"
            "  \"shape_count\": 0,\n"
            "  \"output_position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"output_motion_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"collision_scalar_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            device_name,
            heap[OUTPUT_POSITION_OFFSET],
            heap[OUTPUT_POSITION_OFFSET + 1u],
            heap[OUTPUT_POSITION_OFFSET + 2u],
            heap[OUTPUT_POSITION_OFFSET + 3u],
            heap[OUTPUT_MOTION_OFFSET],
            heap[OUTPUT_MOTION_OFFSET + 1u],
            heap[OUTPUT_MOTION_OFFSET + 2u],
            heap[OUTPUT_MOTION_OFFSET + 3u],
            heap[SCALAR_48_OFFSET],
            heap[SCALAR_52_OFFSET],
            heap[SCALAR_56_OFFSET],
            heap[SCALAR_60_OFFSET],
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    printf("%s  %s: collision=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[SCALAR_48_OFFSET],
           heap[SCALAR_52_OFFSET],
           heap[SCALAR_56_OFFSET],
           heap[SCALAR_60_OFFSET]);
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
