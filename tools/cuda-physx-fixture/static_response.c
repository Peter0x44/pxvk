#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_NAME "_Z41PxgCudaFluidCollisionStaticResponseBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi"
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 224u

#define WORK_UNIT_OFFSET 8u
#define START_ADDRS_OFFSET 16u
#define PARAMETERS_OFFSET 52u
#define NORMAL_OFFSET 96u
#define ACCUMULATED_POSITION_OFFSET 104u
#define BODY_VELOCITY_OFFSET 112u
#define COUNT_OFFSET 120u
#define WEIGHT_OFFSET 124u
#define AVERAGED_POSITION_OFFSET 128u
#define PARTICLE_VELOCITY_OFFSET 136u
#define RESPONSE_OFFSET 144u
#define DEST_ID_OFFSET 152u
#define SOURCE_ID_OFFSET 156u
#define FLAGS_OFFSET 160u
#define ERROR_OFFSET 164u
#define COMPACT_POSITION_OFFSET 168u
#define COMPACT_VELOCITY_OFFSET 176u
#define COMPACT_RESPONSE_OFFSET 184u
#define PARTICLE_PAIR_OFFSET 192u
#define PREVIOUS_POSITION_OFFSET 200u

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
    static const uint32_t input_normal[4] = {
        0x3f800000u, 0u, 0u, 0xdeadbeefu,
    };
    static const uint32_t input_accumulated_position[4] = {
        0x40e00000u, 0x41000000u, 0x41100000u, 0xcafebabeu,
    };
    static const uint32_t input_body_velocity[4] = {
        0u, 0u, 0u, 0xabcdef01u,
    };
    static const uint32_t input_particle_velocity[4] = {
        0xc0000000u, 0x40400000u, 0u, 0xbaadf00du,
    };
    static const uint32_t sentinel_vector[4] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu,
    };
    static const uint32_t expected_normal[4] = {
        0x3f800000u, 0u, 0u, 0xbaadf00du,
    };
    static const uint32_t expected_position[4] = {
        0x40e00000u, 0x41000000u, 0x41100000u, 0u,
    };
    static const uint32_t expected_velocity[4] = {
        0x3f800000u, 0x40400000u, 0u, 0u,
    };
    static const uint32_t expected_response[4] = {
        0x3f800000u, 0u, 0u, 0u,
    };
    static const uint32_t expected_pair[2] = {0u, 0x12345678u};
    static const uint32_t one[1] = {1u};
    static const uint32_t zero[1] = {0u};
    static const uint32_t source_id[1] = {0x12345678u};
    static const uint32_t sentinel_scalar[1] = {0xfeedfaceu};
    static const uint32_t sentinel_pair[2] = {0xaaaaaaaau, 0xbbbbbbbbu};
    static const size_t vector_offsets[] = {
        NORMAL_OFFSET,
        ACCUMULATED_POSITION_OFFSET,
        BODY_VELOCITY_OFFSET,
        AVERAGED_POSITION_OFFSET,
        PARTICLE_VELOCITY_OFFSET,
        RESPONSE_OFFSET,
        COMPACT_POSITION_OFFSET,
        COMPACT_VELOCITY_OFFSET,
        COMPACT_RESPONSE_OFFSET,
        PREVIOUS_POSITION_OFFSET,
    };
    static const size_t scalar_offsets[] = {
        COUNT_OFFSET,
        WEIGHT_OFFSET,
        DEST_ID_OFFSET,
        SOURCE_ID_OFFSET,
        FLAGS_OFFSET,
        ERROR_OFFSET,
    };
    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t work_unit[4] = {0u, 1u, 0u, 0u};
    uint32_t start_addrs[32] = {0};
    uint32_t parameters[40] = {0};
    uint32_t start_array[2];
    uint32_t work_array[2];
    uint32_t error_pointer;
    uint32_t pointer;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool guards_intact = true;
    bool pass = false;
    int exit_code = 1;
    size_t index;

    if (argc != 3) {
        fprintf(stderr,
                "usage: cuda-fluid-static-response-fixture "
                "<static-response.ptx> <output.json>\n");
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

    if (!heap_pointer(heap_device, START_ADDRS_OFFSET, &start_array[0]) ||
        !heap_pointer(heap_device, WORK_UNIT_OFFSET, &work_array[0]) ||
        !heap_pointer(heap_device, ERROR_OFFSET, &error_pointer)) {
        goto cleanup;
    }
    start_array[1] = 1u;
    work_array[1] = 1u;

#define SET_START(word, offset)                               \
    do {                                                      \
        if (!heap_pointer(heap_device, (offset), &pointer)) { \
            goto cleanup;                                     \
        }                                                     \
        start_addrs[(word)] = pointer;                        \
    } while (0)

    SET_START(0u, COMPACT_POSITION_OFFSET);
    SET_START(1u, COMPACT_VELOCITY_OFFSET);
    SET_START(2u, COMPACT_RESPONSE_OFFSET);
    SET_START(7u, PARTICLE_PAIR_OFFSET);
    SET_START(8u, NORMAL_OFFSET);
    SET_START(9u, ACCUMULATED_POSITION_OFFSET);
    SET_START(10u, BODY_VELOCITY_OFFSET);
    SET_START(11u, COUNT_OFFSET);
    SET_START(12u, WEIGHT_OFFSET);
    SET_START(13u, AVERAGED_POSITION_OFFSET);
    SET_START(14u, PARTICLE_VELOCITY_OFFSET);
    SET_START(15u, RESPONSE_OFFSET);
    SET_START(16u, DEST_ID_OFFSET);
    SET_START(17u, SOURCE_ID_OFFSET);
    SET_START(18u, FLAGS_OFFSET);
    SET_START(20u, PREVIOUS_POSITION_OFFSET);
    SET_START(27u, PARAMETERS_OFFSET);
#undef SET_START

    parameters[16] = 0x3f800000u;
    parameters[28] = 0x3f000000u;
    initialize_range(heap, WORK_UNIT_OFFSET, work_unit, 4u);
    initialize_range(heap, START_ADDRS_OFFSET, start_addrs, 32u);
    initialize_range(heap, PARAMETERS_OFFSET, parameters, 40u);
    initialize_range(heap, NORMAL_OFFSET, input_normal, 4u);
    initialize_range(
        heap, ACCUMULATED_POSITION_OFFSET, input_accumulated_position, 4u);
    initialize_range(heap, BODY_VELOCITY_OFFSET, input_body_velocity, 4u);
    initialize_range(heap, COUNT_OFFSET, one, 1u);
    initialize_range(heap, WEIGHT_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, AVERAGED_POSITION_OFFSET, sentinel_vector, 4u);
    initialize_range(
        heap, PARTICLE_VELOCITY_OFFSET, input_particle_velocity, 4u);
    initialize_range(heap, RESPONSE_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, DEST_ID_OFFSET, zero, 1u);
    initialize_range(heap, SOURCE_ID_OFFSET, source_id, 1u);
    initialize_range(heap, FLAGS_OFFSET, one, 1u);
    initialize_range(heap, ERROR_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COMPACT_POSITION_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COMPACT_VELOCITY_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COMPACT_RESPONSE_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, PARTICLE_PAIR_OFFSET, sentinel_pair, 2u);
    initialize_range(heap, PREVIOUS_POSITION_OFFSET, expected_position, 4u);

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    {
        void *launch_parameters[] = {start_array, work_array, &error_pointer};
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          128, 1, 1,
                                          0,
                                          NULL,
                                          launch_parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact = range_guards_match(heap, WORK_UNIT_OFFSET, 4u) &&
                    range_guards_match(heap, START_ADDRS_OFFSET, 32u) &&
                    range_guards_match(heap, PARAMETERS_OFFSET, 40u);
    for (index = 0u; index < sizeof(vector_offsets) / sizeof(vector_offsets[0]);
         ++index) {
        guards_intact =
            guards_intact && range_guards_match(heap, vector_offsets[index], 4u);
    }
    for (index = 0u; index < sizeof(scalar_offsets) / sizeof(scalar_offsets[0]);
         ++index) {
        guards_intact =
            guards_intact && range_guards_match(heap, scalar_offsets[index], 1u);
    }
    guards_intact =
        guards_intact && range_guards_match(heap, PARTICLE_PAIR_OFFSET, 2u);

    pass = guards_intact &&
           words_equal(heap, WORK_UNIT_OFFSET, work_unit, 4u) &&
           words_equal(heap, START_ADDRS_OFFSET, start_addrs, 32u) &&
           words_equal(heap, PARAMETERS_OFFSET, parameters, 40u) &&
           words_equal(heap, NORMAL_OFFSET, expected_normal, 4u) &&
           words_equal(
               heap, ACCUMULATED_POSITION_OFFSET, input_accumulated_position, 4u) &&
           words_equal(heap, BODY_VELOCITY_OFFSET, input_body_velocity, 4u) &&
           words_equal(heap, COUNT_OFFSET, one, 1u) &&
           words_equal(heap, WEIGHT_OFFSET, sentinel_scalar, 1u) &&
           words_equal(heap, AVERAGED_POSITION_OFFSET, expected_position, 4u) &&
           words_equal(
               heap, PARTICLE_VELOCITY_OFFSET, input_particle_velocity, 4u) &&
           words_equal(heap, RESPONSE_OFFSET, expected_response, 4u) &&
           words_equal(heap, DEST_ID_OFFSET, zero, 1u) &&
           words_equal(heap, SOURCE_ID_OFFSET, source_id, 1u) &&
           words_equal(heap, FLAGS_OFFSET, one, 1u) &&
           words_equal(heap, ERROR_OFFSET, sentinel_scalar, 1u) &&
           words_equal(heap, COMPACT_POSITION_OFFSET, expected_position, 4u) &&
           words_equal(heap, COMPACT_VELOCITY_OFFSET, expected_velocity, 4u) &&
           words_equal(heap, COMPACT_RESPONSE_OFFSET, expected_response, 4u) &&
           words_equal(heap, PARTICLE_PAIR_OFFSET, expected_pair, 2u) &&
           words_equal(heap, PREVIOUS_POSITION_OFFSET, expected_position, 4u);

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"PxgCudaFluidCollisionStaticResponseBatch\",\n"
            "  \"particle_count\": 1,\n"
            "  \"input_constraint_flags\": \"0x00000001\",\n"
            "  \"collision_radius_bits\": \"0x%08x\",\n"
            "  \"restitution_bits\": \"0x%08x\",\n"
            "  \"input_normal_bits\": ",
            device_name,
            parameters[16],
            parameters[28]);
    write_words(output, input_normal, 4u);
    fprintf(output, ",\n  \"input_accumulated_position_bits\": ");
    write_words(output, input_accumulated_position, 4u);
    fprintf(output, ",\n  \"input_particle_velocity_bits\": ");
    write_words(output, input_particle_velocity, 4u);
    fprintf(output, ",\n  \"output_normal_bits\": ");
    write_words(output, heap + NORMAL_OFFSET, 4u);
    fprintf(output, ",\n  \"output_position_bits\": ");
    write_words(output, heap + COMPACT_POSITION_OFFSET, 4u);
    fprintf(output, ",\n  \"output_velocity_bits\": ");
    write_words(output, heap + COMPACT_VELOCITY_OFFSET, 4u);
    fprintf(output, ",\n  \"output_response_bits\": ");
    write_words(output, heap + COMPACT_RESPONSE_OFFSET, 4u);
    fprintf(output, ",\n  \"output_particle_pair\": ");
    write_words(output, heap + PARTICLE_PAIR_OFFSET, 2u);
    fprintf(output,
            ",\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    printf("%s  %s: velocity=[0x%08x,0x%08x,0x%08x,0x%08x] "
           "response=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[COMPACT_VELOCITY_OFFSET],
           heap[COMPACT_VELOCITY_OFFSET + 1u],
           heap[COMPACT_VELOCITY_OFFSET + 2u],
           heap[COMPACT_VELOCITY_OFFSET + 3u],
           heap[COMPACT_RESPONSE_OFFSET],
           heap[COMPACT_RESPONSE_OFFSET + 1u],
           heap[COMPACT_RESPONSE_OFFSET + 2u],
           heap[COMPACT_RESPONSE_OFFSET + 3u]);
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
