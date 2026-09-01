#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UPDATE_KERNEL_NAME "_Z24PxgCudaDeformableUpdateDPKhjPi"
#define COLLISION_KERNEL_NAME "_Z29KernelDeformableUberCollisionPi"
#define SOLVER_KERNEL_NAME "_Z26KernelDeformableUberSolverILj512EEvjbPhPi"
#define UPDATE_CONSTANT_NAME "updateInstanceData"
#define COLLISION_INSTANCE_CONSTANT_NAME "gDeformableCollisionInstanceData"
#define COLLISION_BATCH_CONSTANT_NAME "gDeformableCollisionBatchedData"
#define SOLVER_CONSTANT_NAME "solverGpuData"
#define UPDATE_CONSTANT_BYTES 7168u
#define COLLISION_INSTANCE_CONSTANT_BYTES 17920u
#define COLLISION_BATCH_CONSTANT_BYTES 72u
#define SOLVER_CONSTANT_BYTES 5888u
#define SOLVER_DYNAMIC_SHARED_BYTES 7072u
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 320u

#define UPDATE_PACKED_OFFSET 8u
#define UPDATE_POSITION_OFFSET 16u
#define UPDATE_MOTION_OFFSET 24u
#define UPDATE_FLAGS_OFFSET 32u
#define UPDATE_IGNORED_OFFSET 36u
#define COLLISION_BATCH_OFFSET 40u
#define COLLISION_OUTPUT_8_OFFSET 48u
#define COLLISION_OUTPUT_12_OFFSET 56u
#define COLLISION_OUTPUT_20_OFFSET 64u
#define COLLISION_OUTPUT_POSITION_OFFSET 72u
#define COLLISION_OUTPUT_MOTION_OFFSET 80u
#define COLLISION_40_OFFSET 88u
#define COLLISION_44_OFFSET 96u
#define COLLISION_SCALAR_48_OFFSET 104u
#define COLLISION_SCALAR_52_OFFSET 108u
#define COLLISION_SCALAR_56_OFFSET 112u
#define COLLISION_SCALAR_60_OFFSET 116u
#define COLLISION_64_OFFSET 120u
#define COLLISION_ACTIVE_OFFSET 128u
#define SOLVER_A_OFFSET 144u
#define SOLVER_B_OFFSET 152u
#define SOLVER_C_OFFSET 160u
#define SOLVER_FLAGS_OFFSET 168u
#define SOLVER_PACKED_OFFSET 172u
#define SOLVER_OUTPUT_OFFSET 180u

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

static bool vector_equals(const uint32_t *heap, size_t offset, const uint32_t *expected)
{
    return memcmp(heap + offset, expected, 4u * sizeof(uint32_t)) == 0;
}

static void write_vector(FILE *output, const uint32_t *values, size_t words)
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
    static const uint32_t update_position[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0u,
    };
    static const uint32_t update_motion[4] = {
        0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u,
    };
    static const uint32_t expected_update_motion[4] = {
        0x40440000u, 0x408c0000u, 0x40c80000u, 0x40e00000u,
    };
    static const uint32_t solver_a[4] = {
        0x41200000u, 0x41300000u, 0x41400000u, 0u,
    };
    static const uint32_t solver_b[4] = {
        0x41a00000u, 0x41a80000u, 0x41b00000u, 0u,
    };
    static const uint32_t solver_c[4] = {
        0x41f00000u, 0x41f80000u, 0x42000000u, 0u,
    };
    static const uint32_t expected_solver_a[4] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0u,
    };
    static const uint32_t expected_solver_position[4] = {
        0xc1100000u, 0xc1100000u, 0xc1100000u, 0u,
    };
    static const uint32_t expected_solver_output[16] = {
        0x3f800000u, 0x40000000u, 0x40400000u, 0x00000400u,
        0x41f00000u, 0x41f80000u, 0x42000000u, 0xabcdef01u,
        0xc1100000u, 0xc1100000u, 0xc1100000u, 0u,
        0x41a00000u, 0x41a80000u, 0x41b00000u, 0u,
    };
    static const uint32_t sentinel_vector[4] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, 0xddddddddu,
    };
    static const uint32_t zero_vector[4] = {0u, 0u, 0u, 0u};
    static const uint32_t zero_output[16] = {0};
    static const uint32_t batch_record[2] = {0x00010000u, 0u};
    static const uint32_t zero_scalar[1] = {0u};
    static const uint32_t sentinel_scalar[1] = {0xfeedfaceu};
    static const uint32_t active_scalar[1] = {1u};
    static const uint32_t update_packed[1] = {0xfeedfaceu};
    static const uint32_t update_flags[1] = {0x00000014u};
    static const uint32_t update_ignored[1] = {0xcafebabeu};
    static const uint32_t solver_flags[1] = {0x00000400u};
    static const uint32_t solver_packed[1] = {0xabcdef01u};

    static const size_t range_offsets[] = {
        UPDATE_PACKED_OFFSET,
        UPDATE_POSITION_OFFSET,
        UPDATE_MOTION_OFFSET,
        UPDATE_FLAGS_OFFSET,
        UPDATE_IGNORED_OFFSET,
        COLLISION_BATCH_OFFSET,
        COLLISION_OUTPUT_8_OFFSET,
        COLLISION_OUTPUT_12_OFFSET,
        COLLISION_OUTPUT_20_OFFSET,
        COLLISION_OUTPUT_POSITION_OFFSET,
        COLLISION_OUTPUT_MOTION_OFFSET,
        COLLISION_40_OFFSET,
        COLLISION_44_OFFSET,
        COLLISION_SCALAR_48_OFFSET,
        COLLISION_SCALAR_52_OFFSET,
        COLLISION_SCALAR_56_OFFSET,
        COLLISION_SCALAR_60_OFFSET,
        COLLISION_64_OFFSET,
        COLLISION_ACTIVE_OFFSET,
        SOLVER_A_OFFSET,
        SOLVER_B_OFFSET,
        SOLVER_C_OFFSET,
        SOLVER_FLAGS_OFFSET,
        SOLVER_PACKED_OFFSET,
        SOLVER_OUTPUT_OFFSET,
    };
    static const size_t range_words[] = {
        1u, 4u, 4u, 1u, 1u, 2u, 4u, 4u, 1u, 4u, 4u, 4u, 4u,
        1u, 1u, 1u, 1u, 4u, 1u, 4u, 4u, 4u, 1u, 1u, 16u,
    };

    cuda_api api;
    char *update_ptx = NULL;
    char *collision_ptx = NULL;
    char *solver_ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule update_module = NULL;
    CUmodule collision_module = NULL;
    CUmodule solver_module = NULL;
    CUfunction update_function = NULL;
    CUfunction collision_function = NULL;
    CUfunction solver_function = NULL;
    CUdeviceptr update_constant_device = 0;
    CUdeviceptr collision_instance_constant_device = 0;
    CUdeviceptr collision_batch_constant_device = 0;
    CUdeviceptr solver_constant_device = 0;
    size_t update_constant_bytes = 0;
    size_t collision_instance_constant_bytes = 0;
    size_t collision_batch_constant_bytes = 0;
    size_t solver_constant_bytes = 0;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t update_constants[UPDATE_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t collision_instance_constants[COLLISION_INSTANCE_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t collision_batch_constants[COLLISION_BATCH_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t solver_constants[SOLVER_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t update_packed_pointer = 0;
    uint32_t update_position_pointer = 0;
    uint32_t update_motion_pointer = 0;
    uint32_t update_flags_pointer = 0;
    uint32_t update_ignored_pointer = 0;
    uint32_t collision_batch_pointer = 0;
    uint32_t collision_output_8_pointer = 0;
    uint32_t collision_output_12_pointer = 0;
    uint32_t collision_output_20_pointer = 0;
    uint32_t collision_output_position_pointer = 0;
    uint32_t collision_output_motion_pointer = 0;
    uint32_t collision_40_pointer = 0;
    uint32_t collision_44_pointer = 0;
    uint32_t collision_scalar_48_pointer = 0;
    uint32_t collision_scalar_52_pointer = 0;
    uint32_t collision_scalar_56_pointer = 0;
    uint32_t collision_scalar_60_pointer = 0;
    uint32_t collision_64_pointer = 0;
    uint32_t collision_active_pointer = 0;
    uint32_t solver_a_pointer = 0;
    uint32_t solver_b_pointer = 0;
    uint32_t solver_c_pointer = 0;
    uint32_t solver_flags_pointer = 0;
    uint32_t solver_packed_pointer = 0;
    uint32_t solver_output_pointer = 0;
    uint32_t update_instance_count = 1u;
    uint32_t collision_error_pointer = 0u;
    uint32_t solver_iteration_count = 0u;
    uint8_t solver_write_velocity = 0u;
    uint32_t solver_error_pointer = 0u;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool guards_intact = false;
    bool pass = false;
    int exit_code = 1;
    size_t index;

    if (argc != 5) {
        fprintf(stderr,
                "usage: cuda-deformable-sequence-fixture <update.ptx> <collision.ptx> <solver.ptx> <output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) {
        return 1;
    }
    update_ptx = read_text_file(argv[1]);
    collision_ptx = read_text_file(argv[2]);
    solver_ptx = read_text_file(argv[3]);
    if (update_ptx == NULL || collision_ptx == NULL || solver_ptx == NULL) {
        goto cleanup;
    }

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    CUDA_TRY(&api, api.cuModuleLoadData(&update_module, update_ptx));
    CUDA_TRY(&api, api.cuModuleLoadData(&collision_module, collision_ptx));
    CUDA_TRY(&api, api.cuModuleLoadData(&solver_module, solver_ptx));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(&update_function, update_module, UPDATE_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(
                 &collision_function, collision_module, COLLISION_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetFunction(&solver_function, solver_module, SOLVER_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(&update_constant_device,
                                      &update_constant_bytes,
                                      update_module,
                                      UPDATE_CONSTANT_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(&collision_instance_constant_device,
                                      &collision_instance_constant_bytes,
                                      collision_module,
                                      COLLISION_INSTANCE_CONSTANT_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(&collision_batch_constant_device,
                                      &collision_batch_constant_bytes,
                                      collision_module,
                                      COLLISION_BATCH_CONSTANT_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(&solver_constant_device,
                                      &solver_constant_bytes,
                                      solver_module,
                                      SOLVER_CONSTANT_NAME));
    if (update_constant_bytes < UPDATE_CONSTANT_BYTES ||
        collision_instance_constant_bytes < COLLISION_INSTANCE_CONSTANT_BYTES ||
        collision_batch_constant_bytes < COLLISION_BATCH_CONSTANT_BYTES ||
        solver_constant_bytes < SOLVER_CONSTANT_BYTES) {
        fprintf(stderr, "deformable sequence constant symbol is too small\n");
        goto cleanup;
    }

    initialize_range(heap, UPDATE_PACKED_OFFSET, update_packed, 1u);
    initialize_range(heap, UPDATE_POSITION_OFFSET, update_position, 4u);
    initialize_range(heap, UPDATE_MOTION_OFFSET, update_motion, 4u);
    initialize_range(heap, UPDATE_FLAGS_OFFSET, update_flags, 1u);
    initialize_range(heap, UPDATE_IGNORED_OFFSET, update_ignored, 1u);
    initialize_range(heap, COLLISION_BATCH_OFFSET, batch_record, 2u);
    initialize_range(heap, COLLISION_OUTPUT_8_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_OUTPUT_12_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_OUTPUT_20_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_OUTPUT_POSITION_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_OUTPUT_MOTION_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_40_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_44_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_SCALAR_48_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_SCALAR_52_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_SCALAR_56_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_SCALAR_60_OFFSET, sentinel_scalar, 1u);
    initialize_range(heap, COLLISION_64_OFFSET, sentinel_vector, 4u);
    initialize_range(heap, COLLISION_ACTIVE_OFFSET, active_scalar, 1u);
    initialize_range(heap, SOLVER_A_OFFSET, solver_a, 4u);
    initialize_range(heap, SOLVER_B_OFFSET, solver_b, 4u);
    initialize_range(heap, SOLVER_C_OFFSET, solver_c, 4u);
    initialize_range(heap, SOLVER_FLAGS_OFFSET, solver_flags, 1u);
    initialize_range(heap, SOLVER_PACKED_OFFSET, solver_packed, 1u);
    initialize_range(heap, SOLVER_OUTPUT_OFFSET, zero_output, 16u);

    CUDA_TRY(&api, api.cuMemAlloc_v2(&heap_device, sizeof(heap)));
    if (!heap_pointer(heap_device, UPDATE_PACKED_OFFSET, &update_packed_pointer) ||
        !heap_pointer(heap_device, UPDATE_POSITION_OFFSET, &update_position_pointer) ||
        !heap_pointer(heap_device, UPDATE_MOTION_OFFSET, &update_motion_pointer) ||
        !heap_pointer(heap_device, UPDATE_FLAGS_OFFSET, &update_flags_pointer) ||
        !heap_pointer(heap_device, UPDATE_IGNORED_OFFSET, &update_ignored_pointer) ||
        !heap_pointer(heap_device, COLLISION_BATCH_OFFSET, &collision_batch_pointer) ||
        !heap_pointer(heap_device, COLLISION_OUTPUT_8_OFFSET, &collision_output_8_pointer) ||
        !heap_pointer(heap_device, COLLISION_OUTPUT_12_OFFSET, &collision_output_12_pointer) ||
        !heap_pointer(heap_device, COLLISION_OUTPUT_20_OFFSET, &collision_output_20_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_OUTPUT_POSITION_OFFSET,
                      &collision_output_position_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_OUTPUT_MOTION_OFFSET,
                      &collision_output_motion_pointer) ||
        !heap_pointer(heap_device, COLLISION_40_OFFSET, &collision_40_pointer) ||
        !heap_pointer(heap_device, COLLISION_44_OFFSET, &collision_44_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_SCALAR_48_OFFSET,
                      &collision_scalar_48_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_SCALAR_52_OFFSET,
                      &collision_scalar_52_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_SCALAR_56_OFFSET,
                      &collision_scalar_56_pointer) ||
        !heap_pointer(heap_device,
                      COLLISION_SCALAR_60_OFFSET,
                      &collision_scalar_60_pointer) ||
        !heap_pointer(heap_device, COLLISION_64_OFFSET, &collision_64_pointer) ||
        !heap_pointer(heap_device, COLLISION_ACTIVE_OFFSET, &collision_active_pointer) ||
        !heap_pointer(heap_device, SOLVER_A_OFFSET, &solver_a_pointer) ||
        !heap_pointer(heap_device, SOLVER_B_OFFSET, &solver_b_pointer) ||
        !heap_pointer(heap_device, SOLVER_C_OFFSET, &solver_c_pointer) ||
        !heap_pointer(heap_device, SOLVER_FLAGS_OFFSET, &solver_flags_pointer) ||
        !heap_pointer(heap_device, SOLVER_PACKED_OFFSET, &solver_packed_pointer) ||
        !heap_pointer(heap_device, SOLVER_OUTPUT_OFFSET, &solver_output_pointer)) {
        goto cleanup;
    }

    update_constants[0] = 0x3e800000u;
    update_constants[1] = 0xbf000000u;
    update_constants[2] = 0x3f800000u;
    update_constants[7] = 1u;
    update_constants[15] = 0x3f000000u;
    update_constants[16] = update_position_pointer;
    update_constants[17] = update_motion_pointer;
    update_constants[22] = update_flags_pointer;
    update_constants[24] = 0u;
    update_constants[25] = 0u;

    collision_instance_constants[0] = update_position_pointer;
    collision_instance_constants[1] = update_motion_pointer;
    collision_instance_constants[2] = collision_output_8_pointer;
    collision_instance_constants[3] = collision_output_12_pointer;
    collision_instance_constants[5] = collision_output_20_pointer;
    collision_instance_constants[6] = update_flags_pointer;
    collision_instance_constants[8] = collision_output_position_pointer;
    collision_instance_constants[9] = collision_output_motion_pointer;
    collision_instance_constants[10] = collision_40_pointer;
    collision_instance_constants[11] = collision_44_pointer;
    collision_instance_constants[12] = collision_scalar_48_pointer;
    collision_instance_constants[13] = collision_scalar_52_pointer;
    collision_instance_constants[14] = collision_scalar_56_pointer;
    collision_instance_constants[15] = collision_scalar_60_pointer;
    collision_instance_constants[16] = collision_64_pointer;
    collision_instance_constants[17] = collision_active_pointer;
    collision_instance_constants[28] = 0u;
    collision_instance_constants[33] = 0u;
    collision_instance_constants[34] = 0u;
    collision_batch_constants[17] = collision_batch_pointer;

    solver_constants[1] = 1u;
    solver_constants[8] = 0x3f800000u;
    solver_constants[9] = 0u;
    solver_constants[14] = solver_a_pointer;
    solver_constants[15] = collision_output_position_pointer;
    solver_constants[16] = solver_b_pointer;
    solver_constants[17] = solver_c_pointer;
    solver_constants[18] = solver_c_pointer;
    solver_constants[21] = solver_flags_pointer;
    solver_constants[22] = solver_packed_pointer;

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 update_constant_device, update_constants, UPDATE_CONSTANT_BYTES));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(collision_instance_constant_device,
                                  collision_instance_constants,
                                  COLLISION_INSTANCE_CONSTANT_BYTES));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(collision_batch_constant_device,
                                  collision_batch_constants,
                                  COLLISION_BATCH_CONSTANT_BYTES));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 solver_constant_device, solver_constants, SOLVER_CONSTANT_BYTES));

    {
        void *parameters[] = {
            &update_packed_pointer, &update_instance_count, &update_ignored_pointer,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(update_function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    {
        void *parameters[] = {&collision_error_pointer};
        CUDA_TRY(&api, api.cuLaunchKernel(collision_function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    {
        void *parameters[] = {
            &solver_iteration_count,
            &solver_write_velocity,
            &solver_output_pointer,
            &solver_error_pointer,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(solver_function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          SOLVER_DYNAMIC_SHARED_BYTES,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact = true;
    for (index = 0u; index < sizeof(range_offsets) / sizeof(range_offsets[0]); ++index) {
        guards_intact = guards_intact &&
                        range_guards_match(heap, range_offsets[index], range_words[index]);
    }
    pass = guards_intact &&
           vector_equals(heap, UPDATE_POSITION_OFFSET, update_position) &&
           vector_equals(heap, UPDATE_MOTION_OFFSET, expected_update_motion) &&
           heap[UPDATE_FLAGS_OFFSET] == 0u &&
           vector_equals(heap, COLLISION_OUTPUT_8_OFFSET, zero_vector) &&
           vector_equals(heap, COLLISION_OUTPUT_12_OFFSET, zero_vector) &&
           heap[COLLISION_OUTPUT_20_OFFSET] == 0xbf800000u &&
           vector_equals(heap, COLLISION_OUTPUT_MOTION_OFFSET, expected_update_motion) &&
           vector_equals(heap, COLLISION_40_OFFSET, zero_vector) &&
           vector_equals(heap, COLLISION_44_OFFSET, zero_vector) &&
           heap[COLLISION_SCALAR_48_OFFSET] == 0xbf800000u &&
           heap[COLLISION_SCALAR_52_OFFSET] == 0u &&
           heap[COLLISION_SCALAR_56_OFFSET] == 0x0000ffffu &&
           heap[COLLISION_SCALAR_60_OFFSET] == 0u &&
           vector_equals(heap, COLLISION_64_OFFSET, zero_vector) &&
           heap[COLLISION_ACTIVE_OFFSET] == 0u &&
           vector_equals(heap, SOLVER_A_OFFSET, expected_solver_a) &&
           vector_equals(heap,
                         COLLISION_OUTPUT_POSITION_OFFSET,
                         expected_solver_position) &&
           memcmp(heap + SOLVER_OUTPUT_OFFSET,
                  expected_solver_output,
                  sizeof(expected_solver_output)) == 0;

    output = fopen(argv[4], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[4]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel_order\": [\"PxgCudaDeformableUpdate\", \"KernelDeformableUberCollision\", \"KernelDeformableUberSolver<512>\"],\n"
            "  \"particle_count\": 1,\n"
            "  \"shape_count\": 0,\n"
            "  \"iteration_count\": 0,\n"
            "  \"update_motion_bits\": ",
            device_name);
    write_vector(output, heap + UPDATE_MOTION_OFFSET, 4u);
    fprintf(output, ",\n  \"collision_motion_bits\": ");
    write_vector(output, heap + COLLISION_OUTPUT_MOTION_OFFSET, 4u);
    fprintf(output,
            ",\n  \"collision_scalar_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"]",
            heap[COLLISION_SCALAR_48_OFFSET],
            heap[COLLISION_SCALAR_52_OFFSET],
            heap[COLLISION_SCALAR_56_OFFSET],
            heap[COLLISION_SCALAR_60_OFFSET]);
    fprintf(output, ",\n  \"solver_a_bits\": ");
    write_vector(output, heap + SOLVER_A_OFFSET, 4u);
    fprintf(output, ",\n  \"solver_position_bits\": ");
    write_vector(output, heap + COLLISION_OUTPUT_POSITION_OFFSET, 4u);
    fprintf(output, ",\n  \"solver_output_bits\": ");
    write_vector(output, heap + SOLVER_OUTPUT_OFFSET, 16u);
    fprintf(output,
            ",\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    printf("%s  %s: update->collision->solver output[0..3]=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[SOLVER_OUTPUT_OFFSET],
           heap[SOLVER_OUTPUT_OFFSET + 1u],
           heap[SOLVER_OUTPUT_OFFSET + 2u],
           heap[SOLVER_OUTPUT_OFFSET + 3u]);
    exit_code = pass ? 0 : 1;

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    if (solver_module != NULL) {
        api.cuModuleUnload(solver_module);
    }
    if (collision_module != NULL) {
        api.cuModuleUnload(collision_module);
    }
    if (update_module != NULL) {
        api.cuModuleUnload(update_module);
    }
    if (heap_device != 0) {
        api.cuMemFree_v2(heap_device);
    }
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    free(solver_ptx);
    free(collision_ptx);
    free(update_ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return exit_code;
}
