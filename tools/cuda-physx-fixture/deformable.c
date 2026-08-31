#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFORMABLE_UPDATE_KERNEL_NAME "_Z24PxgCudaDeformableUpdateDPKhjPi"
#define INSTANCE_CONSTANT_BYTES 7168u
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u

static const uint32_t INPUT_POSITION[4] = {
    0x3f800000u, 0x40000000u, 0x40400000u, 0x00000000u,
};
static const uint32_t INPUT_MOTION[4] = {
    0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u,
};
static const uint32_t EXPECTED_MOTION[4] = {
    0x40440000u, 0x408c0000u, 0x40c80000u, 0x40e00000u,
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

static void free_device(cuda_api *api, CUdeviceptr *pointer)
{
    if (*pointer != 0) {
        api->cuMemFree_v2(*pointer);
        *pointer = 0;
    }
}

static bool guards_match(const uint32_t *words, size_t count)
{
    return count >= 2u && words[0] == GUARD_BEFORE && words[count - 1u] == GUARD_AFTER;
}

static bool vector_guards_match(const uint32_t *words)
{
    return words[3] == GUARD_BEFORE && words[8] == GUARD_AFTER;
}

int main(int argc, char **argv)
{
    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr constants_device = 0;
    size_t constants_bytes = 0;
    CUdeviceptr packed_device = 0;
    CUdeviceptr position_device = 0;
    CUdeviceptr motion_device = 0;
    CUdeviceptr flags_device = 0;
    CUdeviceptr ignored_device = 0;
    uint32_t packed_words[3] = {GUARD_BEFORE, 0xfeedfaceu, GUARD_AFTER};
    uint32_t position_words[9] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, GUARD_BEFORE,
        0x3f800000u, 0x40000000u, 0x40400000u, 0x00000000u,
        GUARD_AFTER,
    };
    uint32_t motion_words[9] = {
        0xaaaaaaaau, 0xbbbbbbbbu, 0xccccccccu, GUARD_BEFORE,
        0x40800000u, 0x40a00000u, 0x40c00000u, 0x40e00000u,
        GUARD_AFTER,
    };
    uint32_t flags_words[3] = {GUARD_BEFORE, 0x00000014u, GUARD_AFTER};
    uint32_t ignored_words[3] = {GUARD_BEFORE, 0xcafebabeu, GUARD_AFTER};
    uint32_t constant_words[INSTANCE_CONSTANT_BYTES / sizeof(uint32_t)] = {0};
    uint32_t packed_pointer = 0;
    uint32_t position_pointer = 0;
    uint32_t motion_pointer = 0;
    uint32_t flags_pointer = 0;
    uint32_t ignored_pointer = 0;
    uint32_t instance_count = 1u;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool pass = false;
    int exit_code = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-deformable-fixture <deformable.ptx> <output.json>\n");
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
    CUDA_TRY(&api,
             api.cuModuleGetFunction(&function, module, DEFORMABLE_UPDATE_KERNEL_NAME));
    CUDA_TRY(&api,
             api.cuModuleGetGlobal_v2(
                 &constants_device, &constants_bytes, module, "updateInstanceData"));
    if (constants_bytes < INSTANCE_CONSTANT_BYTES) {
        fprintf(stderr, "updateInstanceData symbol is too small\n");
        goto cleanup;
    }

    CUDA_TRY(&api, api.cuMemAlloc_v2(&packed_device, sizeof(packed_words)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&position_device, sizeof(position_words)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&motion_device, sizeof(motion_words)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&flags_device, sizeof(flags_words)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&ignored_device, sizeof(ignored_words)));
    if (!pointer32(packed_device + sizeof(uint32_t), &packed_pointer) ||
        !pointer32(position_device + 4u * sizeof(uint32_t), &position_pointer) ||
        !pointer32(motion_device + 4u * sizeof(uint32_t), &motion_pointer) ||
        !pointer32(flags_device + sizeof(uint32_t), &flags_pointer) ||
        !pointer32(ignored_device + sizeof(uint32_t), &ignored_pointer)) {
        goto cleanup;
    }

    constant_words[0] = 0x3e800000u;
    constant_words[1] = 0xbf000000u;
    constant_words[2] = 0x3f800000u;
    constant_words[7] = 1u;
    constant_words[15] = 0x3f000000u;
    constant_words[16] = position_pointer;
    constant_words[17] = motion_pointer;
    constant_words[22] = flags_pointer;
    constant_words[24] = 0u;
    constant_words[25] = 0u;

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(packed_device, packed_words, sizeof(packed_words)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(position_device, position_words, sizeof(position_words)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(motion_device, motion_words, sizeof(motion_words)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(flags_device, flags_words, sizeof(flags_words)));
    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(ignored_device, ignored_words, sizeof(ignored_words)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(
                 constants_device, constant_words, INSTANCE_CONSTANT_BYTES));

    {
        void *parameters[] = {&packed_pointer, &instance_count, &ignored_pointer};
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          64, 1, 1,
                                          0,
                                          NULL,
                                          parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(packed_words, packed_device, sizeof(packed_words)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(position_words, position_device, sizeof(position_words)));
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(motion_words, motion_device, sizeof(motion_words)));
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(flags_words, flags_device, sizeof(flags_words)));
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(ignored_words, ignored_device, sizeof(ignored_words)));

    pass = guards_match(packed_words, 3u) &&
           vector_guards_match(position_words) &&
           vector_guards_match(motion_words) &&
           guards_match(flags_words, 3u) &&
           guards_match(ignored_words, 3u) &&
           packed_words[1] == 0xfeedfaceu &&
           memcmp(position_words + 4u, INPUT_POSITION, sizeof(INPUT_POSITION)) == 0 &&
           memcmp(motion_words + 4u, EXPECTED_MOTION, sizeof(EXPECTED_MOTION)) == 0 &&
           flags_words[1] == 0u &&
           ignored_words[1] == 0xcafebabeu;

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"PxgCudaDeformableUpdate\",\n"
            "  \"instance_count\": 1,\n"
            "  \"acceleration_bits\": [\"0x3e800000\", \"0xbf000000\", \"0x3f800000\"],\n"
            "  \"time_step_bits\": \"0x3f000000\",\n"
            "  \"input_position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"input_motion_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"input_flags\": \"0x00000014\",\n"
            "  \"output_position_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"output_motion_bits\": [\"0x%08x\", \"0x%08x\", \"0x%08x\", \"0x%08x\"],\n"
            "  \"output_flags\": \"0x%08x\",\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            device_name,
            INPUT_POSITION[0], INPUT_POSITION[1], INPUT_POSITION[2], INPUT_POSITION[3],
            INPUT_MOTION[0], INPUT_MOTION[1], INPUT_MOTION[2], INPUT_MOTION[3],
            position_words[4], position_words[5], position_words[6], position_words[7],
            motion_words[4], motion_words[5], motion_words[6], motion_words[7],
            flags_words[1],
            guards_match(packed_words, 3u) &&
                    vector_guards_match(position_words) &&
                    vector_guards_match(motion_words) &&
                    guards_match(flags_words, 3u) &&
                    guards_match(ignored_words, 3u)
                ? "true"
                : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;
    printf("%s  %s: motion=[0x%08x,0x%08x,0x%08x,0x%08x] flags=0x%08x\n",
           pass ? "PASS" : "FAIL",
           device_name,
           motion_words[4], motion_words[5], motion_words[6], motion_words[7],
           flags_words[1]);
    exit_code = pass ? 0 : 1;

cleanup:
    if (output != NULL) {
        fclose(output);
    }
    if (module != NULL) {
        api.cuModuleUnload(module);
    }
    free_device(&api, &ignored_device);
    free_device(&api, &flags_device);
    free_device(&api, &motion_device);
    free_device(&api, &position_device);
    free_device(&api, &packed_device);
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    free(ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return exit_code;
}
