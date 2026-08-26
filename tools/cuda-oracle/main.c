#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MATH_RESULT_COUNT 14
#define BARRIER_THREAD_COUNT 64
#define TEXTURE_ELEMENT_COUNT 8
#define TEXTURE_PROBE_COUNT 12

typedef struct cuda_api {
    HMODULE library;
    CUresult(CUDAAPI *cuInit)(unsigned int);
    CUresult(CUDAAPI *cuDeviceGet)(CUdevice *, int);
    CUresult(CUDAAPI *cuDeviceGetName)(char *, int, CUdevice);
    CUresult(CUDAAPI *cuCtxCreate_v2)(CUcontext *, unsigned int, CUdevice);
    CUresult(CUDAAPI *cuCtxDestroy_v2)(CUcontext);
    CUresult(CUDAAPI *cuCtxSynchronize)(void);
    CUresult(CUDAAPI *cuModuleLoadData)(CUmodule *, const void *);
    CUresult(CUDAAPI *cuModuleLoadDataEx)(
        CUmodule *, const void *, unsigned int, CUjit_option *, void **);
    CUresult(CUDAAPI *cuModuleUnload)(CUmodule);
    CUresult(CUDAAPI *cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult(CUDAAPI *cuModuleGetTexRef)(CUtexref *, CUmodule, const char *);
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

#define LOAD_API(api, name)                                                               \
    do {                                                                                  \
        if (!load_function((api)->library, #name, &(api)->name, sizeof((api)->name))) {   \
            return false;                                                                 \
        }                                                                                 \
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
    LOAD_API(api, cuModuleLoadDataEx);
    LOAD_API(api, cuModuleUnload);
    LOAD_API(api, cuModuleGetFunction);
    LOAD_API(api, cuModuleGetTexRef);
    LOAD_API(api, cuMemAlloc_v2);
    LOAD_API(api, cuMemFree_v2);
    LOAD_API(api, cuMemcpyHtoD_v2);
    LOAD_API(api, cuMemcpyDtoH_v2);
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
    if (api->cuGetErrorName != NULL) {
        api->cuGetErrorName(result, &name);
    }
    if (api->cuGetErrorString != NULL) {
        api->cuGetErrorString(result, &description);
    }
    fprintf(stderr,
            "%s failed: %d (%s: %s)\n",
            operation,
            (int)result,
            name != NULL ? name : "unknown",
            description != NULL ? description : "no description");
}

#define CUDA_TRY(api, expression)                                    \
    do {                                                             \
        CUresult cuda_result__ = (expression);                        \
        if (cuda_result__ != CUDA_SUCCESS) {                          \
            report_cuda_error((api), #expression, cuda_result__);     \
            goto cleanup;                                            \
        }                                                            \
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
    result = malloc((size_t)length + 1);
    if (result == NULL || fread(result, 1, (size_t)length, file) != (size_t)length) {
        free(result);
        fclose(file);
        return NULL;
    }
    result[length] = '\0';
    fclose(file);
    return result;
}

static CUresult launch_one_pointer(
    cuda_api *api,
    CUfunction function,
    unsigned int grid_x,
    unsigned int block_x,
    unsigned int shared_bytes,
    CUdeviceptr pointer)
{
    void *parameters[] = {&pointer};
    return api->cuLaunchKernel(function,
                               grid_x,
                               1,
                               1,
                               block_x,
                               1,
                               1,
                               shared_bytes,
                               NULL,
                               parameters,
                               NULL);
}

static void write_hex(FILE *output, uint32_t value)
{
    fprintf(output, "\"0x%08x\"", value);
}

int main(int argc, char **argv)
{
    static const char *math_names[MATH_RESULT_COUNT] = {
        "div_one_third",
        "div_min_normal_half_ftz",
        "rcp_three_approx",
        "sqrt_two_approx",
        "sqrt_min_subnormal_ftz",
        "rsqrt_two_approx",
        "fma_fused",
        "mul_add_separate",
        "cvt_rzi_neg3_75",
        "cvt_rn_u32_16777217",
        "add_min_subnormal_zero_ftz",
        "mul_min_subnormal_one_ftz",
        "min_nan_one",
        "max_nan_one",
    };
    cuda_api api;
    char *ptx = NULL;
    FILE *output = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction math_function = NULL;
    CUfunction barrier_function = NULL;
    CUfunction texture_function = NULL;
    CUtexref texture = NULL;
    CUdeviceptr math_device = 0;
    CUdeviceptr barrier_device = 0;
    CUdeviceptr texture_input_device = 0;
    CUdeviceptr texture_output_device = 0;
    uint32_t math_results[MATH_RESULT_COUNT] = {0};
    uint32_t barrier_results[BARRIER_THREAD_COUNT] = {0};
    uint32_t texture_input[TEXTURE_ELEMENT_COUNT][4];
    uint32_t texture_results[TEXTURE_PROBE_COUNT][4] = {{0}};
    char device_name[256] = {0};
    size_t texture_offset = ~(size_t)0;
    bool success = false;
    unsigned int index;
    char jit_error[8192] = {0};
    char jit_info[8192] = {0};
    CUjit_option jit_options[] = {
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
    };
    void *jit_values[] = {
        jit_error,
        (void *)(uintptr_t)sizeof(jit_error),
        jit_info,
        (void *)(uintptr_t)sizeof(jit_info),
    };
    CUresult module_load_result;

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-oracle <probe.ptx> <output.json>\n");
        return 2;
    }
    if (!load_cuda(&api)) {
        return 3;
    }
    ptx = read_text_file(argv[1]);
    if (ptx == NULL) {
        return 4;
    }
    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        free(ptx);
        return 5;
    }

    for (index = 0; index < TEXTURE_ELEMENT_COUNT; ++index) {
        texture_input[index][0] = 0x10000000u | (index << 8) | 0u;
        texture_input[index][1] = 0x20000000u | (index << 8) | 1u;
        texture_input[index][2] = 0x30000000u | (index << 8) | 2u;
        texture_input[index][3] = 0x40000000u | (index << 8) | 3u;
    }

    CUDA_TRY(&api, api.cuInit(0));
    CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
    CUDA_TRY(&api, api.cuDeviceGetName(device_name, sizeof(device_name), device));
    CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
    module_load_result = api.cuModuleLoadDataEx(
        &module,
        ptx,
        sizeof(jit_options) / sizeof(jit_options[0]),
        jit_options,
        jit_values);
    if (module_load_result != CUDA_SUCCESS) {
        report_cuda_error(&api, "cuModuleLoadDataEx", module_load_result);
        fprintf(stderr, "PTX JIT error log:\n%s\nPTX JIT info log:\n%s\n", jit_error, jit_info);
        goto cleanup;
    }
    CUDA_TRY(&api, api.cuModuleGetFunction(&math_function, module, "math_probe"));
    CUDA_TRY(&api, api.cuModuleGetFunction(&barrier_function, module, "barrier_probe"));
    CUDA_TRY(&api, api.cuModuleGetFunction(&texture_function, module, "texture_probe"));
    CUDA_TRY(&api, api.cuModuleGetTexRef(&texture, module, "probe_tex"));

    CUDA_TRY(&api, api.cuMemAlloc_v2(&math_device, sizeof(math_results)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&barrier_device, sizeof(barrier_results)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&texture_input_device, sizeof(texture_input)));
    CUDA_TRY(&api, api.cuMemAlloc_v2(&texture_output_device, sizeof(texture_results)));
    CUDA_TRY(&api,
             api.cuMemcpyHtoD_v2(texture_input_device, texture_input, sizeof(texture_input)));

    CUDA_TRY(&api,
             api.cuTexRefSetFormat(texture, CU_AD_FORMAT_UNSIGNED_INT32, 4));
    CUDA_TRY(&api,
             api.cuTexRefSetFlags(texture, CU_TRSF_READ_AS_INTEGER));
    CUDA_TRY(&api,
             api.cuTexRefSetFilterMode(texture, CU_TR_FILTER_MODE_POINT));
    CUDA_TRY(&api,
             api.cuTexRefSetAddressMode(texture, 0, CU_TR_ADDRESS_MODE_CLAMP));
    CUDA_TRY(&api,
             api.cuTexRefSetAddress_v2(
                 &texture_offset, texture, texture_input_device, sizeof(texture_input)));
    if (texture_offset != 0) {
        fprintf(stderr, "unexpected texture byte offset: %lu\n", (unsigned long)texture_offset);
        goto cleanup;
    }

    CUDA_TRY(&api, launch_one_pointer(&api, math_function, 1, 1, 0, math_device));
    CUDA_TRY(&api,
             launch_one_pointer(
                 &api, barrier_function, 1, BARRIER_THREAD_COUNT, 0, barrier_device));
    CUDA_TRY(&api,
             launch_one_pointer(
                 &api, texture_function, 1, TEXTURE_PROBE_COUNT, 0, texture_output_device));
    CUDA_TRY(&api, api.cuCtxSynchronize());

    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(math_results, math_device, sizeof(math_results)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(barrier_results, barrier_device, sizeof(barrier_results)));
    CUDA_TRY(&api,
             api.cuMemcpyDtoH_v2(
                 texture_results, texture_output_device, sizeof(texture_results)));

    fprintf(output, "{\n  \"device\": \"%s\",\n  \"math\": {\n", device_name);
    for (index = 0; index < MATH_RESULT_COUNT; ++index) {
        fprintf(output, "    \"%s\": ", math_names[index]);
        write_hex(output, math_results[index]);
        fprintf(output, "%s\n", index + 1 == MATH_RESULT_COUNT ? "" : ",");
    }
    fprintf(output, "  },\n  \"barrier\": {\n    \"values\": [");
    for (index = 0; index < BARRIER_THREAD_COUNT; ++index) {
        fprintf(output, "%s%u", index == 0 ? "" : ", ", barrier_results[index]);
    }
    fprintf(output, "]\n  },\n  \"texture\": {\n    \"rows\": [\n");
    for (index = 0; index < TEXTURE_PROBE_COUNT; ++index) {
        unsigned int component;
        fprintf(output, "      {\"coord\": %d, \"value\": [", (int)index - 2);
        for (component = 0; component < 4; ++component) {
            if (component != 0) {
                fputs(", ", output);
            }
            write_hex(output, texture_results[index][component]);
        }
        fprintf(output, "]}%s\n", index + 1 == TEXTURE_PROBE_COUNT ? "" : ",");
    }
    fprintf(output, "    ]\n  }\n}\n");
    success = true;

cleanup:
    if (texture_output_device != 0) {
        api.cuMemFree_v2(texture_output_device);
    }
    if (texture_input_device != 0) {
        api.cuMemFree_v2(texture_input_device);
    }
    if (barrier_device != 0) {
        api.cuMemFree_v2(barrier_device);
    }
    if (math_device != 0) {
        api.cuMemFree_v2(math_device);
    }
    if (module != NULL) {
        api.cuModuleUnload(module);
    }
    if (context != NULL) {
        api.cuCtxDestroy_v2(context);
    }
    if (output != NULL) {
        fclose(output);
    }
    free(ptx);
    if (api.library != NULL) {
        FreeLibrary(api.library);
    }
    return success ? 0 : 1;
}
