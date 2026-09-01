#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_NAME "_Z15radixSortBlocksILj32ELj0ELb0EEvP5uint4S1_S1_S1_j"
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 160u
#define KEYS_OFFSET 8u
#define VALUES_OFFSET 80u
#define ELEMENT_COUNT 64u

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

static uint32_t fixture_key(size_t index)
{
    if (index % 11u == 0u) return 0x12345678u;
    if (index == 1u) return 0u;
    if (index == 2u) return UINT32_MAX;
    if (index == 3u) return 0x80000000u;
    return ((uint32_t)index * 0x45d9f3bu) ^
           ((uint32_t)index << 24u) ^
           0x9e3779b9u;
}

static void initialize_fixture(uint32_t *keys, uint32_t *values)
{
    size_t index;
    for (index = 0u; index < ELEMENT_COUNT; ++index) {
        keys[index] = fixture_key(index);
        values[index] = (uint32_t)index;
    }
}

static void stable_sort_reference(uint32_t *keys, uint32_t *values)
{
    size_t index;
    for (index = 1u; index < ELEMENT_COUNT; ++index) {
        const uint32_t key = keys[index];
        const uint32_t value = values[index];
        size_t position = index;
        while (position > 0u && keys[position - 1u] > key) {
            keys[position] = keys[position - 1u];
            values[position] = values[position - 1u];
            --position;
        }
        keys[position] = key;
        values[position] = value;
    }
}

int main(int argc, char **argv)
{
    uint32_t input_keys[ELEMENT_COUNT];
    uint32_t input_values[ELEMENT_COUNT];
    uint32_t expected_keys[ELEMENT_COUNT];
    uint32_t expected_values[ELEMENT_COUNT];
    cuda_api api;
    char *ptx = NULL;
    CUdevice device = 0;
    CUcontext context = NULL;
    CUmodule module = NULL;
    CUfunction function = NULL;
    CUdeviceptr heap_device = 0;
    uint32_t heap[HEAP_WORDS] = {0};
    uint32_t parameters[5];
    uint32_t element_count = ELEMENT_COUNT;
    char device_name[256] = {0};
    FILE *output = NULL;
    bool guards_intact = true;
    bool pass = false;
    int exit_code = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: cuda-radix-sort-fixture <radix.ptx> <output.json>\n");
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

    if (!heap_pointer(heap_device, KEYS_OFFSET, &parameters[0]) ||
        !heap_pointer(heap_device, VALUES_OFFSET, &parameters[1]) ||
        !heap_pointer(heap_device, KEYS_OFFSET, &parameters[2]) ||
        !heap_pointer(heap_device, VALUES_OFFSET, &parameters[3])) {
        goto cleanup;
    }
    parameters[4] = element_count;

    initialize_fixture(input_keys, input_values);
    memcpy(expected_keys, input_keys, sizeof(expected_keys));
    memcpy(expected_values, input_values, sizeof(expected_values));
    stable_sort_reference(expected_keys, expected_values);
    initialize_range(heap, KEYS_OFFSET, input_keys, ELEMENT_COUNT);
    initialize_range(heap, VALUES_OFFSET, input_values, ELEMENT_COUNT);

    CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
    {
        void *launch_parameters[] = {
            &parameters[0],
            &parameters[1],
            &parameters[2],
            &parameters[3],
            &element_count,
        };
        CUDA_TRY(&api, api.cuLaunchKernel(function,
                                          1, 1, 1,
                                          256, 1, 1,
                                          4096,
                                          NULL,
                                          launch_parameters,
                                          NULL));
    }
    CUDA_TRY(&api, api.cuCtxSynchronize());
    CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

    guards_intact = range_guards_match(heap, KEYS_OFFSET, ELEMENT_COUNT) &&
                    range_guards_match(heap, VALUES_OFFSET, ELEMENT_COUNT);
    pass = guards_intact &&
           words_equal(heap, KEYS_OFFSET, expected_keys, ELEMENT_COUNT) &&
           words_equal(heap, VALUES_OFFSET, expected_values, ELEMENT_COUNT);

    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "failed to open output file: %s\n", argv[2]);
        goto cleanup;
    }
    fprintf(output,
            "{\n"
            "  \"schema_version\": 1,\n"
            "  \"device\": \"%s\",\n"
            "  \"kernel\": \"radixSortBlocks<32,0,false>\",\n"
            "  \"element_count\": 64,\n"
            "  \"in_place\": true,\n"
            "  \"dynamic_shared_bytes\": 4096,\n"
            "  \"input_keys\": ",
            device_name);
    write_words(output, input_keys, ELEMENT_COUNT);
    fprintf(output, ",\n  \"input_values\": ");
    write_words(output, input_values, ELEMENT_COUNT);
    fprintf(output, ",\n  \"expected_keys\": ");
    write_words(output, expected_keys, ELEMENT_COUNT);
    fprintf(output, ",\n  \"expected_values\": ");
    write_words(output, expected_values, ELEMENT_COUNT);
    fprintf(output, ",\n  \"output_keys\": ");
    write_words(output, heap + KEYS_OFFSET, ELEMENT_COUNT);
    fprintf(output, ",\n  \"output_values\": ");
    write_words(output, heap + VALUES_OFFSET, ELEMENT_COUNT);
    fprintf(output,
            ",\n"
            "  \"guards_intact\": %s,\n"
            "  \"pass\": %s\n"
            "}\n",
            guards_intact ? "true" : "false",
            pass ? "true" : "false");
    fclose(output);
    output = NULL;

    printf("%s  %s: keys=[0x%08x,0x%08x,0x%08x,0x%08x,...]\n",
           pass ? "PASS" : "FAIL",
           device_name,
           heap[KEYS_OFFSET],
           heap[KEYS_OFFSET + 1u],
           heap[KEYS_OFFSET + 2u],
           heap[KEYS_OFFSET + 3u]);
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
