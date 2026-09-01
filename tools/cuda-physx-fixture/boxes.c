#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_NAME                                                            \
  "_Z36PxgCudaFluidCollisionBoxShapesBatchD11PxgGpuArrayIK32PxgFluidCollisio" \
  "nShapeStartAddrsES_IK16PxgFluidWorkUnitEPi"
#define GUARD_BEFORE 0x13579bdfu
#define GUARD_AFTER 0x2468ace0u
#define HEAP_WORDS 160u

#define WORK_UNIT_OFFSET 8u
#define SHAPE_START_ADDRS_OFFSET 20u
#define PARAMETERS_OFFSET 64u
#define POSITION_OFFSET 112u
#define OLD_POSITION_OFFSET 120u
#define ERROR_OFFSET 128u

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
  CUresult(CUDAAPI *cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, CUstream,
                                    void **, void **);
  CUresult(CUDAAPI *cuGetErrorName)(CUresult, const char **);
  CUresult(CUDAAPI *cuGetErrorString)(CUresult, const char **);
} cuda_api;

static bool load_function(HMODULE library, const char *name, void *output,
                          size_t output_size) {
  FARPROC raw = GetProcAddress(library, name);
  if (raw == NULL || output_size != sizeof(raw)) {
    fprintf(stderr, "missing CUDA export: %s\n", name);
    return false;
  }
  memcpy(output, &raw, output_size);
  return true;
}

#define LOAD_API(api, name)                                                    \
  do {                                                                         \
    if (!load_function((api)->library, #name, &(api)->name,                    \
                       sizeof((api)->name))) {                                 \
      return false;                                                            \
    }                                                                          \
  } while (0)

static bool load_cuda(cuda_api *api) {
  memset(api, 0, sizeof(*api));
  api->library = LoadLibraryW(L"nvcuda.dll");
  if (api->library == NULL) {
    fprintf(stderr, "failed to load nvcuda.dll: %lu\n",
            (unsigned long)GetLastError());
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

static void report_cuda_error(cuda_api *api, const char *operation,
                              CUresult result) {
  const char *name = NULL;
  const char *description = NULL;
  api->cuGetErrorName(result, &name);
  api->cuGetErrorString(result, &description);
  fprintf(stderr, "%s failed: %d (%s: %s)\n", operation, (int)result,
          name != NULL ? name : "unknown",
          description != NULL ? description : "no description");
}

#define CUDA_TRY(api, expression)                                              \
  do {                                                                         \
    CUresult result__ = (expression);                                          \
    if (result__ != CUDA_SUCCESS) {                                            \
      report_cuda_error((api), #expression, result__);                         \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

static char *read_text_file(const char *path) {
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
  if (result == NULL ||
      fread(result, 1, (size_t)length, file) != (size_t)length) {
    free(result);
    fclose(file);
    return NULL;
  }
  result[length] = '\0';
  fclose(file);
  return result;
}

static bool pointer32(CUdeviceptr pointer, uint32_t *output) {
  if (pointer > UINT32_MAX) {
    fprintf(stderr,
            "CUDA allocation is outside the 32-bit PTX address space\n");
    return false;
  }
  *output = (uint32_t)pointer;
  return true;
}

static bool heap_pointer(CUdeviceptr heap, size_t offset, uint32_t *output) {
  return pointer32(heap + offset * sizeof(uint32_t), output);
}

static void initialize_range(uint32_t *heap, size_t offset,
                             const uint32_t *values, size_t words) {
  heap[offset - 1u] = GUARD_BEFORE;
  memcpy(heap + offset, values, words * sizeof(uint32_t));
  heap[offset + words] = GUARD_AFTER;
}

static bool range_guards_match(const uint32_t *heap, size_t offset,
                               size_t words) {
  return heap[offset - 1u] == GUARD_BEFORE &&
         heap[offset + words] == GUARD_AFTER;
}

static bool words_equal(const uint32_t *heap, size_t offset,
                        const uint32_t *expected, size_t words) {
  return memcmp(heap + offset, expected, words * sizeof(uint32_t)) == 0;
}

static void write_words(FILE *output, const uint32_t *values, size_t words) {
  size_t index;
  fputc('[', output);
  for (index = 0u; index < words; ++index) {
    fprintf(output, "%s\"0x%08x\"", index == 0u ? "" : ", ", values[index]);
  }
  fputc(']', output);
}

int main(int argc, char **argv) {
  static const uint32_t input_position[4] = {
      0x3fc00000u,
      0xc0100000u,
      0x40700000u,
      0x11223344u,
  };
  static const uint32_t input_old_position[4] = {
      0x3f800000u,
      0xc0000000u,
      0x40400000u,
      0x55667788u,
  };
  static const uint32_t error_sentinel[1] = {0xfeedfaceu};
  cuda_api api;
  char *ptx = NULL;
  CUdevice device = 0;
  CUcontext context = NULL;
  CUmodule module = NULL;
  CUfunction function = NULL;
  CUdeviceptr heap_device = 0;
  uint32_t heap[HEAP_WORDS] = {0};
  uint32_t work_units[8] = {0};
  uint32_t shape_start_addrs[40] = {0};
  uint32_t parameters[40] = {0};
  uint32_t shape_array[2];
  uint32_t work_array[2];
  uint32_t error_pointer;
  uint32_t position_pointer;
  uint32_t old_position_pointer;
  uint32_t parameters_pointer;
  char device_name[256] = {0};
  FILE *output = NULL;
  bool guards_intact;
  bool pass = false;
  int exit_code = 1;
  size_t record;

  if (argc != 3) {
    fprintf(stderr,
            "usage: cuda-fluid-boxes-fixture <boxes.ptx> <output.json>\n");
    return 2;
  }
  if (!load_cuda(&api))
    return 1;
  ptx = read_text_file(argv[1]);
  if (ptx == NULL)
    goto cleanup;

  CUDA_TRY(&api, api.cuInit(0));
  CUDA_TRY(&api, api.cuDeviceGet(&device, 0));
  CUDA_TRY(&api,
           api.cuDeviceGetName(device_name, (int)sizeof(device_name), device));
  CUDA_TRY(&api, api.cuCtxCreate_v2(&context, 0, device));
  CUDA_TRY(&api, api.cuModuleLoadData(&module, ptx));
  CUDA_TRY(&api, api.cuModuleGetFunction(&function, module, KERNEL_NAME));
  CUDA_TRY(&api, api.cuMemAlloc_v2(&heap_device, sizeof(heap)));

  if (!heap_pointer(heap_device, SHAPE_START_ADDRS_OFFSET, &shape_array[0]) ||
      !heap_pointer(heap_device, WORK_UNIT_OFFSET, &work_array[0]) ||
      !heap_pointer(heap_device, ERROR_OFFSET, &error_pointer) ||
      !heap_pointer(heap_device, POSITION_OFFSET, &position_pointer) ||
      !heap_pointer(heap_device, OLD_POSITION_OFFSET, &old_position_pointer) ||
      !heap_pointer(heap_device, PARAMETERS_OFFSET, &parameters_pointer)) {
    goto cleanup;
  }
  shape_array[1] = 2u;
  work_array[1] = 2u;
  work_units[1] = 1u;
  work_units[6] = 1u;

  for (record = 0u; record < 2u; ++record) {
    const size_t base = record * 20u;
    shape_start_addrs[base + 1u] = 0u;
    shape_start_addrs[base + 2u] = position_pointer;
    shape_start_addrs[base + 3u] = old_position_pointer;
    shape_start_addrs[base + 4u] = parameters_pointer;
    shape_start_addrs[base + 19u] = 0x87654321u + (uint32_t)record;
  }
  parameters[0] = 0x31415926u;
  parameters[39] = 0x27182818u;

  initialize_range(heap, WORK_UNIT_OFFSET, work_units, 8u);
  initialize_range(heap, SHAPE_START_ADDRS_OFFSET, shape_start_addrs, 40u);
  initialize_range(heap, PARAMETERS_OFFSET, parameters, 40u);
  initialize_range(heap, POSITION_OFFSET, input_position, 4u);
  initialize_range(heap, OLD_POSITION_OFFSET, input_old_position, 4u);
  initialize_range(heap, ERROR_OFFSET, error_sentinel, 1u);

  CUDA_TRY(&api, api.cuMemcpyHtoD_v2(heap_device, heap, sizeof(heap)));
  {
    void *launch_parameters[] = {shape_array, work_array, &error_pointer};
    CUDA_TRY(&api, api.cuLaunchKernel(function, 2, 1, 1, 64, 1, 1, 0, NULL,
                                      launch_parameters, NULL));
  }
  CUDA_TRY(&api, api.cuCtxSynchronize());
  CUDA_TRY(&api, api.cuMemcpyDtoH_v2(heap, heap_device, sizeof(heap)));

  guards_intact =
      range_guards_match(heap, WORK_UNIT_OFFSET, 8u) &&
      range_guards_match(heap, SHAPE_START_ADDRS_OFFSET, 40u) &&
      range_guards_match(heap, PARAMETERS_OFFSET, 40u) &&
      range_guards_match(heap, POSITION_OFFSET, 4u) &&
      range_guards_match(heap, OLD_POSITION_OFFSET, 4u) &&
      range_guards_match(heap, ERROR_OFFSET, 1u);
  pass = guards_intact &&
         words_equal(heap, WORK_UNIT_OFFSET, work_units, 8u) &&
         words_equal(heap, SHAPE_START_ADDRS_OFFSET, shape_start_addrs, 40u) &&
         words_equal(heap, PARAMETERS_OFFSET, parameters, 40u) &&
         words_equal(heap, POSITION_OFFSET, input_position, 4u) &&
         words_equal(heap, OLD_POSITION_OFFSET, input_old_position, 4u) &&
         words_equal(heap, ERROR_OFFSET, error_sentinel, 1u);

  output = fopen(argv[2], "wb");
  if (output == NULL) {
    fprintf(stderr, "failed to open output file: %s\n", argv[2]);
    goto cleanup;
  }
  fprintf(output,
          "{\n"
          "  \"schema_version\": 1,\n"
          "  \"device\": \"%s\",\n"
          "  \"kernel\": \"PxgCudaFluidCollisionBoxShapesBatch\",\n"
          "  \"work_unit_count\": 2,\n"
          "  \"particle_count\": 1,\n"
          "  \"box_count\": 0,\n"
          "  \"input_position_bits\": ",
          device_name);
  write_words(output, input_position, 4u);
  fprintf(output, ",\n  \"input_old_position_bits\": ");
  write_words(output, input_old_position, 4u);
  fprintf(output, ",\n  \"output_position_bits\": ");
  write_words(output, heap + POSITION_OFFSET, 4u);
  fprintf(output, ",\n  \"output_old_position_bits\": ");
  write_words(output, heap + OLD_POSITION_OFFSET, 4u);
  fprintf(output,
          ",\n"
          "  \"guards_intact\": %s,\n"
          "  \"pass\": %s\n"
          "}\n",
          guards_intact ? "true" : "false", pass ? "true" : "false");
  fclose(output);
  output = NULL;
  printf("%s  %s: zero-box position=[0x%08x,0x%08x,0x%08x,0x%08x]\n",
         pass ? "PASS" : "FAIL", device_name, heap[POSITION_OFFSET],
         heap[POSITION_OFFSET + 1u], heap[POSITION_OFFSET + 2u],
         heap[POSITION_OFFSET + 3u]);
  exit_code = pass ? 0 : 1;

cleanup:
  if (output != NULL)
    fclose(output);
  if (module != NULL)
    api.cuModuleUnload(module);
  if (heap_device != 0)
    api.cuMemFree_v2(heap_device);
  if (context != NULL)
    api.cuCtxDestroy_v2(context);
  free(ptx);
  if (api.library != NULL)
    FreeLibrary(api.library);
  return exit_code;
}
