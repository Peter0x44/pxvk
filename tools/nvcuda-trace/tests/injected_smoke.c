#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <string.h>

typedef int(__stdcall *cu_init_fn)(unsigned int flags);
typedef int(__stdcall *cu_device_get_count_fn)(int *count);

int main(void)
{
    HMODULE proxy = GetModuleHandleA("nvcuda.dll");
    FARPROC raw_cu_init;
    FARPROC raw_cu_device_get_count;
    cu_init_fn cu_init = NULL;
    cu_device_get_count_fn cu_device_get_count = NULL;
    int count = -1;
    int result;

    if (proxy == NULL) {
        fputs("nvcuda.dll was not injected before process startup\n", stderr);
        return 2;
    }
    raw_cu_init = GetProcAddress(proxy, "cuInit");
    raw_cu_device_get_count = GetProcAddress(proxy, "cuDeviceGetCount");
    memcpy(&cu_init, &raw_cu_init, sizeof(cu_init));
    memcpy(&cu_device_get_count, &raw_cu_device_get_count, sizeof(cu_device_get_count));
    if (cu_init == NULL || cu_device_get_count == NULL) {
        fputs("injected proxy exports are missing\n", stderr);
        return 3;
    }
    result = cu_init(0);
    if (result != 0) {
        fprintf(stderr, "cuInit failed: %d\n", result);
        return 4;
    }
    result = cu_device_get_count(&count);
    if (result != 0 || count < 1) {
        fprintf(stderr, "cuDeviceGetCount failed: result=%d count=%d\n", result, count);
        return 5;
    }
    printf("injected CUDA proxy smoke passed: devices=%d\n", count);
    return 0;
}
