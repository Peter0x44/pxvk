// pxvk-framelog.dll: records a timestamp for every IDirect3DDevice9::Present,
// so native and pxvk runs of a Direct3D 9 game can be compared by frame time
// without administrator rights (PresentMon and ETW need them).
//
// Injected with pxvk-launch --inject-dll (tools/game-bench.py does this).
// Environment:
//   PXVK_FRAMELOG_OUT   output file, one Present time per line in microseconds
//   PXVK_FRAMELOG_LOAD  optional DLL to load first: pxvk's nvcuda.dll for pxvk
//                       runs, since --inject-dll replaces pxvk's own injection
//
// d3d9.dll's Direct3DCreate9 is patched with a jump to a wrapper, which
// restores the original bytes, calls it, re-patches, and replaces
// IDirect3D9::CreateDevice in the returned object's vtable; that wrapper
// replaces IDirect3DDevice9::Present (vtable slot 17) in the new device's.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef void *(WINAPI *create9_fn)(UINT);
typedef HRESULT(WINAPI *create_device_fn)(void *, UINT, int, HWND, DWORD, void *, void **);
typedef HRESULT(WINAPI *present_fn)(void *, const RECT *, const RECT *, HWND, const void *);

enum { kCreateDeviceSlot = 16, kPresentSlot = 17 };

static create9_fn g_create9;
static unsigned char g_saved[5];
static create_device_fn g_create_device;
static present_fn g_present;
static HANDLE g_out = INVALID_HANDLE_VALUE;
static LARGE_INTEGER g_frequency;
static LONGLONG g_last_flush;
static char g_buffer[1 << 16];
static size_t g_used;
static CRITICAL_SECTION g_lock;

static void patch(void *at, const void *bytes, size_t size)
{
    DWORD old;
    VirtualProtect(at, size, PAGE_EXECUTE_READWRITE, &old);
    memcpy(at, bytes, size);
    VirtualProtect(at, size, old, &old);
    FlushInstructionCache(GetCurrentProcess(), at, size);
}

static void jump(void *from, void *to)
{
    unsigned char code[5] = {0xE9};
    const LONG relative = (LONG)((char *)to - ((char *)from + 5));
    memcpy(code + 1, &relative, 4);
    patch(from, code, 5);
}

static void flush(void)
{
    DWORD written;
    if (g_out != INVALID_HANDLE_VALUE && g_used != 0) WriteFile(g_out, g_buffer, (DWORD)g_used, &written, NULL);
    g_used = 0;
}

static HRESULT WINAPI framelog_present(void *device, const RECT *source, const RECT *destination, HWND window,
                                       const void *dirty)
{
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    EnterCriticalSection(&g_lock);
    g_used += (size_t)sprintf(g_buffer + g_used, "%lld\n", now.QuadPart * 1000000 / g_frequency.QuadPart);
    // Flush at least once a second: a run may end by killing the game, which
    // skips DLL_PROCESS_DETACH.
    if (g_used > sizeof(g_buffer) - 64 || now.QuadPart - g_last_flush > g_frequency.QuadPart) {
        flush();
        g_last_flush = now.QuadPart;
    }
    LeaveCriticalSection(&g_lock);
    return g_present(device, source, destination, window, dirty);
}

static HRESULT WINAPI framelog_create_device(void *d3d, UINT adapter, int type, HWND window, DWORD flags,
                                             void *parameters, void **device)
{
    const HRESULT result = g_create_device(d3d, adapter, type, window, flags, parameters, device);
    if (SUCCEEDED(result) && device != NULL && *device != NULL) {
        void **vtable = *(void ***)*device;
        if (vtable[kPresentSlot] != (void *)framelog_present) {
            void *replacement = (void *)framelog_present;
            g_present = (present_fn)vtable[kPresentSlot];
            patch(&vtable[kPresentSlot], &replacement, sizeof(replacement));
        }
    }
    return result;
}

static void *WINAPI framelog_create9(UINT version)
{
    void *d3d;
    patch((void *)g_create9, g_saved, sizeof(g_saved));
    d3d = g_create9(version);
    jump((void *)g_create9, (void *)framelog_create9);
    if (d3d != NULL) {
        void **vtable = *(void ***)d3d;
        if (vtable[kCreateDeviceSlot] != (void *)framelog_create_device) {
            void *replacement = (void *)framelog_create_device;
            g_create_device = (create_device_fn)vtable[kCreateDeviceSlot];
            patch(&vtable[kCreateDeviceSlot], &replacement, sizeof(replacement));
        }
    }
    return d3d;
}

static bool environment(const wchar_t *name, wchar_t *value, DWORD capacity)
{
    const DWORD length = GetEnvironmentVariableW(name, value, capacity);
    return length != 0 && length < capacity;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    (void)instance;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        wchar_t path[1024];
        HMODULE d3d9;
        InitializeCriticalSection(&g_lock);
        QueryPerformanceFrequency(&g_frequency);
        if (environment(L"PXVK_FRAMELOG_LOAD", path, 1024)) LoadLibraryW(path);
        if (environment(L"PXVK_FRAMELOG_OUT", path, 1024))
            g_out = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, 0, NULL);
        d3d9 = LoadLibraryW(L"d3d9.dll");
        g_create9 = d3d9 != NULL ? (create9_fn)GetProcAddress(d3d9, "Direct3DCreate9") : NULL;
        if (g_create9 != NULL) {
            memcpy(g_saved, (void *)g_create9, sizeof(g_saved));
            jump((void *)g_create9, (void *)framelog_create9);
        }
    } else if (reason == DLL_PROCESS_DETACH) {
        EnterCriticalSection(&g_lock);
        flush();
        LeaveCriticalSection(&g_lock);
        if (g_out != INVALID_HANDLE_VALUE) CloseHandle(g_out);
    }
    return TRUE;
}
