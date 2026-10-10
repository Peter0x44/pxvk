// Redirects CUDART's nvcuda.dll load to this DLL.
//
// NVIDIA's cudart32_65.dll builds an absolute System32 path for nvcuda.dll,
// so the normal DLL search order cannot substitute pxvk. When pxvk is
// injected first (pxvk-launch.exe), a loader notification patches only
// CUDART's LoadLibraryA/W/ExW imports so that its nvcuda.dll request
// resolves to the already loaded pxvk module. (CUDA 4.x runtimes such as
// cudart32_41_4.dll ask for plain "nvcuda.dll" with LoadLibraryA, which the
// loaded pxvk module would satisfy anyway; the redirect makes it explicit.)
//
// The same notification patches PhysXCore.dll's and PhysXDevice.dll's
// LoadLibraryA/W/ExW imports so that their nvapi.dll requests resolve to
// pxvk-nvapi.dll next to this DLL (see nvapi.cpp). Without it PhysX picks
// CPU PhysX on machines without an NVIDIA driver. The redirect is the
// default on every machine, so hardware selection behaves the same with or
// without an NVIDIA driver (including one whose GPU is disabled or
// unsupported). PXVK_NVAPI=system keeps the real NVAPI instead, and NVAPI
// loads are also left alone if pxvk-nvapi.dll is absent.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "runtime.h"

#include <algorithm>
#include <cstring>
#include <cwchar>
#include <iterator>

namespace pxvk::runtime {
namespace {

using LoadLibraryAFn = HMODULE(WINAPI *)(LPCSTR);
using LoadLibraryWFn = HMODULE(WINAPI *)(LPCWSTR);
using LoadLibraryExWFn = HMODULE(WINAPI *)(LPCWSTR, HANDLE, DWORD);
using LdrDllNotificationFn = VOID(CALLBACK *)(ULONG, const void *, void *);
using LdrRegisterDllNotificationFn = LONG(NTAPI *)(ULONG, LdrDllNotificationFn, void *, void **);
using LdrUnregisterDllNotificationFn = LONG(NTAPI *)(void *);

struct UnicodeString {
    USHORT length;
    USHORT maximum_length;
    PWSTR buffer;
};

struct LoadedDllNotification {
    ULONG flags;
    const UnicodeString *full_name;
    const UnicodeString *base_name;
    void *base;
    ULONG image_size;
};

wchar_t g_module_path[32768]{};
wchar_t g_nvapi_path[32768]{};
const char *g_nvapi_mode = "pxvk";
LoadLibraryAFn g_load_library_a = nullptr;
LoadLibraryWFn g_load_library_w = nullptr;
LoadLibraryExWFn g_load_library_ex_w = nullptr;
LdrUnregisterDllNotificationFn g_unregister = nullptr;
void *g_cookie = nullptr;

// Whether the file name part of a path equals `name` (case-insensitive).
bool has_file_name(LPCWSTR path, const wchar_t *name)
{
    if (path == nullptr) return false;
    const wchar_t *file = path;
    for (const wchar_t *c = path; *c; ++c)
        if (*c == L'\\' || *c == L'/') file = c + 1;
    return _wcsicmp(file, name) == 0;
}

bool has_file_name(LPCSTR path, const char *name)
{
    if (path == nullptr) return false;
    const char *file = path;
    for (const char *c = path; *c; ++c)
        if (*c == '\\' || *c == '/') file = c + 1;
    return _stricmp(file, name) == 0;
}

bool is_driver_name(LPCWSTR path) { return has_file_name(path, L"nvcuda.dll"); }
bool is_driver_name(LPCSTR path) { return has_file_name(path, "nvcuda.dll"); }
bool is_nvapi_name(LPCWSTR path) { return g_nvapi_path[0] != L'\0' && has_file_name(path, L"nvapi.dll"); }
bool is_nvapi_name(LPCSTR path) { return g_nvapi_path[0] != L'\0' && has_file_name(path, "nvapi.dll"); }

HMODULE WINAPI redirect_load_library_w(LPCWSTR path)
{
    if (is_driver_name(path)) return g_load_library_w(g_module_path);
    if (is_nvapi_name(path)) return g_load_library_w(g_nvapi_path);
    return g_load_library_w(path);
}

HMODULE WINAPI redirect_load_library_ex_w(LPCWSTR path, HANDLE file, DWORD flags)
{
    // PhysXDevice requests NVAPI with LOAD_LIBRARY_SEARCH_SYSTEM32.
    if (is_driver_name(path)) return g_load_library_ex_w(g_module_path, nullptr, 0);
    if (is_nvapi_name(path)) return g_load_library_ex_w(g_nvapi_path, nullptr, 0);
    return g_load_library_ex_w(path, file, flags);
}

HMODULE WINAPI redirect_load_library_a(LPCSTR path)
{
    if (is_driver_name(path)) return g_load_library_w(g_module_path);
    if (is_nvapi_name(path)) return g_load_library_w(g_nvapi_path);
    return g_load_library_a(path);
}

void patch_slot(void **slot, void *replacement)
{
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old)) return;
    *slot = replacement;
    VirtualProtect(slot, sizeof(*slot), old, &old);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(*slot));
}

void patch_imports(void *module_base)
{
    auto *base = static_cast<unsigned char *>(module_base);
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS32 *>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return;
    const IMAGE_DATA_DIRECTORY &dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (dir.VirtualAddress == 0) return;
    for (auto *desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(base + dir.VirtualAddress); desc->Name; ++desc) {
        if (!desc->OriginalFirstThunk || !desc->FirstThunk) continue;
        auto *names = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + desc->OriginalFirstThunk);
        auto *slots = reinterpret_cast<IMAGE_THUNK_DATA32 *>(base + desc->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL32(names->u1.Ordinal)) continue;
            const auto *import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME *>(base + names->u1.AddressOfData);
            const char *name = reinterpret_cast<const char *>(import->Name);
            if (std::strcmp(name, "LoadLibraryA") == 0)
                patch_slot(reinterpret_cast<void **>(&slots->u1.Function), reinterpret_cast<void *>(&redirect_load_library_a));
            else if (std::strcmp(name, "LoadLibraryW") == 0)
                patch_slot(reinterpret_cast<void **>(&slots->u1.Function), reinterpret_cast<void *>(&redirect_load_library_w));
            else if (std::strcmp(name, "LoadLibraryExW") == 0)
                patch_slot(reinterpret_cast<void **>(&slots->u1.Function), reinterpret_cast<void *>(&redirect_load_library_ex_w));
        }
    }
}

bool is_cudart(const wchar_t *name)
{
    // cudart32_NN.dll for the classic CUDA runtimes shipped with PhysX.
    return _wcsnicmp(name, L"cudart32_", 9) == 0;
}

VOID CALLBACK on_dll_notification(ULONG reason, const void *raw, void *)
{
    constexpr ULONG kLoaded = 1;
    if (reason != kLoaded || raw == nullptr) return;
    const auto *data = static_cast<const LoadedDllNotification *>(raw);
    if (!data->base_name || !data->base_name->buffer) return;
    wchar_t name[64]{};
    const std::size_t count = std::min<std::size_t>(data->base_name->length / sizeof(wchar_t), std::size(name) - 1);
    std::wmemcpy(name, data->base_name->buffer, count);
    if (is_cudart(name)) patch_imports(data->base);
    else if (_wcsicmp(name, L"PhysXCore.dll") == 0 || _wcsicmp(name, L"PhysXDevice.dll") == 0)
        patch_imports(data->base);
}

} // namespace

void install_driver_redirect(HMODULE self)
{
    GetModuleFileNameW(self, g_module_path, static_cast<DWORD>(std::size(g_module_path)));
    // pxvk-nvapi.dll next to nvcuda.dll, used when present unless
    // PXVK_NVAPI=system selects the driver's own NVAPI.
    std::wmemcpy(g_nvapi_path, g_module_path, std::size(g_nvapi_path));
    if (wchar_t *slash = std::wcsrchr(g_nvapi_path, L'\\'))
        lstrcpynW(slash + 1, L"pxvk-nvapi.dll", static_cast<int>(std::size(g_nvapi_path) - (slash + 1 - g_nvapi_path)));
    wchar_t mode[16]{};
    const DWORD mode_length = GetEnvironmentVariableW(L"PXVK_NVAPI", mode, static_cast<DWORD>(std::size(mode)));
    if (mode_length > 0 && mode_length < std::size(mode) && _wcsicmp(mode, L"system") == 0)
        g_nvapi_mode = "system";
    else if (GetFileAttributesW(g_nvapi_path) == INVALID_FILE_ATTRIBUTES)
        g_nvapi_mode = "missing";
    if (std::strcmp(g_nvapi_mode, "pxvk") != 0) g_nvapi_path[0] = L'\0';
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    g_load_library_a = reinterpret_cast<LoadLibraryAFn>(reinterpret_cast<void *>(GetProcAddress(kernel, "LoadLibraryA")));
    g_load_library_w = reinterpret_cast<LoadLibraryWFn>(reinterpret_cast<void *>(GetProcAddress(kernel, "LoadLibraryW")));
    g_load_library_ex_w = reinterpret_cast<LoadLibraryExWFn>(reinterpret_cast<void *>(GetProcAddress(kernel, "LoadLibraryExW")));
    auto register_notification = reinterpret_cast<LdrRegisterDllNotificationFn>(
        reinterpret_cast<void *>(GetProcAddress(ntdll, "LdrRegisterDllNotification")));
    g_unregister = reinterpret_cast<LdrUnregisterDllNotificationFn>(
        reinterpret_cast<void *>(GetProcAddress(ntdll, "LdrUnregisterDllNotification")));
    if (g_load_library_w && g_load_library_ex_w && register_notification)
        register_notification(0, on_dll_notification, nullptr, &g_cookie);
    // CUDART may already be loaded if pxvk was injected late.
    for (const wchar_t *name : {L"cudart32_65.dll", L"cudart32_60.dll", L"cudart32_55.dll", L"cudart32_50_35.dll",
                                L"cudart32_42_9.dll", L"cudart32_41_28.dll", L"cudart32_40_17.dll",
                                L"cudart32_32_16.dll", L"cudart32_31_4.dll", L"cudart32_30_14.dll"})
        if (HMODULE loaded = GetModuleHandleW(name)) patch_imports(loaded);
}

const char *nvapi_mode()
{
    return g_nvapi_mode;
}

void remove_driver_redirect()
{
    if (g_cookie && g_unregister) g_unregister(g_cookie);
    g_cookie = nullptr;
}

} // namespace pxvk::runtime
