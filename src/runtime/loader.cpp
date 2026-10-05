// Redirects CUDART's nvcuda.dll load to this DLL.
//
// NVIDIA's cudart32_65.dll builds an absolute System32 path for nvcuda.dll,
// so the normal DLL search order cannot substitute pxvk. When pxvk is
// injected first (pxvk-launch.exe), a loader notification patches only
// CUDART's LoadLibraryW/LoadLibraryExW imports so that its nvcuda.dll
// request resolves to the already loaded pxvk module.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "runtime.h"

#include <algorithm>
#include <cstring>
#include <cwchar>
#include <iterator>

namespace pxvk::runtime {
namespace {

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
LoadLibraryWFn g_load_library_w = nullptr;
LoadLibraryExWFn g_load_library_ex_w = nullptr;
LdrUnregisterDllNotificationFn g_unregister = nullptr;
void *g_cookie = nullptr;

bool is_driver_name(LPCWSTR path)
{
    if (path == nullptr) return false;
    const wchar_t *name = path;
    for (const wchar_t *c = path; *c; ++c)
        if (*c == L'\\' || *c == L'/') name = c + 1;
    return _wcsicmp(name, L"nvcuda.dll") == 0;
}

HMODULE WINAPI redirect_load_library_w(LPCWSTR path)
{
    return is_driver_name(path) ? g_load_library_w(g_module_path) : g_load_library_w(path);
}

HMODULE WINAPI redirect_load_library_ex_w(LPCWSTR path, HANDLE file, DWORD flags)
{
    return is_driver_name(path) ? g_load_library_ex_w(g_module_path, nullptr, 0)
                                : g_load_library_ex_w(path, file, flags);
}

void patch_slot(void **slot, void *replacement)
{
    DWORD old = 0;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old)) return;
    *slot = replacement;
    VirtualProtect(slot, sizeof(*slot), old, &old);
    FlushInstructionCache(GetCurrentProcess(), slot, sizeof(*slot));
}

void patch_cudart(void *module_base)
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
            if (std::strcmp(name, "LoadLibraryW") == 0)
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
    if (is_cudart(name)) patch_cudart(data->base);
}

} // namespace

void install_driver_redirect(HMODULE self)
{
    GetModuleFileNameW(self, g_module_path, static_cast<DWORD>(std::size(g_module_path)));
    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
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
        if (HMODULE loaded = GetModuleHandleW(name)) patch_cudart(loaded);
}

void remove_driver_redirect()
{
    if (g_cookie && g_unregister) g_unregister(g_cookie);
    g_cookie = nullptr;
}

} // namespace pxvk::runtime
