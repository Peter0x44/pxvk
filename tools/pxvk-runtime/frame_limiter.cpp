#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstring>
#include <cwchar>
#include <iterator>

static_assert(sizeof(void *) == 4, "frame limiter requires a 32-bit build");

namespace {

constexpr LONGLONG kFramesPerSecond = 50;

using SwapBuffersFn = BOOL(WINAPI *)(HDC);
using LdrDllNotificationFn = VOID(CALLBACK *)(ULONG, const void *, void *);
using LdrRegisterDllNotificationFn =
    LONG(NTAPI *)(ULONG, LdrDllNotificationFn, void *, void **);
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

struct DllNotification {
    LoadedDllNotification loaded;
};

SwapBuffersFn g_swap_buffers = nullptr;
LARGE_INTEGER g_counter_frequency{};
LONGLONG g_next_frame = 0;
LdrUnregisterDllNotificationFn g_unregister_notification = nullptr;
void *g_notification_cookie = nullptr;

void wait_until(LONGLONG deadline)
{
    LARGE_INTEGER now{};
    for (;;) {
        QueryPerformanceCounter(&now);
        const LONGLONG remaining = deadline - now.QuadPart;
        if (remaining <= 0) return;

        const LONGLONG milliseconds =
            remaining * 1000 / g_counter_frequency.QuadPart;
        if (milliseconds > 1)
            Sleep(static_cast<DWORD>(milliseconds - 1));
        else
            SwitchToThread();
    }
}

BOOL WINAPI limited_swap_buffers(HDC device_context)
{
    const LONGLONG period = g_counter_frequency.QuadPart / kFramesPerSecond;
    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);

    if (g_next_frame == 0) {
        g_next_frame = now.QuadPart;
    } else if (now.QuadPart < g_next_frame) {
        wait_until(g_next_frame);
    }

    const BOOL result = g_swap_buffers(device_context);
    QueryPerformanceCounter(&now);
    g_next_frame += period;
    if (g_next_frame + period < now.QuadPart)
        g_next_frame = now.QuadPart + period;
    return result;
}

bool patch_glut_swap_buffers(void *module_base)
{
    auto *base = static_cast<unsigned char *>(module_base);
    const auto *dos = reinterpret_cast<const IMAGE_DOS_HEADER *>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return false;
    const auto *nt = reinterpret_cast<const IMAGE_NT_HEADERS32 *>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return false;
    const IMAGE_DATA_DIRECTORY &imports =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (imports.VirtualAddress == 0u) return false;

    auto *descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR *>(
        base + imports.VirtualAddress);
    for (; descriptor->Name != 0u; ++descriptor) {
        const char *library = reinterpret_cast<const char *>(base + descriptor->Name);
        if (_stricmp(library, "GDI32.dll") != 0 ||
            descriptor->OriginalFirstThunk == 0u || descriptor->FirstThunk == 0u) {
            continue;
        }

        auto *names = reinterpret_cast<IMAGE_THUNK_DATA32 *>(
            base + descriptor->OriginalFirstThunk);
        auto *slots = reinterpret_cast<IMAGE_THUNK_DATA32 *>(
            base + descriptor->FirstThunk);
        for (; names->u1.AddressOfData != 0u; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL32(names->u1.Ordinal)) continue;
            const auto *import = reinterpret_cast<const IMAGE_IMPORT_BY_NAME *>(
                base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char *>(import->Name),
                            "SwapBuffers") != 0) {
                continue;
            }

            auto *slot = reinterpret_cast<void **>(&slots->u1.Function);
            g_swap_buffers = reinterpret_cast<SwapBuffersFn>(*slot);
            DWORD old_protection = 0;
            if (VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE,
                               &old_protection) == FALSE) {
                g_swap_buffers = nullptr;
                return false;
            }
            *slot = reinterpret_cast<void *>(&limited_swap_buffers);
            DWORD ignored = 0;
            VirtualProtect(slot, sizeof(*slot), old_protection, &ignored);
            FlushInstructionCache(GetCurrentProcess(), slot, sizeof(*slot));
            return true;
        }
    }
    return false;
}

VOID CALLBACK dll_notification(ULONG reason, const void *raw_data, void *)
{
    constexpr ULONG loaded_reason = 1u;
    if (reason != loaded_reason || raw_data == nullptr || g_swap_buffers != nullptr)
        return;

    const auto *data = static_cast<const DllNotification *>(raw_data);
    if (data->loaded.base_name == nullptr ||
        data->loaded.base_name->buffer == nullptr) {
        return;
    }

    wchar_t name[32]{};
    const std::size_t source_length =
        data->loaded.base_name->length / sizeof(wchar_t);
    const std::size_t characters =
        source_length < std::size(name) - 1u ? source_length : std::size(name) - 1u;
    std::wmemcpy(name, data->loaded.base_name->buffer, characters);
    if (_wcsicmp(name, L"glut32.dll") == 0)
        patch_glut_swap_buffers(data->loaded.base);
}

void install_frame_limiter()
{
    QueryPerformanceFrequency(&g_counter_frequency);
    if (HMODULE glut = GetModuleHandleW(L"glut32.dll"))
        patch_glut_swap_buffers(glut);

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    auto register_notification = reinterpret_cast<LdrRegisterDllNotificationFn>(
        GetProcAddress(ntdll, "LdrRegisterDllNotification"));
    g_unregister_notification = reinterpret_cast<LdrUnregisterDllNotificationFn>(
        GetProcAddress(ntdll, "LdrUnregisterDllNotification"));
    if (g_swap_buffers == nullptr && register_notification != nullptr) {
        register_notification(0, dll_notification, nullptr,
                              &g_notification_cookie);
    }
}

} // namespace

extern "C" BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        install_frame_limiter();
    } else if (reason == DLL_PROCESS_DETACH &&
               g_notification_cookie != nullptr &&
               g_unregister_notification != nullptr) {
        g_unregister_notification(g_notification_cookie);
        g_notification_cookie = nullptr;
    }
    return TRUE;
}
