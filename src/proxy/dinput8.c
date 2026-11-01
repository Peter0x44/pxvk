/* dinput8.dll proxy: loads pxvk into a game without a launcher.
 *
 * Copied into the game's executable folder together with pxvk's nvcuda.dll
 * and pxvk-nvapi.dll. Games that import DINPUT8.dll (most Unreal Engine 3
 * games, e.g. Borderlands 2) load it from there at startup, before PhysX,
 * however they are started: Steam, GOG Galaxy, the EA app or a shortcut.
 * DllMain loads nvcuda.dll from this DLL's folder, which installs pxvk's
 * redirects (src/runtime/loader.cpp) as pxvk-launch's injection does. Its
 * six exports call the system's dinput8.dll.
 *
 * Without PXVK_LOG set, pxvk logs to pxvk.log in the same folder, so a
 * tester has a log without configuring anything. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <wchar.h>

static HMODULE g_system;

static FARPROC system_export(const char *name)
{
    if (g_system == NULL) {
        wchar_t path[MAX_PATH];
        const UINT length = GetSystemDirectoryW(path, MAX_PATH);
        if (length == 0 || length > MAX_PATH - 14) return NULL;
        wcscat(path, L"\\dinput8.dll");
        g_system = LoadLibraryW(path);
        if (g_system == NULL) return NULL;
    }
    return GetProcAddress(g_system, name);
}

typedef HRESULT(WINAPI *create_fn)(HINSTANCE, DWORD, const void *, void **, void *);
typedef HRESULT(WINAPI *void_fn)(void);
typedef HRESULT(WINAPI *class_fn)(const void *, const void *, void **);
typedef const void *(WINAPI *format_fn)(void);

HRESULT WINAPI DirectInput8Create(HINSTANCE instance, DWORD version, const void *riid, void **out, void *outer)
{
    create_fn f = (create_fn)(void *)system_export("DirectInput8Create");
    return f ? f(instance, version, riid, out, outer) : E_FAIL;
}

HRESULT WINAPI DllCanUnloadNow(void)
{
    void_fn f = (void_fn)(void *)system_export("DllCanUnloadNow");
    return f ? f() : S_FALSE;
}

HRESULT WINAPI DllGetClassObject(const void *clsid, const void *riid, void **out)
{
    class_fn f = (class_fn)(void *)system_export("DllGetClassObject");
    return f ? f(clsid, riid, out) : E_FAIL;
}

HRESULT WINAPI DllRegisterServer(void)
{
    void_fn f = (void_fn)(void *)system_export("DllRegisterServer");
    return f ? f() : E_FAIL;
}

HRESULT WINAPI DllUnregisterServer(void)
{
    void_fn f = (void_fn)(void *)system_export("DllUnregisterServer");
    return f ? f() : E_FAIL;
}

const void *WINAPI GetdfDIJoystick(void)
{
    format_fn f = (format_fn)(void *)system_export("GetdfDIJoystick");
    return f ? f() : NULL;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        wchar_t path[MAX_PATH];
        wchar_t *slash;
        DisableThreadLibraryCalls(instance);
        /* Already started through pxvk-launch: one pxvk per process. */
        if (GetModuleHandleW(L"nvcuda.dll") != NULL) return TRUE;
        if (GetModuleFileNameW(instance, path, MAX_PATH) == 0 || (slash = wcsrchr(path, L'\\')) == NULL) return TRUE;
        slash[1] = L'\0';
        if (GetEnvironmentVariableW(L"PXVK_LOG", NULL, 0) == 0 && wcslen(path) < MAX_PATH - 9) {
            wchar_t log[MAX_PATH];
            wcscpy(log, path);
            wcscat(log, L"pxvk.log");
            SetEnvironmentVariableW(L"PXVK_LOG", log);
        }
        if (wcslen(path) < MAX_PATH - 11) {
            wcscat(path, L"nvcuda.dll");
            LoadLibraryW(path);
        }
    } else if (reason == DLL_PROCESS_DETACH && reserved == NULL && g_system != NULL) {
        FreeLibrary(g_system);
    }
    return TRUE;
}
