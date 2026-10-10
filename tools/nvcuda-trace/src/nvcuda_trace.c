#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#include <cuda.h>

/* CUDA headers map legacy entry points to their v2 ABI names. The proxy must
 * export both spellings because PhysX probes the legacy names dynamically. */
#undef cuCtxCreate
#undef cuCtxDestroy
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuDeviceTotalMem
#undef cuModuleGetGlobal
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemGetAddressRange
#undef cuTexRefSetAddress
#undef cuMemcpyDtoDAsync
#undef cuMemcpyDtoHAsync
#undef cuMemcpyHtoDAsync
#undef cuMemsetD8

#include "launch_params.h"
#include "capture.h"

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define PXVK_EXPORT __declspec(dllexport)
#define PXVK_PATH_CAPACITY 1024
#define PXVK_FUNCTION_NAME_CAPACITY 512
#define PXVK_MAX_LAUNCH_PARAMETER_BYTES (64u * 1024u)
#define PXVK_MAX_FATBIN_BYTES (128u * 1024u * 1024u)
#define PXVK_CUDART_TABLE_LENGTH 13

#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif

#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif

typedef struct pxvk_function_info {
    CUfunction handle;
    char name[PXVK_FUNCTION_NAME_CAPACITY];
    struct pxvk_function_info *next;
} pxvk_function_info;

typedef struct pxvk_launch_sample {
    CUfunction function;
    size_t size;
    unsigned int ordinal;
    struct pxvk_launch_sample *next;
} pxvk_launch_sample;

typedef struct pxvk_fatbin_header {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t files_size;
} pxvk_fatbin_header;

typedef struct pxvk_fatbin_wrapper {
    uint32_t magic;
    uint32_t version;
    const pxvk_fatbin_header *data;
    const void *filename_or_fatbins;
} pxvk_fatbin_wrapper;

typedef CUresult(CUDAAPI *pxvk_cudart_get_module_fn)(
    CUmodule *module,
    const pxvk_fatbin_wrapper *wrapper);

static INIT_ONCE g_init_once = INIT_ONCE_STATIC_INIT;
typedef FARPROC(WINAPI *pxvk_get_proc_address_fn)(HMODULE module, LPCSTR name);

static HMODULE g_real_driver;
static HMODULE g_self_module;
static pxvk_get_proc_address_fn g_original_get_proc_address;
static HANDLE g_log_file = INVALID_HANDLE_VALUE;
static CRITICAL_SECTION g_lock;
static bool g_lock_initialized;
static wchar_t g_output_directory[PXVK_PATH_CAPACITY];
static pxvk_function_info *g_functions;
static pxvk_launch_sample *g_launch_samples;
static unsigned int g_launch_sample_ordinal;
static volatile LONG g_fatbin_ordinal;
static const void *g_cudart_proxy_table[PXVK_CUDART_TABLE_LENGTH];
static pxvk_cudart_get_module_fn g_real_cudart_get_module;

static volatile LONG g_launch_calls;
static volatile LONG g_context_sync_calls;
static volatile LONG g_host_to_device_calls;
static volatile LONG g_device_to_host_calls;
static volatile LONG g_device_to_device_calls;
static volatile LONG g_memset_calls;
static volatile LONG g_texture_calls;
static volatile LONG64 g_host_to_device_bytes;
static volatile LONG64 g_device_to_host_bytes;
static volatile LONG64 g_device_to_device_bytes;

static const unsigned char g_cudart_interface_guid[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9};

static void append_log_unlocked(const char *message, size_t length)
{
    DWORD written;
    if (g_log_file == INVALID_HANDLE_VALUE || length == 0) {
        return;
    }
    WriteFile(g_log_file, message, (DWORD)length, &written, NULL);
}

static void trace_log(const char *format, ...);

/* A switch is on when set to anything but empty, 0, off, false or no
   (docs/environment.md). */
static int env_flag(const char *name)
{
    char value[16];
    DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0) return 0;
    if (length >= sizeof(value)) return 1;
    return _stricmp(value, "0") != 0 && _stricmp(value, "off") != 0 && _stricmp(value, "false") != 0 &&
           _stricmp(value, "no") != 0;
}

static int api_log_enabled(void)
{
    static int value = -1;
    if (value < 0) value = env_flag("PXVK_API_LOG");
    return value;
}

/* Same prefix as pxvk's API log: the raw PXVK_FRAME (empty when unset) and
 * the calling thread. */
static const char *api_frame(char text[32])
{
    DWORD length = GetEnvironmentVariableA("PXVK_FRAME", text, 32);
    if (length == 0 || length >= 32) text[0] = 0;
    return text;
}

#define API_LOG(...) do { if (api_log_enabled()) { char api_buffer_[256], api_frame_[32]; snprintf(api_buffer_, sizeof(api_buffer_), __VA_ARGS__); trace_log("api f=%s t=%lu %s", api_frame(api_frame_), (unsigned long)GetCurrentThreadId(), api_buffer_); } } while (0)

static void trace_log(const char *format, ...)
{
    char buffer[4096];
    int length;
    va_list args;

    if (!g_lock_initialized) {
        return;
    }
    va_start(args, format);
    length = vsnprintf(buffer, sizeof(buffer) - 2, format, args);
    va_end(args);
    if (length < 0) {
        return;
    }
    if ((size_t)length > sizeof(buffer) - 2) {
        length = (int)(sizeof(buffer) - 2);
    }
    buffer[length++] = '\r';
    buffer[length++] = '\n';

    EnterCriticalSection(&g_lock);
    append_log_unlocked(buffer, (size_t)length);
    LeaveCriticalSection(&g_lock);
}

static bool create_directories(wchar_t *path)
{
    wchar_t *cursor;
    if (path == NULL || path[0] == L'\0') {
        return false;
    }
    for (cursor = path + 1; *cursor != L'\0'; ++cursor) {
        if (*cursor != L'\\' && *cursor != L'/') {
            continue;
        }
        if (cursor == path + 2 && path[1] == L':') {
            continue;
        }
        {
            wchar_t saved = *cursor;
            *cursor = L'\0';
            if (!CreateDirectoryW(path, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
                *cursor = saved;
                return false;
            }
            *cursor = saved;
        }
    }
    return CreateDirectoryW(path, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
}

static const wchar_t *process_file_name(void)
{
    static wchar_t executable[PXVK_PATH_CAPACITY];
    const wchar_t *file_name = executable;
    const wchar_t *cursor;
    DWORD length = GetModuleFileNameW(NULL, executable, PXVK_PATH_CAPACITY);
    if (length == 0 || length >= PXVK_PATH_CAPACITY) {
        return L"process";
    }
    for (cursor = executable; *cursor != L'\0'; ++cursor) {
        if (*cursor == L'\\' || *cursor == L'/') {
            file_name = cursor + 1;
        }
    }
    return file_name;
}

static bool initialize_output(void)
{
    wchar_t base[PXVK_PATH_CAPACITY];
    wchar_t log_path[PXVK_PATH_CAPACITY];
    DWORD length = GetEnvironmentVariableW(L"PXVK_TRACE_DIR", base, PXVK_PATH_CAPACITY);

    if (length == 0) {
        length = GetTempPathW(PXVK_PATH_CAPACITY, base);
        if (length == 0 || length >= PXVK_PATH_CAPACITY) {
            return false;
        }
        if (swprintf(base + length, PXVK_PATH_CAPACITY - length, L"pxvk-nvcuda-trace") < 0) {
            return false;
        }
    } else if (length >= PXVK_PATH_CAPACITY) {
        return false;
    }

    if (swprintf(g_output_directory,
                 PXVK_PATH_CAPACITY,
                 L"%ls\\%ls_%lu",
                 base,
                 process_file_name(),
                 (unsigned long)GetCurrentProcessId()) < 0) {
        return false;
    }
    if (!create_directories(g_output_directory)) {
        return false;
    }
    if (swprintf(log_path, PXVK_PATH_CAPACITY, L"%ls\\trace.log", g_output_directory) < 0) {
        return false;
    }
    g_log_file = CreateFileW(log_path,
                             GENERIC_WRITE,
                             FILE_SHARE_READ,
                             NULL,
                             CREATE_ALWAYS, /* a reused PID's old folder is overwritten */
                             FILE_ATTRIBUTE_NORMAL,
                             NULL);
    return g_log_file != INVALID_HANDLE_VALUE;
}

static bool load_real_driver(void)
{
    wchar_t system_directory[PXVK_PATH_CAPACITY];
    wchar_t driver_path[PXVK_PATH_CAPACITY];
    UINT length = GetSystemWow64DirectoryW(system_directory, PXVK_PATH_CAPACITY);

    if (length == 0 || length >= PXVK_PATH_CAPACITY) {
        return false;
    }
    if (swprintf(driver_path, PXVK_PATH_CAPACITY, L"%ls\\nvcuda.dll", system_directory) < 0) {
        return false;
    }
    g_real_driver = LoadLibraryExW(driver_path,
                                  NULL,
                                  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                      LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (g_real_driver == NULL && GetLastError() == ERROR_INVALID_PARAMETER) {
        g_real_driver = LoadLibraryExW(driver_path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    return g_real_driver != NULL;
}

static FARPROC WINAPI traced_get_proc_address(HMODULE module, LPCSTR name);

static bool patch_module_get_proc_address(HMODULE module)
{
    unsigned char *base = (unsigned char *)module;
    IMAGE_DOS_HEADER *dos_header;
    IMAGE_NT_HEADERS32 *nt_headers;
    IMAGE_DATA_DIRECTORY imports;
    IMAGE_IMPORT_DESCRIPTOR *descriptor;
    bool patched = false;

    if (module == NULL || module == g_self_module || module == g_real_driver) {
        return false;
    }
    dos_header = (IMAGE_DOS_HEADER *)base;
    if (dos_header->e_magic != IMAGE_DOS_SIGNATURE) {
        return false;
    }
    nt_headers = (IMAGE_NT_HEADERS32 *)(base + dos_header->e_lfanew);
    if (nt_headers->Signature != IMAGE_NT_SIGNATURE ||
        nt_headers->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR32_MAGIC) {
        return false;
    }
    imports = nt_headers->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (imports.VirtualAddress == 0 || imports.Size < sizeof(IMAGE_IMPORT_DESCRIPTOR)) {
        return false;
    }

    descriptor = (IMAGE_IMPORT_DESCRIPTOR *)(base + imports.VirtualAddress);
    while (descriptor->Name != 0 && descriptor->FirstThunk != 0) {
        IMAGE_THUNK_DATA32 *thunk =
            (IMAGE_THUNK_DATA32 *)(base + descriptor->FirstThunk);
        while (thunk->u1.Function != 0) {
            FARPROC current = (FARPROC)(uintptr_t)thunk->u1.Function;
            if ((uintptr_t)current == (uintptr_t)g_original_get_proc_address) {
                DWORD old_protection;
                if (VirtualProtect(&thunk->u1.Function,
                                   sizeof(thunk->u1.Function),
                                   PAGE_READWRITE,
                                   &old_protection)) {
                    DWORD ignored;
                    thunk->u1.Function = (DWORD)(uintptr_t)traced_get_proc_address;
                    VirtualProtect(&thunk->u1.Function,
                                   sizeof(thunk->u1.Function),
                                   old_protection,
                                   &ignored);
                    patched = true;
                }
            }
            thunk++;
        }
        descriptor++;
    }
    return patched;
}

static unsigned int install_get_proc_address_hooks(void)
{
    HANDLE snapshot;
    MODULEENTRY32W module;
    unsigned int patched_modules = 0;

    g_original_get_proc_address = GetProcAddress;
    snapshot = CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    if (snapshot == INVALID_HANDLE_VALUE) {
        return 0;
    }
    ZeroMemory(&module, sizeof(module));
    module.dwSize = sizeof(module);
    if (Module32FirstW(snapshot, &module)) {
        do {
            if (patch_module_get_proc_address(module.hModule)) {
                patched_modules++;
            }
        } while (Module32NextW(snapshot, &module));
    }
    CloseHandle(snapshot);
    return patched_modules;
}

static FARPROC WINAPI traced_get_proc_address(HMODULE module, LPCSTR name)
{
    FARPROC result;
    FARPROC replacement;

    if (g_original_get_proc_address == NULL) {
        return NULL;
    }
    result = g_original_get_proc_address(module, name);
    if (module != g_real_driver || (uintptr_t)name <= UINT16_MAX) {
        return result;
    }
    replacement = g_original_get_proc_address(g_self_module, name);
    if (replacement != NULL) {
        trace_log("GetProcAddress nvcuda name=%s intercepted=1", name);
        return replacement;
    }
    trace_log("GetProcAddress nvcuda name=%s intercepted=0", name);
    return result;
}

static FARPROC resolve_symbol(const char *name);

static int capture_sync(void)
{
    typedef CUresult(CUDAAPI *function_type)(void);
    function_type fn = (function_type)resolve_symbol("cuCtxSynchronize");
    return fn != NULL ? (int)fn() : -1;
}

static int capture_dtoh(void *destination, uint32_t source, uint32_t bytes)
{
    typedef CUresult(CUDAAPI *function_type)(void *, CUdeviceptr, size_t);
    function_type fn = (function_type)resolve_symbol("cuMemcpyDtoH_v2");
    return fn != NULL ? (int)fn(destination, (CUdeviceptr)source, bytes) : -1;
}

/* Runs exactly once: a failure is remembered (g_initialized stays false)
 * rather than retried on every call. */
static bool g_initialized;

static BOOL CALLBACK initialize_once(PINIT_ONCE once, PVOID parameter, PVOID *context)
{
    unsigned int patched_modules;
    (void)once;
    (void)parameter;
    (void)context;

    InitializeCriticalSection(&g_lock);
    g_lock_initialized = true;
    if (!initialize_output()) {
        return TRUE;
    }
    if (!load_real_driver()) {
        trace_log("init error=load_real_driver win32=%lu", (unsigned long)GetLastError());
        return TRUE;
    }
    g_initialized = true;
    capture_initialize(g_output_directory, capture_sync, capture_dtoh);
    patched_modules = install_get_proc_address_hooks();
    trace_log("trace_start pid=%lu pointer_bits=%u getproc_hooks=%u",
              (unsigned long)GetCurrentProcessId(),
              (unsigned int)(sizeof(void *) * 8),
              patched_modules);
    return TRUE;
}

static bool ensure_initialized(void)
{
    return InitOnceExecuteOnce(&g_init_once, initialize_once, NULL, NULL) != FALSE && g_initialized;
}

static FARPROC resolve_symbol(const char *name)
{
    FARPROC result;
    if (!ensure_initialized() || g_real_driver == NULL) {
        return NULL;
    }
    result = g_original_get_proc_address != NULL
                 ? g_original_get_proc_address(g_real_driver, name)
                 : GetProcAddress(g_real_driver, name);
    if (result == NULL) {
        trace_log("resolve_failed name=%s win32=%lu", name, (unsigned long)GetLastError());
    }
    return result;
}

static bool readable_protection(DWORD protection)
{
    protection &= 0xff;
    return protection == PAGE_READONLY || protection == PAGE_READWRITE ||
           protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_READ ||
           protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
}

static bool read_current_process(void *context, const void *address, void *output, size_t size)
{
    uintptr_t cursor = (uintptr_t)address;
    uintptr_t end;
    unsigned char *destination = output;
    (void)context;

    if (size == 0) {
        return true;
    }
    if (address == NULL || output == NULL || cursor > UINTPTR_MAX - size) {
        return false;
    }
    end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION information;
        uintptr_t region_end;
        size_t chunk;
        if (VirtualQuery((const void *)cursor, &information, sizeof(information)) == 0 ||
            information.State != MEM_COMMIT ||
            (information.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
            !readable_protection(information.Protect)) {
            return false;
        }
        region_end = (uintptr_t)information.BaseAddress + information.RegionSize;
        if (region_end <= cursor) {
            return false;
        }
        chunk = (size_t)((end < region_end ? end : region_end) - cursor);
        memcpy(destination, (const void *)cursor, chunk);
        destination += chunk;
        cursor += chunk;
    }
    return true;
}

static void remember_function(CUfunction function, const char *name)
{
    pxvk_function_info *entry;
    if (function == NULL || name == NULL) {
        return;
    }
    entry = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*entry));
    if (entry == NULL) {
        return;
    }
    entry->handle = function;
    strncpy(entry->name, name, sizeof(entry->name) - 1);

    EnterCriticalSection(&g_lock);
    entry->next = g_functions;
    g_functions = entry;
    LeaveCriticalSection(&g_lock);
}

static void lookup_function_name(CUfunction function, char *output, size_t output_size)
{
    pxvk_function_info *entry;
    if (output_size == 0) {
        return;
    }
    output[0] = '\0';
    EnterCriticalSection(&g_lock);
    for (entry = g_functions; entry != NULL; entry = entry->next) {
        if (entry->handle == function) {
            snprintf(output, output_size, "%s", entry->name);
            break;
        }
    }
    LeaveCriticalSection(&g_lock);
}

static bool reserve_launch_sample(CUfunction function, size_t size, unsigned int *ordinal)
{
    pxvk_launch_sample *entry;
    pxvk_launch_sample *new_entry;
    bool result = false;

    EnterCriticalSection(&g_lock);
    for (entry = g_launch_samples; entry != NULL; entry = entry->next) {
        if (entry->function == function && entry->size == size) {
            LeaveCriticalSection(&g_lock);
            return false;
        }
    }
    new_entry = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*new_entry));
    if (new_entry != NULL) {
        new_entry->function = function;
        new_entry->size = size;
        new_entry->ordinal = ++g_launch_sample_ordinal;
        new_entry->next = g_launch_samples;
        g_launch_samples = new_entry;
        *ordinal = new_entry->ordinal;
        result = true;
    }
    LeaveCriticalSection(&g_lock);
    return result;
}

static bool write_buffer_file(const wchar_t *file_name, const void *buffer, size_t size)
{
    wchar_t path[PXVK_PATH_CAPACITY];
    HANDLE file;
    DWORD written;
    bool success;

    if (swprintf(path, PXVK_PATH_CAPACITY, L"%ls\\%ls", g_output_directory, file_name) < 0) {
        return false;
    }
    file = CreateFileW(path,
                       GENERIC_WRITE,
                       FILE_SHARE_READ,
                       NULL,
                       CREATE_NEW,
                       FILE_ATTRIBUTE_NORMAL,
                       NULL);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    success = size <= MAXDWORD &&
              WriteFile(file, buffer, (DWORD)size, &written, NULL) != FALSE &&
              written == (DWORD)size;
    CloseHandle(file);
    return success;
}

static void capture_launch_parameters(
    CUfunction function,
    void **kernel_parameters,
    void **extra,
    char *status_text,
    size_t status_text_size,
    size_t *parameter_size)
{
    pxvk_launch_params parameters;
    pxvk_launch_params_status status;
    unsigned int ordinal;

    *parameter_size = 0;
    if (kernel_parameters != NULL && extra != NULL) {
        snprintf(status_text, status_text_size, "both_parameter_forms");
        return;
    }
    if (kernel_parameters != NULL) {
        snprintf(status_text, status_text_size, "kernel_params_uncaptured");
        return;
    }

    status = pxvk_parse_launch_extra(extra, read_current_process, NULL, &parameters);
    if (status != PXVK_LAUNCH_PARAMS_OK) {
        snprintf(status_text, status_text_size, "%s", pxvk_launch_params_status_name(status));
        return;
    }
    *parameter_size = parameters.size;
    if (parameters.size > PXVK_MAX_LAUNCH_PARAMETER_BYTES) {
        snprintf(status_text, status_text_size, "oversize");
        return;
    }
    if (!reserve_launch_sample(function, parameters.size, &ordinal)) {
        snprintf(status_text, status_text_size, "seen");
        return;
    }
    if (parameters.size == 0) {
        snprintf(status_text, status_text_size, "sampled_empty");
        return;
    }
    {
        void *copy = HeapAlloc(GetProcessHeap(), 0, parameters.size);
        wchar_t file_name[128];
        if (copy == NULL ||
            !read_current_process(NULL, parameters.buffer, copy, parameters.size)) {
            if (copy != NULL) {
                HeapFree(GetProcessHeap(), 0, copy);
            }
            snprintf(status_text, status_text_size, "buffer_unreadable");
            return;
        }
        swprintf(file_name,
                 sizeof(file_name) / sizeof(file_name[0]),
                 L"launch_%04u_fn_%08lx_size_%lu.bin",
                 ordinal,
                 (unsigned long)(uintptr_t)function,
                 (unsigned long)parameters.size);
        if (write_buffer_file(file_name, copy, parameters.size)) {
            snprintf(status_text, status_text_size, "sampled");
        } else {
            snprintf(status_text, status_text_size, "sample_write_failed");
        }
        HeapFree(GetProcessHeap(), 0, copy);
    }
}

static bool write_memory_range_file(const wchar_t *file_name, const void *memory, size_t size)
{
    wchar_t path[PXVK_PATH_CAPACITY];
    HANDLE file;
    const unsigned char *cursor = memory;
    size_t remaining = size;
    unsigned char block[64 * 1024];

    if (swprintf(path, PXVK_PATH_CAPACITY, L"%ls\\%ls", g_output_directory, file_name) < 0) {
        return false;
    }
    file = CreateFileW(path,
                       GENERIC_WRITE,
                       FILE_SHARE_READ,
                       NULL,
                       CREATE_NEW,
                       FILE_ATTRIBUTE_NORMAL,
                       NULL);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    while (remaining != 0) {
        size_t chunk = remaining < sizeof(block) ? remaining : sizeof(block);
        DWORD written;
        if (!read_current_process(NULL, cursor, block, chunk) ||
            !WriteFile(file, block, (DWORD)chunk, &written, NULL) || written != chunk) {
            CloseHandle(file);
            return false;
        }
        cursor += chunk;
        remaining -= chunk;
    }
    CloseHandle(file);
    return true;
}

static void capture_fatbin(const pxvk_fatbin_wrapper *wrapper, unsigned int ordinal)
{
    pxvk_fatbin_wrapper wrapper_copy;
    pxvk_fatbin_header header;
    uint64_t total_size;
    wchar_t file_name[128];

    if (!read_current_process(NULL, wrapper, &wrapper_copy, sizeof(wrapper_copy)) ||
        wrapper_copy.data == NULL ||
        !read_current_process(NULL, wrapper_copy.data, &header, sizeof(header))) {
        trace_log("fatbin ordinal=%u status=unreadable", ordinal);
        return;
    }
    total_size = (uint64_t)header.header_size + header.files_size;
    if (total_size > PXVK_MAX_FATBIN_BYTES || total_size > SIZE_MAX) {
        trace_log("fatbin ordinal=%u status=oversize bytes=%llu",
                  ordinal,
                  (unsigned long long)total_size);
        return;
    }
    swprintf(file_name, sizeof(file_name) / sizeof(file_name[0]), L"fatbin_%04u.bin", ordinal);
    if (write_memory_range_file(file_name, wrapper_copy.data, (size_t)total_size)) {
        trace_log("fatbin ordinal=%u status=saved magic=%08lx version=%u bytes=%llu",
                  ordinal,
                  (unsigned long)header.magic,
                  (unsigned int)header.version,
                  (unsigned long long)total_size);
    } else {
        trace_log("fatbin ordinal=%u status=write_failed bytes=%llu",
                  ordinal,
                  (unsigned long long)total_size);
    }
}

static CUresult CUDAAPI traced_cudart_get_module_from_cubin(
    CUmodule *module,
    const pxvk_fatbin_wrapper *wrapper)
{
    unsigned int ordinal = (unsigned int)InterlockedIncrement(&g_fatbin_ordinal);
    CUresult result;

    capture_fatbin(wrapper, ordinal);
    result = g_real_cudart_get_module(module, wrapper);
    if (result == CUDA_SUCCESS && module != NULL) capture_note_module(*module, ordinal);
    trace_log("cudart_get_module ordinal=%u module=%p result=%d", ordinal,
              module != NULL ? (void *)*module : NULL, (int)result);
    return result;
}

static bool guid_equal(const CUuuid *guid, const unsigned char expected[16])
{
    return guid != NULL && memcmp(guid->bytes, expected, 16) == 0;
}

static void log_summary(void)
{
    trace_log("summary launches=%ld syncs=%ld h2d_calls=%ld h2d_bytes=%lld "
              "d2h_calls=%ld d2h_bytes=%lld d2d_calls=%ld d2d_bytes=%lld "
              "memsets=%ld texture_calls=%ld fatbins=%u launch_samples=%u",
              g_launch_calls,
              g_context_sync_calls,
              g_host_to_device_calls,
              (long long)g_host_to_device_bytes,
              g_device_to_host_calls,
              (long long)g_device_to_host_bytes,
              g_device_to_device_calls,
              (long long)g_device_to_device_bytes,
              g_memset_calls,
              g_texture_calls,
              (unsigned int)g_fatbin_ordinal,
              g_launch_sample_ordinal);
}

#define RESOLVE_OR_RETURN(type, variable, export_name)                                    \
    FARPROC variable##_raw = resolve_symbol(export_name);                                 \
    type variable = NULL;                                                                 \
    _Static_assert(sizeof(variable) == sizeof(variable##_raw), "function pointer size");   \
    memcpy(&(variable), &(variable##_raw), sizeof(variable));                             \
    if ((variable) == NULL) {                                                             \
        return CUDA_ERROR_NOT_FOUND;                                                      \
    }

PXVK_EXPORT CUresult CUDAAPI cuInit(unsigned int flags)
{
    typedef CUresult(CUDAAPI *function_type)(unsigned int);
    RESOLVE_OR_RETURN(function_type, real_function, "cuInit");
    {
        CUresult result = real_function(flags);
        trace_log("cuInit flags=%u result=%d", flags, (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuGetExportTable(
    const void **export_table,
    const CUuuid *export_table_id)
{
    typedef CUresult(CUDAAPI *function_type)(const void **, const CUuuid *);
    RESOLVE_OR_RETURN(function_type, real_function, "cuGetExportTable");
    {
        CUresult result = real_function(export_table, export_table_id);
        if (result == CUDA_SUCCESS && export_table != NULL &&
            guid_equal(export_table_id, g_cudart_interface_guid)) {
            const void *const *real_table = (const void *const *)*export_table;
            size_t index;
            EnterCriticalSection(&g_lock);
            for (index = 0; index < PXVK_CUDART_TABLE_LENGTH; ++index) {
                g_cudart_proxy_table[index] = real_table[index];
            }
            memcpy(&g_real_cudart_get_module,
                   &g_cudart_proxy_table[1],
                   sizeof(g_real_cudart_get_module));
            {
                pxvk_cudart_get_module_fn traced_function =
                    traced_cudart_get_module_from_cubin;
                memcpy(&g_cudart_proxy_table[1],
                       &traced_function,
                       sizeof(traced_function));
            }
            *export_table = g_cudart_proxy_table;
            LeaveCriticalSection(&g_lock);
            trace_log("cuGetExportTable interface=CUDART result=%d", (int)result);
        } else {
            trace_log("cuGetExportTable guid=%02x%02x%02x%02x result=%d",
                      export_table_id != NULL ? (unsigned char)export_table_id->bytes[0] : 0,
                      export_table_id != NULL ? (unsigned char)export_table_id->bytes[1] : 0,
                      export_table_id != NULL ? (unsigned char)export_table_id->bytes[2] : 0,
                      export_table_id != NULL ? (unsigned char)export_table_id->bytes[3] : 0,
                      (int)result);
        }
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuModuleGetFunction(
    CUfunction *function,
    CUmodule module,
    const char *name)
{
    typedef CUresult(CUDAAPI *function_type)(CUfunction *, CUmodule, const char *);
    RESOLVE_OR_RETURN(function_type, real_function, "cuModuleGetFunction");
    {
        CUresult result = real_function(function, module, name);
        if (result == CUDA_SUCCESS && function != NULL) {
            remember_function(*function, name);
            capture_note_function(*function, module, name);
        }
        trace_log("cuModuleGetFunction module=%p function=%p name=%s result=%d",
                  (void *)module,
                  function != NULL ? (void *)*function : NULL,
                  name != NULL ? name : "(null)",
                  (int)result);
        return result;
    }
}

/* PXVK_TRACE_TIMING: per-kernel GPU time from CUDA events around each launch
 * (serializes launches; for profiling comparisons only). */
typedef struct pxvk_kernel_time {
    char name[PXVK_FUNCTION_NAME_CAPACITY];
    unsigned long long count;
    double total_ms;
    struct pxvk_kernel_time *next;
} pxvk_kernel_time;
static pxvk_kernel_time *g_kernel_times;
static unsigned long long g_timed_launches;

static int timing_enabled(void)
{
    static int value = -1;
    if (value < 0) value = env_flag("PXVK_TRACE_TIMING");
    return value;
}

static void dump_kernel_times(void)
{
    for (pxvk_kernel_time *k = g_kernel_times; k != NULL; k = k->next) {
        trace_log("profile %-60.60s count=%llu total=%.2fms avg=%.4fms", k->name, k->count, k->total_ms,
                  k->total_ms / (double)k->count);
        k->count = 0;
        k->total_ms = 0;
    }
}

/* Events are created per launch (PhysX destroys and recreates its context
 * on scene changes, which invalidates longer-lived events) and kept by the
 * launching thread. Returns false when timing is off or failed. */
static bool timing_begin(CUevent events[2])
{
    typedef CUresult(CUDAAPI *create_fn)(CUevent *, unsigned int);
    typedef CUresult(CUDAAPI *record_fn)(CUevent, CUstream);
    typedef CUresult(CUDAAPI *destroy_fn)(CUevent);
    create_fn create;
    events[0] = events[1] = NULL;
    if (!timing_enabled()) return false;
    create = (create_fn)resolve_symbol("cuEventCreate");
    if (create == NULL || create(&events[0], 0) != CUDA_SUCCESS) return false;
    if (create(&events[1], 0) != CUDA_SUCCESS ||
        ((record_fn)resolve_symbol("cuEventRecord"))(events[0], NULL) != CUDA_SUCCESS) {
        destroy_fn destroy = (destroy_fn)resolve_symbol("cuEventDestroy_v2");
        destroy(events[0]);
        if (events[1] != NULL) destroy(events[1]);
        return false;
    }
    return true;
}

static void timing_end(CUevent events[2], const char *name)
{
    typedef CUresult(CUDAAPI *record_fn)(CUevent, CUstream);
    typedef CUresult(CUDAAPI *sync_fn)(CUevent);
    typedef CUresult(CUDAAPI *elapsed_fn)(float *, CUevent, CUevent);
    typedef CUresult(CUDAAPI *destroy_fn)(CUevent);
    float ms = 0;
    bool measured;
    pxvk_kernel_time *k;
    destroy_fn destroy = (destroy_fn)resolve_symbol("cuEventDestroy_v2");
    measured = ((record_fn)resolve_symbol("cuEventRecord"))(events[1], NULL) == CUDA_SUCCESS &&
               ((sync_fn)resolve_symbol("cuEventSynchronize"))(events[1]) == CUDA_SUCCESS &&
               ((elapsed_fn)resolve_symbol("cuEventElapsedTime"))(&ms, events[0], events[1]) == CUDA_SUCCESS;
    destroy(events[0]);
    destroy(events[1]);
    if (!measured) return;
    EnterCriticalSection(&g_lock);
    for (k = g_kernel_times; k != NULL; k = k->next)
        if (strcmp(k->name, name) == 0) break;
    if (k == NULL) {
        k = HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(*k));
        if (k != NULL) {
            strncpy(k->name, name, sizeof(k->name) - 1);
            k->next = g_kernel_times;
            g_kernel_times = k;
        }
    }
    if (k != NULL) {
        ++k->count;
        k->total_ms += ms;
        if (++g_timed_launches % 500 == 0) dump_kernel_times();
    }
    LeaveCriticalSection(&g_lock);
}

PXVK_EXPORT CUresult CUDAAPI cuLaunchKernel(
    CUfunction function,
    unsigned int grid_x,
    unsigned int grid_y,
    unsigned int grid_z,
    unsigned int block_x,
    unsigned int block_y,
    unsigned int block_z,
    unsigned int shared_memory_bytes,
    CUstream stream,
    void **kernel_parameters,
    void **extra)
{
    typedef CUresult(CUDAAPI *function_type)(CUfunction,
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
    RESOLVE_OR_RETURN(function_type, real_function, "cuLaunchKernel");
    {
        char function_name[PXVK_FUNCTION_NAME_CAPACITY];
        char capture_status[64];
        size_t parameter_size;
        CUresult result;
        unsigned int capture_token;
        void *parameter_copy = NULL;
        CUevent timing_events[2];
        bool timed;
        lookup_function_name(function, function_name, sizeof(function_name));
        capture_launch_parameters(function,
                                  kernel_parameters,
                                  extra,
                                  capture_status,
                                  sizeof(capture_status),
                                  &parameter_size);
        {
            pxvk_launch_params launch;
            const void *buffer = NULL;
            size_t buffer_size = 0;
            const unsigned int grid[3] = {grid_x, grid_y, grid_z};
            const unsigned int block[3] = {block_x, block_y, block_z};
            if (extra != NULL &&
                pxvk_parse_launch_extra(extra, read_current_process, NULL, &launch) == PXVK_LAUNCH_PARAMS_OK &&
                launch.size <= PXVK_MAX_LAUNCH_PARAMETER_BYTES && launch.size != 0 &&
                (parameter_copy = malloc(launch.size)) != NULL) {
                if (read_current_process(NULL, launch.buffer, parameter_copy, launch.size)) {
                    buffer = parameter_copy;
                    buffer_size = launch.size;
                }
            }
            capture_token = capture_launch_begin(function, grid, block, shared_memory_bytes, buffer, buffer_size);
        }
        timed = timing_begin(timing_events);
        result = real_function(function,
                               grid_x,
                               grid_y,
                               grid_z,
                               block_x,
                               block_y,
                               block_z,
                               shared_memory_bytes,
                               stream,
                               kernel_parameters,
                               extra);
        if (timed) timing_end(timing_events, function_name);
        capture_launch_end(capture_token);
        free(parameter_copy);
        InterlockedIncrement(&g_launch_calls);
        trace_log("cuLaunchKernel function=%p name=%s grid=%u,%u,%u block=%u,%u,%u "
                  "shared=%u stream=%p params=%s param_bytes=%lu result=%d",
                  (void *)function,
                  function_name[0] != '\0' ? function_name : "(unknown)",
                  grid_x,
                  grid_y,
                  grid_z,
                  block_x,
                  block_y,
                  block_z,
                  shared_memory_bytes,
                  (void *)stream,
                  capture_status,
                  (unsigned long)parameter_size,
                  (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemHostAlloc(void **pointer, size_t size, unsigned int flags)
{
    typedef CUresult(CUDAAPI *function_type)(void **, size_t, unsigned int);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemHostAlloc");
    {
        CUresult result = real_function(pointer, size, flags);
        trace_log("cuMemHostAlloc pointer=%p bytes=%lu flags=%u result=%d",
                  pointer != NULL ? *pointer : NULL,
                  (unsigned long)size,
                  flags,
                  (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr *pointer, size_t size)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr *, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemAlloc_v2");
    {
        CUresult result = real_function(pointer, size);
        if (result == CUDA_SUCCESS && pointer != NULL) capture_note_alloc((uint32_t)*pointer, (uint32_t)size);
        trace_log("cuMemAlloc_v2 pointer=%08lx bytes=%lu result=%d",
                  pointer != NULL ? (unsigned long)*pointer : 0,
                  (unsigned long)size,
                  (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemFreeHost(void *pointer)
{
    typedef CUresult(CUDAAPI *function_type)(void *);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemFreeHost");
    {
        CUresult result = real_function(pointer);
        trace_log("cuMemFreeHost pointer=%p result=%d", pointer, (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemFree_v2(CUdeviceptr pointer)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemFree_v2");
    {
        CUresult result = real_function(pointer);
        if (result == CUDA_SUCCESS) capture_note_free((uint32_t)pointer);
        trace_log("cuMemFree_v2 pointer=%08lx result=%d", (unsigned long)pointer, (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemcpyHtoDAsync_v2(
    CUdeviceptr destination,
    const void *source,
    size_t size,
    CUstream stream)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr, const void *, size_t, CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyHtoDAsync_v2");
    InterlockedIncrement(&g_host_to_device_calls);
    InterlockedExchangeAdd64(&g_host_to_device_bytes, (LONG64)size);
    API_LOG("htod_async dst=%08lx bytes=%lu", (unsigned long)destination, (unsigned long)size);
    return real_function(destination, source, size, stream);
}

PXVK_EXPORT CUresult CUDAAPI cuMemcpyDtoHAsync_v2(
    void *destination,
    CUdeviceptr source,
    size_t size,
    CUstream stream)
{
    typedef CUresult(CUDAAPI *function_type)(void *, CUdeviceptr, size_t, CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyDtoHAsync_v2");
    InterlockedIncrement(&g_device_to_host_calls);
    InterlockedExchangeAdd64(&g_device_to_host_bytes, (LONG64)size);
    API_LOG("dtoh_async src=%08lx bytes=%lu", (unsigned long)source, (unsigned long)size);
    return real_function(destination, source, size, stream);
}

PXVK_EXPORT CUresult CUDAAPI cuMemcpyDtoDAsync_v2(
    CUdeviceptr destination,
    CUdeviceptr source,
    size_t size,
    CUstream stream)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyDtoDAsync_v2");
    InterlockedIncrement(&g_device_to_device_calls);
    InterlockedExchangeAdd64(&g_device_to_device_bytes, (LONG64)size);
    API_LOG("dtod_async dst=%08lx src=%08lx bytes=%lu", (unsigned long)destination, (unsigned long)source, (unsigned long)size);
    return real_function(destination, source, size, stream);
}

PXVK_EXPORT CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr destination, unsigned char value, size_t count)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr, unsigned char, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemsetD8_v2");
    InterlockedIncrement(&g_memset_calls);
    API_LOG("memset dst=%08lx value=%u bytes=%lu", (unsigned long)destination, (unsigned)value, (unsigned long)count);
    return real_function(destination, value, count);
}

PXVK_EXPORT CUresult CUDAAPI cuCtxSynchronize(void)
{
    typedef CUresult(CUDAAPI *function_type)(void);
    RESOLVE_OR_RETURN(function_type, real_function, "cuCtxSynchronize");
    InterlockedIncrement(&g_context_sync_calls);
    API_LOG("sync");
    return real_function();
}

PXVK_EXPORT CUresult CUDAAPI cuCtxDetach(CUcontext context)
{
    typedef CUresult(CUDAAPI *function_type)(CUcontext);
    RESOLVE_OR_RETURN(function_type, real_function, "cuCtxDetach");
    log_summary();
    {
        CUresult result = real_function(context);
        trace_log("cuCtxDetach context=%p result=%d", (void *)context, (int)result);
        return result;
    }
}

#define DEFINE_TEXTURE_FORWARDER_2(name, type2)                                           \
    PXVK_EXPORT CUresult CUDAAPI name(CUtexref texture, type2 value)                       \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(CUtexref, type2);                         \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        InterlockedIncrement(&g_texture_calls);                                             \
        return real_function(texture, value);                                               \
    }

DEFINE_TEXTURE_FORWARDER_2(cuTexRefSetFlags, unsigned int)
DEFINE_TEXTURE_FORWARDER_2(cuTexRefSetFilterMode, CUfilter_mode)
DEFINE_TEXTURE_FORWARDER_2(cuTexRefSetMipmapFilterMode, CUfilter_mode)
DEFINE_TEXTURE_FORWARDER_2(cuTexRefSetMipmapLevelBias, float)
DEFINE_TEXTURE_FORWARDER_2(cuTexRefSetMaxAnisotropy, unsigned int)

PXVK_EXPORT CUresult CUDAAPI cuTexRefSetAddress_v2(
    size_t *byte_offset,
    CUtexref texture,
    CUdeviceptr pointer,
    size_t size)
{
    typedef CUresult(CUDAAPI *function_type)(size_t *, CUtexref, CUdeviceptr, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuTexRefSetAddress_v2");
    InterlockedIncrement(&g_texture_calls);
    API_LOG("texbind ref=%p ptr=%08lx bytes=%lu", (void *)texture, (unsigned long)pointer, (unsigned long)size);
    capture_note_texref_address(texture, (uint32_t)pointer, (uint32_t)size);
    return real_function(byte_offset, texture, pointer, size);
}

PXVK_EXPORT CUresult CUDAAPI cuTexRefSetFormat(
    CUtexref texture,
    CUarray_format format,
    int packed_components)
{
    typedef CUresult(CUDAAPI *function_type)(CUtexref, CUarray_format, int);
    RESOLVE_OR_RETURN(function_type, real_function, "cuTexRefSetFormat");
    InterlockedIncrement(&g_texture_calls);
    capture_note_texref_format(texture, (int)format, packed_components);
    return real_function(texture, format, packed_components);
}

PXVK_EXPORT CUresult CUDAAPI cuTexRefSetMipmapLevelClamp(
    CUtexref texture,
    float minimum,
    float maximum)
{
    typedef CUresult(CUDAAPI *function_type)(CUtexref, float, float);
    RESOLVE_OR_RETURN(function_type, real_function, "cuTexRefSetMipmapLevelClamp");
    InterlockedIncrement(&g_texture_calls);
    return real_function(texture, minimum, maximum);
}

PXVK_EXPORT CUresult CUDAAPI cuTexRefSetAddressMode(
    CUtexref texture,
    int dimension,
    CUaddress_mode mode)
{
    typedef CUresult(CUDAAPI *function_type)(CUtexref, int, CUaddress_mode);
    RESOLVE_OR_RETURN(function_type, real_function, "cuTexRefSetAddressMode");
    InterlockedIncrement(&g_texture_calls);
    return real_function(texture, dimension, mode);
}

#define DEFINE_FORWARDER_0(name)                                                          \
    PXVK_EXPORT CUresult CUDAAPI name(void)                                                \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(void);                                    \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function();                                                            \
    }

#define DEFINE_FORWARDER_1(name, type1, arg1)                                             \
    PXVK_EXPORT CUresult CUDAAPI name(type1 arg1)                                          \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(type1);                                   \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function(arg1);                                                        \
    }

#define DEFINE_FORWARDER_2(name, type1, arg1, type2, arg2)                                \
    PXVK_EXPORT CUresult CUDAAPI name(type1 arg1, type2 arg2)                              \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(type1, type2);                            \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function(arg1, arg2);                                                  \
    }

#define DEFINE_FORWARDER_3(name, type1, arg1, type2, arg2, type3, arg3)                   \
    PXVK_EXPORT CUresult CUDAAPI name(type1 arg1, type2 arg2, type3 arg3)                  \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(type1, type2, type3);                     \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function(arg1, arg2, arg3);                                            \
    }

#define DEFINE_FORWARDER_4(name, type1, arg1, type2, arg2, type3, arg3, type4, arg4)      \
    PXVK_EXPORT CUresult CUDAAPI name(type1 arg1, type2 arg2, type3 arg3, type4 arg4)      \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(type1, type2, type3, type4);              \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function(arg1, arg2, arg3, arg4);                                      \
    }

#define DEFINE_FORWARDER_5(name, type1, arg1, type2, arg2, type3, arg3, type4, arg4,     \
                           type5, arg5)                                                    \
    PXVK_EXPORT CUresult CUDAAPI name(                                                     \
        type1 arg1, type2 arg2, type3 arg3, type4 arg4, type5 arg5)                       \
    {                                                                                      \
        typedef CUresult(CUDAAPI *function_type)(type1, type2, type3, type4, type5);       \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                            \
        return real_function(arg1, arg2, arg3, arg4, arg5);                               \
    }

DEFINE_FORWARDER_1(cuDriverGetVersion, int *, version)
DEFINE_FORWARDER_1(cuDeviceGetCount, int *, count)
DEFINE_FORWARDER_2(cuDeviceGet, CUdevice *, device, int, ordinal)
DEFINE_FORWARDER_3(cuDeviceGetName, char *, name, int, length, CUdevice, device)
DEFINE_FORWARDER_2(cuDeviceTotalMem_v2, size_t *, bytes, CUdevice, device)
DEFINE_FORWARDER_3(cuDeviceGetAttribute,
                   int *,
                   value,
                   CUdevice_attribute,
                   attribute,
                   CUdevice,
                   device)
DEFINE_FORWARDER_1(cuCtxGetCurrent, CUcontext *, context)
DEFINE_FORWARDER_1(cuCtxGetDevice, CUdevice *, device)
DEFINE_FORWARDER_2(cuCtxGetApiVersion, CUcontext, context, unsigned int *, version)
PXVK_EXPORT CUresult CUDAAPI cuModuleGetGlobal_v2(
    CUdeviceptr *pointer, size_t *bytes, CUmodule module, const char *name)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr *, size_t *, CUmodule, const char *);
    RESOLVE_OR_RETURN(function_type, real_function, "cuModuleGetGlobal_v2");
    {
        size_t size = 0;
        CUresult result = real_function(pointer, &size, module, name);
        if (bytes != NULL) *bytes = size;
        if (result == CUDA_SUCCESS && pointer != NULL && name != NULL)
            capture_note_global(module, name, (uint32_t)*pointer, (uint32_t)size);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuModuleGetTexRef(CUtexref *texture, CUmodule module, const char *name)
{
    typedef CUresult(CUDAAPI *function_type)(CUtexref *, CUmodule, const char *);
    RESOLVE_OR_RETURN(function_type, real_function, "cuModuleGetTexRef");
    {
        CUresult result = real_function(texture, module, name);
        if (result == CUDA_SUCCESS && texture != NULL && name != NULL)
            capture_note_texref(*texture, module, name);
        return result;
    }
}
DEFINE_FORWARDER_3(cuMemGetAddressRange_v2,
                   CUdeviceptr *,
                   base,
                   size_t *,
                   size,
                   CUdeviceptr,
                   pointer)

/* Legacy entry points queried by the PhysX 2.8 runtime even when its active
 * path ultimately uses the v2 APIs above. */
DEFINE_FORWARDER_3(cuDeviceComputeCapability,
                   int *,
                   major,
                   int *,
                   minor,
                   CUdevice,
                   device)
DEFINE_FORWARDER_2(cuDeviceGetProperties, CUdevprop *, properties, CUdevice, device)
DEFINE_FORWARDER_2(cuDeviceTotalMem, size_t *, bytes, CUdevice, device)
DEFINE_FORWARDER_3(cuCtxCreate,
                   CUcontext *,
                   context,
                   unsigned int,
                   flags,
                   CUdevice,
                   device)
DEFINE_FORWARDER_3(cuCtxCreate_v2,
                   CUcontext *,
                   context,
                   unsigned int,
                   flags,
                   CUdevice,
                   device)
/* Destroying a context frees its modules and memory: drop their capture
 * records so a later module reusing a handle or address does not pick them up. */
#define DEFINE_CONTEXT_DESTROY(name)                                                      \
    PXVK_EXPORT CUresult CUDAAPI name(CUcontext context)                                  \
    {                                                                                     \
        typedef CUresult(CUDAAPI *function_type)(CUcontext);                              \
        RESOLVE_OR_RETURN(function_type, real_function, #name);                           \
        {                                                                                 \
            CUresult result = real_function(context);                                     \
            if (result == CUDA_SUCCESS) capture_forget_all();                             \
            trace_log(#name " context=%p result=%d", (void *)context, (int)result);       \
            return result;                                                                \
        }                                                                                 \
    }
DEFINE_CONTEXT_DESTROY(cuCtxDestroy)
DEFINE_CONTEXT_DESTROY(cuCtxDestroy_v2)
DEFINE_FORWARDER_1(cuCtxPushCurrent, CUcontext, context)
DEFINE_FORWARDER_1(cuCtxPushCurrent_v2, CUcontext, context)
DEFINE_FORWARDER_1(cuCtxPopCurrent, CUcontext *, context)
DEFINE_FORWARDER_1(cuCtxPopCurrent_v2, CUcontext *, context)
DEFINE_FORWARDER_2(cuModuleLoadData, CUmodule *, module, const void *, image)
DEFINE_FORWARDER_5(cuModuleLoadDataEx,
                   CUmodule *,
                   module,
                   const void *,
                   image,
                   unsigned int,
                   option_count,
                   CUjit_option *,
                   options,
                   void **,
                   option_values)
PXVK_EXPORT CUresult CUDAAPI cuModuleUnload(CUmodule module)
{
    typedef CUresult(CUDAAPI *function_type)(CUmodule);
    RESOLVE_OR_RETURN(function_type, real_function, "cuModuleUnload");
    {
        CUresult result = real_function(module);
        if (result == CUDA_SUCCESS) capture_forget_module(module);
        trace_log("cuModuleUnload module=%p result=%d", (void *)module, (int)result);
        return result;
    }
}
DEFINE_FORWARDER_4(cuModuleGetGlobal,
                   CUdeviceptr *,
                   pointer,
                   size_t *,
                   bytes,
                   CUmodule,
                   module,
                   const char *,
                   name)
PXVK_EXPORT CUresult CUDAAPI cuMemAlloc(CUdeviceptr *pointer, size_t size)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr *, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemAlloc");
    {
        CUresult result = real_function(pointer, size);
        if (result == CUDA_SUCCESS && pointer != NULL) capture_note_alloc((uint32_t)*pointer, (uint32_t)size);
        trace_log("cuMemAlloc pointer=%08lx bytes=%lu result=%d",
                  pointer != NULL ? (unsigned long)*pointer : 0, (unsigned long)size, (int)result);
        return result;
    }
}

PXVK_EXPORT CUresult CUDAAPI cuMemFree(CUdeviceptr pointer)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemFree");
    {
        CUresult result = real_function(pointer);
        if (result == CUDA_SUCCESS) capture_note_free((uint32_t)pointer);
        trace_log("cuMemFree pointer=%08lx result=%d", (unsigned long)pointer, (int)result);
        return result;
    }
}
DEFINE_FORWARDER_3(cuMemGetAddressRange,
                   CUdeviceptr *,
                   base,
                   size_t *,
                   size,
                   CUdeviceptr,
                   pointer)
DEFINE_FORWARDER_4(cuTexRefSetAddress,
                   size_t *,
                   byte_offset,
                   CUtexref,
                   texture,
                   CUdeviceptr,
                   pointer,
                   size_t,
                   bytes)
DEFINE_FORWARDER_4(cuMemcpyHtoDAsync,
                   CUdeviceptr,
                   destination,
                   const void *,
                   source,
                   size_t,
                   bytes,
                   CUstream,
                   stream)
DEFINE_FORWARDER_4(cuMemcpyDtoHAsync,
                   void *,
                   destination,
                   CUdeviceptr,
                   source,
                   size_t,
                   bytes,
                   CUstream,
                   stream)
DEFINE_FORWARDER_4(cuMemcpyDtoDAsync,
                   CUdeviceptr,
                   destination,
                   CUdeviceptr,
                   source,
                   size_t,
                   bytes,
                   CUstream,
                   stream)
DEFINE_FORWARDER_3(cuMemsetD8, CUdeviceptr, destination, unsigned char, value, size_t, count)
DEFINE_FORWARDER_4(cuFuncSetBlockShape,
                   CUfunction,
                   function,
                   int,
                   x,
                   int,
                   y,
                   int,
                   z)
DEFINE_FORWARDER_2(cuFuncSetSharedSize, CUfunction, function, unsigned int, bytes)
DEFINE_FORWARDER_2(cuParamSetSize, CUfunction, function, unsigned int, bytes)
DEFINE_FORWARDER_3(cuParamSeti, CUfunction, function, int, offset, unsigned int, value)
DEFINE_FORWARDER_3(cuParamSetf, CUfunction, function, int, offset, float, value)
DEFINE_FORWARDER_4(cuParamSetv,
                   CUfunction,
                   function,
                   int,
                   offset,
                   void *,
                   pointer,
                   unsigned int,
                   bytes)
DEFINE_FORWARDER_1(cuLaunch, CUfunction, function)
DEFINE_FORWARDER_3(cuLaunchGrid, CUfunction, function, int, width, int, height)
DEFINE_FORWARDER_4(cuLaunchGridAsync,
                   CUfunction,
                   function,
                   int,
                   width,
                   int,
                   height,
                   CUstream,
                   stream)
DEFINE_FORWARDER_2(cuGetErrorName, CUresult, error, const char **, text)
DEFINE_FORWARDER_2(cuGetErrorString, CUresult, error, const char **, text)

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        g_self_module = instance;
        DisableThreadLibraryCalls(instance);
    } else if (reason == DLL_PROCESS_DETACH && g_log_file != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(g_log_file);
        CloseHandle(g_log_file);
        g_log_file = INVALID_HANDLE_VALUE;
    }
    return TRUE;
}

/* Logged forwarders for synchronization-relevant APIs (PXVK_API_LOG). */
PXVK_EXPORT CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr d, const void *s, size_t n)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr, const void *, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyHtoD_v2");
    API_LOG("htod dst=%08lx bytes=%lu", (unsigned long)d, (unsigned long)n);
    return real_function(d, s, n);
}
PXVK_EXPORT CUresult CUDAAPI cuMemcpyDtoH_v2(void *d, CUdeviceptr s, size_t n)
{
    typedef CUresult(CUDAAPI *function_type)(void *, CUdeviceptr, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyDtoH_v2");
    API_LOG("dtoh src=%08lx bytes=%lu", (unsigned long)s, (unsigned long)n);
    return real_function(d, s, n);
}
PXVK_EXPORT CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr d, CUdeviceptr s, size_t n)
{
    typedef CUresult(CUDAAPI *function_type)(CUdeviceptr, CUdeviceptr, size_t);
    RESOLVE_OR_RETURN(function_type, real_function, "cuMemcpyDtoD_v2");
    API_LOG("dtod dst=%08lx src=%08lx bytes=%lu", (unsigned long)d, (unsigned long)s, (unsigned long)n);
    return real_function(d, s, n);
}
PXVK_EXPORT CUresult CUDAAPI cuEventRecord(CUevent e, CUstream st)
{
    typedef CUresult(CUDAAPI *function_type)(CUevent, CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuEventRecord");
    API_LOG("event_record %p stream=%p", (void *)e, (void *)st);
    return real_function(e, st);
}
PXVK_EXPORT CUresult CUDAAPI cuEventQuery(CUevent e)
{
    typedef CUresult(CUDAAPI *function_type)(CUevent);
    RESOLVE_OR_RETURN(function_type, real_function, "cuEventQuery");
    {
        CUresult r = real_function(e);
        API_LOG("event_query %p result=%d", (void *)e, (int)r);
        return r;
    }
}
PXVK_EXPORT CUresult CUDAAPI cuEventSynchronize(CUevent e)
{
    typedef CUresult(CUDAAPI *function_type)(CUevent);
    RESOLVE_OR_RETURN(function_type, real_function, "cuEventSynchronize");
    API_LOG("event_sync %p", (void *)e);
    return real_function(e);
}
PXVK_EXPORT CUresult CUDAAPI cuStreamSynchronize(CUstream st)
{
    typedef CUresult(CUDAAPI *function_type)(CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuStreamSynchronize");
    API_LOG("stream_sync %p", (void *)st);
    return real_function(st);
}
PXVK_EXPORT CUresult CUDAAPI cuStreamQuery(CUstream st)
{
    typedef CUresult(CUDAAPI *function_type)(CUstream);
    RESOLVE_OR_RETURN(function_type, real_function, "cuStreamQuery");
    {
        CUresult r = real_function(st);
        API_LOG("stream_query %p result=%d", (void *)st, (int)r);
        return r;
    }
}
PXVK_EXPORT CUresult CUDAAPI cuStreamCreate(CUstream *st, unsigned int flags)
{
    typedef CUresult(CUDAAPI *function_type)(CUstream *, unsigned int);
    RESOLVE_OR_RETURN(function_type, real_function, "cuStreamCreate");
    {
        CUresult r = real_function(st, flags);
        API_LOG("stream_create %p flags=%u result=%d", st ? (void *)*st : NULL, flags, (int)r);
        return r;
    }
}
PXVK_EXPORT CUresult CUDAAPI cuEventCreate(CUevent *e, unsigned int flags)
{
    typedef CUresult(CUDAAPI *function_type)(CUevent *, unsigned int);
    RESOLVE_OR_RETURN(function_type, real_function, "cuEventCreate");
    {
        CUresult r = real_function(e, flags);
        API_LOG("event_create %p flags=%u result=%d", e ? (void *)*e : NULL, flags, (int)r);
        return r;
    }
}
