#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#define PXVK_PATH_CAPACITY 32768

static void print_last_error(const wchar_t *operation)
{
    fwprintf(stderr, L"%ls failed: %lu\n", operation, (unsigned long)GetLastError());
}

static bool append_character(wchar_t **cursor, size_t *remaining, wchar_t character)
{
    if (*remaining <= 1) {
        return false;
    }
    **cursor = character;
    (*cursor)++;
    (*remaining)--;
    **cursor = L'\0';
    return true;
}

static bool append_repeated(
    wchar_t **cursor,
    size_t *remaining,
    wchar_t character,
    size_t count)
{
    while (count-- != 0) {
        if (!append_character(cursor, remaining, character)) {
            return false;
        }
    }
    return true;
}

static bool append_windows_argument(
    wchar_t **cursor,
    size_t *remaining,
    const wchar_t *argument)
{
    const wchar_t *source;
    bool needs_quotes = argument[0] == L'\0' || wcspbrk(argument, L" \t\n\v\"") != NULL;

    if (!needs_quotes) {
        size_t length = wcslen(argument);
        if (length >= *remaining) {
            return false;
        }
        memcpy(*cursor, argument, (length + 1) * sizeof(wchar_t));
        *cursor += length;
        *remaining -= length;
        return true;
    }

    if (!append_character(cursor, remaining, L'\"')) {
        return false;
    }
    source = argument;
    while (*source != L'\0') {
        size_t backslashes = 0;
        while (*source == L'\\') {
            backslashes++;
            source++;
        }
        if (*source == L'\"') {
            if (!append_repeated(cursor, remaining, L'\\', backslashes * 2 + 1) ||
                !append_character(cursor, remaining, *source++)) {
                return false;
            }
        } else if (*source == L'\0') {
            if (!append_repeated(cursor, remaining, L'\\', backslashes * 2)) {
                return false;
            }
        } else {
            if (!append_repeated(cursor, remaining, L'\\', backslashes) ||
                !append_character(cursor, remaining, *source++)) {
                return false;
            }
        }
    }
    return append_character(cursor, remaining, L'\"');
}

static wchar_t *build_command_line(int argument_count, wchar_t **arguments)
{
    size_t capacity = 1;
    wchar_t *command_line;
    wchar_t *cursor;
    size_t remaining;
    int index;

    for (index = 0; index < argument_count; ++index) {
        capacity += wcslen(arguments[index]) * 2 + 4;
    }
    command_line = calloc(capacity, sizeof(wchar_t));
    if (command_line == NULL) {
        return NULL;
    }
    cursor = command_line;
    remaining = capacity;
    for (index = 0; index < argument_count; ++index) {
        if (index != 0 && !append_character(&cursor, &remaining, L' ')) {
            free(command_line);
            return NULL;
        }
        if (!append_windows_argument(&cursor, &remaining, arguments[index])) {
            free(command_line);
            return NULL;
        }
    }
    return command_line;
}

static bool sibling_tracer_path(wchar_t path[PXVK_PATH_CAPACITY])
{
    DWORD length = GetModuleFileNameW(NULL, path, PXVK_PATH_CAPACITY);
    wchar_t *cursor;
    if (length == 0 || length >= PXVK_PATH_CAPACITY) {
        return false;
    }
    cursor = path + length;
    while (cursor != path && cursor[-1] != L'\\' && cursor[-1] != L'/') {
        cursor--;
    }
    if ((size_t)(cursor - path) + wcslen(L"nvcuda.dll") + 1 > PXVK_PATH_CAPACITY) {
        return false;
    }
    wcscpy(cursor, L"nvcuda.dll");
    return true;
}

static bool inject_library(HANDLE process, const wchar_t *library_path)
{
    size_t path_bytes = (wcslen(library_path) + 1) * sizeof(wchar_t);
    void *remote_path = VirtualAllocEx(
        process, NULL, path_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    HMODULE kernel32;
    FARPROC load_library_raw;
    LPTHREAD_START_ROUTINE load_library = NULL;
    HANDLE thread;
    DWORD remote_module = 0;
    bool success = false;

    if (remote_path == NULL) {
        print_last_error(L"VirtualAllocEx");
        return false;
    }
    if (!WriteProcessMemory(process, remote_path, library_path, path_bytes, NULL)) {
        print_last_error(L"WriteProcessMemory");
        goto cleanup_path;
    }
    kernel32 = GetModuleHandleW(L"kernel32.dll");
    load_library_raw = kernel32 != NULL ? GetProcAddress(kernel32, "LoadLibraryW") : NULL;
    if (load_library_raw == NULL) {
        print_last_error(L"GetProcAddress(LoadLibraryW)");
        goto cleanup_path;
    }
    _Static_assert(sizeof(load_library) == sizeof(load_library_raw), "function pointer size");
    memcpy(&load_library, &load_library_raw, sizeof(load_library));
    thread = CreateRemoteThread(process, NULL, 0, load_library, remote_path, 0, NULL);
    if (thread == NULL) {
        print_last_error(L"CreateRemoteThread");
        goto cleanup_path;
    }
    if (WaitForSingleObject(thread, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeThread(thread, &remote_module)) {
        print_last_error(L"waiting for injected LoadLibraryW");
    } else if (remote_module == 0) {
        fwprintf(stderr, L"remote LoadLibraryW returned NULL\n");
    } else {
        success = true;
    }
    CloseHandle(thread);

cleanup_path:
    VirtualFreeEx(process, remote_path, 0, MEM_RELEASE);
    return success;
}

static void usage(void)
{
    fwprintf(stderr,
             L"usage: pxvk-trace [--output <directory>] -- <executable> [arguments...]\n");
}

int wmain(int argc, wchar_t **argv)
{
    int target_index = 1;
    const wchar_t *output_directory = NULL;
    wchar_t tracer_path[PXVK_PATH_CAPACITY];
    wchar_t *command_line;
    STARTUPINFOW startup_info;
    PROCESS_INFORMATION process_info;
    DWORD exit_code = 1;

    if (argc > 3 && wcscmp(argv[target_index], L"--output") == 0) {
        output_directory = argv[target_index + 1];
        target_index += 2;
    }
    if (target_index >= argc || wcscmp(argv[target_index], L"--") != 0) {
        usage();
        return 2;
    }
    target_index++;
    if (target_index >= argc) {
        usage();
        return 2;
    }
    if (output_directory != NULL &&
        !SetEnvironmentVariableW(L"PXVK_TRACE_DIR", output_directory)) {
        print_last_error(L"SetEnvironmentVariableW(PXVK_TRACE_DIR)");
        return 3;
    }
    if (!sibling_tracer_path(tracer_path)) {
        print_last_error(L"locating sibling nvcuda.dll");
        return 4;
    }
    if (GetFileAttributesW(tracer_path) == INVALID_FILE_ATTRIBUTES) {
        fwprintf(stderr, L"tracer DLL not found: %ls\n", tracer_path);
        return 4;
    }
    command_line = build_command_line(argc - target_index, argv + target_index);
    if (command_line == NULL) {
        fwprintf(stderr, L"failed to build child command line\n");
        return 5;
    }

    ZeroMemory(&startup_info, sizeof(startup_info));
    startup_info.cb = sizeof(startup_info);
    ZeroMemory(&process_info, sizeof(process_info));
    if (!CreateProcessW(argv[target_index],
                        command_line,
                        NULL,
                        NULL,
                        FALSE,
                        CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                        NULL,
                        NULL,
                        &startup_info,
                        &process_info)) {
        print_last_error(L"CreateProcessW");
        free(command_line);
        return 6;
    }
    free(command_line);

    if (!inject_library(process_info.hProcess, tracer_path)) {
        TerminateProcess(process_info.hProcess, 7);
        WaitForSingleObject(process_info.hProcess, INFINITE);
        CloseHandle(process_info.hThread);
        CloseHandle(process_info.hProcess);
        return 7;
    }
    if (ResumeThread(process_info.hThread) == (DWORD)-1) {
        print_last_error(L"ResumeThread");
        TerminateProcess(process_info.hProcess, 8);
    }
    CloseHandle(process_info.hThread);
    WaitForSingleObject(process_info.hProcess, INFINITE);
    if (!GetExitCodeProcess(process_info.hProcess, &exit_code)) {
        print_last_error(L"GetExitCodeProcess");
        exit_code = 9;
    }
    CloseHandle(process_info.hProcess);
    return (int)exit_code;
}
