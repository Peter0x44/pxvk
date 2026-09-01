#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#include <iterator>
#include <string>
#include <vector>

namespace {

int fail(const wchar_t *message, DWORD error = 0)
{
    std::wstring text(message);
    if (error != 0) {
        text += L" (Windows error ";
        text += std::to_wstring(error);
        text += L")";
    }
    MessageBoxW(nullptr, text.c_str(), L"pxvk launcher", MB_OK | MB_ICONERROR);
    return 2;
}

std::wstring quote(const wchar_t *argument)
{
    std::wstring output = L"\"";
    unsigned int slashes = 0;
    for (const wchar_t *cursor = argument; ; ++cursor) {
        if (*cursor == L'\\') {
            ++slashes;
            continue;
        }
        if (*cursor == L'\"') {
            output.append(slashes * 2u + 1u, L'\\');
            output.push_back(L'\"');
            slashes = 0;
            continue;
        }
        if (*cursor == L'\0') {
            output.append(slashes * 2u, L'\\');
            break;
        }
        output.append(slashes, L'\\');
        slashes = 0;
        output.push_back(*cursor);
    }
    output.push_back(L'\"');
    return output;
}

std::wstring directory(const std::wstring &path)
{
    const std::size_t separator = path.find_last_of(L"\\/");
    return separator == std::wstring::npos ? L"." : path.substr(0, separator);
}

} // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int)
{
    int argument_count = 0;
    wchar_t **arguments = CommandLineToArgvW(GetCommandLineW(), &argument_count);
    if (arguments == nullptr || argument_count < 2)
        return fail(L"Usage: pxvk-launch.exe [--inject-dll <path>] "
                    L"<32-bit application> [arguments]");

    int application_argument = 1;
    std::wstring injected_path;
    if (lstrcmpW(arguments[1], L"--inject-dll") == 0) {
        if (argument_count < 4) {
            LocalFree(arguments);
            return fail(L"--inject-dll requires a DLL and an application");
        }
        wchar_t injected_buffer[32768]{};
        if (GetFullPathNameW(arguments[2],
                             static_cast<DWORD>(std::size(injected_buffer)),
                             injected_buffer, nullptr) == 0) {
            const DWORD error = GetLastError();
            LocalFree(arguments);
            return fail(L"Cannot resolve the injected DLL path", error);
        }
        injected_path = injected_buffer;
        application_argument = 3;
    }

    wchar_t application_buffer[32768]{};
    if (GetFullPathNameW(arguments[application_argument],
                         static_cast<DWORD>(std::size(application_buffer)),
                         application_buffer, nullptr) == 0) {
        const DWORD error = GetLastError();
        LocalFree(arguments);
        return fail(L"Cannot resolve the application path", error);
    }
    const std::wstring application(application_buffer);
    std::wstring command_line;
    for (int i = application_argument; i < argument_count; ++i) {
        if (!command_line.empty()) command_line.push_back(L' ');
        command_line += quote(arguments[i]);
    }
    LocalFree(arguments);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    std::vector<wchar_t> mutable_command(command_line.begin(), command_line.end());
    mutable_command.push_back(L'\0');
    const std::wstring working_directory = directory(application);
    if (CreateProcessW(application.c_str(), mutable_command.data(), nullptr, nullptr,
                       FALSE, CREATE_SUSPENDED, nullptr, working_directory.c_str(),
                       &startup, &process) == FALSE) {
        return fail(L"Cannot create the target process", GetLastError());
    }

    wchar_t launcher_path[32768]{};
    GetModuleFileNameW(nullptr, launcher_path,
                       static_cast<DWORD>(std::size(launcher_path)));
    const std::wstring runtime_path =
        injected_path.empty()
            ? directory(launcher_path) + L"\\nvcuda.dll"
            : injected_path;
    const SIZE_T path_bytes = (runtime_path.size() + 1u) * sizeof(wchar_t);
    void *remote_path = VirtualAllocEx(process.hProcess, nullptr, path_bytes,
                                       MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (remote_path == nullptr) {
        TerminateProcess(process.hProcess, 2);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return fail(L"Cannot allocate the injected DLL path", GetLastError());
    }
    SIZE_T written = 0;
    if (WriteProcessMemory(process.hProcess, remote_path, runtime_path.c_str(),
                           path_bytes, &written) == FALSE || written != path_bytes) {
        TerminateProcess(process.hProcess, 2);
        VirtualFreeEx(process.hProcess, remote_path, 0, MEM_RELEASE);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return fail(L"Cannot write the injected DLL path", GetLastError());
    }

    HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    const auto load_library = reinterpret_cast<LPTHREAD_START_ROUTINE>(
        GetProcAddress(kernel, "LoadLibraryW"));
    HANDLE injection = CreateRemoteThread(process.hProcess, nullptr, 0, load_library,
                                          remote_path, 0, nullptr);
    if (injection == nullptr) {
        TerminateProcess(process.hProcess, 2);
        VirtualFreeEx(process.hProcess, remote_path, 0, MEM_RELEASE);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return fail(L"Cannot inject the selected DLL", GetLastError());
    }
    WaitForSingleObject(injection, INFINITE);
    DWORD injected_module = 0;
    GetExitCodeThread(injection, &injected_module);
    CloseHandle(injection);
    VirtualFreeEx(process.hProcess, remote_path, 0, MEM_RELEASE);
    if (injected_module == 0u) {
        TerminateProcess(process.hProcess, 2);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return fail(L"The target process rejected the selected DLL");
    }

    ResumeThread(process.hThread);
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 2;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}
