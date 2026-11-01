// pxvk-launch.exe [--inject-dll <path>] [--] <32-bit application> [arguments]
//
// Starts the application suspended, loads pxvk's nvcuda.dll (or the given
// DLL) into it and resumes it. pxvk then follows the application's 32-bit
// child processes itself (loader.cpp), so store launchers work: as a Steam
// launch option, `"C:\pxvk\pxvk-launch.exe" %command%` (or `-- %command%`).
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>

#include "inject.h"

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

// The part of a command line after its first `count` arguments, verbatim.
// Games often parse GetCommandLineW() themselves (UE3's ParseParam needs
// `-windowed` followed by a space, not a quote), so their arguments must be
// passed on exactly as typed. Follows CommandLineToArgvW's splitting: the
// program name ends at its closing quote or first whitespace; later
// arguments end at whitespace outside quotes, and a quote toggles quoting
// unless an odd number of backslashes precedes it.
const wchar_t *skip_arguments(const wchar_t *line, int count)
{
    const wchar_t *p = line;
    for (int i = 0; i < count && *p; ++i) {
        if (i == 0) {
            if (*p == L'"') {
                ++p;
                while (*p && *p != L'"') ++p;
                if (*p) ++p;
            }
            while (*p && *p != L' ' && *p != L'\t') ++p;
        } else {
            bool quoted = false;
            unsigned slashes = 0;
            for (; *p && (quoted || (*p != L' ' && *p != L'\t')); ++p) {
                if (*p == L'\\') {
                    ++slashes;
                    continue;
                }
                if (*p == L'"' && slashes % 2 == 0) quoted = !quoted;
                slashes = 0;
            }
        }
        while (*p == L' ' || *p == L'\t') ++p;
    }
    return p;
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
        return fail(L"Usage: pxvk-launch.exe [--inject-dll <path>] [--] "
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
    // An optional "--" before the application, as in `pxvk-launch -- %command%`.
    if (application_argument < argument_count && lstrcmpW(arguments[application_argument], L"--") == 0)
        ++application_argument;
    if (application_argument >= argument_count) {
        LocalFree(arguments);
        return fail(L"No application given");
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
    // The application path is quoted; its arguments follow exactly as given.
    std::wstring command_line = quote(arguments[application_argument]);
    const wchar_t *rest = skip_arguments(GetCommandLineW(), application_argument + 1);
    if (*rest) command_line += L" " + std::wstring(rest);
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
    const DWORD error = pxvk::runtime::inject_library(process.hProcess, runtime_path.c_str());
    if (error != 0) {
        TerminateProcess(process.hProcess, 2);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        return error == ERROR_DLL_INIT_FAILED ? fail(L"The target process rejected the selected DLL")
                                              : fail(L"Cannot inject the selected DLL", error);
    }

    ResumeThread(process.hThread);
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 2;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess);
    return static_cast<int>(exit_code);
}
