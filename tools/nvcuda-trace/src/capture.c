/* Kernel launch capture for differential testing.
 *
 * With PXVK_CAPTURE_LAUNCHES="a-b,c,..." (1-based launch ordinals) each
 * selected cuLaunchKernel is bracketed by full snapshots of every live device
 * allocation and module global. A capture is three files:
 *
 *   capture_NNNNNN.txt     metadata (kernel, fatbin, launch shape, params,
 *                          module globals and texture bindings)
 *   capture_NNNNNN.before  sparse memory image before the launch
 *   capture_NNNNNN.after   sparse memory image after the launch
 *
 * Sparse images are a sequence of {u32 address, u32 bytes, data} records
 * covering only 4 KiB pages that contain a nonzero byte.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "capture.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAPTURE_MAX_RANGES 64
#define CAPTURE_PAGE 4096u

typedef struct record {
    struct record *next;
    uintptr_t key;    /* handle or device address */
    uintptr_t owner;  /* module handle */
    uint32_t base, size;
    int format, components;
    unsigned int ordinal;
    char name[512];
} record;

static CRITICAL_SECTION g_capture_lock;
static bool g_capture_enabled;
static unsigned int g_ranges[CAPTURE_MAX_RANGES][2];
static unsigned int g_range_count;
static unsigned int g_frames[CAPTURE_MAX_RANGES];
static unsigned int g_frame_count;
static volatile LONG g_launch_ordinal;
static record *g_modules, *g_functions, *g_allocations, *g_globals, *g_texrefs;
static wchar_t g_directory[1024];
static capture_sync_fn g_sync;
static capture_dtoh_fn g_dtoh;

static record *find(record *list, uintptr_t key)
{
    for (; list != NULL; list = list->next)
        if (list->key == key) return list;
    return NULL;
}

static record *upsert(record **list, uintptr_t key)
{
    record *r = find(*list, key);
    if (r != NULL) return r;
    r = calloc(1, sizeof(*r));
    if (r == NULL) return NULL;
    r->key = key;
    r->next = *list;
    *list = r;
    return r;
}

void capture_initialize(const wchar_t *directory, capture_sync_fn sync, capture_dtoh_fn dtoh)
{
    char spec[1024];
    DWORD length;
    InitializeCriticalSection(&g_capture_lock);
    wcsncpy(g_directory, directory, 1023);
    g_sync = sync;
    g_dtoh = dtoh;
    length = GetEnvironmentVariableA("PXVK_CAPTURE_FRAMES", spec, sizeof(spec));
    if (length != 0 && length < sizeof(spec)) {
        for (char *cursor = spec; *cursor != '\0' && g_frame_count < CAPTURE_MAX_RANGES;) {
            char *end = NULL;
            g_frames[g_frame_count++] = (unsigned int)strtoul(cursor, &end, 10);
            if (end == cursor) break;
            cursor = *end == ',' ? end + 1 : end;
        }
        g_capture_enabled = g_frame_count != 0;
    }
    length = GetEnvironmentVariableA("PXVK_CAPTURE_LAUNCHES", spec, sizeof(spec));
    if (length == 0 || length >= sizeof(spec)) return;
    for (char *cursor = spec; *cursor != '\0' && g_range_count < CAPTURE_MAX_RANGES;) {
        char *end = NULL;
        unsigned long first = strtoul(cursor, &end, 10);
        unsigned long last = first;
        if (end == cursor) break;
        if (*end == '-') {
            cursor = end + 1;
            last = strtoul(cursor, &end, 10);
        }
        g_ranges[g_range_count][0] = (unsigned int)first;
        g_ranges[g_range_count][1] = (unsigned int)last;
        ++g_range_count;
        cursor = *end == ',' ? end + 1 : end;
    }
    g_capture_enabled = g_capture_enabled || g_range_count != 0;
}

static unsigned int current_frame(void)
{
    char text[32];
    const DWORD length = GetEnvironmentVariableA("PXVK_FRAME", text, sizeof(text));
    return length == 0 || length >= sizeof(text) ? 0u : (unsigned int)strtoul(text, NULL, 10);
}

void capture_note_module(void *module, unsigned int fatbin_ordinal)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_modules, (uintptr_t)module);
    if (r != NULL) r->ordinal = fatbin_ordinal;
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_function(void *function, void *module, const char *name)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_functions, (uintptr_t)function);
    if (r != NULL) {
        r->owner = (uintptr_t)module;
        strncpy(r->name, name, sizeof(r->name) - 1);
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_alloc(uint32_t base, uint32_t size)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_allocations, base);
    if (r != NULL) {
        r->base = base;
        r->size = size;
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_free(uint32_t base)
{
    EnterCriticalSection(&g_capture_lock);
    for (record **link = &g_allocations; *link != NULL; link = &(*link)->next) {
        if ((*link)->key == base) {
            record *dead = *link;
            *link = dead->next;
            free(dead);
            break;
        }
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_global(void *module, const char *name, uint32_t base, uint32_t size)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_globals, base);
    if (r != NULL) {
        r->owner = (uintptr_t)module;
        r->base = base;
        r->size = size;
        strncpy(r->name, name, sizeof(r->name) - 1);
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_texref(void *texref, void *module, const char *name)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_texrefs, (uintptr_t)texref);
    if (r != NULL) {
        r->owner = (uintptr_t)module;
        strncpy(r->name, name, sizeof(r->name) - 1);
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_texref_address(void *texref, uint32_t base, uint32_t bytes)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_texrefs, (uintptr_t)texref);
    if (r != NULL) {
        r->base = base;
        r->size = bytes;
    }
    LeaveCriticalSection(&g_capture_lock);
}

void capture_note_texref_format(void *texref, int format, int components)
{
    record *r;
    EnterCriticalSection(&g_capture_lock);
    r = upsert(&g_texrefs, (uintptr_t)texref);
    if (r != NULL) {
        r->format = format;
        r->components = components;
    }
    LeaveCriticalSection(&g_capture_lock);
}

static FILE *open_file(unsigned int ordinal, const wchar_t *suffix)
{
    wchar_t path[1200];
    swprintf(path, 1200, L"%ls\\capture_%06u.%ls", g_directory, ordinal, suffix);
    return _wfopen(path, L"wb");
}

static void write_region(FILE *file, uint32_t base, uint32_t size, unsigned char *buffer)
{
    if (g_dtoh(buffer, base, size) != 0) return;
    /* Emit runs of nonzero pages. */
    for (uint32_t page = 0; page < size;) {
        uint32_t length = size - page < CAPTURE_PAGE ? size - page : CAPTURE_PAGE;
        bool nonzero = false;
        for (uint32_t i = 0; i < length && !nonzero; ++i) nonzero = buffer[page + i] != 0;
        if (!nonzero) {
            page += length;
            continue;
        }
        uint32_t run_end = page + length;
        while (run_end < size) {
            uint32_t next = size - run_end < CAPTURE_PAGE ? size - run_end : CAPTURE_PAGE;
            bool any = false;
            for (uint32_t i = 0; i < next && !any; ++i) any = buffer[run_end + i] != 0;
            if (!any) break;
            run_end += next;
        }
        {
            uint32_t header[2] = {base + page, run_end - page};
            fwrite(header, sizeof(header), 1, file);
            fwrite(buffer + page, 1, run_end - page, file);
        }
        page = run_end;
    }
}

static void snapshot(unsigned int ordinal, const wchar_t *suffix)
{
    FILE *file = open_file(ordinal, suffix);
    uint32_t largest = 0;
    unsigned char *buffer;
    if (file == NULL) return;
    for (record *r = g_allocations; r != NULL; r = r->next)
        if (r->size > largest) largest = r->size;
    for (record *r = g_globals; r != NULL; r = r->next)
        if (r->size > largest) largest = r->size;
    buffer = malloc(largest ? largest : 1);
    if (buffer != NULL) {
        for (record *r = g_allocations; r != NULL; r = r->next) write_region(file, r->base, r->size, buffer);
        for (record *r = g_globals; r != NULL; r = r->next) write_region(file, r->base, r->size, buffer);
        free(buffer);
    }
    fclose(file);
}

static bool selected(unsigned int ordinal)
{
    if (g_frame_count != 0) {
        const unsigned int frame = current_frame();
        for (unsigned int i = 0; i < g_frame_count; ++i)
            if (g_frames[i] == frame) return true;
    }
    for (unsigned int i = 0; i < g_range_count; ++i)
        if (ordinal >= g_ranges[i][0] && ordinal <= g_ranges[i][1]) return true;
    return false;
}

unsigned int capture_launch_begin(void *function, const unsigned int grid[3], const unsigned int block[3],
                                  unsigned int shared_bytes, const void *params, size_t param_bytes)
{
    const unsigned int ordinal = (unsigned int)InterlockedIncrement(&g_launch_ordinal);
    FILE *meta;
    record *fn, *module;
    if (!g_capture_enabled || !selected(ordinal)) return 0;
    EnterCriticalSection(&g_capture_lock);
    g_sync();
    fn = find(g_functions, (uintptr_t)function);
    module = fn != NULL ? find(g_modules, fn->owner) : NULL;
    meta = open_file(ordinal, L"txt");
    if (meta != NULL) {
        fprintf(meta, "kernel %s\n", fn != NULL ? fn->name : "?");
        fprintf(meta, "frame %u\nlaunch %u\n", current_frame(), ordinal);
        fprintf(meta, "fatbin %u\n", module != NULL ? module->ordinal : 0u);
        fprintf(meta, "grid %u %u %u\nblock %u %u %u\nshared %u\n", grid[0], grid[1], grid[2],
                block[0], block[1], block[2], shared_bytes);
        fprintf(meta, "params ");
        for (size_t i = 0; i < param_bytes; ++i) fprintf(meta, "%02x", ((const unsigned char *)params)[i]);
        fprintf(meta, "\n");
        for (record *g = g_globals; g != NULL; g = g->next)
            if (fn != NULL && g->owner == fn->owner)
                fprintf(meta, "global %s %08x %u\n", g->name, g->base, g->size);
        for (record *t = g_texrefs; t != NULL; t = t->next)
            if (fn != NULL && t->owner == fn->owner)
                fprintf(meta, "texref %s %08x %u %d %d\n", t->name, t->base, t->size, t->format, t->components);
        for (record *a = g_allocations; a != NULL; a = a->next)
            fprintf(meta, "allocation %08x %u\n", a->base, a->size);
        fclose(meta);
    }
    snapshot(ordinal, L"before");
    return ordinal;
}

void capture_launch_end(unsigned int token)
{
    if (token == 0) return;
    g_sync();
    snapshot(token, L"after");
    LeaveCriticalSection(&g_capture_lock);
}
