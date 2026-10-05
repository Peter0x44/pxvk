#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

typedef int (*capture_sync_fn)(void);
typedef int (*capture_dtoh_fn)(void *destination, uint32_t source, uint32_t bytes);

void capture_initialize(const wchar_t *directory, capture_sync_fn sync, capture_dtoh_fn dtoh);
void capture_note_module(void *module, unsigned int fatbin_ordinal);
void capture_note_function(void *function, void *module, const char *name);
void capture_note_alloc(uint32_t base, uint32_t size);
void capture_note_free(uint32_t base);
void capture_note_global(void *module, const char *name, uint32_t base, uint32_t size);
void capture_note_texref(void *texref, void *module, const char *name);
void capture_note_texref_address(void *texref, uint32_t base, uint32_t bytes);
void capture_note_texref_format(void *texref, int format, int components);
/* Returns a nonzero token when this launch is captured; pass it to _end. */
unsigned int capture_launch_begin(void *function, const unsigned int grid[3],
                                  const unsigned int block[3], unsigned int shared_bytes,
                                  const void *params, size_t param_bytes);
void capture_launch_end(unsigned int token);
