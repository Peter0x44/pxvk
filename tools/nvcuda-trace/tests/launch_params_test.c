#include "launch_params.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct bounded_memory {
    const unsigned char *begin;
    const unsigned char *end;
} bounded_memory;

static bool read_bounded(void *context, const void *address, void *output, size_t size)
{
    const bounded_memory *memory = context;
    const unsigned char *source = address;
    if (source < memory->begin || source > memory->end ||
        size > (size_t)(memory->end - source)) {
        return false;
    }
    memcpy(output, source, size);
    return true;
}

static int failures;

#define CHECK(condition)                                                                  \
    do {                                                                                  \
        if (!(condition)) {                                                               \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
            failures++;                                                                   \
        }                                                                                 \
    } while (0)

static void test_standard_order(void)
{
    unsigned char storage[128] = {0};
    unsigned char *parameters = storage;
    size_t *size = (size_t *)(storage + 64);
    void **extra = (void **)(storage + 80);
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;
    pxvk_launch_params_status status;

    *size = 24;
    extra[0] = (void *)(uintptr_t)1;
    extra[1] = parameters;
    extra[2] = (void *)(uintptr_t)2;
    extra[3] = size;
    extra[4] = NULL;

    status = pxvk_parse_launch_extra(extra, read_bounded, &memory, &result);
    CHECK(status == PXVK_LAUNCH_PARAMS_OK);
    CHECK(result.buffer == parameters);
    CHECK(result.size == *size);
}

static void test_reversed_order(void)
{
    unsigned char storage[128] = {0};
    unsigned char *parameters = storage;
    size_t *size = (size_t *)(storage + 64);
    void **extra = (void **)(storage + 80);
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;

    *size = 17;
    extra[0] = (void *)(uintptr_t)2;
    extra[1] = size;
    extra[2] = (void *)(uintptr_t)1;
    extra[3] = parameters;
    extra[4] = NULL;

    CHECK(pxvk_parse_launch_extra(extra, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_OK);
    CHECK(result.buffer == parameters);
    CHECK(result.size == 17);
}

static void test_none(void)
{
    unsigned char storage[16] = {0};
    void **extra = (void **)storage;
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;

    CHECK(pxvk_parse_launch_extra(NULL, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_NONE);
    CHECK(pxvk_parse_launch_extra(extra, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_NONE);
}

static void test_missing_size(void)
{
    unsigned char storage[64] = {0};
    void **extra = (void **)(storage + 32);
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;

    extra[0] = (void *)(uintptr_t)1;
    extra[1] = storage;
    extra[2] = NULL;
    CHECK(pxvk_parse_launch_extra(extra, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_MALFORMED);
}

static void test_unknown_token(void)
{
    unsigned char storage[32] = {0};
    void **extra = (void **)storage;
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;

    extra[0] = (void *)(uintptr_t)99;
    extra[1] = NULL;
    CHECK(pxvk_parse_launch_extra(extra, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_MALFORMED);
}

static void test_unreadable_size(void)
{
    unsigned char storage[64] = {0};
    void **extra = (void **)storage;
    bounded_memory memory = {storage, storage + sizeof(storage)};
    pxvk_launch_params result;

    extra[0] = (void *)(uintptr_t)1;
    extra[1] = storage + 32;
    extra[2] = (void *)(uintptr_t)2;
    extra[3] = (void *)((uintptr_t)storage + sizeof(storage) + 4);
    extra[4] = NULL;
    CHECK(pxvk_parse_launch_extra(extra, read_bounded, &memory, &result) ==
          PXVK_LAUNCH_PARAMS_UNREADABLE);
}

int main(void)
{
    test_standard_order();
    test_reversed_order();
    test_none();
    test_missing_size();
    test_unknown_token();
    test_unreadable_size();
    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    puts("launch parameter parser tests passed");
    return 0;
}
