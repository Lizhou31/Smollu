/**
 * @file smollu_native_tables.h
 * @brief Device-specific native function registries for the Smollu VM.
 *
 *  This header collects, for every supported `device_id`, the ordered array of
 *  native host functions (`NativeFn`) together with the parallel list of their
 *  string names.  The compiler uses the names to translate `native foo(...)`
 *  into the correct index, while the VM uses the table to initialise its
 *  dispatch vector at start-up.
 *
 *  To add a new device:
 *      1. Implement the native C functions matching `Value (*)(Value *,uint8)`.
 *      2. Create an array `<device>_table[]` with the desired order.
 *      3. Create a matching `<device>_names[]` array of `const char *`.
 *      4. Append a new entry to `smollu_device_native_tables` below.
 */

#ifndef SMOLLU_NATIVE_TABLES_H
#define SMOLLU_NATIVE_TABLES_H

#include "../vm/smollu_vm.h" /* Value, NativeFn */

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Demo device (device_id = 0x00)                                             */
/* ──────────────────────────────────────────────────────────────────────────── */

/* For compiler use - only names matter, not function pointers */
static const char *const demo_device_names[] = {
    "print",
    "rand"
};
#define DEMO_DEVICE_NATIVE_COUNT 2
#define DEMO_DEVICE_ID           0x00

/* ──────────────────────────────────────────────────────────────────────────── */
/*  RS Emulator device (device_id = 0x01)                                        */
/* ──────────────────────────────────────────────────────────────────────────── */


/* For compiler use - only names matter, not function pointers */
static const char *const rs_emulator_names[] = {
    "print",
    "led_matrix_init",
    "led_set",
    "led_set_color"
};
#define RS_EMULATOR_NATIVE_COUNT 4
#define RS_EMULATOR_ID           0x01

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Registry abstraction                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint8_t              device_id;
    uint8_t              native_count;
    const char *const   *names;  /* length native_count */
    const NativeFn      *table;  /* length native_count */
} DeviceNativeTable;

/* Compiler-only registry - no function pointers needed */
static const DeviceNativeTable smollu_device_native_tables[] = {
    { DEMO_DEVICE_ID, DEMO_DEVICE_NATIVE_COUNT, demo_device_names, NULL },
    { RS_EMULATOR_ID, RS_EMULATOR_NATIVE_COUNT, rs_emulator_names, NULL },
    /* Extend here for new devices */
};
static const size_t smollu_device_native_tables_len = sizeof(smollu_device_native_tables)/sizeof(smollu_device_native_tables[0]);

static inline const DeviceNativeTable *smollu_get_device_native_table(uint8_t device_id)
{
    for (size_t i = 0; i < smollu_device_native_tables_len; ++i) {
        if (smollu_device_native_tables[i].device_id == device_id) {
            return &smollu_device_native_tables[i];
        }
    }
    return NULL; /* unknown device */
}

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SMOLLU_NATIVE_TABLES_H */
