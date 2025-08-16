#include "smollu_vm.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SMOLLU_NATIVE_IMPLEMENTATIONS_AVAILABLE
#include "smollu_native_tables.h"
#define SMOLLU_INTERPRETER_MAIN

/* ────────────────────────────────────────────────────────────────────────── */
/*  Sample native bindings                                                    */
/* ────────────────────────────────────────────────────────────────────────── */

/* Native #0  : print (variadic) */
Value nat_print(Value *args, uint8_t argc) {
    for (uint8_t i = 0; i < argc; ++i) {
        Value v = args[i];
        switch (v.type) {
            case VAL_NIL:   printf("nil"); break;
            case VAL_BOOL:  printf(v.as.boolean ? "true" : "false"); break;
            case VAL_INT:   printf("%d", v.as.i); break;
            case VAL_FLOAT: printf("%f", v.as.f); break;
        }
        if (i + 1 < argc) printf(" ");
    }
    printf("\n");
    return value_make_nil();
}

/* Native #1  : random int in range [0, arg0) */
Value nat_rand(Value *args, uint8_t argc) {
    (void)argc;
    if (argc == 0 || args[0].type != VAL_INT) return value_from_int(0);
    int limit = args[0].as.i;
    return value_from_int(rand() % (limit ? limit : 1));
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Public helper                                                             */
/* ────────────────────────────────────────────────────────────────────────── */

int smollu_interpreter_run(void) {

    /* Load bytecode from file */
    const char *filename = "demo.smolbc";
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "[VM] Failed to open bytecode file %s\n", filename);
        return -1;
    }

    /* Determine file size */
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "[VM] fseek failed\n");
        fclose(fp);
        return -1;
    }
    long fsize = ftell(fp);
    if (fsize <= 0) {
        fprintf(stderr, "[VM] Bytecode file is empty or ftell failed\n");
        fclose(fp);
        return -1;
    }
    rewind(fp);

    /* Read entire file into memory */
    uint8_t *buffer = (uint8_t *)malloc((size_t)fsize);
    if (!buffer) {
        fprintf(stderr, "[VM] Out of memory allocating %ld bytes\n", fsize);
        fclose(fp);
        return -1;
    }
    if (fread(buffer, 1, (size_t)fsize, fp) != (size_t)fsize) {
        fprintf(stderr, "[VM] Failed to read bytecode file\n");
        free(buffer);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    /* Sanity check header size */
    if ((size_t)fsize < 16) {
        fprintf(stderr, "[VM] Invalid bytecode file (too small)\n");
        free(buffer);
        return -1;
    }

    /* Read native function table */
    uint8_t device_id    = buffer[5];
    uint8_t native_count = buffer[7];
    const DeviceNativeTable *dtable = smollu_get_device_native_table(device_id);
    if (!dtable) {
        fprintf(stderr, "[VM] Unknown device id 0x%02X\n", device_id);
        free(buffer);
        return -1;
    }

    /* Initialize VM */
    SmolluVM vm;
    smollu_vm_init(&vm);

    /* Prepare VM with header & native function table */
    smollu_vm_prepare(&vm, buffer, dtable->table);

    /* Calculate code section offset and length */
    size_t code_offset = 16u + (size_t)native_count * 2u;
    if ((size_t)fsize < code_offset) {
        fprintf(stderr, "[VM] Invalid bytecode file (truncated native table)\n");
        free(buffer);
        return -1;
    }
    const uint8_t *code_ptr = buffer + code_offset;
    size_t code_len = (size_t)fsize - code_offset;

    /* Load code section */
    smollu_vm_load(&vm, code_ptr, code_len);

    /* Run program */
    int rc = smollu_vm_run(&vm);
    if (rc == 0) {
        printf("[VM] Program terminated normally.\n");
    } else {
        printf("[VM] Program terminated with error code %d.\n", rc);
    }

    smollu_vm_destroy(&vm);
    free(buffer);
    return rc;
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Stand-alone driver (optional)                                             */
/* ────────────────────────────────────────────────────────────────────────── */

#ifdef SMOLLU_INTERPRETER_MAIN
int main(int argc, char **argv) {
    int rc = smollu_interpreter_run();
    return rc;
}
#endif /* SMOLLU_INTERPRETER_MAIN */
