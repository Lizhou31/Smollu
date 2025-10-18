/**
 * @file test_streaming.c
 * @brief Test program for VM streaming mode
 */

#define CONFIG_SMOLLU_STREAMING 1

#include "smollu_vm.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ────────────────────────────────────────────────────────────────────────── */
/*  Simulated streaming source (file-backed)                                  */
/* ────────────────────────────────────────────────────────────────────────── */

typedef struct {
    FILE *fp;
    size_t base_offset;  /* Base offset in file */
} file_stream_ctx_t;

/**
 * @brief Read region callback for file-based streaming
 */
static int file_read_region(void *ctx, uint32_t offset, uint8_t *dst, size_t len) {
    file_stream_ctx_t *fctx = (file_stream_ctx_t *)ctx;

    /* Seek to offset */
    if (fseek(fctx->fp, fctx->base_offset + offset, SEEK_SET) != 0) {
        fprintf(stderr, "fseek failed for offset %u\n", offset);
        return -1;
    }

    /* Read data */
    size_t bytes_read = fread(dst, 1, len, fctx->fp);
    if (bytes_read != len) {
        fprintf(stderr, "Read failed: expected %zu bytes, got %zu\n", len, bytes_read);
        return -1;
    }

    return 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Sample native functions                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

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

Value nat_rand(Value *args, uint8_t argc) {
    (void)argc;
    if (argc == 0 || args[0].type != VAL_INT) return value_from_int(0);
    int limit = args[0].as.i;
    return value_from_int(rand() % (limit ? limit : 1));
}

/* Native function table */
static const NativeFn native_table[] = {
    nat_print,  /* 0 */
    nat_rand,   /* 1 */
};

/* ────────────────────────────────────────────────────────────────────────── */
/*  Main test                                                                 */
/* ────────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *filename = "demo.smolbc";

    if (argc > 1) {
        filename = argv[1];
    }

    printf("=== Smollu VM Streaming Mode Test ===\n");
    printf("Testing with bytecode file: %s\n\n", filename);

    /* Open bytecode file */
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open bytecode file: %s\n", filename);
        return -1;
    }

    /* Setup streaming context */
    file_stream_ctx_t stream_ctx = {
        .fp = fp,
        .base_offset = 0
    };

    /* Initialize VM */
    SmolluVM vm;
    smollu_vm_init(&vm);

    /* Prepare VM in streaming mode */
    printf("Initializing VM in streaming mode...\n");
    int ret = smollu_vm_prepare_streaming(&vm, file_read_region, &stream_ctx, native_table);
    if (ret != 0) {
        fprintf(stderr, "Failed to prepare streaming VM: %d\n", ret);
        fclose(fp);
        return -1;
    }

    printf("VM initialized successfully\n");
    printf("  Code size: %u bytes\n", vm.code_size);
    printf("  Code offset: %u\n", vm.code_offset);
    printf("  Native count: %u\n", vm.native_count);
    printf("  Cache size: %d bytes\n\n", SMOLLU_STREAM_CACHE_SIZE);

    /* Run program */
    printf("Running program...\n");
    printf("----------------------------------------\n");
    ret = smollu_vm_run(&vm);
    printf("----------------------------------------\n");

    if (ret == 0) {
        printf("\n[SUCCESS] Program terminated normally\n");
    } else {
        printf("\n[ERROR] Program terminated with error code %d\n", ret);
    }

    /* Cleanup */
    smollu_vm_destroy(&vm);
    fclose(fp);

    return ret;
}
