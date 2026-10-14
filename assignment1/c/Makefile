# C implementation of the basic matrix multiplication.
#
# The benchmark harness passes CC and CFLAGS from config/bench.json, which is the
# single source of truth; the defaults below are the same values for manual builds.
#   make          build build/matmul
#   make test     build and run the tests
#   make asm      emit build/matmul.s (evidence of the generated kernel code)

CC     = clang
CFLAGS = -std=c17 -O2 -ffp-contract=off -Wall -Wextra

BUILD := build

.PHONY: all test asm clean

all: $(BUILD)/matmul

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/matmul.o: src/matmul.c src/matmul.h | $(BUILD)
	$(CC) $(CFLAGS) -c src/matmul.c -o $@

$(BUILD)/matmul: src/main.c $(BUILD)/matmul.o | $(BUILD)
	$(CC) $(CFLAGS) src/main.c $(BUILD)/matmul.o -o $@ -lm

$(BUILD)/test_matmul: tests/test_matmul.c $(BUILD)/matmul.o | $(BUILD)
	$(CC) $(CFLAGS) -Isrc tests/test_matmul.c $(BUILD)/matmul.o -o $@

test: $(BUILD)/matmul $(BUILD)/test_matmul
	./$(BUILD)/test_matmul

asm: | $(BUILD)
	$(CC) $(CFLAGS) -S src/matmul.c -o $(BUILD)/matmul.s

clean:
	rm -rf $(BUILD)
