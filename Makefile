# Janus Shadow — single-file C, vendored notorch core (Chuck + hand-SIMD matvec).
# No external deps beyond libm. Build: make   Portable scalar: make portable
CC     ?= cc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
LDLIBS := -lm
UNAME_M := $(shell uname -m)

ifneq ($(PORTABLE),1)
  ifeq ($(UNAME_M),arm64)
    CFLAGS += -march=armv8.2-a+fp16+dotprod
  else ifeq ($(UNAME_M),aarch64)
    CFLAGS += -march=armv8.2-a+fp16+dotprod
  else ifeq ($(FAST_X86),1)
    CFLAGS += -mavx2 -mfma
  endif
endif

BIN := janus-shadow
SRC := janus.shadow.c

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDLIBS) -o $(BIN)

test: $(BIN)
	./$(BIN) --test

# scalar-only reference build (no NEON/AVX)
portable:
	$(MAKE) PORTABLE=1 CFLAGS="-std=c11 -O2 -Wall -Wextra -Wpedantic -DJS_SCALAR_ONLY" clean $(BIN)

# x86 with AVX2+FMA (opt-in; arm64 gets NEON automatically)
fast-x86:
	$(MAKE) FAST_X86=1 clean $(BIN)

clean:
	rm -f $(BIN)

.PHONY: test portable fast-x86 clean
