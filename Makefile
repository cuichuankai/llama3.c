NDK ?= /Users/cuick/Library/Android/sdk/ndk/29.0.13846066
HOST_TAG := $(shell if [ -d "$(NDK)/toolchains/llvm/prebuilt/darwin-arm64" ]; then echo darwin-arm64; else echo darwin-x86_64; fi)
TOOLCHAIN := $(NDK)/toolchains/llvm/prebuilt/$(HOST_TAG)/bin

SRC := run.c server_mode.c mongoose.c
SRCQ := runq.c server_mode.c mongoose.c
CFLAGS := -O3 -DNDEBUG -fPIC -fopenmp
LDFLAGS := -lm -fopenmp -static-openmp
OUT := bin/android

ARM64_CC := $(TOOLCHAIN)/aarch64-linux-android21-clang
X86_64_CC := $(TOOLCHAIN)/x86_64-linux-android21-clang
ARM64_CFLAGS := $(CFLAGS) -march=armv8.2-a+dotprod -ffast-math -fno-math-errno

.PHONY: android-all android-arm64 android-x86_64 androidq-all androidq-arm64 androidq-x86_64 macq clean

android-all: android-arm64 android-x86_64

android-arm64: $(OUT)/arm64-v8a/run
$(OUT)/arm64-v8a/run: $(SRC)
	@mkdir -p $(@D)
	$(ARM64_CC) $(ARM64_CFLAGS) -o $@ $^ $(LDFLAGS)


android-x86_64: $(OUT)/x86_64/run
$(OUT)/x86_64/run: $(SRC)
	@mkdir -p $(@D)
	$(X86_64_CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

androidq-all: androidq-arm64 androidq-x86_64

androidq-arm64: $(OUT)/arm64-v8a/runq
$(OUT)/arm64-v8a/runq: $(SRCQ)
	@mkdir -p $(@D)
	$(ARM64_CC) $(ARM64_CFLAGS) -o $@ $^ $(LDFLAGS)

androidq-x86_64: $(OUT)/x86_64/runq
$(OUT)/x86_64/runq: $(SRCQ)
	@mkdir -p $(@D)
	$(X86_64_CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf $(OUT)

macq:
	@mkdir -p bin/mac
	clang -O3 -DNDEBUG -fPIC -o bin/mac/runq runq.c server_mode.c mongoose.c -lm
