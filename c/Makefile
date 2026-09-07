# Star tracker flight-path build.
#
#   make            build both test programs on the host
#   make run        build and run the self-test
#   make pi         cross-compile for a 64-bit Raspberry Pi 5
#   make clean
#
# -ffp-contract=off matters more than it looks. By default GCC may fuse a
# multiply and an add into a single fused multiply-add, which is MORE accurate
# but gives a different result from the Python reference. Turning it off is
# what lets the cross-check use a tight tolerance instead of a loose one that
# would hide real bugs.

CC      ?= gcc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
           -Wstrict-prototypes -Wmissing-prototypes -ffp-contract=off \
           -D_POSIX_C_SOURCE=199309L
INCS    := -Iinclude -I../gen
LDLIBS  := -lm

SRC     := src/st_centroid.c src/st_camera.c src/st_feature.c \
           src/st_mlp.c src/st_select.c src/st_pipeline.c
OBJ     := $(SRC:.c=.o)

BIN     := st_crosscheck st_pipeline_test

all: $(BIN)

st_crosscheck: tests/crosscheck_main.c $(OBJ)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(LDLIBS)

st_pipeline_test: tests/pipeline_main.c $(OBJ)
	$(CC) $(CFLAGS) $(INCS) -o $@ $^ $(LDLIBS)

# -MMD -MP emits a .d file listing every header each object depends on, and the
# -include pulls those in. Without this, editing st_config.h leaves stale .o
# files whose structs are a different size than the ones the rest of the build
# expects — which shows up as a segfault, not a compile error.
%.o: %.c
	$(CC) $(CFLAGS) $(INCS) -MMD -MP -c -o $@ $<

-include $(OBJ:.o=.d)

# Raspberry Pi 5 is a Cortex-A76. -mcpu lets GCC use the right scheduling and
# NEON width; on a model this small that is worth a solid factor on the dense
# layers.
pi: CC := aarch64-linux-gnu-gcc
pi: CFLAGS := -std=c99 -O3 -mcpu=cortex-a76 -Wall -Wextra -ffp-contract=off \
              -D_POSIX_C_SOURCE=199309L
pi: clean all

run: st_pipeline_test
	./st_pipeline_test ../gen/test_frame.pgm

clean:
	rm -f $(OBJ) $(OBJ:.o=.d) $(BIN) libstartracker.so

.PHONY: all clean run pi

# Linux host bridge for rendered training, evaluation, and regression tests.
libstartracker.so: $(SRC) tests/bridge.c $(wildcard include/*.h) ../gen/model_params.h
	$(CC) $(CFLAGS) -fPIC -shared $(INCS) -o $@ $(SRC) tests/bridge.c $(LDLIBS)

bridge: libstartracker.so
.PHONY: bridge
