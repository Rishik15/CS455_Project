# Minimal Makefile for the skeleton. Andrew will add the real targets
# (serial, mpi, cuda, mpi_cuda, test) as those files get written.

CC     = cc
CFLAGS = -O2 -Wall -std=gnu17 -Isrc -MMD -MP
LDLIBS = -lm

MODEL_SRC = src/transformer.c src/vocab.c $(wildcard src/layers/*.c)
MODEL_OBJ = $(patsubst src/%.c,build/%.o,$(MODEL_SRC))

# There is no main program yet, so this just checks that the model code compiles.
check: $(MODEL_OBJ)

build/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Makes data/train.bin, val.bin, vocab.bin from the TinyStories text.
prep_data: prep_data.c src/config.h src/vocab.h
	$(CC) -O2 -Wall -std=gnu17 -o $@ prep_data.c

clean:
	rm -rf build prep_data

-include $(MODEL_OBJ:.o=.d)

.PHONY: check clean
