CC = gcc
MPICC = mpicc
CFLAGS = -Wall -Wextra -O2 -Iinclude
LDFLAGS = -lm

SERIAL_SRCS = \
    train/train_serial.c \
    src/model.c \
    src/optimizer.c \
    src/utils.c \
    src/cpu/attention.c \
    src/cpu/encoder.c \
    src/cpu/gelu.c \
    src/cpu/layernorm.c \
    src/cpu/matmul.c \
    src/cpu/residual.c \
    src/cpu/crossentropy.c \
    src/dataloader.c \
    src/checkpoint.c \
    src/eval.c

MPI_SRCS = \
	train/train_mpi.c \
    src/model.c \
    src/optimizer.c \
    src/utils.c \
    src/cpu/attention.c \
    src/cpu/encoder.c \
    src/cpu/gelu.c \
    src/cpu/layernorm.c \
    src/cpu/matmul.c \
    src/cpu/residual.c \
    src/cpu/crossentropy.c \
    src/dataloader.c \
    src/checkpoint.c \
    src/eval.c



train_serial: $(SERIAL_SRCS)
	$(CC) $(CFLAGS) $(SERIAL_SRCS) -o train_serial $(LDFLAGS)

prep_data: tools/prep_data.c tools/pretokenize.c tools/bpe_train.c src/utils.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test_checkpoint: tests/test_checkpoint.c $(filter-out train/train_serial.c,$(SERIAL_SRCS))
	$(CC) $(CFLAGS) $^ -o /tmp/cs455-test-checkpoint $(LDFLAGS)
	/tmp/cs455-test-checkpoint

generate: sample/generate.c $(filter-out train/train_serial.c,$(SERIAL_SRCS))
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

train_mpi: $(MPI_SRCS)
	$(MPICC) $(CFLAGS) $(MPI_SRCS) -o train_mpi $(LDFLAGS)

.PHONY: train_serial prep_data test_checkpoint generate train_mpi
