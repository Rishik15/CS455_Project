CC = gcc
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

train_serial: $(SERIAL_SRCS)
	$(CC) $(CFLAGS) $(SERIAL_SRCS) -o train_serial $(LDFLAGS)

.PHONY: train_serial
