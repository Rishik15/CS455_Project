# Remaining work

In order of implementation.

1. **Data prep:** implement BPE in `tools/prep_data.c` (`pretokenize.*`, `bpe_train.*`), producing `train.bin`, `val.bin`, `vocab.bin`.
2. **Tokenizer** (`src/tokenizer.c`, `include/tokenizer.h`): load `vocab.bin`, decode ids to text.
3. **Dataloader** (`src/dataloader.c`, `include/dataloader.h`): batches of `inputs` and `targets` (shifted by one) from `train.bin` / `val.bin`, with a per-rank slice for MPI.
4. **Optimizer** (`src/optimizer.c`, `include/optimizer.h`): AdamW over the flat `params_memory` and `grads_memory`.
5. **Makefile:** targets `prep`, `serial`, `test`, `clean`, later `mpi`, `cuda`, `mpi_cuda`.
6. **Tests** (`tests/test_ops.c`, `tests/test_tokenizer.c`): finite-difference gradient check per op and for the full model; `decode(encode(text)) == text`.
7. **Serial training** (`train/train_serial.c`): training loop, validation, sampling, checkpoint save.
8. **Generation** (`sample/generate.c`): load a checkpoint and generate a story.
9. **MPI training** (`train/train_mpi.c`): broadcast params, per-rank data, allreduce grads each step.
10. **CUDA ops** (`src/cuda/`): the `ops.h` functions as kernels, matching the CPU versions.
11. **CUDA training** (`train/train_cuda.cu`): all buffers on the GPU.
12. **MPI + CUDA training** (`train/train_mpi_cuda.cu`): CUDA loop plus the allreduce.
13. **Measurements:** ms per step, tokens per second, speedup vs serial, scaling efficiency, allreduce vs compute time, final val loss for every version.
