# CS455_Project

## Data

The token files the model trains on are already in the repo:

```
data/
├── train.bin    5M training tokens
├── val.bin      250K validation tokens
└── vocab.bin    the 75 characters + end-of-story token
```

`prep_data.c` makes them from the TinyStories text. You only need the text if you want to re-run it:

1. Download [data.zip](https://drive.google.com/file/d/1kJMImMGOJ1NKqi7hmqHsEl7zp-U8H6HQ/view?usp=drive_link) from Google Drive.
2. Move it into the root of this repo and unzip it:

   ```
   unzip data.zip
   ```

   This adds `data/TinyStories-train.txt` and `data/TinyStories-valid.txt`.

3. Rebuild the token files:

   ```
   make prep_data
   ./prep_data
   ```

## Roles

| Person | Serial model | Parallel version | Also |
|---|---|---|---|
| Manny | embedding | MPI | data prep, data loader |
| Rishik | matmul, attention | CUDA | GPU optimization |
| Andrew | layernorm, loss, AdamW, training loop | MPI+CUDA | Makefile, benchmarks, plots |
| Marcus | skeleton, GELU, residual | inference | gradient checks, helps wherever needed |

Everyone helps with debugging, testing, and the final report.

## Project layout

Each file lists who writes it. Some files don't exist yet.

```
prep_data.c                Manny    text → data/train.bin, val.bin, vocab.bin
Makefile                   Andrew   build targets: serial, mpi, cuda, mpi_cuda, test
src/
├── config.h               Marcus   model size + training settings (all versions)
├── model.h                Marcus   model structs and functions
├── transformer.c          Marcus   memory setup, forward pass, backward pass
├── rng.h                  Marcus   fixed-seed random numbers
├── vocab.h, vocab.c       Marcus   reads data/vocab.bin, token file formats
├── layers/
│   ├── layers.h           Marcus   inputs and outputs of every layer function
│   ├── embedding.c        Manny
│   ├── matmul.c           Rishik
│   ├── attention.c        Rishik
│   ├── layernorm.c        Andrew
│   ├── loss.c             Andrew
│   └── gelu.c             Marcus   GELU + residual
├── optimizer.c            Andrew   AdamW
├── dataloader.c           Manny    batches, split between MPI processes
├── train_serial.c         Andrew
├── train_mpi.c            Manny
├── kernels.cu             Rishik
├── train_cuda.cu          Rishik
├── train_mpi_cuda.cu      Andrew
└── generate.c             Marcus   text generation (inference)
tests/test_grads.c         Marcus   gradient checks for every layer
scripts/                   Andrew   cluster job scripts, benchmarks, plots
```

The rules every layer function follows are at the top of `src/layers/layers.h`.

## Building and running on the cluster

- **Build everything on box2.** It matches the grading machine (Fedora 37, gcc 12, CUDA 12.3). nvcc doesn't work on box1/3/4, and programs built there won't run on box2.
- CUDA code targets the M2000M: `nvcc -arch=sm_50`. Never use CUDA 13; it can't target this GPU.
- `mpirun` doesn't copy files. Put the project folder (programs + `data/*.bin`) at the same path on every node, then run from that folder:

  ```
  for h in box1 box3 box4; do
    ssh $h "mkdir -p $(pwd)" && rsync -a --exclude 'data/*.txt' ./ $h:$(pwd)/
  done
  mpirun -np 4 -hostfile ~/hosts.4 --map-by node $(pwd)/train_mpi_cuda
  ```

- MPI is **not** CUDA-aware. Copy GPU data to the CPU before any MPI call, and back after.
- In GPU code, use `float`, not `double` (much slower on this GPU), and `cudaMalloc` + `cudaMemcpy`, not `cudaMallocManaged`.
- For benchmarks, turn the desktop off on every node (`sudo systemctl isolate multi-user.target`) so the screen isn't sharing the GPU.
