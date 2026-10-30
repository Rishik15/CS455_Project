# CS455 Project

A small language model written from scratch in C. It learns to write short children's stories by reading the TinyStories dataset and predicting the next piece of text.

We train it four ways and compare the speed: on one CPU, across several machines with MPI, on one GPU with CUDA, and across several machines with GPUs (MPI + CUDA).
