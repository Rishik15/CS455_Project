# CS455 Project

A small language model written from scratch in C. It learns to write short children's stories by reading the TinyStories dataset and predicting the next piece of text.

We train it four ways and compare the speed: on one CPU, across several machines with MPI, on one GPU with CUDA, and across several machines with GPUs (MPI + CUDA).


## Running the Serial Implementation Locally ##

download tinystories training and test data to data/
	Use this syntax: TinyStories-(train/valid).txt

compile and run the prep_data program from root 
	(info in tools/prep_data.c)

run make train_serial; ./train_serial
	this uses a preset data path that can be changed via macro in train/train_serial.c
	
