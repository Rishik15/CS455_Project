#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "config.h"
#include "model.h"
#include "optimizer.h"
#include "dataloader.h"

#define DATA_PATH "data/train.bin"

int main(){

	const int B = 2;
	const int T = 8;
	const int num_steps = 5;
	const float learning_rate = 0.001f;
	const uint64_t seed = 0xbad;
	const float weight_decay = 0.01f;
	const float epsilon = 1e-8f;
	
	//GPT	
	//	config 
	
	GPTConfig config = gpt_config_default();
	
	//	init
	GPT model = {0};
	gpt_build(&model, config, B, T, seed);
	gpt_print_summary(&model);

	
	//AdamW
	//	init
	AdamW optimizer = {0};
	adamw_init(&optimizer, model.num_parameters, weight_decay, epsilon);

	//dataloader
	const char *data_path = DATA_PATH; 

	DataLoader data_loadr = {0};
	dataloader_init(&data_loadr, data_path, B, T);

	//training loop
	for(int step = 0; step < num_steps; step++){
		
		dataloader_next(&data_loadr);
		
		float loss = gpt_forward(&model, data_loadr.inputs, data_loadr.targets, B, T);

		if(!isfinite(loss)){
			fprintf(stderr,"Infinite loss calculation at step %d", step + 1);
			exit(EXIT_FAILURE);
			
		}

		gpt_zero_grad(&model);

		gpt_backward(&model);

		adamw_update(
			&optimizer,
        		model.grads_memory,
        		model.params_memory,
             		learning_rate
		);
		

		printf("Step %d | Loss: %.6f\n", step + 1, loss);
	}

	// Free allocated resources
	dataloader_free(&data_loadr);
	adamw_free(&optimizer);
	gpt_free(&model);

	return 0;
}
