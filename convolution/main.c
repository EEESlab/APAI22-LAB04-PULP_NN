/*
 * test.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pmsis.h"
#include "pulp_nn_utils.h"
#include "pulp_nn_kernels.h"


// include data allocators and golden samples
#include "golden.h"
#include "data_allocation.h"


typedef struct {
  uint8_t *input;
  uint8_t *output;
  uint8_t *im2col;
  int8_t  *weights;
  int8_t  *bias;
} conv_args_t;

void convolution(void *args) {
  conv_args_t conv_args = *(conv_args_t *)args;

  pulp_nn_conv_u8_u8_i8(
      conv_args.input,
      conv_args.im2col,
      conv_args.bias,
      conv_args.output,
      conv_args.weights,
      OUT_SHIFT,
      DIM_IM_IN_X,
      DIM_IM_IN_Y,
      CH_IM_IN,
      DIM_IM_OUT_X,
      DIM_IM_OUT_Y,
      CH_IM_OUT,
      DIM_KERNEL_X,
      DIM_KERNEL_Y,
      PADDING_Y_TOP,
      PADDING_Y_BOTTOM,
      PADDING_X_LEFT,
      PADDING_X_RIGHT,
      STRIDE_X,
      STRIDE_Y);
}

void cluster_entry() {
  for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    IN_INT8_L1[i] = IN_INT8_L2[i];

  for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT); i++)
    WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];

  printf("\n\nRunning the Convolution layer\n");

  conv_args_t conv_args = {
    .input = IN_INT8_L1,
    .output = OUT_L1,
    .im2col = IM2COL_L1,
    .weights = WEIGHT_INT8_L1,
    .bias = BIAS_L1
  };

  // configure perf counters
  pi_perf_conf(1<<PI_PERF_CYCLES);
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start(); 

  // call the convolution kernel
  pi_cl_team_fork(NUM_CORES, (void *)convolution, (void *)&conv_args);

  // measure performance
  pi_perf_stop();          
  int cid = pi_core_id();   
  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);

  printf("Convolution layer completed, running on %d cores\n", NUM_CORES);
  printf(" - num_cycles: %d\n", perf_cyc); 

  int errors = 0;
  for (int i = 0; i < (DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    if(OUT_L1[i] != OUT_INT8_L2[i])
    {
      printf("Error at index %d: got %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
      errors++;
    }

  if (errors == 0)
    printf("Convolution layer executed without errors.\n");
  else
    printf("ERROR: Convolution layer executed with %d errors.\n", errors);
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main() {
  struct pi_device cl_dev;
  struct pi_cluster_conf cl_conf;

  // First open the cluster
  pi_cluster_conf_init(&cl_conf);
  pi_open_from_conf(&cl_dev, &cl_conf);
  if (pi_cluster_open(&cl_dev))
    return -1;

  // Then offload an entry point, this will get executed on the cluster controller
  struct pi_cluster_task cl_task;
  pi_cluster_send_task_to_cl(&cl_dev, pi_cluster_task(&cl_task, cluster_entry, NULL));

  // closing of the cluster
  pi_cluster_close(&cl_dev);

  return 0;
}
