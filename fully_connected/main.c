/*
 * test.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 * Manuele Rusci <manuele.rusci@unibo.it>
 *
 * Copyright (C) 2019-2021 University of Bologna
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

// data allocation and golden models
#include "golden.h"
#include "data_allocation.h"

typedef struct {
  uint8_t *input;
  int8_t  *weights;
  int32_t *output;
  int      channels_in;
  int      channels_out;
} fc_args_t;

int32_t dotp_u8_i8_i32(uint8_t *a, int8_t *b, size_t length) {
  int32_t sum = 0;
  for (int i = 0; i < length; i++) 
    sum += a[i] * b[i];
  return sum;
}

int32_t dotp_u8_i8_i32_simd(uint8_t *a, int8_t *b, size_t length) {
  v4u *vA = (v4u *)a;
  v4s *vB = (v4s *)b;

  int32_t sum = 0;
  for (int i = 0; i < length / 4; i++) 
    sum = SumDotp4(vA[i], vB[i], sum);

  // Remainder
  size_t begin = (length / 4) * 4; // length & -4
  size_t end = begin + (length % 4);
  for (int i = begin; i < end; i++)
    sum += a[i] * b[i];

  return sum;
}

int calculate_per_core_size(int size) {
  const int log2core = log2(NUM_CORES);
  return size >> log2core + ((size & ((1 << log2core)-1)) != 0); 
}

void fully_connected(void *args) {
  const fc_args_t fc_args = *(fc_args_t *)args;

  const int core_id = pi_core_id();
  const int log2core = log2(NUM_CORES);

  // parallelize over output feature dimension
  const int size = calculate_per_core_size(fc_args.channels_out); 
  const int begin = min(size * core_id, fc_args.channels_out);
  const int end = min(begin + size, fc_args.channels_out);

  for (int i = begin; i < end; i++) {
    int8_t *weights_row = fc_args.weights + (i * fc_args.channels_in);
    #if USE_SIMD == 0
    fc_args.output[i] = dotp_u8_i8_i32(fc_args.input, weights_row, fc_args.channels_in);
    #else
    fc_args.output[i] = dotp_u8_i8_i32_simd(fc_args.input, weights_row, fc_args.channels_in);
    #endif
  }
}

void cluster_entry()
{
  // copy inputs and weights from L2 to L1
  for(int i=0; i<(CH_IM_IN); i++)
    IN_INT8_L1[i] = IN_INT8_L2[i];

  for(int i=0; i<(CH_IM_IN * CH_IM_OUT); i++)
    WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];

  fc_args_t fc_args = {
    .input = IN_INT8_L1,
    .weights = WEIGHT_INT8_L1,
    .output = OUT_L1,
    .channels_in = CH_IM_IN,
    .channels_out = CH_IM_OUT
  };

  printf("\n\nRunning the FullyConnected layer (%dx%d)!\n", fc_args.channels_in, fc_args.channels_out);

  // setup and start performance counters
  pi_perf_conf(1<<PI_PERF_CYCLES | 1<<PI_PERF_INSTR);          
  pi_perf_reset();                      
  pi_perf_stop();                       
  pi_perf_start(); 

  // call the fully connected
  pi_cl_team_fork(NUM_CORES, (void *)fully_connected, (void *)&fc_args);

  // compute print performance
  pi_perf_stop();          

  int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
  int perf_ins =  pi_perf_read(PI_PERF_INSTR);
  int MACs = CH_IM_IN * CH_IM_OUT;
  float perf_MAC =  (float)MACs/perf_cyc;

  printf("Fully-connected layer completed!\nRuntime statistics on %d cores:\n", NUM_CORES);
  printf("  - num_cycles: %d\n", perf_cyc); 
  printf("  - num_inst: %d\n", perf_ins);
  printf("  - MACs: %d\n", MACs); 
  printf("  - MAC/cycle: %f\n", perf_MAC); 

  //check results
  int errors = 0;
  for (int i = 0; i < CH_IM_OUT; i++)
    if(OUT_L1[i] != OUT_L2[i]) {
      printf("Erraneous result found at index %d: calculated %d vs. golden %d\n", i, OUT_L1[i], OUT_L2[i]);
      errors++;
    }

  if (errors == 0)
    printf("FullyConnected layer executed without errors.\n");
  else
    printf("ERROR: FullyConnected layer executed with %d errors.\n", errors);
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main()
{
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
