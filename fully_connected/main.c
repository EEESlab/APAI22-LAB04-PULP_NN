/*
 * main.c
 *
 * Luka Macan <luka.macan@unibo.it>
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
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
#include "fully_connected.h"

// data allocation and golden models
#include "golden.h"
#include "data_allocation.h"


void cluster_fn(void *args) {
  const int core_id = pi_core_id();

  #ifndef PER_CORE_PERF
  if (core_id == 0) {
  #endif
    // Setup and start performance counters
    pi_perf_conf(1<<PI_PERF_CYCLES | 1<<PI_PERF_INSTR);          
    pi_perf_reset();                      
    pi_perf_stop();                       
    pi_perf_start();
  #ifndef PER_CORE_PERF
  }
  #endif

  fully_connected(*(fc_args_t *)args);

  #ifndef PER_CORE_PERF
  if (core_id == 0) {
  #endif
    // Stop performance counters and print out the performance
    pi_perf_stop();          

    int perf_cyc =  pi_perf_read(PI_PERF_CYCLES);
    int perf_ins =  pi_perf_read(PI_PERF_INSTR);

    printf("[CORE %d] FullyConnected layer performance: #cycles: %d, #inst: %d\n\n",
          core_id, perf_cyc, perf_ins);
  #ifndef PER_CORE_PERF
  }
  #endif
}

void cluster_entry() {
  // Copy inputs and weights from L2 to L1
  for(int i=0; i<(CH_IM_IN); i++)
    IN_INT8_L1[i] = IN_INT8_L2[i];

  for(int i=0; i<(CH_IM_IN * CH_IM_OUT); i++)
    WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];

  // Arguments for the `cluster_fn` function
  fc_args_t fc_args = {
    .input = IN_INT8_L1,
    .weights = WEIGHT_INT8_L1,
    .output = OUT_L1,
    .channels_in = CH_IM_IN,
    .channels_out = CH_IM_OUT
  };

  printf("\n\nRunning the FullyConnected layer\n");

  // Execute function `cluster_fn` over NUM_CORES cores
  pi_cl_team_fork(NUM_CORES, (void *)cluster_fn, (void *)&fc_args);

  // Check results
  int errors = 0;
  for (int i = 0; i < CH_IM_OUT; i++)
    if(OUT_L1[i] != OUT_L2[i]) {
      printf("Error at index %d: got %d instead of %d", i, OUT_L1[i], OUT_L2[i]);
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
