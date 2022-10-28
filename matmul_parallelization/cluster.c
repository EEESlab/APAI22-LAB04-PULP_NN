#include "pmsis.h"

/* TASK 1.2: Parallelizazion technique with 8 cores
 *
 * Change the size of the rows of matrix A and see what happens.
 * To parallelize the task, you need to create small chunks to
 * assign to each core. The matrix multiplication kernel splits the
 * task along the matrix A rows accordingly with the number of cores
 * involved.
 * 
 * 1.2.1: Fill the table with the MACs/cycle with 8 cores execution varying
 * the the row size with 4 and 8. Is the parallelization good enough?
 * Amdhal's law can help you answer.
 * What about if you set 80 and 81? What do you expect? Fill the table.
 * 
 * 1.2.2: The IPC (instruction per cycle) is a good metric to see the
 * efficiency of your code, and the ideal value is 1.
 * Calculate the IPC as:
 * 
 *                                 N_EXECUTED_INSNS
 *                       IPC = -----------------------
 *                                 N_TOTAL_CYCLES
 * 
 * What is yours? Fill the table with the performance from each core
 * to see what is happening.
 * You have the pool of performance counters in each core, and they are
 * already used in this code, but only the core 0 is printing them on the terminal.
 * to analyze the performance.
 *
 */
#define N 32
#define M 16
#define K 16

// AxB = C -> [NxK] x [KxM] = [NxM]
PI_L1 int A[N*K];
PI_L1 int B[K*M];
PI_L1 int C[N*M];

void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK);
void gemm_unroll(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK);

void fill_matrix(int * Mat, int height, int width, int val){
    for (int i=0; i<height*width; i++)
    {
      Mat[i] = val;
    }
}

void checksum(int * MatA, int a_val, int b_val, int NN, int MM, int KK){
  printf("\n\n**************************************************************\n\n");
  for (int i=0; i<NN*MM; i++){
    if (MatA[i]!=a_val*b_val*KK){
        printf("ERROR - CHECKSUM WRONG! \n");
        break;
    }
    if (i==NN*MM-1){
      printf("CHECKSUM CORRECT!\n");
    }
  }
  printf("\n**************************************************************\n\n");
}


void cluster_fn() {

  // INIT MATRICES, e.g. with the same value each cell
  int mat_a_val = 2;
  int mat_b_val = 1;

  fill_matrix(A, N, K, mat_a_val);
  fill_matrix(B, K, M, mat_b_val);
  fill_matrix(C, N, M, 0);

  // define other variables
  uint32_t cycles_cnt,instr_cnt,ld_stall_cnt;

  /* TASK 1.3: Performance analysis
   *
   * Set again the row size equal to 32.
   * The IPC is not higher enough! Your code does not fit perfectly
   * the underlying architecture. What is happening?
   * The pattern:
   * 
   * lw A, off(B)
   * lw C, off(D)
   * mac A, C, Z   <- load stall
   * 
   * generates a load stall each iteration. This is an intrinsic effect
   * of the core pipeline and memory control unit, and you must know it
   * to produce optimal code. In fact, in this case, the load has
   * 1 cycle of latency. This means that you can not use the data loaded
   * from memory (lw instruction) the next clock cycle. The core,
   * before computing the mac instruction, needs to stall the pipeline
   * waiting for the valid signal from the memory, telling it that now
   * operand C is ready. The effect is the incrementing of the
   * cycles counter but not of the instructions counter, degrading
   * the overall IPC.
   * The PULP core has a dedicated performance counter to count the number
   * of load stalls. Activate it on each core to evaluate the effect
   * of the pipeline on your code.
   * 
   * 1.3.1: Fill the table with the total number of load stalls
   * with 8 cores.
   *
   */
  pi_perf_conf(
      1 << PI_PERF_CYCLES
      // | 1 << PI_PERF_INSTR
      // PUT YOUR CODE HERE (add load stall)
  );

  //pi_perf_stop();
  //pi_perf_reset();
  //pi_perf_start(); // start the performance counters

  /* TASK 1.1: Speed-up and Amdhal's law
   *
   * Profile the execution of the matrix multiplication kernel. 
   * The speed-up represents the gain you are getting by splitting
   * the workload among multiple processing units with respect
   * to execute it sequentially.
   * The maximum achievable speed-up can be calculated with
   * the Amdhal's law and, at the first order, it is equal
   * to the available number of cores.
   * 
   * 1.1.1: Fill the table with the total execution cycles varying the
   * number of cores with 2, 4, and 8 and compute the speed-up
   * for each iteration. Is it equal to the maximum achievable?
   * Why? The Amdhal's law tells you that:
   * 
   *                  1
   * SU(N) = -------------------, p <= 1, N = available cores
   *                       p
   *            (1 - p) + ---
   *                       N
   * where
   *  - p is the part of the code that ca be parallelized
   *  - 1-p is the part of the code that can not be parallelized
   *  - N is the number of cores
   *  - S is the speedup
   * 
   * Take away message: You can speed up only the parallel code!!
   * 
   */
  // task to profile
  gemm(A, B, C, N, M, K); 
  //gemm_unroll(A, B, C, N, M, K);
  //gemm_reuse(A, B, C, N, M, K);

  //pi_perf_stop(); // stop the performance counters

  // core 1, ... , NUM_CORES do not execute the if-statement body..
  // comment it to see the results from all the cores
  if (pi_core_id()==0) {

    cycles_cnt = pi_perf_read(PI_PERF_CYCLES);
    //instr_cnt = pi_perf_read(PI_PERF_INSTR);
	  // float ipc = (float) instr_cnt / (float) cycles_cnt;
    // /*YOUR CODE HERE*/ // (read the ld stalls counter)

    printf("coreID [%d]: Clock Cycles: %d | %d cores execution\n",
        pi_core_id(), cycles_cnt, NUM_CORES);
    printf("coreID [%d]: Number of Instructions: %-6d\tClock Cycles: %d\tIPC= %.2f | %d cores execution\n",
        pi_core_id(), instr_cnt, cycles_cnt, ipc, NUM_CORES);
    // (print the ld stalls counter)
    // printf("coreID [%d]: Stalls: %d, instructions %d, cycles %d, IPC %.2f| %d cores execution\n",
    //     pi_core_id(), /*YOUR CODE HERE*/, instr_cnt, cycles_cnt, ipc, NUM_CORES);    
  }


  pi_cl_team_barrier();

  /* RESULTS CHECKSUM */
  if (pi_core_id()==0)
    checksum(C, mat_a_val, mat_b_val, N, M, K);

}
