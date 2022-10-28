#include "pmsis.h"


// generic matrix multiplication
void gemm(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){
    uint32_t core_id, i_chunk, i_start, i_end;
    uint32_t i = 0;

    core_id = pi_core_id();
    i_chunk = (NN + NUM_CORES-1) / NUM_CORES;
    i_start = core_id * i_chunk;
    i_end   = i_start + i_chunk < NN ? i_start + i_chunk : NN;

    // task to profile
    for (i = i_start; i < i_end; i ++) {
      for (int j = 0; j < MM; j++) {
        int acc = 0;
        for (int k = 0; k < KK; k++) {
          acc += MatA[i*KK+k] * MatB[k*MM+j];
        } //k
        MatC[i*MM+j] = acc;
      }//j
    }//i
    pi_cl_team_barrier();

}
/* TASK 1.3: Loop unrolling technique
 *
 * Does the scheduling of the instructions avoid the stalls?
 * If you have the following pattern:
 * 
 * lw A, off(B)
 * lw C, off(D)
 * lw E, off(F)
 * lw G, off(H)
 * mac A, C, Z
 * mac E, G, Y
 * 
 * would there be any load stall?
 * This approach is known as loop unrolling. This method is
 * one of the most powerful technique you have to optimize code.
 * Especially in regular accesses tasks such as array operations,
 * loops might be always unrolled and, in fact, the compiler tends
 * to apply the unrolling whenever he can.
 * 
 * 1.3.2: Implement loop unrolling with factor of 2, 4, 8, 16 and
 * fill the table with the overall MACs/cycle for 8 cores execution.
 * Do you see a regression doing more aggressive loop unrolling? Why?
 * The problem is that implenting unrolling, you are forcing multiple
 * loads from the memory, loading the values in the internal core
 * registers. They are not infinite, especially in embedded architecture
 * such as PULP. The available registers in the register file
 * of the core are 32, but not all are general purpose registers.
 * This fact actually limits the unrolling, because, if you saturate
 * the available registers in your architecture, you would face with
 * spilling issues, where the compiler pushes and pops over and over
 * to and from the stack (memory) the values on the registers to reuse
 * the register for another operand. This effect would distroy all the
 * performance metrics.
 * 
 * BONUS 1.4: Is there a smarter way to compute it? If you look deeper
 * into the kernel implemetation you should see that the index of matrix A
 * depends on i and k, which are the outermost and the innermost loop indexes
 * respectively, while the one of matrix B depends on the j and k. This means
 * that this operation exposes a kind of reuse of the matrix A to compute different
 * output. The reuse in general increases the performance and especially the
 * OPEF (operation efficiency). This metric tells us the ratio between the
 * "useful" operations with respect the total amount of executed instructions.
 * The higher is the OPEF the higher is the efficiency of the code.
 * Therefore, in our case the OPEF is the following:
 * 
 *                                 NB_MACS 
 *                  OPEF = -----------------------
 *                             NB_INSTRUCTIONS
 * 
 * To calculate it you just have to consider the number of MACs in the
 * innermost loop with respect to the total amount of instructions (loads
 * + macs).
 * Fill the table with the OPEF for 8 cores execution, varying the reusing
 * with 2, 4, 8 factor.
 *
 */
void gemm_unroll(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){

  // int core_id = pi_core_id();
  // int i_chunk = (NN + NUM_CORES-1) / NUM_CORES;
  // int i_start = core_id * i_chunk;
  // int i_end   = i_start + i_chunk < NN ? i_start + i_chunk : NN;

  // for (int i = i_start; i < i_end; i ++) {
  //   for (int j = 0; j < MM; j++) {
  //     int acc = 0;
  //     for (int k = 0; k < KK; /*YOUR CORE HERE*/) {
  //       // memory insns
  //       int A0  = MatA[i*KK+(k+0)];
  //       int A1  = MatA[i*KK+(k+1)];
  //       int A2  = MatA[i*KK+(k+2)];
  //       int A3  = MatA[i*KK+(k+3)];
  //       int A4  = MatA[i*KK+(k+4)];
  //       int A5  = MatA[i*KK+(k+4)];
  //       int A6  = MatA[i*KK+(k+6)];
  //       int A7  = MatA[i*KK+(k+7)];
  //       int A8  = MatA[i*KK+(k+8)];
  //       int A9  = MatA[i*KK+(k+9)];
  //       int A10 = MatA[i*KK+(k+10)];
  //       int A11 = MatA[i*KK+(k+11)];
  //       int A12 = MatA[i*KK+(k+12)];
  //       int A13 = MatA[i*KK+(k+13)];
  //       int A14 = MatA[i*KK+(k+14)];
  //       int A15 = MatA[i*KK+(k+15)];
  //       int B0  = MatB[(k+0)*MM+j];
  //       int B1  = MatB[(k+1)*MM+j];
  //       int B2  = MatB[(k+2)*MM+j];
  //       int B3  = MatB[(k+3)*MM+j];
  //       int B4  = MatB[(k+4)*MM+j];
  //       int B5  = MatB[(k+5)*MM+j];
  //       int B6  = MatB[(k+6)*MM+j];
  //       int B7  = MatB[(k+7)*MM+j];
  //       int B8  = MatB[(k+8)*MM+j];
  //       int B9  = MatB[(k+9)*MM+j];
  //       int B10 = MatB[(k+10)*MM+j];
  //       int B11 = MatB[(k+11)*MM+j];
  //       int B12 = MatB[(k+12)*MM+j];
  //       int B13 = MatB[(k+13)*MM+j];
  //       int B14 = MatB[(k+14)*MM+j];
  //       int B15 = MatB[(k+15)*MM+j];

  //       // compiler fence to explicitly separate the memory insns from alu insns
  //       asm volatile("":::"memory");

  //       // alu insns
  //       acc += A0 * B0;
  //       acc += A1 * B1;
  //       acc += A2 * B2;
  //       acc += A3 * B3;
  //       acc += A4 * B4;
  //       acc += A5 * B5;
  //       acc += A6 * B6;
  //       acc += A7 * B7;
  //       acc += A8 * B8;
  //       acc += A9 * B9;
  //       acc += A10 * B10;
  //       acc += A11 * B11;
  //       acc += A12 * B12;
  //       acc += A13 * B13;
  //       acc += A14 * B14;
  //       acc += A15 * B15;
  //     } //k
  //     MatC[i*MM+j] = acc;
  //   }//j
  // }//i

  // pi_cl_team_barrier();

}

void gemm_reuse(int * MatA, int * MatB, int* MatC, int NN, int MM, int KK){

  // int core_id = pi_core_id();
  // int i_chunk = (NN + NUM_CORES-1) / NUM_CORES;
  // int i_start = core_id * i_chunk;
  // int i_end   = i_start + i_chunk < NN ? i_start + i_chunk : NN;

  // for (int i = i_start; i < i_end; i ++) {
  //   for (int j = 0; j < MM; PUT YOUR CODE HERE) {
  //     int acc0 = 0;
  //     int acc1 = 0;
  //     int acc2 = 0;
  //     int acc3 = 0;
  //     int acc4 = 0;
  //     int acc5 = 0;
  //     int acc6 = 0;
  //     int acc7 = 0;
  //     for (int k = 0; k < KK; k++) {
  //       int shared_op = MatA[i*KK+k];
  //       acc0 += shared_op * MatB[k*MM+(j+0)];
  //       acc1 += shared_op * MatB[k*MM+(j+1)];
  //       acc2 += shared_op * MatB[k*MM+(j+2)];
  //       acc3 += shared_op * MatB[k*MM+(j+3)];
  //       acc4 += shared_op * MatB[k*MM+(j+4)];
  //       acc5 += shared_op * MatB[k*MM+(j+5)];
  //       acc6 += shared_op * MatB[k*MM+(j+6)];
  //       acc7 += shared_op * MatB[k*MM+(j+7)];
  //     } //k
  //     MatC[i*MM+(j+0)] = acc0;
  //     MatC[i*MM+(j+1)] = acc1;
  //     MatC[i*MM+(j+2)] = acc2;
  //     MatC[i*MM+(j+3)] = acc3;
  //     MatC[i*MM+(j+4)] = acc4;
  //     MatC[i*MM+(j+5)] = acc5;
  //     MatC[i*MM+(j+6)] = acc6;
  //     MatC[i*MM+(j+7)] = acc7;
  //   }//j
  // }//i

  // pi_cl_team_barrier();

}