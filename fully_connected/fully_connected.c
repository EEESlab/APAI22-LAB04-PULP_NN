#include "fully_connected.h"
#include "pmsis.h"
#include "pulp_nn_utils.h"


/* dotp_u8_i8_i32
 *
 * Dot-product operation that does:
 *
 *        sum(a[i] * b[i])
 * 
 * Function name: dotp_<type of first vector>_<type of second vector>_<type of the result>
 * 
 * Arguments:
 *   - a: pointer to the first vector
 *   - b: pointer to the second vector
 *   - length: vector length
 *   - result: dot-product result
 */
int32_t dotp_u8_i8_i32(uint8_t *a, int8_t *b, size_t length) {
  int32_t sum = 0;
  for (int i = 0; i < length; i++)  {
    sum += a[i] * b[i];
  }
  return sum;
}


/* dotp_u8_i8_i32_simd
 *
 * SIMD implementation of the dot-product operation.
 * All the arguments are the same as in the function above.
 */
int32_t dotp_u8_i8_i32_simd(uint8_t *a, int8_t *b, size_t length) {
  v4u *vA = (v4u *)a;
  v4s *vB = (v4s *)b;

  int32_t sum = 0;
  for (int i = 0; i < length / 4; i++) 
    // The SumDotp4 considers that one word (32 bits) consists of 4 8bit operands
    sum = SumDotp4(vA[i], vB[i], sum);

  // Left over: handling the remaining output features
  size_t begin = (length / 4) * 4; // length & -4
  size_t end = begin + (length % 4);
  for (int i = begin; i < end; i++)
    sum += a[i] * b[i];

  return sum;
}


/* calculate_chunk_size
 * 
 * Calculate the chunk size from the a priori known number of cores (NUM_CORES).
 */
int calculate_chunk_size(const int total_size) {
  // Divide the workload by the number of cores
  int chunk_size = total_size / NUM_CORES;

  // If there is a remainder from the division, increment the chunk_size
  if (total_size % NUM_CORES > 0) chunk_size++;

  return chunk_size; 
}


/* fully_connected
 * 
 * Execute a FullyConnected layer with given inputs and weights.
 * The `fc_args_t` structure defined in the `include/fully_connected.h`.
 */
void fully_connected(const fc_args_t args) {
  // Parallelize over the output channel dimension
  const int chunk_size = calculate_chunk_size(args.channels_out); 

  // Begin and end of the core's chunk
  const int core_id = pi_core_id();
  const int begin = min(chunk_size * core_id, args.channels_out);
  const int end = min(begin + chunk_size, args.channels_out);

  /* Weights * Input
   *
   * The FullyConnected layer executes a Matrix-Vector multiplication
   * between the Weights matrix and the Input vector.
   * Matrix-Vector multiplication implemented as a series of dot-products.
   */
  for (int i = begin; i < end; i++) {
    // Pointer to the weight row corresponding to the processed output channel
    int8_t *weights_row = args.weights + (i * args.channels_in);
    
    // Calculate the channel output at index i by doing a dot product between the input and the weight row
    args.output[i] = dotp_u8_i8_i32(args.input, weights_row, args.channels_in);
  }
}
