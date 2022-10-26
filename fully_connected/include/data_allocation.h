/*
 * data_allocation_8_32_8.h
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

#ifndef __PULPNN_TEST_DATA_ALLOCATION__
#define __PULPNN_TEST_DATA_ALLOCATION__

#define CH_IM_IN 1024
#define CH_IM_OUT 16
#define DIM_KERNEL_X 3
#define DIM_KERNEL_Y 3
#define PADDING_Y_TOP 1
#define PADDING_Y_BOTTOM 1
#define PADDING_X_LEFT 1
#define PADDING_X_RIGHT 1
#define STRIDE_X 1
#define STRIDE_Y 1



PI_L2 uint8_t IN_INT8_L2[CH_IM_IN] = IN_INT8;
PI_L1 uint8_t IN_INT8_L1[CH_IM_IN];
PI_L2 int8_t WEIGHT_INT8_L2[(CH_IM_IN * CH_IM_OUT)] = WEIGHT_INT8;
PI_L1 int8_t WEIGHT_INT8_L1[(CH_IM_IN * CH_IM_OUT)];
PI_L2 int32_t OUT_L2[CH_IM_OUT] = OUT_INT32;
PI_L1 int32_t OUT_L1[CH_IM_OUT];
PI_L1 int8_t BIAS_L1[CH_IM_OUT] = {0};


#endif
