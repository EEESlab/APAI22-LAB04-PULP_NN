/*
 * pulp_nn_utils.h
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

#ifndef __PULPNN_UTILS__
#define __PULPNN_UTILS__

#include "pmsis.h"


#define bitext(x,size,off)                                   __builtin_pulp_bextract(x,size,off)
#define bitextu(x,size,off)                                  __builtin_pulp_bextractu(x,size,off)
#define bitins(dst,not_mask_imm,src,mask_imm,off)            __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define pack(x,y,z,t)                                        __builtin_pulp_pack4(x,y,z,t)
#define max4(a,b)                                            __builtin_pulp_maxu4(a,b)
#define max8(a,b)                                            __builtin_pulp_maxu8(a,b)
#define max16(a,b)                                           __builtin_pulp_maxu16(a,b)
#define avg4(a,b)                                            __builtin_pulp_avgu4(a,b)
#define avg8(a,b)                                            __builtin_pulp_avgu8(a,b)
#define avg16(a,b)                                           __builtin_pulp_avgu16(a,b)
       
#define log2(x)                                              __builtin_pulp_fl1(x)
#define min(a,b)                                             ((a)<(b)?(a):(b))
#define SumDotp4(a, b, c)                                    __builtin_pulp_sdotusp4(a, b, c)
#define SumDotp8(a, b, c)                                    __builtin_pulp_sdotusp8(a, b, c)
#define SumDotp16(a, b, c)                                   __builtin_pulp_sdotusp16(a, b, c)
#define clip4(x)                                             __builtin_pulp_clipu_r(x, 15)
#define clip2(x)                                             __builtin_pulp_clipu_r(x, 3)
#define clip8(x)                                             __builtin_pulp_clipu_r(x, 255)

#define MacLoadInit(a_update, b_update, a_reg, b_reg, ptr)   __builtin_pulp_mlinitspr_v3(a_update, b_update, a_reg, b_reg, ptr)
#define MacLoadUpdate(ptr)                                   __builtin_pulp_mlupdatespr_v3(ptr)
#define MacLoadAssign(ptr)                                   __builtin_pulp_mlassignspr_v3(ptr)
#define MacLoad4(a_update, b_update, a_reg, b_reg, ptr, sum) __builtin_pulp_mlsdotsup4_v3(a_update, b_update, a_reg, b_reg, ptr, sum)
#define MacLoad8(a_update, b_update, a_reg, b_reg, ptr, sum) __builtin_pulp_mlsdotsup8_v3(a_update, b_update, a_reg, b_reg, ptr, sum)
#define MacLoad16(a_update, b_update, a_reg, b_reg, ptr, sum)__builtin_pulp_mlsdotsup16_v3(a_update, b_update, a_reg, b_reg, ptr, sum)

#define PACK_INT8_SIZE(x)                                    (x)
#define PACK_INT4_SIZE(x)                                    ((x) >> 1)
#define PACK_INT2_SIZE(x)                                    ((x) >> 2)

#define MemoryFence()                                        asm volatile("":::"memory")



static void __attribute__((noinline)) pulp_zero_mem(uint8_t * pBuffer, unsigned int size)
{
  int lfover = size &0x3;
  for (int i=0; i<(size>>2); i++)
  {
    *((v4u *)pBuffer) = (v4u){0,0,0,0};
    MemoryFence();
    pBuffer+=4;
  }
  while(lfover)
  {
    *pBuffer++=0;
    lfover--;
  }
}

static void __attribute__((noinline)) pulp_nn_im2col_u8_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  int lfover = blockSize & 0x3;

  for (int i = 0; i<blkCnt; i++)
  {
    *((v4u*)pOutput) = *((v4u*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}

#endif