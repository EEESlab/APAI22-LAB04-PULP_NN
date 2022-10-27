APP = convolution
APP_SRCS = main.c pulp_nn_conv_u8_u8_i8.c pulp_nn_matmul_u8_i8.c

CORES ?= 1

APP_CFLAGS += -O3 -Iinclude 
APP_CFLAGS += -DNUM_CORES=$(CORES)

include $(RULES_DIR)/pmsis_rules.mk
