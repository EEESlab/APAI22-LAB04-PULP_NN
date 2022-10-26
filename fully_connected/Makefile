APP = fully_connected
APP_SRCS = main.c

CORES ?= 1
SIMD ?= 0

APP_CFLAGS += -O3 -Iinclude 
APP_CFLAGS += -DNUM_CORES=$(CORES) -DUSE_SIMD=$(SIMD)

include $(RULES_DIR)/pmsis_rules.mk
