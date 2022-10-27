# APPLICATION NAME
APP = fullyConnected

# SOURCE FILES
APP_SRCS = main.c fully_connected.c

# NUMBER OF USED CLUSTER CORES
CORES ?= 1

# MEASURE PERFORMANCE ON ALL CORES
# Uncomment the line below to measure performance on all cores.
#APP_CFLAGS += -DPER_CORE_PERF

# COMPILER FLAGS
APP_CFLAGS += -O3 -Iinclude -DNUM_CORES=$(CORES)
APP_LDFLAGS +=

include $(RULES_DIR)/pmsis_rules.mk
