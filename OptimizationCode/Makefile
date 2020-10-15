# MakeFile
# This file includes automatic dependency tracking
# Please see example of "utils" to see how to add additional file
# to your project

all: ASFContinuousOptimizer ASFDiscreteOptimizer filter_optimizer_int filter_optimizer_bloom full_workflow_filter_optimizer full_workflow_bloom

# C-compiler settings
CC = g++ -std=c++17 -g -ggdb3

# Default optimization level
O ?= 3

# Flags and other libraries
override CFLAGS += -Wall -Wextra -pedantic -pthread -O$(O) -I$(INCLUDES)
LDFLAGS =
LIBS = -lnlopt -lm
INCLUDES = include

####### Automatic dependency magic #######
# Set-up dependency directory
DEPSDIR := .deps
BUILDSTAMP := $(DEPSDIR)/rebuildstamp
DEPFILES := $(wildcard $(DEPSDIR)/*.d)
ifneq ($(DEPFILES),)
include $(DEPFILES)
endif
DEPCFLAGS = -MD -MF $(DEPSDIR)/$*.d -MP

# Dependency compilation
ifneq ($(DEP_CC),$(CC) $(CFLAGS) $(DEPCFLAGS) $(O))
DEP_CC := $(shell mkdir -p $(DEPSDIR); echo >$(BUILDSTAMP); echo "DEP_CC:=$(CC) $(CFLAGS) $(DEPCFLAGS) $(O)" >$(DEPSDIR)/_cc.d)
endif

# Make sure dependency directories are generated
$(DEPSDIR)/stamp $(BUILDSTAMP):
	mkdir -p $(@D)
	touch $@

####### Automatic dependency magic #######

%.o : %.cpp $(BUILDSTAMP)
	$(CC) $(CFLAGS) $(DEPCFLAGS) -O$(O) -o $@ -c $<

##
# To include additional non-executable files (e.g. selects.c, utils.c, etc),
# you'll need to add an additional build dependency to the file that requires
# the new file.  For example, see that client and server both require utils.o
#
# If you create a new file such as selects.c, then you will need a "selects.o"
# dependency on the right side of whichever one requires the file.
##

ASFContinuousOptimizer: asf_continuous_filter_optimization.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

ASFDiscreteOptimizer: asf_discrete_filter_optimization.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

filter_optimizer_int: filter_optimizer_int_based.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

filter_optimizer_bloom: filter_optimizer_bloom.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

full_workflow_bloom: full_workflow_bloom.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

full_workflow_filter_optimizer: full_workflow_filter_optimizer.o workload_generator.o
	$(CC) $(CFLAGS) $(DEPCFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -f ASFDiscreteOptimizer ASFContinuousOptimizer full_workflow_bloom full_workflow_filter_optimizer filter_optimizer_int filter_optimizer_bloom
	rm -f toy_game_for_proof
	rm ./*.o
	rm -rf .deps

distclean: clean
	rm -rf $(DEPSDIR)

.PHONY: all clean distclean
