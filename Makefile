#Get nvcc compiler, compiler flags and OS/arch parameters
include settings.mk

##General flags
CC=g++
CFLAGS+= -Wall -O3
LDFLAGS+= -lpng -lm

#pass some flags through to gcc when defined
ifdef DEBUG
CFLAGS+= -DDEBUG -g
endif
ifdef SILENT
CFLAGS+= -DSILENT
endif
ifdef CPU
CFLAGS+= -DCPU_ONLY
endif
ifdef TIMING
CFLAGS+= -DTIMING
endif

CUSRCS=$(wildcard *.cu)
SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o) $(CUSRCS:.cu=.o)
DEPS=$(OBJS:.o=.d)
EXE=mobilenetv2

default:check

#include file that defines checks for a set of verification images
include images.mk

.PHONY:check
check:converted_$(basename $(notdir $(word 1,$(IMAGE_URLS)))).png $(EXE)
	./$(EXE) $< | tee $@
	@grep -q "Detected class: $(strip $(word 1, $(CLASS_IDX)))" $@ && printf "$(GREEN)correctly identified image $<$(NC)\n" ||  printf "$(RED)Did not correctly identify image $<$(NC)\n"
	python PerformancePlotter.py

#link the executable
.PRECIOUS:$(EXE)
$(EXE):$(OBJS)
	$(NVCC) $^  -o $@ $(NVCCFLAGS) $(LDFLAGS) --compiler-options $(CFLAGS)

#compile c files
%.o:%.c
	$(CC) -MMD -MF $(subst .o,.d,$@) $(INCLUDES) -c $(CFLAGS) $< -o $@

%.o:%.cu
	@$(NVCC) $(NVCCFLAGS) --generate-dependencies $< -o $(subst .o,.d,$@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ --compiler-options $(CFLAGS)

#Create an archive of the code
USER=$(shell whoami)
ZIPFILE=$(USER).zip
zip:$(ZIPFILE)
$(ZIPFILE):clean
	zip -r $@ ./* -x ./bins/*

#build dependency files if we are not cleaning
ifneq ($(filter clean,$(MAKECMDGOALS)),clean)
-include $(DEPS)
endif

CLEAN=$(OBJS) $(DEPS) $(EXE) check $(ZIPFILE)

#downloaded images
CLEAN+=$(foreach URL, $(IMAGE_URLS), $(notdir $(URL)))

#converted images
CLEAN+=$(foreach URL, $(IMAGE_URLS), converted_$(basename $(notdir $(URL))).png)

#check files
CLEAN+=$(foreach URL, $(IMAGE_URLS), check_$(basename $(notdir $(URL))))


ifdef DEBUG
#add extra cleanup when debug is set (to clean up dumped blobs)
CLEAN+= *.txt *.bin
endif

clean:
	@rm -f $(CLEAN)
