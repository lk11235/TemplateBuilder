#makefile 


CC   =   g++

#UCFLAGS = -O0 -g3 -Wall -gstabs+  
UCFLAGS = -O3 -Wall -gstabs+ -std=c++0x 


RUCFLAGS := $(shell root-config --cflags) -I./include/ -I./include/external/
LIBS :=  $(shell root-config --libs) -lTreePlayer
GLIBS := $(shell root-config --glibs)

VPATH = ./src/:./src/external/:./bin/

SRCPP = main.cpp\
	json_reader.cpp\
	json_value.cpp\
	json_writer.cpp\
	BinTree.cpp\
	GaussKernelSmoother.cpp\
	Smoother1D.cpp\
	Template.cpp\
	TemplateManager.cpp\
	TemplateBuilder.cpp\
	TemplateParameters.cpp

	
         
#OBJCPP = $(SRCPP:.cpp=.o)
OBJCPP = $(patsubst %.cpp,obj/%.o,$(SRCPP))


all : smoothHist.exe

obj/%.o : %.cpp
	@echo ">> compiling $*"
	@mkdir -p obj/
	@$(CC) -c $< $(UCFLAGS) $(RUCFLAGS) -o $@

smoothHist.exe : $(OBJCPP) 
	@echo ">> linking..."
	@$(CC) $^ $(ACLIBS) $(LIBS) $(GLIBS)  -o $@

clean:
	@echo ">> cleaning objects and executable"
	@rm  -f obj/*.o
	@rm -f smoothHist.exe
