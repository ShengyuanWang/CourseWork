

PROGS = vectorAdd

all: $(PROGS)

vectorAdd: vectorAdd.cu
	nvcc -o vectorAdd vectorAdd.cu

clean:
	rm -f $(PROGS)
