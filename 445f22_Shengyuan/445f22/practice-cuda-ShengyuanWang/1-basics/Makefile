CC = nvcc

PROGS = device_info simpleBlockPrintf

all: $(PROGS)

device_info: device_info.cu
	$(CC) -o device_info device_info.cu

simpleBlockPrintf: simpleBlockPrintf.cu
	$(CC) -o simpleBlockPrintf simpleBlockPrintf.cu

clean:
	rm -f $(PROGS)
