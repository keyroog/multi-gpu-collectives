.SUFFIXES:
CXX=icpx
CXXFLAGS=-fsycl
LDLIBS=-lccl -lmpi

OPS = allgather allgatherv allreduce alltoall alltoallv broadcast reduce reduce_scatter scatter

TARGETS = \
 src/oneccl/allgather/allgather \
 src/oneccl/allgatherv/allgatherv \
 src/oneccl/allreduce/allreduce \
 src/oneccl/alltoall/alltoall \
 src/oneccl/alltoallv/alltoallv \
 src/oneccl/broadcast/broadcast \
 src/oneccl/reduce/reduce \
 src/oneccl/reduce_scatter/reduce_scatter \
 src/oneccl/scatter/scatter

all: $(TARGETS)

src/oneccl/allgather/allgather: src/oneccl/allgather/allgather.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/allgatherv/allgatherv: src/oneccl/allgatherv/allgatherv.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/allreduce/allreduce: src/oneccl/allreduce/allreduce.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/alltoall/alltoall: src/oneccl/alltoall/alltoall.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/alltoallv/alltoallv: src/oneccl/alltoallv/alltoallv.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/broadcast/broadcast: src/oneccl/broadcast/broadcast.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/reduce/reduce: src/oneccl/reduce/reduce.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/reduce_scatter/reduce_scatter: src/oneccl/reduce_scatter/reduce_scatter.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

src/oneccl/scatter/scatter: src/oneccl/scatter/scatter.cpp
	$(CXX) -o $@ $< $(LDLIBS) $(CXXFLAGS)

allgather: src/oneccl/allgather/allgather
allgatherv: src/oneccl/allgatherv/allgatherv
allreduce: src/oneccl/allreduce/allreduce
alltoall: src/oneccl/alltoall/alltoall
alltoallv: src/oneccl/alltoallv/alltoallv
broadcast: src/oneccl/broadcast/broadcast
reduce: src/oneccl/reduce/reduce
reduce_scatter: src/oneccl/reduce_scatter/reduce_scatter
scatter: src/oneccl/scatter/scatter

clean:
	rm -f $(TARGETS)

.PHONY: all clean $(OPS)
