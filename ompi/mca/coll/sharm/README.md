# coll/sharm component

## Structure

- Files *_cma.c contains CMA-based algoriothm versions
- Files *_xpmem.c contains XPMEM-based algoriothm versions
- Files *_cico.c contains Copy-in Copy-out based alogrithm versions
- File [coll_sharm.h](coll_sharm.h) is a main component header. Define tech supports (CMA, XPMEM, etc...)
- File [coll_sharm_segment.c](coll_sharm_segment.c) initialize shared memory segment and local flags/systems
- File [coll_sharm_algorithm_types.h](coll_sharm_algorithm_types.h) contain constants for algorithm types (COLL_SHARM_BCAST_ALG_CICO, COLL_SHARM_BCAST_ALG_CMA, etc...)
- File [coll_sharm_func.h](coll_sharm_func.h) contains function prototypes and main macros of the component.
- File [coll_queue.h](coll_queue.h) contains function prototypes for working with the queue.
- File [coll_sharm_profiling.h](coll_sharm_profiling.h) contains functions for working with the profiling subsystem.
- File [coll_sharm_module.c](coll_sharm_module.c) contains functions for creating and deleting a module.
- File [coll_sharm_component.c](coll_sharm_component.c) contains functions for initializing the component and its parameters.
- File [coll_sharm_util.h](coll_sharm_util.h) contains helpers macros for NUMA, KNEM, XPMEM, CMA, AVX2, etc...
- File [coll_sharm_util.c](coll_sharm_util.c) contains helpers functions.

## Requirements

## Queue

| Process |  Queue  |
|---------|---------|
|    0    |    0    |
|    0    |   ...   |
|    0    |  p - 1  |
|    1    |    0    |
|    1    |   ...   |
|    1    |  p - 1  |
|   ...   |   ...   |
|  p - 1  |    0    |
|  p - 1  |   ...   |
|  p - 1  |  p - 1  |

### Queue system (p = 4, s = 8, f = 8192)

![Alt text](docs/memmap.drawio.svg)
