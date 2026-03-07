// NVTX stub: when PX_NVTX is 0 or undefined, all nvtx calls compile to no-ops.
// Toggle via cmake: -DPX_USE_NVTX=ON/OFF
#pragma once

#if PX_NVTX
	#include <nvtx3/nvToolsExt.h>
#else
	static inline int  nvtxRangePush(const char*) { return 0; }
	static inline void nvtxRangePop() {}
#endif
