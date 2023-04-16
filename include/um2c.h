#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void um2_malloc(void ** const p, int const size);

void um2_free(void * const p);

//void um2_initialize(char const * const verbosity,
//                    int const init_gmsh,
//                    int * const ierr);
//
//void um2_finalize(int * const ierr);


#ifdef __cplusplus
}
#endif
