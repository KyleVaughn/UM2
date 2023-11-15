from ctypes import c_double, c_int, CDLL
import ctypes
import sys

lib_path = "libum2.dylib" #name of shared lib

um2_function_lib = None
try:
    um2_function_lib = CDLL(lib_path)
    print(um2_function_lib)
except:
    print('OS %s not recognized' % (sys.platform))
#void um2Malloc(void ** p, Size size);
py_um2Malloc = um2_function_lib.um2Malloc
py_um2Malloc.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int32)
py_um2Malloc.restype = None

#void um2Free(void * p);
py_um2Free = um2_function_lib.um2Free
py_um2Malloc.argtypes = (ctypes.c_void_p,)
py_um2Malloc.restype = None

#void um2Initialize(char const * verbosity, Int init_gmsh, Int gmsh_verbosity, Int * ierr);
py_um2Initialize = um2_function_lib.um2Initialize
py_um2Initialize.argtypes = (ctypes.c_char_p,
                             ctypes.c_int32,ctypes.c_int32,
                             ctypes.POINTER(ctypes.c_int32),)
py_um2Initialize.restype = None

# void um2Finalize(Int * ierr);
py_um2Finalize = um2_function_lib.um2Finalize
py_um2Finalize.argtypes = (ctypes.POINTER(ctypes.c_int32),)
py_um2Finalize.restype = None
    
# void getSizeOfInt(int * size);
py_getSizeOfInt = um2_function_lib.getSizeOfInt
py_getSizeOfInt.argtypes = (ctypes.POINTER(ctypes.c_int32),)
py_getSizeOfInt.restype = None

# void getSizeOfFloat(int * size);
py_getSizeOfFloat = um2_function_lib.getSizeOfFloat
py_getSizeOfFloat.argtypes = (ctypes.POINTER(ctypes.c_int32),)
py_getSizeOfInt.restype = None

# // MPACT Spatial Partition
# void um2NewMPACTSpatialPartition(void ** model, Int * ierr);
py_um2NewMPACTSpatialPartition = um2_function_lib.um2NewMPACTSpatialPartition
py_um2NewMPACTSpatialPartition.argtypes = (ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_int32,)
py_um2NewMPACTSpatialPartition.restype = None

# void um2DeleteMPACTSpatialPartition(void * model, Int * ierr);
py_um2DeleteMPACTSpatialPartition = um2_function_lib.um2DeleteMPACTSpatialPartition
py_um2DeleteMPACTSpatialPartition.argtypes = (ctypes.c_void_p, 
                                              ctypes.POINTER(ctypes.c_int32),)
py_um2DeleteMPACTSpatialPartition.restype = None

# void um2ImportMPACTModel(char const * path, void ** model, Int * ierr);
py_um2ImportMPACTModel = um2_function_lib.um2ImportMPACTModel
py_um2ImportMPACTModel.argtypes = (ctypes.c_char_p, 
                                   ctypes.POINTER(ctypes.c_void_p), 
                                   ctypes.POINTER(ctypes.c_int32),)
py_um2ImportMPACTModel.restype = None

# // Num
# void um2MPACTNumCoarseCells(void * model, Int * n, Int * ierr);
py_um2MPACTNumCoarseCells = um2_function_lib.um2MPACTNumCoarseCells
py_um2MPACTNumCoarseCells.argtypes = (ctypes.c_void_p, 
                                      ctypes.POINTER(ctypes.c_int32), 
                                      ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTNumCoarseCells.restype = None

# void um2MPACTNumRTMs(void * model, Int * n, Int * ierr);
py_um2MPACTNumRTMs = um2_function_lib.um2MPACTNumRTMs
py_um2MPACTNumRTMs.argtypes = (ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_int32),
                               ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTNumRTMs.restype = None

# void um2MPACTNumLattices(void * model, Int * n, Int * ierr);
py_um2MPACTNumLattices = um2_function_lib.um2MPACTNumLattices
py_um2MPACTNumLattices.argtypes = (ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32,))
py_um2MPACTNumLattices.restype = None


# void um2MPACTNumAssemblies(void * model, Int * n, Int * ierr);
py_um2MPACTNumAssemblies = um2_function_lib.um2MPACTNumAssemblies
py_um2MPACTNumAssemblies.argtypes = (ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTNumAssemblies.restype = None


# // NumCells
# void um2MPACTCoreNumCells(void * model, Int * nx, Int * ny, Int * ierr);
py_um2MPACTCoreNumCells = um2_function_lib.um2MPACTCoreNumCells
py_um2MPACTCoreNumCells.argtypes = (ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoreNumCells.restype = None

# void um2MPACTAssemblyNumCells(void * model, Int asy_id, Int * nx, Int * ierr);
py_um2MPACTAssemblyNumCells = um2_function_lib.um2MPACTAssemblyNumCells
py_um2MPACTAssemblyNumCells.argtypes = (ctypes.c_void_p,
                                    ctypes.c_int32, 
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTAssemblyNumCells = None

# void um2MPACTLatticeNumCells(void * model, Int lat_id, Int * nx, Int * ny, Int * ierr);
py_um2MPACTLatticeNumCells = um2_function_lib.um2MPACTLatticeNumCells
py_um2MPACTLatticeNumCells.argtypes = (ctypes.c_void_p,
                                        ctypes.c_int32,
                                        ctypes.POINTER(ctypes.c_int32),
                                        ctypes.POINTER(ctypes.c_int32),
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTLatticeNumCells = None

# void um2MPACTRTMNumCells(void * model, Int rtm_id, Int * nx, Int * ny, Int * ierr);
py_um2MPACTRTMNumCells = um2_function_lib.um2MPACTRTMNumCells
py_um2MPACTRTMNumCells.argtypes = (ctypes.c_void_p,
                                    ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTRTMNumCells.restype = None


# // Child
# void um2MPACTCoreGetChild(void * model, Int ix, Int iy, Int * child, Int * ierr);
py_um2MPACTCoreGetChild = um2_function_lib.um2MPACTCoreGetChild
py_um2MPACTCoreGetChild.argtypes = (ctypes.c_void_p, 
                                    ctypes.c_int32, 
                                    ctypes.c_int32, 
                                    ctypes.POINTER(ctypes.c_int32), 
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoreGetChild.restype = None

# void um2MPACTAssemblyGetChild(void * model, Int asy_id, Int ix, Int * child, Int * ierr);
py_um2MPACTAssemblyGetChild = um2_function_lib.um2MPACTAssemblyGetChild
py_um2MPACTAssemblyGetChild.argtypes = (ctypes.c_void_p, 
                                        ctypes.c_int32, 
                                        ctypes.c_int32, 
                                        ctypes.POINTER(ctypes.c_int32),
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTAssemblyGetChild.restype = None

# void um2MPACTLatticeGetChild(void * model, Int lat_id, Int ix, Int iy, Int * child,
#                         Int * ierr);
py_um2MPACTLatticeGetChild = um2_function_lib.um2MPACTLatticeGetChild
py_um2MPACTLatticeGetChild.argtypes = (ctypes.c_void_p, 
                                        ctypes.c_int32, 
                                        ctypes.c_int32,
                                        ctypes.c_int32,
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTLatticeGetChild.restype = None

# void um2MPACTRTMGetChild(void * model, Int rtm_id, Int ix, Int iy, Int * child, Int * ierr);

py_um2MPACTRTMGetChild = um2_function_lib.um2MPACTRTMGetChild
py_um2MPACTRTMGetChild.argtypes = (ctypes.c_void_p, 
                                        ctypes.c_int32, 
                                        ctypes.c_int32, 
                                        ctypes.c_int32, 
                                        ctypes.POINTER(ctypes.c_int32), 
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTRTMGetChild.restype = None


# // CoarseCell
# void um2MPACTCoarseCellNumFaces(void * model, Int cc_id, Int * num_faces, Int * ierr);
py_um2MPACTCoarseCellNumFaces = um2_function_lib.um2MPACTCoarseCellNumFaces
py_um2MPACTCoarseCellNumFaces.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32,
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoarseCellNumFaces = None

# void um2MPACTCoarseCellWidth(void * model, Int cc_id, Float * width, Int * ierr);
py_um2MPACTCoarseCellWidth = um2_function_lib.um2MPACTCoarseCellWidth
py_um2MPACTCoarseCellWidth.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32, 
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_int32))
py_um2MPACTCoarseCellWidth.restype = None

# void um2MPACTCoarseCellHeight(void * model, Int cc_id, Float * height, Int * ierr);
py_um2MPACTCoarseCellHeight = um2_function_lib.um2MPACTCoarseCellHeight
py_um2MPACTCoarseCellHeight.argtypes = (ctypes.c_void_p,
                                        ctypes.c_int32, 
                                        ctypes.POINTER(ctypes.c_float), 
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoarseCellHeight.restype = None

# void um2MPACTCoarseCellFaceAreas(void * model, Int cc_id, Int * n, Float ** areas, Int * ierr);
py_um2MPACTCoarseCellFaceAreas = um2_function_lib.um2MPACTCoarseCellFaceAreas
py_um2MPACTCoarseCellFaceAreas.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32, 
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.POINTER(ctypes.c_int32))
py_um2MPACTCoarseCellFaceAreas.restype = None

# void um2MPACTCoarseCellFaceContaining(void * model, Int cc_id, Float x, Float y, Int * face_id,
#                                  Int * ierr);
py_um2MPACTCoarseCellFaceContaining = um2_function_lib.um2MPACTCoarseCellFaceContaining
py_um2MPACTCoarseCellFaceContaining.argtypes = (ctypes.c_void_p, 
                                                ctypes.c_int32, 
                                                ctypes.c_float, 
                                                ctypes.c_float,
                                                ctypes.POINTER(ctypes.c_int32),
                                                ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoarseCellFaceContaining.restype = None

# void um2MPACTCoarseCellFaceCentroid(void * model, Int cc_id, Int face_id, Float * x, Float * y,
#                                Int * ierr);
py_um2MPACTCoarseCellFaceCentroid = um2_function_lib.um2MPACTCoarseCellFaceCentroid
py_um2MPACTCoarseCellFaceCentroid.argtypes = (ctypes.c_void_p, 
                                                ctypes.c_int32, 
                                                ctypes.c_int32, 
                                                ctypes.POINTER(ctypes.c_float),
                                                ctypes.POINTER(ctypes.c_float),)
py_um2MPACTCoarseCellFaceCentroid.restype = None

# void um2MPACTCoarseCellMaterialIDs(void * model, Int cc_id, MaterialID ** mat_ids, Int * n,
#                               Int * ierr);
py_um2MPACTCoarseCellMaterialIDs = um2_function_lib.um2MPACTCoarseCellMaterialIDs
py_um2MPACTCoarseCellMaterialIDs.argtypes = (ctypes.c_void_p, 
                                                ctypes.c_int32, 
                                                ctypes.POINTER(ctypes.POINTER(ctypes.c_int8)), 
                                                ctypes.POINTER(ctypes.c_int32), 
                                                ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoarseCellMaterialIDs.restype = None

# void um2MPACTIntersectCoarseCell(void * model, Int cc_id, Float origin_x, Float origin_y,
#                             Float direction_x, Float direction_y, Float * intersections,
#                             Int * n, Int * ierr);
py_um2MPACTIntersectCoarseCell = um2_function_lib.um2MPACTIntersectCoarseCell
py_um2MPACTIntersectCoarseCell.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32,
                                            ctypes.c_float,
                                            ctypes.c_float,
                                            ctypes.c_float,
                                            ctypes.c_float,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTIntersectCoarseCell.restype = None

# void um2MPACTRTMWidth(void * model, Int rtm_id, Float * width, Int * ierr);
py_um2MPACTRTMWidth = um2_function_lib.um2MPACTRTMWidth
py_um2MPACTRTMWidth.argtypes = (ctypes.c_void_p,
                                ctypes.c_int32,
                                ctypes.POINTER(ctypes.c_float),
                                ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTRTMWidth.restype = None

# void um2MPACTRTMHeight(void * model, Int rtm_id, Float * height, Int * ierr);
py_um2MPACTRTMHeight = um2_function_lib.um2MPACTRTMHeight
py_um2MPACTRTMHeight.argtypes = (ctypes.c_void_p,
                                    ctypes.c_int32,
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTRTMHeight.restype = None

# void um2MPACTCoarseCellHeights(void * model, Int * n, Int ** cc_ids, Float ** heights,
#                           Int * ierr);
py_um2MPACTCoarseCellHeights = um2_function_lib.um2MPACTCoarseCellHeights
py_um2MPACTCoarseCellHeights.argtypes = (ctypes.c_void_p,
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTCoarseCellHeights.restype = None

# void um2MPACTRTMHeights(void * model, Int * n, Int ** rtm_ids, Float ** heights, Int * ierr);
py_um2MPACTRTMHeights = um2_function_lib.um2MPACTRTMHeights
py_um2MPACTRTMHeights.argtypes = (ctypes.c_void_p,
                                    ctypes.POINTER(ctypes.c_int32),
                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                    ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTRTMHeights.restype = None

# void um2MPACTLatticeHeights(void * model, Int * n, Int ** lat_ids, Float ** heights,
#                        Int * ierr);
py_um2MPACTLatticeHeights = um2_function_lib.um2MPACTLatticeHeights
py_um2MPACTLatticeHeights.argtypes = (ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_int32),
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                        ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTLatticeHeights.restype = None

# void um2MPACTAssemblyHeights(void * model, Int asy_id, Int * n, Float ** heights, Int * ierr);
py_um2MPACTAssemblyHeights = um2_function_lib.um2MPACTAssemblyHeights
py_um2MPACTAssemblyHeights.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32,
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.POINTER(ctypes.c_int32),)
py_um2MPACTAssemblyHeights.restype = None


# void um2MPACTCoarseCellFaceData(void * model, Int cc_id, Int * mesh_type, Int * num_vertices,
#                            Int * num_faces, Float ** vertices, Int ** fv, Int * ierr);
py_um2MPACTCoarseCellFaceData = um2_function_lib.um2MPACTCoarseCellFaceData
py_um2MPACTCoarseCellFaceData.argtypes = (ctypes.c_void_p,
                                            ctypes.c_int32,
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.c_int32),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                            ctypes.POINTER(ctypes.c_int32))
py_um2MPACTCoarseCellFaceData.restype = None

def initialize(verbosity = "info" , init_gmsh = True , gmsh_verbosity = 2):
    error = None
    py_um2Initialize(verbosity,init_gmsh,gmsh_verbosity,error)
    if error.contents != 0:
        print("um2 initialize failed")
        sys.exit(error)

def finalize():
    error = None
    py_um2Finalize(error = None)
    if error.contents != 0:
        print("um2 finalize failed")
        sys.exit(error)

class SpatialPartion:
    def __newMPACTSpatialPartition(self):
        #intializes a spatial partion model and passes the created void pointer as a value of self
        return
    
    def __deleteMPACTSpatialPartition(self):
        #frees spatial partition model's data
        return
    
    def __um2ImportMPACTModel(self, filepath):
        #imports spatial partion model from filepath
        return
    
    def __init__(self):
        self.__newMPACTSpatialPartion()
        
    def __init__(self,filepath):
        self.__um2ImportMPACTModel(filepath)
    
    def __del__(self):
        self.__deleteMPACTSpatialPartition()
        
    # def make_coarse_cell(self):
        
def main():
    #void um2Initialize(char const * verbosity, Int init_gmsh, Int gmsh_verbosity, Int * ierr);
    verbosity = ctypes.c_char_p("info".encode('utf-8'))
    ierr = ctypes.c_int32(3)
    ierr_p = ctypes.pointer(ierr)
    py_um2Initialize(verbosity, 0, 0, ierr)
    print(ierr.value)

main()
        
    
