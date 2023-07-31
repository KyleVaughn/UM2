#pragma once

#include <um2/config.hpp> // UM2_HAS_GMSH

#if UM2_ENABLE_GMSH

// NOLINTBEGIN(readability*, modernize*)

#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"

#  include <cmath>
#  include <functional>
#  include <string>
#  include <utility>
#  include <vector>

// A wrapper around the gmsh library to make namespace management with um2 gmsh
// related code easier.

#  ifndef M_PI
#    define M_PI (3.14159265358979323846)
#  endif

namespace um2
{

namespace gmsh
{
typedef std::vector<std::pair<int, int>> vectorpair;
}

namespace gmsh
{

void
initialize(int argc = 0, char ** argv = nullptr, const bool readConfigFiles = true,
           const bool run = false);
int
isInitialized();
void
finalize();
void
open(const std::string & fileName);
void
merge(const std::string & fileName);
void
write(const std::string & fileName);
void
clear();

namespace option
{ // Option handling functions

void
setNumber(const std::string & name, const double value);
void
getNumber(const std::string & name, double & value);
void
setString(const std::string & name, const std::string & value);
void
getString(const std::string & name, std::string & value);
void
setColor(const std::string & name, const int r, const int g, const int b,
         const int a = 255);
void
getColor(const std::string & name, int & r, int & g, int & b, int & a);

} // namespace option

namespace model
{ // Model functions

void
add(const std::string & name);
void
remove();
void
list(std::vector<std::string> & names);
void
getCurrent(std::string & name);
void
setCurrent(const std::string & name);
void
getFileName(std::string & fileName);
void
setFileName(const std::string & fileName);
void
getEntities(gmsh::vectorpair & dimTags, const int dim = -1);
void
setEntityName(const int dim, const int tag, const std::string & name);
void
getEntityName(const int dim, const int tag, std::string & name);
void
getPhysicalGroups(gmsh::vectorpair & dimTags, const int dim = -1);
void
getEntitiesForPhysicalGroup(const int dim, const int tag, std::vector<int> & tags);
//        void getEntitiesForPhysicalName(const std::string & name, gmsh::vectorpair &
//        dimTags);
void
getPhysicalGroupsForEntity(const int dim, const int tag, std::vector<int> & physicalTags);
int
addPhysicalGroup(const int dim, const std::vector<int> & tags, const int tag = -1,
                 const std::string & name = "");
void
removePhysicalGroups(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
setPhysicalName(const int dim, const int tag, const std::string & name);
void
removePhysicalName(const std::string & name);
void
getPhysicalName(const int dim, const int tag, std::string & name);
void
setTag(const int dim, const int tag, const int newTag);
void
getBoundary(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags,
            const bool combined = true, const bool oriented = true,
            const bool recursive = false);
void
getAdjacencies(const int dim, const int tag, std::vector<int> & upward,
               std::vector<int> & downward);
void
getEntitiesInBoundingBox(const double xmin, const double ymin, const double zmin,
                         const double xmax, const double ymax, const double zmax,
                         gmsh::vectorpair & dimTags, const int dim = -1);
void
getBoundingBox(const int dim, const int tag, double & xmin, double & ymin, double & zmin,
               double & xmax, double & ymax, double & zmax);
int
getDimension();
int
addDiscreteEntity(const int dim, const int tag = -1,
                  const std::vector<int> & boundary = std::vector<int>());
void
removeEntities(const gmsh::vectorpair & dimTags, const bool recursive = false);
void
removeEntityName(const std::string & name);
void
getType(const int dim, const int tag, std::string & entityType);
void
getParent(const int dim, const int tag, int & parentDim, int & parentTag);
int
getNumberOfPartitions();
void
getPartitions(const int dim, const int tag, std::vector<int> & partitions);
void
getValue(const int dim, const int tag, const std::vector<double> & parametricCoord,
         std::vector<double> & coord);
void
getDerivative(const int dim, const int tag, const std::vector<double> & parametricCoord,
              std::vector<double> & derivatives);
void
getSecondDerivative(const int dim, const int tag,
                    const std::vector<double> & parametricCoord,
                    std::vector<double> & derivatives);
void
getCurvature(const int dim, const int tag, const std::vector<double> & parametricCoord,
             std::vector<double> & curvatures);
void
getPrincipalCurvatures(const int tag, const std::vector<double> & parametricCoord,
                       std::vector<double> & curvatureMax,
                       std::vector<double> & curvatureMin,
                       std::vector<double> & directionMax,
                       std::vector<double> & directionMin);
void
getNormal(const int tag, const std::vector<double> & parametricCoord,
          std::vector<double> & normals);
void
getParametrization(const int dim, const int tag, const std::vector<double> & coord,
                   std::vector<double> & parametricCoord);
void
getParametrizationBounds(const int dim, const int tag, std::vector<double> & min,
                         std::vector<double> & max);
int
isInside(const int dim, const int tag, const std::vector<double> & coord,
         const bool parametric = false);
void
getClosestPoint(const int dim, const int tag, const std::vector<double> & coord,
                std::vector<double> & closestCoord,
                std::vector<double> & parametricCoord);
void
reparametrizeOnSurface(const int dim, const int tag,
                       const std::vector<double> & parametricCoord, const int surfaceTag,
                       std::vector<double> & surfaceParametricCoord, const int which = 0);
void
setVisibility(const gmsh::vectorpair & dimTags, const int value,
              const bool recursive = false);
void
getVisibility(const int dim, const int tag, int & value);
void
setVisibilityPerWindow(const int value, const int windowIndex = 0);
void
setColor(const gmsh::vectorpair & dimTags, const int r, const int g, const int b,
         const int a = 255, const bool recursive = false);
void
getColor(const int dim, const int tag, int & r, int & g, int & b, int & a);
void
setCoordinates(const int tag, const double x, const double y, const double z);
void
getAttributeNames(std::vector<std::string> & names);
void
getAttribute(const std::string & name, std::vector<std::string> & values);
void
setAttribute(const std::string & name, const std::vector<std::string> & values);
void
removeAttribute(const std::string & name);

namespace mesh
{ // Mesh functions

void
generate(const int dim = 3);
void
partition(const int numPart,
          const std::vector<std::size_t> & elementTags = std::vector<std::size_t>(),
          const std::vector<int> & partitions = std::vector<int>());
void
unpartition();
void
optimize(const std::string & method = "", const bool force = false, const int niter = 1,
         const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
recombine();
void
refine();
void
setOrder(const int order);
void
getLastEntityError(gmsh::vectorpair & dimTags);
void
getLastNodeError(std::vector<std::size_t> & nodeTags);
void
clear(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
reverse(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
affineTransform(const std::vector<double> & affineTransform,
                const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
getNodes(std::vector<std::size_t> & nodeTags, std::vector<double> & coord,
         std::vector<double> & parametricCoord, const int dim = -1, const int tag = -1,
         const bool includeBoundary = false, const bool returnParametricCoord = true);
void
getNodesByElementType(const int elementType, std::vector<std::size_t> & nodeTags,
                      std::vector<double> & coord, std::vector<double> & parametricCoord,
                      const int tag = -1, const bool returnParametricCoord = true);
void
getNode(const std::size_t nodeTag, std::vector<double> & coord,
        std::vector<double> & parametricCoord, int & dim, int & tag);
void
setNode(const std::size_t nodeTag, const std::vector<double> & coord,
        const std::vector<double> & parametricCoord);
void
rebuildNodeCache(const bool onlyIfNecessary = true);
void
rebuildElementCache(const bool onlyIfNecessary = true);
void
getNodesForPhysicalGroup(const int dim, const int tag,
                         std::vector<std::size_t> & nodeTags,
                         std::vector<double> & coord);
void
getMaxNodeTag(std::size_t & maxTag);
void
addNodes(const int dim, const int tag, const std::vector<std::size_t> & nodeTags,
         const std::vector<double> & coord,
         const std::vector<double> & parametricCoord = std::vector<double>());
void
reclassifyNodes();
void
relocateNodes(const int dim = -1, const int tag = -1);
void
getElements(std::vector<int> & elementTypes,
            std::vector<std::vector<std::size_t>> & elementTags,
            std::vector<std::vector<std::size_t>> & nodeTags, const int dim = -1,
            const int tag = -1);
void
getElement(const std::size_t elementTag, int & elementType,
           std::vector<std::size_t> & nodeTags, int & dim, int & tag);
void
getElementByCoordinates(const double x, const double y, const double z,
                        std::size_t & elementTag, int & elementType,
                        std::vector<std::size_t> & nodeTags, double & u, double & v,
                        double & w, const int dim = -1, const bool strict = false);
void
getElementsByCoordinates(const double x, const double y, const double z,
                         std::vector<std::size_t> & elementTags, const int dim = -1,
                         const bool strict = false);
void
getLocalCoordinatesInElement(const std::size_t elementTag, const double x, const double y,
                             const double z, double & u, double & v, double & w);
void
getElementTypes(std::vector<int> & elementTypes, const int dim = -1, const int tag = -1);
int
getElementType(const std::string & familyName, const int order,
               const bool serendip = false);
void
getElementProperties(const int elementType, std::string & elementName, int & dim,
                     int & order, int & numNodes, std::vector<double> & localNodeCoord,
                     int & numPrimaryNodes);
void
getElementsByType(const int elementType, std::vector<std::size_t> & elementTags,
                  std::vector<std::size_t> & nodeTags, const int tag = -1,
                  const std::size_t task = 0, const std::size_t numTasks = 1);
void
getMaxElementTag(std::size_t & maxTag);
void
preallocateElementsByType(const int elementType, const bool elementTag,
                          const bool nodeTag, std::vector<std::size_t> & elementTags,
                          std::vector<std::size_t> & nodeTags, const int tag = -1);
void
getElementQualities(const std::vector<std::size_t> & elementTags,
                    std::vector<double> & elementsQuality,
                    const std::string & qualityName = "minSICN",
                    const std::size_t task = 0, const std::size_t numTasks = 1);
void
addElements(const int dim, const int tag, const std::vector<int> & elementTypes,
            const std::vector<std::vector<std::size_t>> & elementTags,
            const std::vector<std::vector<std::size_t>> & nodeTags);
void
addElementsByType(const int tag, const int elementType,
                  const std::vector<std::size_t> & elementTags,
                  const std::vector<std::size_t> & nodeTags);
void
getIntegrationPoints(const int elementType, const std::string & integrationType,
                     std::vector<double> & localCoord, std::vector<double> & weights);
void
getJacobians(const int elementType, const std::vector<double> & localCoord,
             std::vector<double> & jacobians, std::vector<double> & determinants,
             std::vector<double> & coord, const int tag = -1, const std::size_t task = 0,
             const std::size_t numTasks = 1);
void
preallocateJacobians(const int elementType, const int numEvaluationPoints,
                     const bool allocateJacobians, const bool allocateDeterminants,
                     const bool allocateCoord, std::vector<double> & jacobians,
                     std::vector<double> & determinants, std::vector<double> & coord,
                     const int tag = -1);
void
getJacobian(const std::size_t elementTag, const std::vector<double> & localCoord,
            std::vector<double> & jacobians, std::vector<double> & determinants,
            std::vector<double> & coord);
void
getBasisFunctions(const int elementType, const std::vector<double> & localCoord,
                  const std::string & functionSpaceType, int & numComponents,
                  std::vector<double> & basisFunctions, int & numOrientations,
                  const std::vector<int> & wantedOrientations = std::vector<int>());
void
getBasisFunctionsOrientation(const int elementType, const std::string & functionSpaceType,
                             std::vector<int> & basisFunctionsOrientation,
                             const int tag = -1, const std::size_t task = 0,
                             const std::size_t numTasks = 1);
void
getBasisFunctionsOrientationForElement(const std::size_t elementTag,
                                       const std::string & functionSpaceType,
                                       int & basisFunctionsOrientation);
int
getNumberOfOrientations(const int elementType, const std::string & functionSpaceType);
void
preallocateBasisFunctionsOrientation(const int elementType,
                                     std::vector<int> & basisFunctionsOrientation,
                                     const int tag = -1);
void
getEdges(const std::vector<std::size_t> & nodeTags, std::vector<std::size_t> & edgeTags,
         std::vector<int> & edgeOrientations);
void
getFaces(const int faceType, const std::vector<std::size_t> & nodeTags,
         std::vector<std::size_t> & faceTags, std::vector<int> & faceOrientations);
void
createEdges(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
createFaces(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
getAllEdges(std::vector<std::size_t> & edgeTags, std::vector<std::size_t> & edgeNodes);
void
getAllFaces(const int faceType, std::vector<std::size_t> & faceTags,
            std::vector<std::size_t> & faceNodes);
void
addEdges(const std::vector<std::size_t> & edgeTags,
         const std::vector<std::size_t> & edgeNodes);
void
addFaces(const int faceType, const std::vector<std::size_t> & faceTags,
         const std::vector<std::size_t> & faceNodes);
void
getKeys(const int elementType, const std::string & functionSpaceType,
        std::vector<int> & typeKeys, std::vector<std::size_t> & entityKeys,
        std::vector<double> & coord, const int tag = -1, const bool returnCoord = true);
void
getKeysForElement(const std::size_t elementTag, const std::string & functionSpaceType,
                  std::vector<int> & typeKeys, std::vector<std::size_t> & entityKeys,
                  std::vector<double> & coord, const bool returnCoord = true);
int
getNumberOfKeys(const int elementType, const std::string & functionSpaceType);
void
getKeysInformation(const std::vector<int> & typeKeys,
                   const std::vector<std::size_t> & entityKeys, const int elementType,
                   const std::string & functionSpaceType, gmsh::vectorpair & infoKeys);
void
getBarycenters(const int elementType, const int tag, const bool fast, const bool primary,
               std::vector<double> & barycenters, const std::size_t task = 0,
               const std::size_t numTasks = 1);
void
preallocateBarycenters(const int elementType, std::vector<double> & barycenters,
                       const int tag = -1);
void
getElementEdgeNodes(const int elementType, std::vector<std::size_t> & nodeTags,
                    const int tag = -1, const bool primary = false,
                    const std::size_t task = 0, const std::size_t numTasks = 1);
void
getElementFaceNodes(const int elementType, const int faceType,
                    std::vector<std::size_t> & nodeTags, const int tag = -1,
                    const bool primary = false, const std::size_t task = 0,
                    const std::size_t numTasks = 1);
void
getGhostElements(const int dim, const int tag, std::vector<std::size_t> & elementTags,
                 std::vector<int> & partitions);
void
setSize(const gmsh::vectorpair & dimTags, const double size);
void
getSizes(const gmsh::vectorpair & dimTags, std::vector<double> & sizes);
void
setSizeAtParametricPoints(const int dim, const int tag,
                          const std::vector<double> & parametricCoord,
                          const std::vector<double> & sizes);
void
setSizeCallback(std::function<double(int, int, double, double, double, double)> callback);
void
removeSizeCallback();
void
setTransfiniteCurve(const int tag, const int numNodes,
                    const std::string & meshType = "Progression", const double coef = 1.);
void
setTransfiniteSurface(const int tag, const std::string & arrangement = "Left",
                      const std::vector<int> & cornerTags = std::vector<int>());
void
setTransfiniteVolume(const int tag,
                     const std::vector<int> & cornerTags = std::vector<int>());
void
setTransfiniteAutomatic(const gmsh::vectorpair & dimTags = gmsh::vectorpair(),
                        const double cornerAngle = 2.35, const bool recombine = true);
void
setRecombine(const int dim, const int tag, const double angle = 45.);
void
setSmoothing(const int dim, const int tag, const int val);
void
setReverse(const int dim, const int tag, const bool val = true);
void
setAlgorithm(const int dim, const int tag, const int val);
void
setSizeFromBoundary(const int dim, const int tag, const int val);
void
setCompound(const int dim, const std::vector<int> & tags);
void
setOutwardOrientation(const int tag);
void
removeConstraints(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
embed(const int dim, const std::vector<int> & tags, const int inDim, const int inTag);
void
removeEmbedded(const gmsh::vectorpair & dimTags, const int dim = -1);
void
getEmbedded(const int dim, const int tag, gmsh::vectorpair & dimTags);
void
reorderElements(const int elementType, const int tag,
                const std::vector<std::size_t> & ordering);
void
renumberNodes();
void
renumberElements();
void
setPeriodic(const int dim, const std::vector<int> & tags,
            const std::vector<int> & tagsMaster,
            const std::vector<double> & affineTransform);
void
getPeriodic(const int dim, const std::vector<int> & tags, std::vector<int> & tagMaster);
void
getPeriodicNodes(const int dim, const int tag, int & tagMaster,
                 std::vector<std::size_t> & nodeTags,
                 std::vector<std::size_t> & nodeTagsMaster,
                 std::vector<double> & affineTransform,
                 const bool includeHighOrderNodes = false);
void
getPeriodicKeys(const int elementType, const std::string & functionSpaceType,
                const int tag, int & tagMaster, std::vector<int> & typeKeys,
                std::vector<int> & typeKeysMaster, std::vector<std::size_t> & entityKeys,
                std::vector<std::size_t> & entityKeysMaster, std::vector<double> & coord,
                std::vector<double> & coordMaster, const bool returnCoord = true);
void
importStl();
void
getDuplicateNodes(std::vector<std::size_t> & tags,
                  const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
removeDuplicateNodes(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
removeDuplicateElements(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
splitQuadrangles(const double quality = 1., const int tag = -1);
void
setVisibility(const std::vector<std::size_t> & elementTags, const int value);
void
classifySurfaces(const double angle, const bool boundary = true,
                 const bool forReparametrization = false, const double curveAngle = M_PI,
                 const bool exportDiscrete = true);
void
createGeometry(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
createTopology(const bool makeSimplyConnected = true, const bool exportDiscrete = true);
void
addHomologyRequest(const std::string & type = "Homology",
                   const std::vector<int> & domainTags = std::vector<int>(),
                   const std::vector<int> & subdomainTags = std::vector<int>(),
                   const std::vector<int> & dims = std::vector<int>());
void
clearHomologyRequests();
void
computeHomology(gmsh::vectorpair & dimTags);
void
computeCrossField(std::vector<int> & viewTags);
void
triangulate(const std::vector<double> & coord, std::vector<std::size_t> & tri);
void
tetrahedralize(const std::vector<double> & coord, std::vector<std::size_t> & tetra);

namespace field
{ // Mesh size field functions

int
add(const std::string & fieldType, const int tag = -1);
void
remove(const int tag);
void
list(std::vector<int> & tags);
void
getType(const int tag, std::string & fileType);
void
setNumber(const int tag, const std::string & option, const double value);
void
getNumber(const int tag, const std::string & option, double & value);
void
setString(const int tag, const std::string & option, const std::string & value);
void
getString(const int tag, const std::string & option, std::string & value);
void
setNumbers(const int tag, const std::string & option, const std::vector<double> & values);
void
getNumbers(const int tag, const std::string & option, std::vector<double> & values);
void
setAsBackgroundMesh(const int tag);
void
setAsBoundaryLayer(const int tag);

} // namespace field

} // namespace mesh

namespace geo
{ // Built-in CAD kernel functions

int
addPoint(const double x, const double y, const double z, const double meshSize = 0.,
         const int tag = -1);
int
addLine(const int startTag, const int endTag, const int tag = -1);
int
addCircleArc(const int startTag, const int centerTag, const int endTag,
             const int tag = -1, const double nx = 0., const double ny = 0.,
             const double nz = 0.);
int
addEllipseArc(const int startTag, const int centerTag, const int majorTag,
              const int endTag, const int tag = -1, const double nx = 0.,
              const double ny = 0., const double nz = 0.);
int
addSpline(const std::vector<int> & pointTags, const int tag = -1);
int
addBSpline(const std::vector<int> & pointTags, const int tag = -1);
int
addBezier(const std::vector<int> & pointTags, const int tag = -1);
int
addPolyline(const std::vector<int> & pointTags, const int tag = -1);
int
addCompoundSpline(const std::vector<int> & curveTags, const int numIntervals = 5,
                  const int tag = -1);
int
addCompoundBSpline(const std::vector<int> & curveTags, const int numIntervals = 20,
                   const int tag = -1);
int
addCurveLoop(const std::vector<int> & curveTags, const int tag = -1,
             const bool reorient = false);
void
addCurveLoops(const std::vector<int> & curveTags, std::vector<int> & tags);
int
addPlaneSurface(const std::vector<int> & wireTags, const int tag = -1);
int
addSurfaceFilling(const std::vector<int> & wireTags, const int tag = -1,
                  const int sphereCenterTag = -1);
void
revolve(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
        const double ax, const double ay, const double az, const double angle,
        gmsh::vectorpair & outDimTags,
        const std::vector<int> & numElements = std::vector<int>(),
        const std::vector<double> & heights = std::vector<double>(),
        const bool recombine = false);
void
twist(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
      const double dx, const double dy, const double dz, const double ax, const double ay,
      const double az, const double angle, gmsh::vectorpair & outDimTags,
      const std::vector<int> & numElements = std::vector<int>(),
      const std::vector<double> & heights = std::vector<double>(),
      const bool recombine = false);
void
extrudeBoundaryLayer(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags,
                     const std::vector<int> & numElements = std::vector<int>(1, 1),
                     const std::vector<double> & heights = std::vector<double>(),
                     const bool recombine = false, const bool second = false,
                     const int viewIndex = -1);
void
translate(const gmsh::vectorpair & dimTags, const double dx, const double dy,
          const double dz);
void
rotate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double ax, const double ay, const double az, const double angle);
void
dilate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double a, const double b, const double c);
void
mirror(const gmsh::vectorpair & dimTags, const double a, const double b, const double c,
       const double d);
void
symmetrize(const gmsh::vectorpair & dimTags, const double a, const double b,
           const double c, const double d);
void
copy(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags);
void
remove(const gmsh::vectorpair & dimTags, const bool recursive = false);
void
removeAllDuplicates();
void
splitCurve(const int tag, const std::vector<int> & pointTags,
           std::vector<int> & curveTags);
int
getMaxTag(const int dim);
void
setMaxTag(const int dim, const int maxTag);
int
addPhysicalGroup(const int dim, const std::vector<int> & tags, const int tag = -1,
                 const std::string & name = "");
void
removePhysicalGroups(const gmsh::vectorpair & dimTags = gmsh::vectorpair());
void
synchronize();

namespace mesh
{ // Built-in CAD kernel meshing constraints

void
setSize(const gmsh::vectorpair & dimTags, const double size);
void
setTransfiniteCurve(const int tag, const int nPoints,
                    const std::string & meshType = "Progression", const double coef = 1.);
void
setTransfiniteSurface(const int tag, const std::string & arrangement = "Left",
                      const std::vector<int> & cornerTags = std::vector<int>());
void
setTransfiniteVolume(const int tag,
                     const std::vector<int> & cornerTags = std::vector<int>());
void
setRecombine(const int dim, const int tag, const double angle = 45.);
void
setSmoothing(const int dim, const int tag, const int val);
void
setReverse(const int dim, const int tag, const bool val = true);
void
setAlgorithm(const int dim, const int tag, const int val);
void
setSizeFromBoundary(const int dim, const int tag, const int val);

} // namespace mesh

} // namespace geo

namespace occ
{ // OpenCASCADE CAD kernel functions

int
addPoint(const double x, const double y, const double z, const double meshSize = 0.,
         const int tag = -1);
int
addLine(const int startTag, const int endTag, const int tag = -1);
int
addCircleArc(const int startTag, const int centerTag, const int endTag,
             const int tag = -1);
int
addCircle(const double x, const double y, const double z, const double r,
          const int tag = -1, const double angle1 = 0., const double angle2 = 2 * M_PI,
          const std::vector<double> & zAxis = std::vector<double>(),
          const std::vector<double> & xAxis = std::vector<double>());
int
addEllipseArc(const int startTag, const int centerTag, const int majorTag,
              const int endTag, const int tag = -1);
int
addEllipse(const double x, const double y, const double z, const double r1,
           const double r2, const int tag = -1, const double angle1 = 0.,
           const double angle2 = 2 * M_PI,
           const std::vector<double> & zAxis = std::vector<double>(),
           const std::vector<double> & xAxis = std::vector<double>());
int
addSpline(const std::vector<int> & pointTags, const int tag = -1,
          const std::vector<double> & tangents = std::vector<double>());
int
addBSpline(const std::vector<int> & pointTags, const int tag = -1, const int degree = 3,
           const std::vector<double> & weights = std::vector<double>(),
           const std::vector<double> & knots = std::vector<double>(),
           const std::vector<int> & multiplicities = std::vector<int>());
int
addBezier(const std::vector<int> & pointTags, const int tag = -1);
int
addWire(const std::vector<int> & curveTags, const int tag = -1,
        const bool checkClosed = false);
int
addCurveLoop(const std::vector<int> & curveTags, const int tag = -1);
int
addRectangle(const double x, const double y, const double z, const double dx,
             const double dy, const int tag = -1, const double roundedRadius = 0.);
int
addDisk(const double xc, const double yc, const double zc, const double rx,
        const double ry, const int tag = -1,
        const std::vector<double> & zAxis = std::vector<double>(),
        const std::vector<double> & xAxis = std::vector<double>());
int
addPlaneSurface(const std::vector<int> & wireTags, const int tag = -1);
int
addSurfaceFilling(const int wireTag, const int tag = -1,
                  const std::vector<int> & pointTags = std::vector<int>(),
                  const int degree = 3, const int numPointsOnCurves = 15,
                  const int numIter = 2, const bool anisotropic = false,
                  const double tol2d = 0.00001, const double tol3d = 0.0001,
                  const double tolAng = 0.01, const double tolCurv = 0.1,
                  const int maxDegree = 8, const int maxSegments = 9);
int
addBSplineFilling(const int wireTag, const int tag = -1, const std::string & type = "");
int
addBezierFilling(const int wireTag, const int tag = -1, const std::string & type = "");
int
addBSplineSurface(const std::vector<int> & pointTags, const int numPointsU,
                  const int tag = -1, const int degreeU = 3, const int degreeV = 3,
                  const std::vector<double> & weights = std::vector<double>(),
                  const std::vector<double> & knotsU = std::vector<double>(),
                  const std::vector<double> & knotsV = std::vector<double>(),
                  const std::vector<int> & multiplicitiesU = std::vector<int>(),
                  const std::vector<int> & multiplicitiesV = std::vector<int>(),
                  const std::vector<int> & wireTags = std::vector<int>(),
                  const bool wire3D = false);
int
addBezierSurface(const std::vector<int> & pointTags, const int numPointsU,
                 const int tag = -1,
                 const std::vector<int> & wireTags = std::vector<int>(),
                 const bool wire3D = false);
int
addTrimmedSurface(const int surfaceTag,
                  const std::vector<int> & wireTags = std::vector<int>(),
                  const bool wire3D = false, const int tag = -1);
int
addSurfaceLoop(const std::vector<int> & surfaceTags, const int tag = -1,
               const bool sewing = false);
int
addVolume(const std::vector<int> & shellTags, const int tag = -1);
int
addSphere(const double xc, const double yc, const double zc, const double radius,
          const int tag = -1, const double angle1 = -M_PI / 2,
          const double angle2 = M_PI / 2, const double angle3 = 2 * M_PI);
int
addBox(const double x, const double y, const double z, const double dx, const double dy,
       const double dz, const int tag = -1);
int
addCylinder(const double x, const double y, const double z, const double dx,
            const double dy, const double dz, const double r, const int tag = -1,
            const double angle = 2 * M_PI);
int
addCone(const double x, const double y, const double z, const double dx, const double dy,
        const double dz, const double r1, const double r2, const int tag = -1,
        const double angle = 2 * M_PI);
int
addWedge(const double x, const double y, const double z, const double dx, const double dy,
         const double dz, const int tag = -1, const double ltx = 0.,
         const std::vector<double> & zAxis = std::vector<double>());
int
addTorus(const double x, const double y, const double z, const double r1, const double r2,
         const int tag = -1, const double angle = 2 * M_PI,
         const std::vector<double> & zAxis = std::vector<double>());
void
addThruSections(const std::vector<int> & wireTags, gmsh::vectorpair & outDimTags,
                const int tag = -1, const bool makeSolid = true,
                const bool makeRuled = false, const int maxDegree = -1,
                const std::string & continuity = "",
                const std::string & parametrization = "", const bool smoothing = false);
void
addThickSolid(const int volumeTag, const std::vector<int> & excludeSurfaceTags,
              const double offset, gmsh::vectorpair & outDimTags, const int tag = -1);
void
extrude(const gmsh::vectorpair & dimTags, const double dx, const double dy,
        const double dz, gmsh::vectorpair & outDimTags,
        const std::vector<int> & numElements = std::vector<int>(),
        const std::vector<double> & heights = std::vector<double>(),
        const bool recombine = false);
void
revolve(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
        const double ax, const double ay, const double az, const double angle,
        gmsh::vectorpair & outDimTags,
        const std::vector<int> & numElements = std::vector<int>(),
        const std::vector<double> & heights = std::vector<double>(),
        const bool recombine = false);
void
addPipe(const gmsh::vectorpair & dimTags, const int wireTag,
        gmsh::vectorpair & outDimTags, const std::string & trihedron = "");
void
fillet(const std::vector<int> & volumeTags, const std::vector<int> & curveTags,
       const std::vector<double> & radii, gmsh::vectorpair & outDimTags,
       const bool removeVolume = true);
void
chamfer(const std::vector<int> & volumeTags, const std::vector<int> & curveTags,
        const std::vector<int> & surfaceTags, const std::vector<double> & distances,
        gmsh::vectorpair & outDimTags, const bool removeVolume = true);
void
fuse(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
     gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
     const int tag = -1, const bool removeObject = true, const bool removeTool = true);
void
intersect(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
          gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
          const int tag = -1, const bool removeObject = true,
          const bool removeTool = true);
void
cut(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
    gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
    const int tag = -1, const bool removeObject = true, const bool removeTool = true);
void
fragment(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
         gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
         const int tag = -1, const bool removeObject = true,
         const bool removeTool = true);
void
translate(const gmsh::vectorpair & dimTags, const double dx, const double dy,
          const double dz);
void
rotate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double ax, const double ay, const double az, const double angle);
void
dilate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double a, const double b, const double c);
void
mirror(const gmsh::vectorpair & dimTags, const double a, const double b, const double c,
       const double d);
void
symmetrize(const gmsh::vectorpair & dimTags, const double a, const double b,
           const double c, const double d);
void
affineTransform(const gmsh::vectorpair & dimTags,
                const std::vector<double> & affineTransform);
void
copy(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags);
void
remove(const gmsh::vectorpair & dimTags, const bool recursive = false);
void
removeAllDuplicates();
void
healShapes(gmsh::vectorpair & outDimTags,
           const gmsh::vectorpair & dimTags = gmsh::vectorpair(),
           const double tolerance = 1e-8, const bool fixDegenerated = true,
           const bool fixSmallEdges = true, const bool fixSmallFaces = true,
           const bool sewFaces = true, const bool makeSolids = true);
void
convertToNURBS(const gmsh::vectorpair & dimTags);
void
importShapes(const std::string & fileName, gmsh::vectorpair & outDimTags,
             const bool highestDimOnly = true, const std::string & format = "");
void
importShapesNativePointer(const void * shape, gmsh::vectorpair & outDimTags,
                          const bool highestDimOnly = true);
void
getEntities(gmsh::vectorpair & dimTags, const int dim = -1);
void
getEntitiesInBoundingBox(const double xmin, const double ymin, const double zmin,
                         const double xmax, const double ymax, const double zmax,
                         gmsh::vectorpair & dimTags, const int dim = -1);
void
getBoundingBox(const int dim, const int tag, double & xmin, double & ymin, double & zmin,
               double & xmax, double & ymax, double & zmax);
void
getCurveLoops(const int surfaceTag, std::vector<int> & curveLoopTags,
              std::vector<std::vector<int>> & curveTags);
void
getSurfaceLoops(const int volumeTag, std::vector<int> & surfaceLoopTags,
                std::vector<std::vector<int>> & surfaceTags);
void
getMass(const int dim, const int tag, double & mass);
void
getCenterOfMass(const int dim, const int tag, double & x, double & y, double & z);
void
getMatrixOfInertia(const int dim, const int tag, std::vector<double> & mat);
int
getMaxTag(const int dim);
void
setMaxTag(const int dim, const int maxTag);
void
synchronize();

namespace mesh
{ // OpenCASCADE CAD kernel meshing constraints

void
setSize(const gmsh::vectorpair & dimTags, const double size);

} // namespace mesh

} // namespace occ

} // namespace model

namespace view
{ // Post-processing view functions

int
add(const std::string & name, const int tag = -1);
void
remove(const int tag);
int
getIndex(const int tag);
void
getTags(std::vector<int> & tags);
void
addModelData(const int tag, const int step, const std::string & modelName,
             const std::string & dataType, const std::vector<std::size_t> & tags,
             const std::vector<std::vector<double>> & data, const double time = 0.,
             const int numComponents = -1, const int partition = 0);
void
addHomogeneousModelData(const int tag, const int step, const std::string & modelName,
                        const std::string & dataType,
                        const std::vector<std::size_t> & tags,
                        const std::vector<double> & data, const double time = 0.,
                        const int numComponents = -1, const int partition = 0);
void
getModelData(const int tag, const int step, std::string & dataType,
             std::vector<std::size_t> & tags, std::vector<std::vector<double>> & data,
             double & time, int & numComponents);
void
getHomogeneousModelData(const int tag, const int step, std::string & dataType,
                        std::vector<std::size_t> & tags, std::vector<double> & data,
                        double & time, int & numComponents);
void
addListData(const int tag, const std::string & dataType, const int numEle,
            const std::vector<double> & data);
void
getListData(const int tag, std::vector<std::string> & dataType,
            std::vector<int> & numElements, std::vector<std::vector<double>> & data);
void
addListDataString(const int tag, const std::vector<double> & coord,
                  const std::vector<std::string> & data,
                  const std::vector<std::string> & style = std::vector<std::string>());
void
getListDataStrings(const int tag, const int dim, std::vector<double> & coord,
                   std::vector<std::string> & data, std::vector<std::string> & style);
void
setInterpolationMatrices(const int tag, const std::string & type, const int d,
                         const std::vector<double> & coef,
                         const std::vector<double> & exp, const int dGeo = 0,
                         const std::vector<double> & coefGeo = std::vector<double>(),
                         const std::vector<double> & expGeo = std::vector<double>());
int
addAlias(const int refTag, const bool copyOptions = false, const int tag = -1);
void
combine(const std::string & what, const std::string & how, const bool remove = true,
        const bool copyOptions = true);
void
probe(const int tag, const double x, const double y, const double z,
      std::vector<double> & values, double & distance, const int step = -1,
      const int numComp = -1, const bool gradient = false, const double distanceMax = 0.,
      const std::vector<double> & xElemCoord = std::vector<double>(),
      const std::vector<double> & yElemCoord = std::vector<double>(),
      const std::vector<double> & zElemCoord = std::vector<double>(), const int dim = -1);
void
write(const int tag, const std::string & fileName, const bool append = false);
void
setVisibilityPerWindow(const int tag, const int value, const int windowIndex = 0);

namespace option
{ // View option handling functions

void
setNumber(const int tag, const std::string & name, const double value);
void
getNumber(const int tag, const std::string & name, double & value);
void
setString(const int tag, const std::string & name, const std::string & value);
void
getString(const int tag, const std::string & name, std::string & value);
void
setColor(const int tag, const std::string & name, const int r, const int g, const int b,
         const int a = 255);
void
getColor(const int tag, const std::string & name, int & r, int & g, int & b, int & a);
void
copy(const int refTag, const int tag);

} // namespace option

} // namespace view

namespace plugin
{ // Plugin functions

void
setNumber(const std::string & name, const std::string & option, const double value);
void
setString(const std::string & name, const std::string & option,
          const std::string & value);
int
run(const std::string & name);

} // namespace plugin

namespace graphics
{ // Graphics functions

void
draw();

} // namespace graphics

namespace fltk
{ // FLTK graphical user interface functions

void
initialize();
void
finalize();
void
wait(const double time = -1.);
void
update();
void
awake(const std::string & action = "");
void
lock();
void
unlock();
void
run();
int
isAvailable();
int
selectEntities(gmsh::vectorpair & dimTags, const int dim = -1);
int
selectElements(std::vector<std::size_t> & elementTags);
int
selectViews(std::vector<int> & viewTags);
void
splitCurrentWindow(const std::string & how = "v", const double ratio = 0.5);
void
setCurrentWindow(const int windowIndex = 0);
void
setStatusMessage(const std::string & message, const bool graphics = false);
void
showContextWindow(const int dim, const int tag);
void
openTreeItem(const std::string & name);
void
closeTreeItem(const std::string & name);

} // namespace fltk

namespace parser
{ // Parser functions

void
getNames(std::vector<std::string> & names, const std::string & search = "");
void
setNumber(const std::string & name, const std::vector<double> & value);
void
setString(const std::string & name, const std::vector<std::string> & value);
void
getNumber(const std::string & name, std::vector<double> & value);
void
getString(const std::string & name, std::vector<std::string> & value);
void
clear(const std::string & name = "");
void
parse(const std::string & fileName);

} // namespace parser

namespace onelab
{ // ONELAB server functions

void
set(const std::string & data, const std::string & format = "json");
void
get(std::string & data, const std::string & name = "",
    const std::string & format = "json");
void
getNames(std::vector<std::string> & names, const std::string & search = "");
void
setNumber(const std::string & name, const std::vector<double> & value);
void
setString(const std::string & name, const std::vector<std::string> & value);
void
getNumber(const std::string & name, std::vector<double> & value);
void
getString(const std::string & name, std::vector<std::string> & value);
int
getChanged(const std::string & name);
void
setChanged(const std::string & name, const int value);
void
clear(const std::string & name = "");
void
run(const std::string & name = "", const std::string & command = "");

} // namespace onelab

namespace logger
{ // Information logging functions
void
write(const std::string & message, const std::string & level = "info");
void
start();
void
get(std::vector<std::string> & log);
void
stop();
double
getWallTime();
double
getCpuTime();
void
getLastError(std::string & error);

} // namespace logger
} // namespace gmsh
} // namespace um2

#  pragma GCC diagnostic pop
// NOLINTEND(readability*, modernize*)

#endif // UM2_ENABLE_GMSH
