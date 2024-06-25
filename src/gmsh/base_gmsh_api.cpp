#include <um2/config.hpp>

// NOLINTBEGIN(readability*, modernize*) justification: simply a gmsh wrapper (not our
// code).

#if UM2_USE_GMSH
#  include <cstddef>
#  include <functional>
#  include <gmsh.h>
#  include <string>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <vector>

// A wrapper around the gmsh library to make namespace management with um2 gmsh
// related code easier.

namespace um2
{
namespace gmsh
{

void
initialize(int argc, char ** argv, const bool readConfigFiles, const bool run)
{
  ::gmsh::initialize(argc, argv, readConfigFiles, run);
}
int
isInitialized()
{
  return ::gmsh::isInitialized();
}
void
finalize()
{
  ::gmsh::finalize();
}
void
open(const std::string & fileName)
{
  ::gmsh::open(fileName);
}
void
merge(const std::string & fileName)
{
  ::gmsh::merge(fileName);
}
void
write(const std::string & fileName)
{
  ::gmsh::write(fileName);
}
void
clear()
{
  ::gmsh::clear();
}

namespace option
{ // Option handling functions

void
setNumber(const std::string & name, const double value)
{
  ::gmsh::option::setNumber(name, value);
}
void
getNumber(const std::string & name, double & value)
{
  ::gmsh::option::getNumber(name, value);
}
void
setString(const std::string & name, const std::string & value)
{
  ::gmsh::option::setString(name, value);
}
void
getString(const std::string & name, std::string & value)
{
  ::gmsh::option::getString(name, value);
}
void
setColor(const std::string & name, const int r, const int g, const int b, const int a)
{
  ::gmsh::option::setColor(name, r, g, b, a);
}
void
getColor(const std::string & name, int & r, int & g, int & b, int & a)
{
  ::gmsh::option::getColor(name, r, g, b, a);
}

} // namespace option

namespace model
{ // Model functions

void
add(const std::string & name)
{
  ::gmsh::model::add(name);
}
void
remove()
{
  ::gmsh::model::remove();
}
void
list(std::vector<std::string> & names)
{
  ::gmsh::model::list(names);
}
void
getCurrent(std::string & name)
{
  ::gmsh::model::getCurrent(name);
}
void
setCurrent(const std::string & name)
{
  ::gmsh::model::setCurrent(name);
}
void
getFileName(std::string & fileName)
{
  ::gmsh::model::getFileName(fileName);
}
void
setFileName(const std::string & fileName)
{
  ::gmsh::model::setFileName(fileName);
}
void
getEntities(gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::getEntities(dimTags, dim);
}
void
setEntityName(const int dim, const int tag, const std::string & name)
{
  ::gmsh::model::setEntityName(dim, tag, name);
}
void
getEntityName(const int dim, const int tag, std::string & name)
{
  ::gmsh::model::getEntityName(dim, tag, name);
}
void
getPhysicalGroups(gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::getPhysicalGroups(dimTags, dim);
}
void
getEntitiesForPhysicalGroup(const int dim, const int tag, std::vector<int> & tags)
{
  ::gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags);
}
//        void getEntitiesForPhysicalName(const std::string & name, gmsh::vectorpair &
//        dimTags) {
//            ::gmsh::model::getEntitiesForPhysicalName(name, dimTags);
//        }
void
getPhysicalGroupsForEntity(const int dim, const int tag, std::vector<int> & physicalTags)
{
  ::gmsh::model::getPhysicalGroupsForEntity(dim, tag, physicalTags);
}
int
addPhysicalGroup(const int dim, const std::vector<int> & tags, const int tag,
                 const std::string & name)
{
  return ::gmsh::model::addPhysicalGroup(dim, tags, tag, name);
}
void
removePhysicalGroups(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::removePhysicalGroups(dimTags);
}
void
setPhysicalName(const int dim, const int tag, const std::string & name)
{
  ::gmsh::model::setPhysicalName(dim, tag, name);
}
void
removePhysicalName(const std::string & name)
{
  ::gmsh::model::removePhysicalName(name);
}
void
getPhysicalName(const int dim, const int tag, std::string & name)
{
  ::gmsh::model::getPhysicalName(dim, tag, name);
}
void
setTag(const int dim, const int tag, const int newTag)
{
  ::gmsh::model::setTag(dim, tag, newTag);
}
void
getBoundary(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags,
            const bool combined, const bool oriented, const bool recursive)
{
  ::gmsh::model::getBoundary(dimTags, outDimTags, combined, oriented, recursive);
}
void
getAdjacencies(const int dim, const int tag, std::vector<int> & upward,
               std::vector<int> & downward)
{
  ::gmsh::model::getAdjacencies(dim, tag, upward, downward);
}
void
getEntitiesInBoundingBox(const double xmin, const double ymin, const double zmin,
                         const double xmax, const double ymax, const double zmax,
                         gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax, dimTags,
                                          dim);
}
void
getBoundingBox(const int dim, const int tag, double & xmin, double & ymin, double & zmin,
               double & xmax, double & ymax, double & zmax)
{
  ::gmsh::model::getBoundingBox(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax);
}
int
getDimension()
{
  return ::gmsh::model::getDimension();
}
int
addDiscreteEntity(const int dim, const int tag, const std::vector<int> & boundary)
{
  return ::gmsh::model::addDiscreteEntity(dim, tag, boundary);
}
void
removeEntities(const gmsh::vectorpair & dimTags, const bool recursive)
{
  ::gmsh::model::removeEntities(dimTags, recursive);
}
void
removeEntityName(const std::string & name)
{
  ::gmsh::model::removeEntityName(name);
}
void
getType(const int dim, const int tag, std::string & entityType)
{
  ::gmsh::model::getType(dim, tag, entityType);
}
void
getParent(const int dim, const int tag, int & parentDim, int & parentTag)
{
  ::gmsh::model::getParent(dim, tag, parentDim, parentTag);
}
int
getNumberOfPartitions()
{
  return ::gmsh::model::getNumberOfPartitions();
}
void
getPartitions(const int dim, const int tag, std::vector<int> & partitions)
{
  ::gmsh::model::getPartitions(dim, tag, partitions);
}
void
getValue(const int dim, const int tag, const std::vector<double> & parametricCoord,
         std::vector<double> & coord)
{
  ::gmsh::model::getValue(dim, tag, parametricCoord, coord);
}
void
getDerivative(const int dim, const int tag, const std::vector<double> & parametricCoord,
              std::vector<double> & derivatives)
{
  ::gmsh::model::getDerivative(dim, tag, parametricCoord, derivatives);
}
void
getSecondDerivative(const int dim, const int tag,
                    const std::vector<double> & parametricCoord,
                    std::vector<double> & derivatives)
{
  ::gmsh::model::getSecondDerivative(dim, tag, parametricCoord, derivatives);
}
void
getCurvature(const int dim, const int tag, const std::vector<double> & parametricCoord,
             std::vector<double> & curvatures)
{
  ::gmsh::model::getCurvature(dim, tag, parametricCoord, curvatures);
}
void
getPrincipalCurvatures(const int tag, const std::vector<double> & parametricCoord,
                       std::vector<double> & curvatureMax,
                       std::vector<double> & curvatureMin,
                       std::vector<double> & directionMax,
                       std::vector<double> & directionMin)
{
  ::gmsh::model::getPrincipalCurvatures(tag, parametricCoord, curvatureMax, curvatureMin,
                                        directionMax, directionMin);
}
void
getNormal(const int tag, const std::vector<double> & parametricCoord,
          std::vector<double> & normals)
{
  ::gmsh::model::getNormal(tag, parametricCoord, normals);
}
void
getParametrization(const int dim, const int tag, const std::vector<double> & coord,
                   std::vector<double> & parametricCoord)
{
  ::gmsh::model::getParametrization(dim, tag, coord, parametricCoord);
}
void
getParametrizationBounds(const int dim, const int tag, std::vector<double> & min,
                         std::vector<double> & max)
{
  ::gmsh::model::getParametrizationBounds(dim, tag, min, max);
}
int
isInside(const int dim, const int tag, const std::vector<double> & coord,
         const bool parametric)
{
  return ::gmsh::model::isInside(dim, tag, coord, parametric);
}
void
getClosestPoint(const int dim, const int tag, const std::vector<double> & coord,
                std::vector<double> & closestCoord, std::vector<double> & parametricCoord)
{
  ::gmsh::model::getClosestPoint(dim, tag, coord, closestCoord, parametricCoord);
}
void
reparametrizeOnSurface(const int dim, const int tag,
                       const std::vector<double> & parametricCoord, const int surfaceTag,
                       std::vector<double> & surfaceParametricCoord, const int which)
{
  ::gmsh::model::reparametrizeOnSurface(dim, tag, parametricCoord, surfaceTag,
                                        surfaceParametricCoord, which);
}
void
setVisibility(const gmsh::vectorpair & dimTags, const int value, const bool recursive)
{
  ::gmsh::model::setVisibility(dimTags, value, recursive);
}
void
getVisibility(const int dim, const int tag, int & value)
{
  ::gmsh::model::getVisibility(dim, tag, value);
}
void
setVisibilityPerWindow(const int value, const int windowIndex)
{
  ::gmsh::model::setVisibilityPerWindow(value, windowIndex);
}
void
setColor(const gmsh::vectorpair & dimTags, const int r, const int g, const int b,
         const int a, const bool recursive)
{
  ::gmsh::model::setColor(dimTags, r, g, b, a, recursive);
}
void
getColor(const int dim, const int tag, int & r, int & g, int & b, int & a)
{
  ::gmsh::model::getColor(dim, tag, r, g, b, a);
}
void
setCoordinates(const int tag, const double x, const double y, const double z)
{
  ::gmsh::model::setCoordinates(tag, x, y, z);
}
//        void getAttributeNames(std::vector<std::string> & names) {
//            ::gmsh::model::getAttributeNames(names);
//        }
//        void getAttribute(const std::string & name,
//                          std::vector<std::string> & values) {
//            ::gmsh::model::getAttribute(name, values);
//        }
//        void setAttribute(const std::string & name,
//                          const std::vector<std::string> & values) {
//            ::gmsh::model::setAttribute(name, values);
//        }
//        void removeAttribute(const std::string & name) {
//            ::gmsh::model::removeAttribute(name);
//        }

namespace mesh
{ // Mesh functions

void
generate(const int dim)
{
  ::gmsh::model::mesh::generate(dim);
}
void
partition(const int numPart, const std::vector<std::size_t> & elementTags,
          const std::vector<int> & partitions)
{
  ::gmsh::model::mesh::partition(numPart, elementTags, partitions);
}
void
unpartition()
{
  ::gmsh::model::mesh::unpartition();
}
void
optimize(const std::string & method, const bool force, const int niter,
         const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::optimize(method, force, niter, dimTags);
}
void
recombine()
{
  ::gmsh::model::mesh::recombine();
}
void
refine()
{
  ::gmsh::model::mesh::refine();
}
void
setOrder(const int order)
{
  ::gmsh::model::mesh::setOrder(order);
}
void
getLastEntityError(gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::getLastEntityError(dimTags);
}
void
getLastNodeError(std::vector<std::size_t> & nodeTags)
{
  ::gmsh::model::mesh::getLastNodeError(nodeTags);
}
void
clear(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::clear(dimTags);
}
void
reverse(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::reverse(dimTags);
}
void
affineTransform(const std::vector<double> & affineTransform,
                const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::affineTransform(affineTransform, dimTags);
}
void
getNodes(std::vector<std::size_t> & nodeTags, std::vector<double> & coord,
         std::vector<double> & parametricCoord, const int dim, const int tag,
         const bool includeBoundary, const bool returnParametricCoord)
{
  ::gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord, dim, tag,
                                includeBoundary, returnParametricCoord);
}
void
getNodesByElementType(const int elementType, std::vector<std::size_t> & nodeTags,
                      std::vector<double> & coord, std::vector<double> & parametricCoord,
                      const int tag, const bool returnParametricCoord)
{
  ::gmsh::model::mesh::getNodesByElementType(elementType, nodeTags, coord,
                                             parametricCoord, tag, returnParametricCoord);
}
void
getNode(const std::size_t nodeTag, std::vector<double> & coord,
        std::vector<double> & parametricCoord, int & dim, int & tag)
{
  ::gmsh::model::mesh::getNode(nodeTag, coord, parametricCoord, dim, tag);
}
void
setNode(const std::size_t nodeTag, const std::vector<double> & coord,
        const std::vector<double> & parametricCoord)
{
  ::gmsh::model::mesh::setNode(nodeTag, coord, parametricCoord);
}
void
rebuildNodeCache(const bool onlyIfNecessary)
{
  ::gmsh::model::mesh::rebuildNodeCache(onlyIfNecessary);
}
void
rebuildElementCache(const bool onlyIfNecessary)
{
  ::gmsh::model::mesh::rebuildElementCache(onlyIfNecessary);
}
void
getNodesForPhysicalGroup(const int dim, const int tag,
                         std::vector<std::size_t> & nodeTags, std::vector<double> & coord)
{
  ::gmsh::model::mesh::getNodesForPhysicalGroup(dim, tag, nodeTags, coord);
}
void
getMaxNodeTag(std::size_t & maxTag)
{
  ::gmsh::model::mesh::getMaxNodeTag(maxTag);
}
void
addNodes(const int dim, const int tag, const std::vector<std::size_t> & nodeTags,
         const std::vector<double> & coord, const std::vector<double> & parametricCoord)
{
  ::gmsh::model::mesh::addNodes(dim, tag, nodeTags, coord, parametricCoord);
}
void
reclassifyNodes()
{
  ::gmsh::model::mesh::reclassifyNodes();
}
void
relocateNodes(const int dim, const int tag)
{
  ::gmsh::model::mesh::relocateNodes(dim, tag);
}
void
getElements(std::vector<int> & elementTypes,
            std::vector<std::vector<std::size_t>> & elementTags,
            std::vector<std::vector<std::size_t>> & nodeTags, const int dim,
            const int tag)
{
  ::gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTags, dim, tag);
}
void
getElement(const std::size_t elementTag, int & elementType,
           std::vector<std::size_t> & nodeTags, int & dim, int & tag)
{
  ::gmsh::model::mesh::getElement(elementTag, elementType, nodeTags, dim, tag);
}
void
getElementByCoordinates(const double x, const double y, const double z,
                        std::size_t & elementTag, int & elementType,
                        std::vector<std::size_t> & nodeTags, double & u, double & v,
                        double & w, const int dim, const bool strict)
{
  ::gmsh::model::mesh::getElementByCoordinates(x, y, z, elementTag, elementType, nodeTags,
                                               u, v, w, dim, strict);
}
void
getElementsByCoordinates(const double x, const double y, const double z,
                         std::vector<std::size_t> & elementTags, const int dim,
                         const bool strict)
{
  ::gmsh::model::mesh::getElementsByCoordinates(x, y, z, elementTags, dim, strict);
}
void
getLocalCoordinatesInElement(const std::size_t elementTag, const double x, const double y,
                             const double z, double & u, double & v, double & w)
{
  ::gmsh::model::mesh::getLocalCoordinatesInElement(elementTag, x, y, z, u, v, w);
}
void
getElementTypes(std::vector<int> & elementTypes, const int dim, const int tag)
{
  ::gmsh::model::mesh::getElementTypes(elementTypes, dim, tag);
}
int
getElementType(const std::string & familyName, const int order, const bool serendip)
{
  return ::gmsh::model::mesh::getElementType(familyName, order, serendip);
}
void
getElementProperties(const int elementType, std::string & elementName, int & dim,
                     int & order, int & numNodes, std::vector<double> & localNodeCoord,
                     int & numPrimaryNodes)
{
  ::gmsh::model::mesh::getElementProperties(elementType, elementName, dim, order,
                                            numNodes, localNodeCoord, numPrimaryNodes);
}
void
getElementsByType(const int elementType, std::vector<std::size_t> & elementTags,
                  std::vector<std::size_t> & nodeTags, const int tag,
                  const std::size_t task, const std::size_t numTasks)
{
  ::gmsh::model::mesh::getElementsByType(elementType, elementTags, nodeTags, tag, task,
                                         numTasks);
}
void
getMaxElementTag(std::size_t & maxTag)
{
  ::gmsh::model::mesh::getMaxElementTag(maxTag);
}
void
preallocateElementsByType(const int elementType, const bool elementTag,
                          const bool nodeTag, std::vector<std::size_t> & elementTags,
                          std::vector<std::size_t> & nodeTags, const int tag)
{
  ::gmsh::model::mesh::preallocateElementsByType(elementType, elementTag, nodeTag,
                                                 elementTags, nodeTags, tag);
}
void
getElementQualities(const std::vector<std::size_t> & elementTags,
                    std::vector<double> & elementsQuality,
                    const std::string & qualityName, const std::size_t task,
                    const std::size_t numTasks)
{
  ::gmsh::model::mesh::getElementQualities(elementTags, elementsQuality, qualityName,
                                           task, numTasks);
}
void
addElements(const int dim, const int tag, const std::vector<int> & elementTypes,
            const std::vector<std::vector<std::size_t>> & elementTags,
            const std::vector<std::vector<std::size_t>> & nodeTags)
{
  ::gmsh::model::mesh::addElements(dim, tag, elementTypes, elementTags, nodeTags);
}
void
addElementsByType(const int tag, const int elementType,
                  const std::vector<std::size_t> & elementTags,
                  const std::vector<std::size_t> & nodeTags)
{
  ::gmsh::model::mesh::addElementsByType(tag, elementType, elementTags, nodeTags);
}
void
getIntegrationPoints(const int elementType, const std::string & integrationType,
                     std::vector<double> & localCoord, std::vector<double> & weights)
{
  ::gmsh::model::mesh::getIntegrationPoints(elementType, integrationType, localCoord,
                                            weights);
}
void
getJacobians(const int elementType, const std::vector<double> & localCoord,
             std::vector<double> & jacobians, std::vector<double> & determinants,
             std::vector<double> & coord, const int tag, const std::size_t task,
             const std::size_t numTasks)
{
  ::gmsh::model::mesh::getJacobians(elementType, localCoord, jacobians, determinants,
                                    coord, tag, task, numTasks);
}
void
preallocateJacobians(const int elementType, const int numEvaluationPoints,
                     const bool allocateJacobians, const bool allocateDeterminants,
                     const bool allocateCoord, std::vector<double> & jacobians,
                     std::vector<double> & determinants, std::vector<double> & coord,
                     const int tag)
{
  ::gmsh::model::mesh::preallocateJacobians(
      elementType, numEvaluationPoints, allocateJacobians, allocateDeterminants,
      allocateCoord, jacobians, determinants, coord, tag);
}
void
getJacobian(const std::size_t elementTag, const std::vector<double> & localCoord,
            std::vector<double> & jacobians, std::vector<double> & determinants,
            std::vector<double> & coord)
{
  ::gmsh::model::mesh::getJacobian(elementTag, localCoord, jacobians, determinants,
                                   coord);
}
void
getBasisFunctions(const int elementType, const std::vector<double> & localCoord,
                  const std::string & functionSpaceType, int & numComponents,
                  std::vector<double> & basisFunctions, int & numOrientations,
                  const std::vector<int> & wantedOrientations)
{
  ::gmsh::model::mesh::getBasisFunctions(elementType, localCoord, functionSpaceType,
                                         numComponents, basisFunctions, numOrientations,
                                         wantedOrientations);
}
void
getBasisFunctionsOrientation(const int elementType, const std::string & functionSpaceType,
                             std::vector<int> & basisFunctionsOrientation, const int tag,
                             const std::size_t task, const std::size_t numTasks)
{
  ::gmsh::model::mesh::getBasisFunctionsOrientation(
      elementType, functionSpaceType, basisFunctionsOrientation, tag, task, numTasks);
}
void
getBasisFunctionsOrientationForElement(const std::size_t elementTag,
                                       const std::string & functionSpaceType,
                                       int & basisFunctionsOrientation)
{
  ::gmsh::model::mesh::getBasisFunctionsOrientationForElement(
      elementTag, functionSpaceType, basisFunctionsOrientation);
}
int
getNumberOfOrientations(const int elementType, const std::string & functionSpaceType)
{
  return ::gmsh::model::mesh::getNumberOfOrientations(elementType, functionSpaceType);
}
void
preallocateBasisFunctionsOrientation(const int elementType,
                                     std::vector<int> & basisFunctionsOrientation,
                                     const int tag)
{
  ::gmsh::model::mesh::preallocateBasisFunctionsOrientation(
      elementType, basisFunctionsOrientation, tag);
}
void
getEdges(const std::vector<std::size_t> & nodeTags, std::vector<std::size_t> & edgeTags,
         std::vector<int> & edgeOrientations)
{
  ::gmsh::model::mesh::getEdges(nodeTags, edgeTags, edgeOrientations);
}
void
getFaces(const int faceType, const std::vector<std::size_t> & nodeTags,
         std::vector<std::size_t> & faceTags, std::vector<int> & faceOrientations)
{
  ::gmsh::model::mesh::getFaces(faceType, nodeTags, faceTags, faceOrientations);
}
void
createEdges(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::createEdges(dimTags);
}
void
createFaces(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::createFaces(dimTags);
}
void
getAllEdges(std::vector<std::size_t> & edgeTags, std::vector<std::size_t> & edgeNodes)
{
  ::gmsh::model::mesh::getAllEdges(edgeTags, edgeNodes);
}
void
getAllFaces(const int faceType, std::vector<std::size_t> & faceTags,
            std::vector<std::size_t> & faceNodes)
{
  ::gmsh::model::mesh::getAllFaces(faceType, faceTags, faceNodes);
}
void
addEdges(const std::vector<std::size_t> & edgeTags,
         const std::vector<std::size_t> & edgeNodes)
{
  ::gmsh::model::mesh::addEdges(edgeTags, edgeNodes);
}
void
addFaces(const int faceType, const std::vector<std::size_t> & faceTags,
         const std::vector<std::size_t> & faceNodes)
{
  ::gmsh::model::mesh::addFaces(faceType, faceTags, faceNodes);
}
void
getKeys(const int elementType, const std::string & functionSpaceType,
        std::vector<int> & typeKeys, std::vector<std::size_t> & entityKeys,
        std::vector<double> & coord, const int tag, const bool returnCoord)
{
  ::gmsh::model::mesh::getKeys(elementType, functionSpaceType, typeKeys, entityKeys,
                               coord, tag, returnCoord);
}
void
getKeysForElement(const std::size_t elementTag, const std::string & functionSpaceType,
                  std::vector<int> & typeKeys, std::vector<std::size_t> & entityKeys,
                  std::vector<double> & coord, const bool returnCoord)
{
  ::gmsh::model::mesh::getKeysForElement(elementTag, functionSpaceType, typeKeys,
                                         entityKeys, coord, returnCoord);
}
int
getNumberOfKeys(const int elementType, const std::string & functionSpaceType)
{
  return ::gmsh::model::mesh::getNumberOfKeys(elementType, functionSpaceType);
}
void
getKeysInformation(const std::vector<int> & typeKeys,
                   const std::vector<std::size_t> & entityKeys, const int elementType,
                   const std::string & functionSpaceType, gmsh::vectorpair & infoKeys)
{
  ::gmsh::model::mesh::getKeysInformation(typeKeys, entityKeys, elementType,
                                          functionSpaceType, infoKeys);
}
void
getBarycenters(const int elementType, const int tag, const bool fast, const bool primary,
               std::vector<double> & barycenters, const std::size_t task,
               const std::size_t numTasks)
{
  ::gmsh::model::mesh::getBarycenters(elementType, tag, fast, primary, barycenters, task,
                                      numTasks);
}
void
preallocateBarycenters(const int elementType, std::vector<double> & barycenters,
                       const int tag)
{
  ::gmsh::model::mesh::preallocateBarycenters(elementType, barycenters, tag);
}
void
getElementEdgeNodes(const int elementType, std::vector<std::size_t> & nodeTags,
                    const int tag, const bool primary, const std::size_t task,
                    const std::size_t numTasks)
{
  ::gmsh::model::mesh::getElementEdgeNodes(elementType, nodeTags, tag, primary, task,
                                           numTasks);
}
void
getElementFaceNodes(const int elementType, const int faceType,
                    std::vector<std::size_t> & nodeTags, const int tag,
                    const bool primary, const std::size_t task,
                    const std::size_t numTasks)
{
  ::gmsh::model::mesh::getElementFaceNodes(elementType, faceType, nodeTags, tag, primary,
                                           task, numTasks);
}
void
getGhostElements(const int dim, const int tag, std::vector<std::size_t> & elementTags,
                 std::vector<int> & partitions)
{
  ::gmsh::model::mesh::getGhostElements(dim, tag, elementTags, partitions);
}
void
setSize(const gmsh::vectorpair & dimTags, const double size)
{
  ::gmsh::model::mesh::setSize(dimTags, size);
}
void
getSizes(const gmsh::vectorpair & dimTags, std::vector<double> & sizes)
{
  ::gmsh::model::mesh::getSizes(dimTags, sizes);
}
void
setSizeAtParametricPoints(const int dim, const int tag,
                          const std::vector<double> & parametricCoord,
                          const std::vector<double> & sizes)
{
  ::gmsh::model::mesh::setSizeAtParametricPoints(dim, tag, parametricCoord, sizes);
}
void
setSizeCallback(std::function<double(int, int, double, double, double, double)> callback)
{
  // NOLINTNEXTLINE(performance-unnecessary-value-param) justification: this is a TPL
  ::gmsh::model::mesh::setSizeCallback(callback);
}
void
removeSizeCallback()
{
  ::gmsh::model::mesh::removeSizeCallback();
}
void
setTransfiniteCurve(const int tag, const int numNodes, const std::string & meshType,
                    const double coef)
{
  ::gmsh::model::mesh::setTransfiniteCurve(tag, numNodes, meshType, coef);
}
void
setTransfiniteSurface(const int tag, const std::string & arrangement,
                      const std::vector<int> & cornerTags)
{
  ::gmsh::model::mesh::setTransfiniteSurface(tag, arrangement, cornerTags);
}
void
setTransfiniteVolume(const int tag, const std::vector<int> & cornerTags)
{
  ::gmsh::model::mesh::setTransfiniteVolume(tag, cornerTags);
}
void
setTransfiniteAutomatic(const gmsh::vectorpair & dimTags, const double cornerAngle,
                        const bool recombine)
{
  ::gmsh::model::mesh::setTransfiniteAutomatic(dimTags, cornerAngle, recombine);
}
void
setRecombine(const int dim, const int tag, const double angle)
{
  ::gmsh::model::mesh::setRecombine(dim, tag, angle);
}
void
setSmoothing(const int dim, const int tag, const int val)
{
  ::gmsh::model::mesh::setSmoothing(dim, tag, val);
}
void
setReverse(const int dim, const int tag, const bool val)
{
  ::gmsh::model::mesh::setReverse(dim, tag, val);
}
void
setAlgorithm(const int dim, const int tag, const int val)
{
  ::gmsh::model::mesh::setAlgorithm(dim, tag, val);
}
void
setSizeFromBoundary(const int dim, const int tag, const int val)
{
  ::gmsh::model::mesh::setSizeFromBoundary(dim, tag, val);
}
void
setCompound(const int dim, const std::vector<int> & tags)
{
  ::gmsh::model::mesh::setCompound(dim, tags);
}
void
setOutwardOrientation(const int tag)
{
  ::gmsh::model::mesh::setOutwardOrientation(tag);
}
void
removeConstraints(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::removeConstraints(dimTags);
}
void
embed(const int dim, const std::vector<int> & tags, const int inDim, const int inTag)
{
  ::gmsh::model::mesh::embed(dim, tags, inDim, inTag);
}
void
removeEmbedded(const gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::mesh::removeEmbedded(dimTags, dim);
}
void
getEmbedded(const int dim, const int tag, gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::getEmbedded(dim, tag, dimTags);
}
void
reorderElements(const int elementType, const int tag,
                const std::vector<std::size_t> & ordering)
{
  ::gmsh::model::mesh::reorderElements(elementType, tag, ordering);
}
void
renumberNodes()
{
  ::gmsh::model::mesh::renumberNodes();
}
void
renumberElements()
{
  ::gmsh::model::mesh::renumberElements();
}
void
setPeriodic(const int dim, const std::vector<int> & tags,
            const std::vector<int> & tagsMaster,
            const std::vector<double> & affineTransform)
{
  ::gmsh::model::mesh::setPeriodic(dim, tags, tagsMaster, affineTransform);
}
void
getPeriodic(const int dim, const std::vector<int> & tags, std::vector<int> & tagMaster)
{
  ::gmsh::model::mesh::getPeriodic(dim, tags, tagMaster);
}
void
getPeriodicNodes(const int dim, const int tag, int & tagMaster,
                 std::vector<std::size_t> & nodeTags,
                 std::vector<std::size_t> & nodeTagsMaster,
                 std::vector<double> & affineTransform, const bool includeHighOrderNodes)
{
  ::gmsh::model::mesh::getPeriodicNodes(dim, tag, tagMaster, nodeTags, nodeTagsMaster,
                                        affineTransform, includeHighOrderNodes);
}
void
getPeriodicKeys(const int elementType, const std::string & functionSpaceType,
                const int tag, int & tagMaster, std::vector<int> & typeKeys,
                std::vector<int> & typeKeysMaster, std::vector<std::size_t> & entityKeys,
                std::vector<std::size_t> & entityKeysMaster, std::vector<double> & coord,
                std::vector<double> & coordMaster, const bool returnCoord)
{
  ::gmsh::model::mesh::getPeriodicKeys(elementType, functionSpaceType, tag, tagMaster,
                                       typeKeys, typeKeysMaster, entityKeys,
                                       entityKeysMaster, coord, coordMaster, returnCoord);
}
void
importStl()
{
  ::gmsh::model::mesh::importStl();
}
void
getDuplicateNodes(std::vector<std::size_t> & tags, const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::getDuplicateNodes(tags, dimTags);
}
void
removeDuplicateNodes(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::removeDuplicateNodes(dimTags);
}
//            void removeDuplicateElements(const gmsh::vectorpair & dimTags ) {
//                ::gmsh::model::mesh::removeDuplicateElements(dimTags);
//            }
void
splitQuadrangles(const double quality, const int tag)
{
  ::gmsh::model::mesh::splitQuadrangles(quality, tag);
}
void
setVisibility(const std::vector<std::size_t> & elementTags, const int value)
{
  ::gmsh::model::mesh::setVisibility(elementTags, value);
}
void
classifySurfaces(const double angle, const bool boundary, const bool forReparametrization,
                 const double curveAngle, const bool exportDiscrete)
{
  ::gmsh::model::mesh::classifySurfaces(angle, boundary, forReparametrization, curveAngle,
                                        exportDiscrete);
}
void
createGeometry(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::mesh::createGeometry(dimTags);
}
void
createTopology(const bool makeSimplyConnected, const bool exportDiscrete)
{
  ::gmsh::model::mesh::createTopology(makeSimplyConnected, exportDiscrete);
}
void
addHomologyRequest(const std::string & type, const std::vector<int> & domainTags,
                   const std::vector<int> & subdomainTags, const std::vector<int> & dims)
{
  ::gmsh::model::mesh::addHomologyRequest(type, domainTags, subdomainTags, dims);
}
void
clearHomologyRequests()
{
  ::gmsh::model::mesh::clearHomologyRequests();
}
//            void computeHomology(gmsh::vectorpair & dimTags) {
//                ::gmsh::model::mesh::computeHomology(dimTags);
//            }
void
computeCrossField(std::vector<int> & viewTags)
{
  ::gmsh::model::mesh::computeCrossField(viewTags);
}
void
triangulate(const std::vector<double> & coord, std::vector<std::size_t> & tri)
{
  ::gmsh::model::mesh::triangulate(coord, tri);
}
void
tetrahedralize(const std::vector<double> & coord, std::vector<std::size_t> & tetra)
{
  ::gmsh::model::mesh::tetrahedralize(coord, tetra);
}

namespace field
{ // Mesh size field functions

int
add(const std::string & fieldType, const int tag)
{
  return ::gmsh::model::mesh::field::add(fieldType, tag);
}
void
remove(const int tag)
{
  ::gmsh::model::mesh::field::remove(tag);
}
void
list(std::vector<int> & tags)
{
  ::gmsh::model::mesh::field::list(tags);
}
void
getType(const int tag, std::string & fileType)
{
  ::gmsh::model::mesh::field::getType(tag, fileType);
}
void
setNumber(const int tag, const std::string & option, const double value)
{
  ::gmsh::model::mesh::field::setNumber(tag, option, value);
}
void
getNumber(const int tag, const std::string & option, double & value)
{
  ::gmsh::model::mesh::field::getNumber(tag, option, value);
}
void
setString(const int tag, const std::string & option, const std::string & value)
{
  ::gmsh::model::mesh::field::setString(tag, option, value);
}
void
getString(const int tag, const std::string & option, std::string & value)
{
  ::gmsh::model::mesh::field::getString(tag, option, value);
}
void
setNumbers(const int tag, const std::string & option, const std::vector<double> & values)
{
  ::gmsh::model::mesh::field::setNumbers(tag, option, values);
}
void
getNumbers(const int tag, const std::string & option, std::vector<double> & values)
{
  ::gmsh::model::mesh::field::getNumbers(tag, option, values);
}
void
setAsBackgroundMesh(const int tag)
{
  ::gmsh::model::mesh::field::setAsBackgroundMesh(tag);
}
void
setAsBoundaryLayer(const int tag)
{
  ::gmsh::model::mesh::field::setAsBoundaryLayer(tag);
}

} // namespace field

} // namespace mesh

namespace geo
{ // Built-in CAD kernel functions

int
addPoint(const double x, const double y, const double z, const double meshSize,
         const int tag)
{
  return ::gmsh::model::geo::addPoint(x, y, z, meshSize, tag);
}
int
addLine(const int startTag, const int endTag, const int tag)
{
  return ::gmsh::model::geo::addLine(startTag, endTag, tag);
}
int
addCircleArc(const int startTag, const int centerTag, const int endTag, const int tag,
             const double nx, const double ny, const double nz)
{
  return ::gmsh::model::geo::addCircleArc(startTag, centerTag, endTag, tag, nx, ny, nz);
}
int
addEllipseArc(const int startTag, const int centerTag, const int majorTag,
              const int endTag, const int tag, const double nx, const double ny,
              const double nz)
{
  return ::gmsh::model::geo::addEllipseArc(startTag, centerTag, majorTag, endTag, tag, nx,
                                           ny, nz);
}
//            int addSpline(const std::vector<int> & pointTags,
//                          const int tag ) {
//                return ::gmsh::model::geo::addSpline(pointTags, tag);
//            }
int
addBSpline(const std::vector<int> & pointTags, const int tag)
{
  return ::gmsh::model::geo::addBSpline(pointTags, tag);
}
int
addBezier(const std::vector<int> & pointTags, const int tag)
{
  return ::gmsh::model::geo::addBezier(pointTags, tag);
}
int
addPolyline(const std::vector<int> & pointTags, const int tag)
{
  return ::gmsh::model::geo::addPolyline(pointTags, tag);
}
int
addCompoundSpline(const std::vector<int> & curveTags, const int numIntervals,
                  const int tag)
{
  return ::gmsh::model::geo::addCompoundSpline(curveTags, numIntervals, tag);
}
int
addCompoundBSpline(const std::vector<int> & curveTags, const int numIntervals,
                   const int tag)
{
  return ::gmsh::model::geo::addCompoundBSpline(curveTags, numIntervals, tag);
}
int
addCurveLoop(const std::vector<int> & curveTags, const int tag, const bool reorient)
{
  return ::gmsh::model::geo::addCurveLoop(curveTags, tag, reorient);
}
void
addCurveLoops(const std::vector<int> & curveTags, std::vector<int> & tags)
{
  ::gmsh::model::geo::addCurveLoops(curveTags, tags);
}
int
addPlaneSurface(const std::vector<int> & wireTags, const int tag)
{
  return ::gmsh::model::geo::addPlaneSurface(wireTags, tag);
}
int
addSurfaceFilling(const std::vector<int> & wireTags, const int tag,
                  const int sphereCenterTag)
{
  return ::gmsh::model::geo::addSurfaceFilling(wireTags, tag, sphereCenterTag);
}
void
revolve(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
        const double ax, const double ay, const double az, const double angle,
        gmsh::vectorpair & outDimTags, const std::vector<int> & numElements,
        const std::vector<double> & heights, const bool recombine)
{
  ::gmsh::model::geo::revolve(dimTags, x, y, z, ax, ay, az, angle, outDimTags,
                              numElements, heights, recombine);
}
void
twist(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
      const double dx, const double dy, const double dz, const double ax, const double ay,
      const double az, const double angle, gmsh::vectorpair & outDimTags,
      const std::vector<int> & numElements, const std::vector<double> & heights,
      const bool recombine)
{
  ::gmsh::model::geo::twist(dimTags, x, y, z, dx, dy, dz, ax, ay, az, angle, outDimTags,
                            numElements, heights, recombine);
}
void
extrudeBoundaryLayer(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags,
                     const std::vector<int> & numElements,
                     const std::vector<double> & heights, const bool recombine,
                     const bool second, const int viewIndex)
{
  ::gmsh::model::geo::extrudeBoundaryLayer(dimTags, outDimTags, numElements, heights,
                                           recombine, second, viewIndex);
}
void
translate(const gmsh::vectorpair & dimTags, const double dx, const double dy,
          const double dz)
{
  ::gmsh::model::geo::translate(dimTags, dx, dy, dz);
}
void
rotate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double ax, const double ay, const double az, const double angle)
{
  ::gmsh::model::geo::rotate(dimTags, x, y, z, ax, ay, az, angle);
}
void
dilate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double a, const double b, const double c)
{
  ::gmsh::model::geo::dilate(dimTags, x, y, z, a, b, c);
}
void
mirror(const gmsh::vectorpair & dimTags, const double a, const double b, const double c,
       const double d)
{
  ::gmsh::model::geo::mirror(dimTags, a, b, c, d);
}
void
symmetrize(const gmsh::vectorpair & dimTags, const double a, const double b,
           const double c, const double d)
{
  ::gmsh::model::geo::symmetrize(dimTags, a, b, c, d);
}
void
copy(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags)
{
  ::gmsh::model::geo::copy(dimTags, outDimTags);
}
void
remove(const gmsh::vectorpair & dimTags, const bool recursive)
{
  ::gmsh::model::geo::remove(dimTags, recursive);
}
void
removeAllDuplicates()
{
  ::gmsh::model::geo::removeAllDuplicates();
}
void
splitCurve(const int tag, const std::vector<int> & pointTags,
           std::vector<int> & curveTags)
{
  ::gmsh::model::geo::splitCurve(tag, pointTags, curveTags);
}
int
getMaxTag(const int dim)
{
  return ::gmsh::model::geo::getMaxTag(dim);
}
void
setMaxTag(const int dim, const int maxTag)
{
  ::gmsh::model::geo::setMaxTag(dim, maxTag);
}
int
addPhysicalGroup(const int dim, const std::vector<int> & tags, const int tag,
                 const std::string & name)
{
  return ::gmsh::model::geo::addPhysicalGroup(dim, tags, tag, name);
}
void
removePhysicalGroups(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::geo::removePhysicalGroups(dimTags);
}
void
synchronize()
{
  ::gmsh::model::geo::synchronize();
}

namespace mesh
{ // Built-in CAD kernel meshing constraints

void
setSize(const gmsh::vectorpair & dimTags, const double size)
{
  ::gmsh::model::geo::mesh::setSize(dimTags, size);
}
void
setTransfiniteCurve(const int tag, const int nPoints, const std::string & meshType,
                    const double coef)
{
  ::gmsh::model::geo::mesh::setTransfiniteCurve(tag, nPoints, meshType, coef);
}
void
setTransfiniteSurface(const int tag, const std::string & arrangement,
                      const std::vector<int> & cornerTags)
{
  ::gmsh::model::geo::mesh::setTransfiniteSurface(tag, arrangement, cornerTags);
}
void
setTransfiniteVolume(const int tag, const std::vector<int> & cornerTags)
{
  ::gmsh::model::geo::mesh::setTransfiniteVolume(tag, cornerTags);
}
void
setRecombine(const int dim, const int tag, const double angle)
{
  ::gmsh::model::geo::mesh::setRecombine(dim, tag, angle);
}
void
setSmoothing(const int dim, const int tag, const int val)
{
  ::gmsh::model::geo::mesh::setSmoothing(dim, tag, val);
}
void
setReverse(const int dim, const int tag, const bool val)
{
  ::gmsh::model::geo::mesh::setReverse(dim, tag, val);
}
void
setAlgorithm(const int dim, const int tag, const int val)
{
  ::gmsh::model::geo::mesh::setAlgorithm(dim, tag, val);
}
void
setSizeFromBoundary(const int dim, const int tag, const int val)
{
  ::gmsh::model::geo::mesh::setSizeFromBoundary(dim, tag, val);
}

} // namespace mesh

} // namespace geo

namespace occ
{ // OpenCASCADE CAD kernel functions

int
addPoint(const double x, const double y, const double z, const double meshSize,
         const int tag)
{
  return ::gmsh::model::occ::addPoint(x, y, z, meshSize, tag);
}
int
addLine(const int startTag, const int endTag, const int tag)
{
  return ::gmsh::model::occ::addLine(startTag, endTag, tag);
}
int
addCircleArc(const int startTag, const int centerTag, const int endTag, const int tag)
{
  return ::gmsh::model::occ::addCircleArc(startTag, centerTag, endTag, tag);
}
int
addCircle(const double x, const double y, const double z, const double r, const int tag,
          const double angle1, const double angle2, const std::vector<double> & zAxis,
          const std::vector<double> & xAxis)
{
  return ::gmsh::model::occ::addCircle(x, y, z, r, tag, angle1, angle2, zAxis, xAxis);
}
int
addEllipseArc(const int startTag, const int centerTag, const int majorTag,
              const int endTag, const int tag)
{
  return ::gmsh::model::occ::addEllipseArc(startTag, centerTag, majorTag, endTag, tag);
}
int
addEllipse(const double x, const double y, const double z, const double r1,
           const double r2, const int tag, const double angle1, const double angle2,
           const std::vector<double> & zAxis, const std::vector<double> & xAxis)
{
  return ::gmsh::model::occ::addEllipse(x, y, z, r1, r2, tag, angle1, angle2, zAxis,
                                        xAxis);
}
//            int addSpline(const std::vector<int> & pointTags,
//                          const int tag ,
//                          const std::vector<double> & tangents ) {
//                return ::gmsh::model::occ::addSpline(pointTags, tag, tangents);
//            }
int
addBSpline(const std::vector<int> & pointTags, const int tag, const int degree,
           const std::vector<double> & weights, const std::vector<double> & knots,
           const std::vector<int> & multiplicities)
{
  return ::gmsh::model::occ::addBSpline(pointTags, tag, degree, weights, knots,
                                        multiplicities);
}
int
addBezier(const std::vector<int> & pointTags, const int tag)
{
  return ::gmsh::model::occ::addBezier(pointTags, tag);
}
int
addWire(const std::vector<int> & curveTags, const int tag, const bool checkClosed)
{
  return ::gmsh::model::occ::addWire(curveTags, tag, checkClosed);
}
int
addCurveLoop(const std::vector<int> & curveTags, const int tag)
{
  return ::gmsh::model::occ::addCurveLoop(curveTags, tag);
}
int
addRectangle(const double x, const double y, const double z, const double dx,
             const double dy, const int tag, const double roundedRadius)
{
  return ::gmsh::model::occ::addRectangle(x, y, z, dx, dy, tag, roundedRadius);
}
int
addDisk(const double xc, const double yc, const double zc, const double rx,
        const double ry, const int tag, const std::vector<double> & zAxis,
        const std::vector<double> & xAxis)
{
  return ::gmsh::model::occ::addDisk(xc, yc, zc, rx, ry, tag, zAxis, xAxis);
}
int
addPlaneSurface(const std::vector<int> & wireTags, const int tag)
{
  return ::gmsh::model::occ::addPlaneSurface(wireTags, tag);
}
int
addSurfaceFilling(const int wireTag, const int tag, const std::vector<int> & pointTags,
                  const int degree, const int numPointsOnCurves, const int numIter,
                  const bool anisotropic, const double tol2d, const double tol3d,
                  const double tolAng, const double tolCurv, const int maxDegree,
                  const int maxSegments)
{
  return ::gmsh::model::occ::addSurfaceFilling(
      wireTag, tag, pointTags, degree, numPointsOnCurves, numIter, anisotropic, tol2d,
      tol3d, tolAng, tolCurv, maxDegree, maxSegments);
}
int
addBSplineFilling(const int wireTag, const int tag, const std::string & type)
{
  return ::gmsh::model::occ::addBSplineFilling(wireTag, tag, type);
}
int
addBezierFilling(const int wireTag, const int tag, const std::string & type)
{
  return ::gmsh::model::occ::addBezierFilling(wireTag, tag, type);
}
int
addBSplineSurface(const std::vector<int> & pointTags, const int numPointsU, const int tag,
                  const int degreeU, const int degreeV,
                  const std::vector<double> & weights, const std::vector<double> & knotsU,
                  const std::vector<double> & knotsV,
                  const std::vector<int> & multiplicitiesU,
                  const std::vector<int> & multiplicitiesV,
                  const std::vector<int> & wireTags, const bool wire3D)
{
  return ::gmsh::model::occ::addBSplineSurface(
      pointTags, numPointsU, tag, degreeU, degreeV, weights, knotsU, knotsV,
      multiplicitiesU, multiplicitiesV, wireTags, wire3D);
}
int
addBezierSurface(const std::vector<int> & pointTags, const int numPointsU, const int tag,
                 const std::vector<int> & wireTags, const bool wire3D)
{
  return ::gmsh::model::occ::addBezierSurface(pointTags, numPointsU, tag, wireTags,
                                              wire3D);
}
int
addTrimmedSurface(const int surfaceTag, const std::vector<int> & wireTags,
                  const bool wire3D, const int tag)
{
  return ::gmsh::model::occ::addTrimmedSurface(surfaceTag, wireTags, wire3D, tag);
}
int
addSurfaceLoop(const std::vector<int> & surfaceTags, const int tag, const bool sewing)
{
  return ::gmsh::model::occ::addSurfaceLoop(surfaceTags, tag, sewing);
}
int
addVolume(const std::vector<int> & shellTags, const int tag)
{
  return ::gmsh::model::occ::addVolume(shellTags, tag);
}
int
addSphere(const double xc, const double yc, const double zc, const double radius,
          const int tag, const double angle1, const double angle2, const double angle3)
{
  return ::gmsh::model::occ::addSphere(xc, yc, zc, radius, tag, angle1, angle2, angle3);
}
int
addBox(const double x, const double y, const double z, const double dx, const double dy,
       const double dz, const int tag)
{
  return ::gmsh::model::occ::addBox(x, y, z, dx, dy, dz, tag);
}
int
addCylinder(const double x, const double y, const double z, const double dx,
            const double dy, const double dz, const double r, const int tag,
            const double angle)
{
  return ::gmsh::model::occ::addCylinder(x, y, z, dx, dy, dz, r, tag, angle);
}
int
addCone(const double x, const double y, const double z, const double dx, const double dy,
        const double dz, const double r1, const double r2, const int tag,
        const double angle)
{
  return ::gmsh::model::occ::addCone(x, y, z, dx, dy, dz, r1, r2, tag, angle);
}
int
addWedge(const double x, const double y, const double z, const double dx, const double dy,
         const double dz, const int tag, const double ltx,
         const std::vector<double> & zAxis)
{
  return ::gmsh::model::occ::addWedge(x, y, z, dx, dy, dz, tag, ltx, zAxis);
}
int
addTorus(const double x, const double y, const double z, const double r1, const double r2,
         const int tag, const double angle, const std::vector<double> & zAxis)
{
  return ::gmsh::model::occ::addTorus(x, y, z, r1, r2, tag, angle, zAxis);
}
//            void addThruSections(const std::vector<int> & wireTags,
//                                 gmsh::vectorpair & outDimTags,
//                                 const int tag ,
//                                 const bool makeSolid ,
//                                 const bool makeRuled ,
//                                 const int maxDegree ,
//                                 const std::string & continuity ,
//                                 const std::string & parametrization ,
//                                 const bool smoothing ) {
//                ::gmsh::model::occ::addThruSections(wireTags, outDimTags, tag,
//                makeSolid, makeRuled,
//                                                    maxDegree, continuity,
//                                                    parametrization, smoothing);
//            }
void
addThickSolid(const int volumeTag, const std::vector<int> & excludeSurfaceTags,
              const double offset, gmsh::vectorpair & outDimTags, const int tag)
{
  ::gmsh::model::occ::addThickSolid(volumeTag, excludeSurfaceTags, offset, outDimTags,
                                    tag);
}
void
extrude(const gmsh::vectorpair & dimTags, const double dx, const double dy,
        const double dz, gmsh::vectorpair & outDimTags,
        const std::vector<int> & numElements, const std::vector<double> & heights,
        const bool recombine)
{
  ::gmsh::model::occ::extrude(dimTags, dx, dy, dz, outDimTags, numElements, heights,
                              recombine);
}
void
revolve(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
        const double ax, const double ay, const double az, const double angle,
        gmsh::vectorpair & outDimTags, const std::vector<int> & numElements,
        const std::vector<double> & heights, const bool recombine)
{
  ::gmsh::model::occ::revolve(dimTags, x, y, z, ax, ay, az, angle, outDimTags,
                              numElements, heights, recombine);
}
void
addPipe(const gmsh::vectorpair & dimTags, const int wireTag,
        gmsh::vectorpair & outDimTags, const std::string & trihedron)
{
  ::gmsh::model::occ::addPipe(dimTags, wireTag, outDimTags, trihedron);
}
void
fillet(const std::vector<int> & volumeTags, const std::vector<int> & curveTags,
       const std::vector<double> & radii, gmsh::vectorpair & outDimTags,
       const bool removeVolume)
{
  ::gmsh::model::occ::fillet(volumeTags, curveTags, radii, outDimTags, removeVolume);
}
void
chamfer(const std::vector<int> & volumeTags, const std::vector<int> & curveTags,
        const std::vector<int> & surfaceTags, const std::vector<double> & distances,
        gmsh::vectorpair & outDimTags, const bool removeVolume)
{
  ::gmsh::model::occ::chamfer(volumeTags, curveTags, surfaceTags, distances, outDimTags,
                              removeVolume);
}
void
fuse(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
     gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
     const int tag, const bool removeObject, const bool removeTool)
{
  ::gmsh::model::occ::fuse(objectDimTags, toolDimTags, outDimTags, outDimTagsMap, tag,
                           removeObject, removeTool);
}
void
intersect(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
          gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
          const int tag, const bool removeObject, const bool removeTool)
{
  ::gmsh::model::occ::intersect(objectDimTags, toolDimTags, outDimTags, outDimTagsMap,
                                tag, removeObject, removeTool);
}
void
cut(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
    gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
    const int tag, const bool removeObject, const bool removeTool)
{
  ::gmsh::model::occ::cut(objectDimTags, toolDimTags, outDimTags, outDimTagsMap, tag,
                          removeObject, removeTool);
}
void
fragment(const gmsh::vectorpair & objectDimTags, const gmsh::vectorpair & toolDimTags,
         gmsh::vectorpair & outDimTags, std::vector<gmsh::vectorpair> & outDimTagsMap,
         const int tag, const bool removeObject, const bool removeTool)
{
  ::gmsh::model::occ::fragment(objectDimTags, toolDimTags, outDimTags, outDimTagsMap, tag,
                               removeObject, removeTool);
}
void
translate(const gmsh::vectorpair & dimTags, const double dx, const double dy,
          const double dz)
{
  ::gmsh::model::occ::translate(dimTags, dx, dy, dz);
}
void
rotate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double ax, const double ay, const double az, const double angle)
{
  ::gmsh::model::occ::rotate(dimTags, x, y, z, ax, ay, az, angle);
}
void
dilate(const gmsh::vectorpair & dimTags, const double x, const double y, const double z,
       const double a, const double b, const double c)
{
  ::gmsh::model::occ::dilate(dimTags, x, y, z, a, b, c);
}
void
mirror(const gmsh::vectorpair & dimTags, const double a, const double b, const double c,
       const double d)
{
  ::gmsh::model::occ::mirror(dimTags, a, b, c, d);
}
void
symmetrize(const gmsh::vectorpair & dimTags, const double a, const double b,
           const double c, const double d)
{
  ::gmsh::model::occ::symmetrize(dimTags, a, b, c, d);
}
void
affineTransform(const gmsh::vectorpair & dimTags,
                const std::vector<double> & affineTransform)
{
  ::gmsh::model::occ::affineTransform(dimTags, affineTransform);
}
void
copy(const gmsh::vectorpair & dimTags, gmsh::vectorpair & outDimTags)
{
  ::gmsh::model::occ::copy(dimTags, outDimTags);
}
void
remove(const gmsh::vectorpair & dimTags, const bool recursive)
{
  ::gmsh::model::occ::remove(dimTags, recursive);
}
void
removeAllDuplicates()
{
  ::gmsh::model::occ::removeAllDuplicates();
}
void
healShapes(gmsh::vectorpair & outDimTags, const gmsh::vectorpair & dimTags,
           const double tolerance, const bool fixDegenerated, const bool fixSmallEdges,
           const bool fixSmallFaces, const bool sewFaces, const bool makeSolids)
{
  ::gmsh::model::occ::healShapes(outDimTags, dimTags, tolerance, fixDegenerated,
                                 fixSmallEdges, fixSmallFaces, sewFaces, makeSolids);
}
void
convertToNURBS(const gmsh::vectorpair & dimTags)
{
  ::gmsh::model::occ::convertToNURBS(dimTags);
}
void
importShapes(const std::string & fileName, gmsh::vectorpair & outDimTags,
             const bool highestDimOnly, const std::string & format)
{
  ::gmsh::model::occ::importShapes(fileName, outDimTags, highestDimOnly, format);
}
void
importShapesNativePointer(const void * shape, gmsh::vectorpair & outDimTags,
                          const bool highestDimOnly)
{
  ::gmsh::model::occ::importShapesNativePointer(shape, outDimTags, highestDimOnly);
}
void
getEntities(gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::occ::getEntities(dimTags, dim);
}
void
getEntitiesInBoundingBox(const double xmin, const double ymin, const double zmin,
                         const double xmax, const double ymax, const double zmax,
                         gmsh::vectorpair & dimTags, const int dim)
{
  ::gmsh::model::occ::getEntitiesInBoundingBox(xmin, ymin, zmin, xmax, ymax, zmax,
                                               dimTags, dim);
}
void
getBoundingBox(const int dim, const int tag, double & xmin, double & ymin, double & zmin,
               double & xmax, double & ymax, double & zmax)
{
  ::gmsh::model::occ::getBoundingBox(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax);
}
void
getCurveLoops(const int surfaceTag, std::vector<int> & curveLoopTags,
              std::vector<std::vector<int>> & curveTags)
{
  ::gmsh::model::occ::getCurveLoops(surfaceTag, curveLoopTags, curveTags);
}
void
getSurfaceLoops(const int volumeTag, std::vector<int> & surfaceLoopTags,
                std::vector<std::vector<int>> & surfaceTags)
{
  ::gmsh::model::occ::getSurfaceLoops(volumeTag, surfaceLoopTags, surfaceTags);
}
void
getMass(const int dim, const int tag, double & mass)
{
  ::gmsh::model::occ::getMass(dim, tag, mass);
}
void
getCenterOfMass(const int dim, const int tag, double & x, double & y, double & z)
{
  ::gmsh::model::occ::getCenterOfMass(dim, tag, x, y, z);
}
void
getMatrixOfInertia(const int dim, const int tag, std::vector<double> & mat)
{
  ::gmsh::model::occ::getMatrixOfInertia(dim, tag, mat);
}
int
getMaxTag(const int dim)
{
  return ::gmsh::model::occ::getMaxTag(dim);
}
void
setMaxTag(const int dim, const int maxTag)
{
  ::gmsh::model::occ::setMaxTag(dim, maxTag);
}
void
synchronize()
{
  ::gmsh::model::occ::synchronize();
}

namespace mesh
{ // OpenCASCADE CAD kernel meshing constraints

void
setSize(const gmsh::vectorpair & dimTags, const double size)
{
  ::gmsh::model::occ::mesh::setSize(dimTags, size);
}

} // namespace mesh

} // namespace occ

} // namespace model

namespace view
{ // Post-processing view functions

int
add(const std::string & name, const int tag)
{
  return ::gmsh::view::add(name, tag);
}
void
remove(const int tag)
{
  ::gmsh::view::remove(tag);
}
int
getIndex(const int tag)
{
  return ::gmsh::view::getIndex(tag);
}
void
getTags(std::vector<int> & tags)
{
  ::gmsh::view::getTags(tags);
}
void
addModelData(const int tag, const int step, const std::string & modelName,
             const std::string & dataType, const std::vector<std::size_t> & tags,
             const std::vector<std::vector<double>> & data, const double time,
             const int numComponents, const int partition)
{
  ::gmsh::view::addModelData(tag, step, modelName, dataType, tags, data, time,
                             numComponents, partition);
}
void
addHomogeneousModelData(const int tag, const int step, const std::string & modelName,
                        const std::string & dataType,
                        const std::vector<std::size_t> & tags,
                        const std::vector<double> & data, const double time,
                        const int numComponents, const int partition)
{
  ::gmsh::view::addHomogeneousModelData(tag, step, modelName, dataType, tags, data, time,
                                        numComponents, partition);
}
void
getModelData(const int tag, const int step, std::string & dataType,
             std::vector<std::size_t> & tags, std::vector<std::vector<double>> & data,
             double & time, int & numComponents)
{
  ::gmsh::view::getModelData(tag, step, dataType, tags, data, time, numComponents);
}
void
getHomogeneousModelData(const int tag, const int step, std::string & dataType,
                        std::vector<std::size_t> & tags, std::vector<double> & data,
                        double & time, int & numComponents)
{
  ::gmsh::view::getHomogeneousModelData(tag, step, dataType, tags, data, time,
                                        numComponents);
}
void
addListData(const int tag, const std::string & dataType, const int numEle,
            const std::vector<double> & data)
{
  ::gmsh::view::addListData(tag, dataType, numEle, data);
}
void
getListData(const int tag, std::vector<std::string> & dataType,
            std::vector<int> & numElements, std::vector<std::vector<double>> & data)
{
  ::gmsh::view::getListData(tag, dataType, numElements, data);
}
void
addListDataString(const int tag, const std::vector<double> & coord,
                  const std::vector<std::string> & data,
                  const std::vector<std::string> & style)
{
  ::gmsh::view::addListDataString(tag, coord, data, style);
}
void
getListDataStrings(const int tag, const int dim, std::vector<double> & coord,
                   std::vector<std::string> & data, std::vector<std::string> & style)
{
  ::gmsh::view::getListDataStrings(tag, dim, coord, data, style);
}
void
setInterpolationMatrices(const int tag, const std::string & type, const int d,
                         const std::vector<double> & coef,
                         const std::vector<double> & exp, const int dGeo,
                         const std::vector<double> & coefGeo,
                         const std::vector<double> & expGeo)
{
  ::gmsh::view::setInterpolationMatrices(tag, type, d, coef, exp, dGeo, coefGeo, expGeo);
}
int
addAlias(const int refTag, const bool copyOptions, const int tag)
{
  return ::gmsh::view::addAlias(refTag, copyOptions, tag);
}
void
combine(const std::string & what, const std::string & how, const bool remove,
        const bool copyOptions)
{
  ::gmsh::view::combine(what, how, remove, copyOptions);
}
void
probe(const int tag, const double x, const double y, const double z,
      std::vector<double> & values, double & distance, const int step, const int numComp,
      const bool gradient, const double distanceMax,
      const std::vector<double> & xElemCoord, const std::vector<double> & yElemCoord,
      const std::vector<double> & zElemCoord, const int dim)
{
  ::gmsh::view::probe(tag, x, y, z, values, distance, step, numComp, gradient,
                      distanceMax, xElemCoord, yElemCoord, zElemCoord, dim);
}
void
write(const int tag, const std::string & fileName, const bool append)
{
  ::gmsh::view::write(tag, fileName, append);
}
void
setVisibilityPerWindow(const int tag, const int value, const int windowIndex)
{
  ::gmsh::view::setVisibilityPerWindow(tag, value, windowIndex);
}

namespace option
{ // View option handling functions

void
setNumber(const int tag, const std::string & name, const double value)
{
  ::gmsh::view::option::setNumber(tag, name, value);
}
void
getNumber(const int tag, const std::string & name, double & value)
{
  ::gmsh::view::option::getNumber(tag, name, value);
}
void
setString(const int tag, const std::string & name, const std::string & value)
{
  ::gmsh::view::option::setString(tag, name, value);
}
void
getString(const int tag, const std::string & name, std::string & value)
{
  ::gmsh::view::option::getString(tag, name, value);
}
void
setColor(const int tag, const std::string & name, const int r, const int g, const int b,
         const int a)
{
  ::gmsh::view::option::setColor(tag, name, r, g, b, a);
}
void
getColor(const int tag, const std::string & name, int & r, int & g, int & b, int & a)
{
  ::gmsh::view::option::getColor(tag, name, r, g, b, a);
}
void
copy(const int refTag, const int tag)
{
  ::gmsh::view::option::copy(refTag, tag);
}

} // namespace option

} // namespace view

namespace plugin
{ // Plugin functions

void
setNumber(const std::string & name, const std::string & option, const double value)
{
  ::gmsh::plugin::setNumber(name, option, value);
}
void
setString(const std::string & name, const std::string & option, const std::string & value)
{
  ::gmsh::plugin::setString(name, option, value);
}
int
run(const std::string & name)
{
  return ::gmsh::plugin::run(name);
}

} // namespace plugin

namespace graphics
{ // Graphics functions

void
draw()
{
  ::gmsh::graphics::draw();
}

} // namespace graphics

namespace fltk
{ // FLTK graphical user interface functions

void
initialize()
{
  ::gmsh::fltk::initialize();
}
void
finalize()
{
  ::gmsh::fltk::finalize();
}
void
wait(const double time)
{
  ::gmsh::fltk::wait(time);
}
void
update()
{
  ::gmsh::fltk::update();
}
void
awake(const std::string & action)
{
  ::gmsh::fltk::awake(action);
}
void
lock()
{
  ::gmsh::fltk::lock();
}
void
unlock()
{
  ::gmsh::fltk::unlock();
}
void
run()
{
  ::gmsh::fltk::run();
}
int
isAvailable()
{
  return ::gmsh::fltk::isAvailable();
}
int
selectEntities(gmsh::vectorpair & dimTags, const int dim)
{
  return ::gmsh::fltk::selectEntities(dimTags, dim);
}
int
selectElements(std::vector<std::size_t> & elementTags)
{
  return ::gmsh::fltk::selectElements(elementTags);
}
int
selectViews(std::vector<int> & viewTags)
{
  return ::gmsh::fltk::selectViews(viewTags);
}
void
splitCurrentWindow(const std::string & how, const double ratio)
{
  ::gmsh::fltk::splitCurrentWindow(how, ratio);
}
void
setCurrentWindow(const int windowIndex)
{
  ::gmsh::fltk::setCurrentWindow(windowIndex);
}
void
setStatusMessage(const std::string & message, const bool graphics)
{
  ::gmsh::fltk::setStatusMessage(message, graphics);
}
void
showContextWindow(const int dim, const int tag)
{
  ::gmsh::fltk::showContextWindow(dim, tag);
}
void
openTreeItem(const std::string & name)
{
  ::gmsh::fltk::openTreeItem(name);
}
void
closeTreeItem(const std::string & name)
{
  ::gmsh::fltk::closeTreeItem(name);
}

} // namespace fltk

namespace parser
{ // Parser functions

void
getNames(std::vector<std::string> & names, const std::string & search)
{
  ::gmsh::parser::getNames(names, search);
}
void
setNumber(const std::string & name, const std::vector<double> & value)
{
  ::gmsh::parser::setNumber(name, value);
}
void
setString(const std::string & name, const std::vector<std::string> & value)
{
  ::gmsh::parser::setString(name, value);
}
void
getNumber(const std::string & name, std::vector<double> & value)
{
  ::gmsh::parser::getNumber(name, value);
}
void
getString(const std::string & name, std::vector<std::string> & value)
{
  ::gmsh::parser::getString(name, value);
}
void
clear(const std::string & name)
{
  ::gmsh::parser::clear(name);
}
void
parse(const std::string & fileName)
{
  ::gmsh::parser::parse(fileName);
}

} // namespace parser

namespace onelab
{ // ONELAB server functions

void
set(const std::string & data, const std::string & format)
{
  ::gmsh::onelab::set(data, format);
}
void
get(std::string & data, const std::string & name, const std::string & format)
{
  ::gmsh::onelab::get(data, name, format);
}
void
getNames(std::vector<std::string> & names, const std::string & search)
{
  ::gmsh::onelab::getNames(names, search);
}
void
setNumber(const std::string & name, const std::vector<double> & value)
{
  ::gmsh::onelab::setNumber(name, value);
}
void
setString(const std::string & name, const std::vector<std::string> & value)
{
  ::gmsh::onelab::setString(name, value);
}
void
getNumber(const std::string & name, std::vector<double> & value)
{
  ::gmsh::onelab::getNumber(name, value);
}
void
getString(const std::string & name, std::vector<std::string> & value)
{
  ::gmsh::onelab::getString(name, value);
}
int
getChanged(const std::string & name)
{
  return ::gmsh::onelab::getChanged(name);
}
void
setChanged(const std::string & name, const int value)
{
  ::gmsh::onelab::setChanged(name, value);
}
void
clear(const std::string & name)
{
  ::gmsh::onelab::clear(name);
}
void
run(const std::string & name, const std::string & command)
{
  ::gmsh::onelab::run(name, command);
}

} // namespace onelab

namespace logger
{ // Information logging functions
void
write(const std::string & message, const std::string & level)
{
  ::gmsh::logger::write(message, level);
}
void
start()
{
  ::gmsh::logger::start();
}
void
get(std::vector<std::string> & log)
{
  ::gmsh::logger::get(log);
}
void
stop()
{
  ::gmsh::logger::stop();
}
double
getWallTime()
{
  return ::gmsh::logger::getWallTime();
}
double
getCpuTime()
{
  return ::gmsh::logger::getCpuTime();
}
void
getLastError(std::string & error)
{
  ::gmsh::logger::getLastError(error);
}

} // namespace logger
} // namespace gmsh
} // namespace um2
#endif // UM2_USE_GMSH
// NOLINTEND(readability*, modernize*)
