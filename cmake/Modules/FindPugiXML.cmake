find_library(PUGIXML_LIB "pugixml" REQUIRED)
find_path(PUGIXML_INC "pugixml.hpp" REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PugiXML
  "PugiXML could not be found."
  PUGIXML_LIB PUGIXML_INC)
