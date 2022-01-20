
vvas_xboundingbox = library('vvas_xstereopipeline',
  'src/vvas_xstereopipeline.cpp',
  cpp_args : [gst_plugins_ivas_args, '-std=c++11'],
  include_directories : [configinc],
  dependencies : [gst_dep, jansson_dep, ivasutils_dep, gstivasinfermeta_dep, opencv_dep],
  install : true,
)
