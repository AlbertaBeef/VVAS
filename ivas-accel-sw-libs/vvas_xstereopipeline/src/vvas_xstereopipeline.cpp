/*
 * Copyright 2022 Avnet, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <math.h>
#include <ivas/ivas_kernel.h>
#include <gst/ivas/gstinferencemeta.h>

#include "vvas_xstereopipeline.hpp"

int log_level = LOG_LEVEL_WARNING;

using namespace cv;
using namespace std;

struct color
{
  unsigned int blue;
  unsigned int green;
  unsigned int red;
};

struct overlayframe_info
{
  IVASFrame *inframe;
  IVASFrame *outframe;
  Mat image;
  Mat I420image;
  Mat NV12image;
  Mat lumaImg;
  Mat chromaImg;
};



struct vvas_xstereopipelinepriv
{
  unsigned int frame_counter;
  float font_size;
  unsigned int font;
  int line_thickness;
  int x_offset;
  int y_offset;
  color text_color;
  struct overlayframe_info frameinfo;
};

/* Get y and uv color components corresponding to givne RGB color */
void
convert_rgb_to_yuv_clrs (color clr, unsigned char *y, unsigned short *uv)
{
  Mat YUVmat;
  Mat BGRmat (2, 2, CV_8UC3, Scalar (clr.red, clr.green, clr.blue));
  cvtColor (BGRmat, YUVmat, cv::COLOR_BGR2YUV_I420);
  *y = YUVmat.at < uchar > (0, 0);
  *uv = YUVmat.at < uchar > (2, 0) << 8 | YUVmat.at < uchar > (2, 1);
  return;
}

static gboolean
display_frame_counter (gpointer kpriv_ptr)
{
  vvas_xstereopipelinepriv *kpriv = (vvas_xstereopipelinepriv *) kpriv_ptr;
  struct overlayframe_info *frameinfo = &(kpriv->frameinfo);
  LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

  char text_string[1024];
  Size textsize;

  sprintf (text_string, "Frame : %04d", kpriv->frame_counter++ );

  LOG_MESSAGE (LOG_LEVEL_INFO, text_string );

  /* Check whether the frame is NV12 or BGR and act accordingly */
  if (frameinfo->outframe->props.fmt == IVAS_VFMT_Y_UV8_420) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Displaying frame counter for NV12 image");
    unsigned char yScalar;
    unsigned short uvScalar;

    /* Display frame counter */
    convert_rgb_to_yuv_clrs (kpriv->text_color, &yScalar, &uvScalar);
    putText (frameinfo->lumaImg, text_string, cv::Point (kpriv->x_offset, kpriv->y_offset), 
             kpriv->font, kpriv->font_size, Scalar (yScalar), 1, 1);
    putText (frameinfo->chromaImg, text_string, cv::Point (kpriv->x_offset / 2, kpriv->y_offset / 2),
             kpriv->font, kpriv->font_size / 2, Scalar (uvScalar), 1, 1);

  } else if (frameinfo->outframe->props.fmt == IVAS_VFMT_BGR8) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Displaying frame counter for BGR image");

    /* Display frame counter */
    putText (frameinfo->image, text_string, cv::Point (kpriv->x_offset, kpriv->y_offset),
             kpriv->font, kpriv->font_size, 
             Scalar (kpriv->text_color.blue, kpriv->text_color.green, kpriv->text_color.red), 1, 1);
  } else if (frameinfo->outframe->props.fmt == IVAS_VFMT_YUYV8) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Displaying frame counter for YUV 4:2:2 image");
    unsigned char yScalar;
    unsigned short uvScalar;

    /* Display frame counter */
    convert_rgb_to_yuv_clrs (kpriv->text_color, &yScalar, &uvScalar);
    putText (frameinfo->image, text_string, cv::Point (kpriv->x_offset, kpriv->y_offset),
             kpriv->font, kpriv->font_size, 
             Scalar (yScalar, uvScalar), 1, 1);
  }

  return FALSE;
}

extern "C"
{
  int32_t xlnx_kernel_init (IVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

    vvas_xstereopipelinepriv *kpriv =
        (vvas_xstereopipelinepriv *) calloc (1, sizeof (vvas_xstereopipelinepriv));

    json_t *jconfig = handle->kernel_config;
    json_t *val, *karray = NULL;

    /* Initialize config params with default values */
    log_level = LOG_LEVEL_WARNING;
    kpriv->frame_counter = 0;
    kpriv->font_size = 1;
    kpriv->font = 3;
    kpriv->line_thickness = 2;
    kpriv->x_offset = 16;
    kpriv->y_offset = 32;
    kpriv->text_color = {255, 255, 255};


    /* parse config */

    val = json_object_get (jconfig, "debug_level");
    if (!val || !json_is_integer (val))
        log_level = LOG_LEVEL_WARNING;
    else
        log_level = json_integer_value (val);

    val = json_object_get (jconfig, "font_size");
    if (!val || !json_is_integer (val))
        kpriv->font_size = 0.5;
    else
        kpriv->font_size = json_integer_value (val);

    val = json_object_get (jconfig, "font");
    if (!val || !json_is_integer (val))
        kpriv->font = 0;
    else
        kpriv->font = json_integer_value (val);

    val = json_object_get (jconfig, "thickness");
    if (!val || !json_is_integer (val))
        kpriv->line_thickness = 1;
    else
        kpriv->line_thickness = json_integer_value (val);

    val = json_object_get (jconfig, "x_offset");
    if (!val || !json_is_integer (val))
        kpriv->x_offset = 10;
    else
        kpriv->x_offset = json_integer_value (val);

    val = json_object_get (jconfig, "y_offset");
    if (!val || !json_is_integer (val))
        kpriv->y_offset = 10;
    else
        kpriv->y_offset = json_integer_value (val);

    /* get label color array */
    karray = json_object_get (jconfig, "text_color");
    if (!karray)
    {
      LOG_MESSAGE (LOG_LEVEL_ERROR, "failed to find text_color");
      return -1;
    } else
    {
      kpriv->text_color.blue =
          json_integer_value (json_object_get (karray, "blue"));
      kpriv->text_color.green =
          json_integer_value (json_object_get (karray, "green"));
      kpriv->text_color.red =
          json_integer_value (json_object_get (karray, "red"));
    }

    handle->kernel_priv = (void *) kpriv;
    return 0;
  }

  uint32_t xlnx_kernel_deinit (IVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    vvas_xstereopipelinepriv *kpriv = (vvas_xstereopipelinepriv *) handle->kernel_priv;

    if (kpriv)
      free (kpriv);

    return 0;
  }


  uint32_t xlnx_kernel_start (IVASKernel * handle, int start,
      IVASFrame * input[MAX_NUM_OBJECT], IVASFrame * output[MAX_NUM_OBJECT])
  {
    int plane_id;
    unsigned int buffer_size;

    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

    vvas_xstereopipelinepriv *kpriv = (vvas_xstereopipelinepriv *) handle->kernel_priv;
    struct overlayframe_info *frameinfo = &(kpriv->frameinfo);

    if ( input[0] == NULL )
    {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input frame is not available");
      return false;
    }
    if ( output[0] == NULL )
    {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "output frame is not available");
      return false;
    }

    frameinfo->inframe = input[0];
    frameinfo->outframe = output[0];

    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input format = %d", frameinfo->inframe->props.fmt);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input height = %d", frameinfo->inframe->props.height);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input width  = %d", frameinfo->inframe->props.width);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input stride = %d", frameinfo->inframe->props.stride);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input memtype= %d", frameinfo->inframe->mem_type);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, " input planes = %d", frameinfo->inframe->n_planes);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output format = %d", frameinfo->outframe->props.fmt);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output height = %d", frameinfo->outframe->props.height);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output width  = %d", frameinfo->outframe->props.width);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output stride = %d", frameinfo->outframe->props.stride);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output planes = %d", frameinfo->outframe->n_planes);
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "output memtype= %d", frameinfo->outframe->mem_type);

    if (frameinfo->inframe->props.fmt != frameinfo->outframe->props.fmt) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input/output frames are not of same format");
      return false;
    }
    if (frameinfo->inframe->props.height != frameinfo->outframe->props.height) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input/output frames are not of same height");
      return false;
    }
    //if (frameinfo->inframe->props.width != frameinfo->outframe->props.width) {
    //  LOG_MESSAGE (LOG_LEVEL_DEBUG, "input/output frames are not of same width");
    //  return false;
    //}
    //if (frameinfo->inframe->props.stride != frameinfo->outframe->props.stride) {
    //  LOG_MESSAGE (LOG_LEVEL_DEBUG, "input/output frames are not of same stride");
    //  return false;
    //}
    if (frameinfo->inframe->props.width != frameinfo->outframe->props.width * 2) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input frame width is not 2x of output frame width");
      return false;
    }
    if (frameinfo->inframe->props.stride != frameinfo->outframe->props.stride * 2) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input frame stride is not 2x of output frame stride");
      return false;
    }
    if ( frameinfo->inframe->n_planes != frameinfo->outframe->n_planes )
    {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "input/output planes are not same");
      return false;
    }

    //for ( plane_id = 0; plane_id < frameinfo->inframe->n_planes; plane_id++ )
    //{
    //  LOG_MESSAGE (LOG_LEVEL_DEBUG, "copying plane %d of size %d", plane_id, frameinfo->inframe->size[plane_id] );
    //  memcpy( frameinfo->outframe->vaddr[plane_id], frameinfo->inframe->vaddr[plane_id], frameinfo->inframe->size[plane_id] );
    //}
    if ( frameinfo->outframe->n_planes == 1 )
    {
      char *inPtr  = (char *)frameinfo->inframe->vaddr[0];
      char *outPtr = (char *)frameinfo->outframe->vaddr[0];
      int row;
      for ( row = 0; row < frameinfo->outframe->props.height; row++ )
      {
        buffer_size = frameinfo->outframe->props.stride * frameinfo->outframe->props.height;
        memcpy( outPtr, inPtr, frameinfo->outframe->props.stride );

        inPtr  += frameinfo->inframe->props.stride;
        outPtr += frameinfo->outframe->props.stride;
      }
    }

    char *outdata = (char *) frameinfo->outframe->vaddr[0];
    char *lumaBuf = (char *) frameinfo->outframe->vaddr[0];
    char *chromaBuf = (char *) frameinfo->outframe->vaddr[1];

    if (frameinfo->outframe->props.fmt == IVAS_VFMT_Y_UV8_420) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Output frame is in NV12 format\n");
      frameinfo->lumaImg.create (input[0]->props.height, input[0]->props.stride,
          CV_8UC1);
      frameinfo->lumaImg.data = (unsigned char *) lumaBuf;
      frameinfo->chromaImg.create (output[0]->props.height / 2, output[0]->props.stride / 2, CV_16UC1);
      frameinfo->chromaImg.data = (unsigned char *) chromaBuf;
    } else if (frameinfo->outframe->props.fmt == IVAS_VFMT_BGR8) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Output frame is in BGR format\n");
      frameinfo->image.create (output[0]->props.height, output[0]->props.stride / 3, CV_8UC3);
      frameinfo->image.data = (unsigned char *) outdata;
    } else if (frameinfo->outframe->props.fmt == IVAS_VFMT_YUYV8) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Output frame is in YUV 4:2:2 format\n");
      frameinfo->image.create (output[0]->props.height, output[0]->props.stride / 2, CV_8UC2);
      frameinfo->image.data = (unsigned char *) outdata;
    } else {
      LOG_MESSAGE (LOG_LEVEL_WARNING, "Output frame is in Unsupported color format\n");
      return 0;
    }

    display_frame_counter( kpriv );

    return 0;
  }


  int32_t xlnx_kernel_done (IVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    return 0;
  }
}
