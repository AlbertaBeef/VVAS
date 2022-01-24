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

#include "vvas_xpassthru.hpp"

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
  Mat image;
  Mat I420image;
  Mat NV12image;
  Mat lumaImg;
  Mat chromaImg;
  int y_offset;
};



struct vvas_xpassthrupriv
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
  vvas_xpassthrupriv *kpriv = (vvas_xpassthrupriv *) kpriv_ptr;
  struct overlayframe_info *frameinfo = &(kpriv->frameinfo);
  LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

  char text_string[1024];
  Size textsize;

  sprintf (text_string, "Frame : %04d", kpriv->frame_counter++ );

  LOG_MESSAGE (LOG_LEVEL_INFO, text_string );

  /* Check whether the frame is NV12 or BGR and act accordingly */
  if (frameinfo->inframe->props.fmt == IVAS_VFMT_Y_UV8_420) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Displaying frame counter for NV12 image");
    unsigned char yScalar;
    unsigned short uvScalar;

    /* Display frame counter */
    convert_rgb_to_yuv_clrs (kpriv->text_color, &yScalar, &uvScalar);
    putText (frameinfo->lumaImg, text_string, cv::Point (kpriv->x_offset, kpriv->y_offset), 
             kpriv->font, kpriv->font_size, Scalar (yScalar), 1, 1);
    putText (frameinfo->chromaImg, text_string, cv::Point (kpriv->x_offset / 2, kpriv->y_offset / 2),
             kpriv->font, kpriv->font_size / 2, Scalar (uvScalar), 1, 1);

  } else if (frameinfo->inframe->props.fmt == IVAS_VFMT_BGR8) {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "Displaying frame counter for BGR image");

    /* Display frame counter */
    putText (frameinfo->image, text_string, cv::Point (kpriv->x_offset, kpriv->y_offset),
             kpriv->font, kpriv->font_size, 
             Scalar (kpriv->text_color.blue, kpriv->text_color.green, kpriv->text_color.red), 1, 1);
  }

  return FALSE;
}

extern "C"
{
  int32_t xlnx_kernel_init (IVASKernel * handle)
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");

    vvas_xpassthrupriv *kpriv =
        (vvas_xpassthrupriv *) calloc (1, sizeof (vvas_xpassthrupriv));

    json_t *jconfig = handle->kernel_config;
    json_t *val, *karray = NULL;

    /* Initialize config params with default values */
    log_level = LOG_LEVEL_WARNING;
    kpriv->frame_counter = 0;
    kpriv->font_size = 0.5;
    kpriv->font = 0;
    kpriv->line_thickness = 1;
    kpriv->x_offset = 10;
    kpriv->y_offset = 10;
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
    vvas_xpassthrupriv *kpriv = (vvas_xpassthrupriv *) handle->kernel_priv;

    if (kpriv)
      free (kpriv);

    return 0;
  }


  uint32_t xlnx_kernel_start (IVASKernel * handle, int start,
      IVASFrame * input[MAX_NUM_OBJECT], IVASFrame * output[MAX_NUM_OBJECT])
  {
    LOG_MESSAGE (LOG_LEVEL_DEBUG, "enter");
    GstInferenceMeta *infer_meta = NULL;
    char *pstr;

    vvas_xpassthrupriv *kpriv = (vvas_xpassthrupriv *) handle->kernel_priv;
    struct overlayframe_info *frameinfo = &(kpriv->frameinfo);

    frameinfo->y_offset = 0;
    frameinfo->inframe = input[0];
    char *indata = (char *) frameinfo->inframe->vaddr[0];
    char *lumaBuf = (char *) frameinfo->inframe->vaddr[0];
    char *chromaBuf = (char *) frameinfo->inframe->vaddr[1];
    infer_meta = ((GstInferenceMeta *) gst_buffer_get_meta ((GstBuffer *)
            frameinfo->inframe->app_priv, gst_inference_meta_api_get_type ()));
    if (infer_meta == NULL) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "ivas meta data is not available");
    }

    if (frameinfo->inframe->props.fmt == IVAS_VFMT_Y_UV8_420) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Input frame is in NV12 format\n");
      frameinfo->lumaImg.create (input[0]->props.height, input[0]->props.stride,
          CV_8UC1);
      frameinfo->lumaImg.data = (unsigned char *) lumaBuf;
      frameinfo->chromaImg.create (input[0]->props.height / 2,
          input[0]->props.stride / 2, CV_16UC1);
      frameinfo->chromaImg.data = (unsigned char *) chromaBuf;
    } else if (frameinfo->inframe->props.fmt == IVAS_VFMT_BGR8) {
      LOG_MESSAGE (LOG_LEVEL_DEBUG, "Input frame is in BGR format\n");
      frameinfo->image.create (input[0]->props.height,
          input[0]->props.stride / 3, CV_8UC3);
      frameinfo->image.data = (unsigned char *) indata;
    } else {
      LOG_MESSAGE (LOG_LEVEL_WARNING, "Unsupported color format\n");
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
