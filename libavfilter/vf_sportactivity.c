/*
 * Copyright (c) 2022 Deniz Ugur
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * libopencv wrapper functions
 */

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/file.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

// CONSTANTS
#define MAX_SNOOKER_BALL_COUNT 22

static void fill_iplimage_from_frame(IplImage *img, const AVFrame *frame, enum AVPixelFormat pixfmt)
{
    IplImage *tmpimg;
    int depth, channels_nb;

    if (pixfmt == AV_PIX_FMT_GRAY8)
    {
        depth = IPL_DEPTH_8U;
        channels_nb = 1;
    }
    else if (pixfmt == AV_PIX_FMT_BGRA)
    {
        depth = IPL_DEPTH_8U;
        channels_nb = 4;
    }
    else if (pixfmt == AV_PIX_FMT_BGR24)
    {
        depth = IPL_DEPTH_8U;
        channels_nb = 3;
    }
    else
        return;

    tmpimg = cvCreateImageHeader((CvSize){frame->width, frame->height}, depth, channels_nb);
    *img = *tmpimg;
    img->imageData = img->imageDataOrigin = frame->data[0];
    img->dataOrder = IPL_DATA_ORDER_PIXEL;
    img->origin = IPL_ORIGIN_TL;
    img->widthStep = frame->linesize[0];
}

static void fill_frame_from_iplimage(AVFrame *frame, const IplImage *img, enum AVPixelFormat pixfmt)
{
    frame->linesize[0] = img->widthStep;
    frame->data[0] = img->imageData;
}

static void sort_swap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}

static void sort_input(int input[])
{
    int i, j;
    int n = sizeof(input) / sizeof(input[0]);

    for (i = 0; i < n - 1; i++)
        for (j = 0; j < n - i - 1; j++)
            if (input[j] > input[j + 1])
                sort_swap(&input[j], &input[j + 1]);
}

typedef struct SAContext
{
    const AVClass *class;
    char *name;
    char *params;
    int (*init)(AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    void (*end_frame_filter)(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg, enum AVPixelFormat pixfmt);
    void *priv;
    int debug_level;
} SAContext;

typedef struct SnookerContext
{
    int previous_balls[MAX_SNOOKER_BALL_COUNT];
    int previous_balls_count;
    double momentum;
    double momentum_hit;
    int state;
    int prev_stable_state;
    int precision;
} SnookerContext;

static av_cold int snooker_init(AVFilterContext *ctx, const char *args)
{
    SAContext *s = ctx->priv;
    SnookerContext *snooker = s->priv;

    snooker->momentum = 0;
    snooker->momentum_hit = 0.17; // TODO: Variable
    snooker->state = -1;
    snooker->prev_stable_state = -1;
    snooker->precision = 100; // TODO: Variable

    return 0;
}

static av_cold int snooker_uninit(AVFilterContext *ctx)
{
    SAContext *s = ctx->priv;
    SnookerContext *snooker = s->priv;

    return 0;
}

static void snooker_end_frame_filter(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg, enum AVPixelFormat pixfmt)
{
    SAContext *s = ctx->priv;
    SnookerContext *snooker = s->priv;

    // ! Copy input frame
    IplImage *img;

    if (pixfmt == AV_PIX_FMT_BGRA)
        img = cvCreateImage(cvGetSize(inimg), 8, 4);
    else if (pixfmt == AV_PIX_FMT_BGR24)
        img = cvCreateImage(cvGetSize(inimg), 8, 3);

    cvCopy(inimg, img, 0);

    // ! Convert input frame to HSV
    IplImage *hsv = cvCreateImage(cvGetSize(inimg), 8, 3);
    cvCvtColor(inimg, hsv, CV_BGR2HSV);

    // ! Apply inRangeS to mask the snooker table
    IplImage *table = cvCreateImage(cvGetSize(inimg), 8, 1);
    IplImage *balls = cvCreateImage(cvGetSize(inimg), 8, 1);
    cvInRangeS(hsv, cvScalar(40, 200, 80, 0), cvScalar(70, 255, 240, 255), balls);
    cvCopy(balls, table, 0);

    // ! Apply invert & smooth for better contour detection
    cvThreshold(balls, balls, 5, 255, CV_THRESH_BINARY_INV);
    cvSmooth(balls, balls, CV_GAUSSIAN, 3, 0, 0, 0);

    // ! Find ball contours
    CvMemStorage *ball_storage = cvCreateMemStorage(0);
    CvSeq *ball_contours = NULL;
    cvFindContours(balls, ball_storage, &ball_contours, sizeof(CvContour),
                   CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

    // ! Find table contour
    CvMemStorage *table_storage = cvCreateMemStorage(0);
    CvSeq *table_contour = NULL;
    cvFindContours(table, table_storage, &table_contour, sizeof(CvContour),
                   CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

    // ! Draw ball contours
    CvPoint ball_points[MAX_SNOOKER_BALL_COUNT];
    int ball_count = 0;

    while (ball_contours)
    {
        double a = cvContourArea(ball_contours, CV_WHOLE_SEQ, 0);

        if (a > 200 && a < 1300)
        {
            CvRect rect = cvBoundingRect(ball_contours, 1);
            double x = rect.x + rect.width / 2;
            double y = rect.y + rect.height / 2;

            ball_points[ball_count++] = cvPoint(x, y);

            if (s->debug_level >= 3)
                cvCircle(img, cvPoint(x, y), 15, cvScalar(0, 255, 255, 255), 4, 8, 0);

            if (s->debug_level > 1)
                av_log(ctx, AV_LOG_INFO, "[BALL] i=%d x=%.2f y=%.2f\t\n", ball_count, x, y);
        }

        //obtain the next contour
        ball_contours = ball_contours->h_next;
    }

    // ! Draw table contour
    CvRect snooker_rect;
    int table_found = 0;

    while (table_contour)
    {
        double a = cvContourArea(table_contour, CV_WHOLE_SEQ, 0);

        if (a < 10000)
        {
            // obtain the next contour
            table_contour = table_contour->h_next;
            continue;
        }

        snooker_rect = cvBoundingRect(table_contour, 1);
        table_found = 1;

        if (s->debug_level >= 3)
            cvRectangle(img,
                        cvPoint(snooker_rect.x, snooker_rect.y),
                        cvPoint(snooker_rect.x + snooker_rect.width, snooker_rect.y + snooker_rect.height),
                        cvScalar(0, 0, 255, 255),
                        4, 8, 0);

        if (s->debug_level > 1)
            av_log(ctx, AV_LOG_INFO, "[TABLE] x=%d y=%d w=%d h=%d\t\n", snooker_rect.x, snooker_rect.y, snooker_rect.width, snooker_rect.height);

        break;
    }

    // ! Init before ball coordinates
    int coords[ball_count];
    double ux = snooker_rect.width / snooker->precision;
    double uy = snooker_rect.height / snooker->precision;

    // ! Calculate ball coordinates
    for (int i = 0; i < ball_count; i++)
    {
        CvPoint *cur_ball = &ball_points[i];
        int x = round(cur_ball->x / ux);
        int y = round(cur_ball->y / uy);
        coords[i] = x + y * snooker->precision;
    }
    sort_input(coords);

    // ! FUN PART, Decide the state
    if (snooker->state == -1)
    {
        snooker->state = 0;
        goto snooker_copyto_prev;
    }

    if (!table_found || ball_count == 0)
    {
        snooker->momentum -= snooker->momentum_hit;
        goto snooker_momentum_end;
    }

    if (snooker->previous_balls_count == ball_count)
    {
        int equal_arrays = 1;
        for (int i = 0; i < ball_count; i++)
        {
            if (snooker->previous_balls[i] != coords[i])
            {
                equal_arrays = 0;
                break;
            }
        }

        if (!equal_arrays)
            snooker->momentum += snooker->momentum_hit;
        else
            snooker->momentum -= snooker->momentum_hit;
    }
    else
        snooker->momentum -= snooker->momentum_hit;

snooker_momentum_end:
    snooker->momentum = fmax(fmin(snooker->momentum, 1.0), 0.0);

    if (snooker->momentum == 1)
        snooker->state = 2;
    else if (snooker->momentum > 0.25 && snooker->momentum < 0.75)
        snooker->state = 1;
    else if (snooker->momentum == 0)
        snooker->state = 0;

    if (snooker->state == 2)
        snooker->prev_stable_state = 2;
    else if (snooker->state == 0)
        snooker->prev_stable_state = 0;

snooker_copyto_prev:
    snooker->previous_balls_count = ball_count;
    memcpy(snooker->previous_balls, coords, sizeof(coords));

    // ! Here we have snooker rect (cvRect) and ball coordinates (cvPoint) inside it
    if (s->debug_level > 0)
    {
        av_log(ctx, AV_LOG_INFO, "[STATS] momentum=%.2f  state=%d      \t\n", snooker->momentum, snooker->state);
    }

    if (s->debug_level < 2)
        goto snooker_end;

    CvScalar color = cvScalar(0, 0, 255, 255);
    if (snooker->prev_stable_state == 2)
        color = cvScalar(0, 255, 0, 255);

    if (snooker->state == 1)
    {
        cvRectangle(img, cvPoint(0, round(720 / 3)), cvPoint(50, 2 * round(720 / 3)), cvScalar(0, 255, 255, 255), -1, 8, 0);
        cvRectangle(img, cvPoint(1280 - 50, round(720 / 3)), cvPoint(1280, 2 * round(720 / 3)), cvScalar(0, 255, 255, 255), -1, 8, 0);
    }
    // * Display stable state
    int offset_top = round(720 / 3) * snooker->prev_stable_state;
    int offset_bottom = round(720 / 3) * (snooker->prev_stable_state + 1);

    cvRectangle(img, cvPoint(0, offset_top), cvPoint(50, offset_bottom), color, -1, 8, 0);
    cvRectangle(img, cvPoint(1280 - 50, offset_top), cvPoint(1280, offset_bottom), color, -1, 8, 0);

snooker_end:
    // ! Pipe the result to output frame
    cvCopy(img, outimg, 0);

    // ! Release memory
    cvReleaseImage(&img);
    cvReleaseImage(&hsv);
    cvReleaseImage(&table);
    cvReleaseImage(&balls);
    cvReleaseMemStorage(&ball_storage);
    cvReleaseMemStorage(&table_storage);
}

typedef struct SAFilterEntry
{
    const char *name;
    size_t priv_size;
    int (*init)(AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    void (*end_frame_filter)(AVFilterContext *ctx, IplImage *inimg, IplImage *outimg, enum AVPixelFormat pixfmt);
} SAFilterEntry;

static const SAFilterEntry sa_filter_entries[] = {
    {"snooker", sizeof(SnookerContext), snooker_init, snooker_uninit, snooker_end_frame_filter},
};

static av_cold int init(AVFilterContext *ctx)
{
    SAContext *s = ctx->priv;
    int i;

    if (!s->name)
    {
        av_log(ctx, AV_LOG_ERROR, "No sportactivity filter name specified\n");
        return AVERROR(EINVAL);
    }
    for (i = 0; i < FF_ARRAY_ELEMS(sa_filter_entries); i++)
    {
        const SAFilterEntry *entry = &sa_filter_entries[i];
        if (!strcmp(s->name, entry->name))
        {
            s->init = entry->init;
            s->uninit = entry->uninit;
            s->end_frame_filter = entry->end_frame_filter;

            if (!(s->priv = av_mallocz(entry->priv_size)))
                return AVERROR(ENOMEM);
            return s->init(ctx, s->params);
        }
    }

    av_log(ctx, AV_LOG_ERROR, "No sportactivity filter named '%s'\n", s->name);
    return AVERROR(EINVAL);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    SAContext *s = ctx->priv;

    if (s->uninit)
        s->uninit(ctx);
    av_freep(&s->priv);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    SAContext *s = ctx->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    AVFrame *out;
    IplImage inimg, outimg;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out)
    {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    fill_iplimage_from_frame(&inimg, in, inlink->format);
    fill_iplimage_from_frame(&outimg, out, inlink->format);
    s->end_frame_filter(ctx, &inimg, &outimg, inlink->format);
    fill_frame_from_iplimage(out, &outimg, inlink->format);

    SnookerContext *snooker = s->priv;
    AVDictionary **metadata = &out->metadata;
    av_dict_set_int(metadata, "lavfi.sa.state", snooker->state, 0);

    av_frame_free(&in);

    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(SAContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption sa_options[] = {
    {"sportactivity_name", "Name of the sport activity in the content", OFFSET(name), AV_OPT_TYPE_STRING, .flags = FLAGS},
    {"sportactivity_debug_level", "Display debug overlays (0 -> none, 1 -> stats on logs, 2 -> detailed logs + screen indicator, 3 -> contours)", OFFSET(debug_level), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 3, .flags = FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(sa);

static const AVFilterPad avfilter_vf_sa_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad avfilter_vf_sa_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
};

AVFilter ff_vf_sa = {
    .name = "sa",
    .description = NULL_IF_CONFIG_SMALL("Detect content importance in sport videos"),
    .priv_size = sizeof(SAContext),
    .priv_class = &sa_class,
    .init = init,
    .uninit = uninit,
    FILTER_INPUTS(avfilter_vf_sa_inputs),
    FILTER_OUTPUTS(avfilter_vf_sa_outputs),
    FILTER_PIXFMTS(AV_PIX_FMT_BGR24, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8),
};
