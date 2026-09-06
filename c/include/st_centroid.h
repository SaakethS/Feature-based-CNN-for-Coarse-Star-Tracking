/* st_centroid.h — image to star centroids. */
#ifndef ST_CENTROID_H
#define ST_CENTROID_H

#include "st_types.h"

/* Scratch workspace for the blob finder. Declare one of these ONCE, statically,
 * in your application. It is large (a label image plus a flood-fill stack), so
 * you do not want it on the stack. */
typedef struct {
    uint8_t  mask[ST_IMG_W * ST_IMG_H];   /* 1 = pixel is above threshold  */
    int32_t  stack[ST_FILL_STACK];        /* flood-fill work stack         */
} st_centroid_ws_t;

/* Estimate the background level and noise sigma of an 8-bit image.
 *
 *   img       : row-major, ST_IMG_W * ST_IMG_H bytes
 *   out_median: estimated background level in counts
 *   out_sigma : estimated background noise standard deviation in counts,
 *               computed robustly as 1.4826 * median-absolute-deviation so
 *               that the stars themselves do not inflate it.
 *
 * Returns ST_OK or ST_ERR_NULL.
 */
st_status_t st_estimate_background(const uint8_t *img,
                                   float *out_median,
                                   float *out_sigma);

/* Detect stars and compute intensity-weighted sub-pixel centroids.
 *
 *   img        : row-major 8-bit image
 *   ws         : caller-provided scratch (contents are clobbered)
 *   thresh_sigma: detection threshold in units of background sigma
 *   out_stars  : caller-provided array of at least max_stars entries
 *   max_stars  : capacity of out_stars
 *   out_n      : number of stars actually written
 *
 * Stars are written in detection order, NOT sorted. Sorting by flux happens in
 * the feature stage. Returns ST_OK, or ST_ERR_OVERFLOW if the blob table
 * filled (in which case out_n is still valid and usable).
 */
st_status_t st_detect_stars(const uint8_t *img,
                            st_centroid_ws_t *ws,
                            float thresh_sigma,
                            st_star_t *out_stars,
                            int max_stars,
                            int *out_n);

#endif /* ST_CENTROID_H */
