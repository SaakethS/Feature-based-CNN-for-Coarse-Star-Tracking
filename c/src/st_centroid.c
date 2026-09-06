#include "st_centroid.h"
#include <string.h>

/* ---- background estimation ---------------------------------------------
 * We want a threshold that adapts to whatever the sensor is doing right now:
 * dark current drifts with temperature, and stray light raises the floor on
 * one side of the frame. A fixed threshold in counts would work in the lab
 * and fail in orbit.
 *
 * The estimator is deliberately robust. Stars are a small fraction of pixels
 * but they are extreme, so a mean and standard deviation would be dragged
 * upward by them and the threshold would drift with the star field itself.
 * The median is immune to that, and the median absolute deviation (MAD) is the
 * matching robust spread. For Gaussian noise, sigma = 1.4826 * MAD.
 *
 * Both medians are computed from a 256-bin histogram of the 8-bit values,
 * which is exact for 8-bit data, needs 1 KB of stack, and is a single pass.
 */
st_status_t st_estimate_background(const uint8_t *img,
                                   float *out_median,
                                   float *out_sigma)
{
    uint32_t hist[256];
    uint32_t mad_hist[256];
    const uint32_t n = (uint32_t)ST_IMG_W * (uint32_t)ST_IMG_H;
    uint32_t cum, i, med, half;
    int mad;

    if (!img || !out_median || !out_sigma) return ST_ERR_NULL;

    memset(hist, 0, sizeof(hist));
    for (i = 0; i < n; ++i) hist[img[i]]++;

    /* median = smallest value where the cumulative count passes half */
    half = n / 2u;
    cum = 0; med = 0;
    for (i = 0; i < 256; ++i) {
        cum += hist[i];
        if (cum > half) { med = i; break; }
    }

    /* MAD: median of |value - med|. Because the value histogram is already
     * built, we can fold it into a distance histogram without touching the
     * image a second time. */
    memset(mad_hist, 0, sizeof(mad_hist));
    for (i = 0; i < 256; ++i) {
        int d = (int)i - (int)med;
        if (d < 0) d = -d;
        mad_hist[d] += hist[i];
    }
    cum = 0; mad = 0;
    for (i = 0; i < 256; ++i) {
        cum += mad_hist[i];
        if (cum > half) { mad = (int)i; break; }
    }

    *out_median = (float)med;
    /* Floor the sigma at 0.5 counts. On a very clean image the MAD can come
     * out as exactly 0, which would make the threshold equal to the median and
     * flag half the frame as stars. */
    *out_sigma = 1.4826f * (float)mad;
    if (*out_sigma < 0.5f) *out_sigma = 0.5f;
    return ST_OK;
}

/* ---- blob finding -------------------------------------------------------
 * Iterative flood fill with an explicit stack. Recursion is banned in flight
 * software for exactly the reason it would bite here: a large smeared blob
 * would blow the call stack.
 */
st_status_t st_detect_stars(const uint8_t *img,
                            st_centroid_ws_t *ws,
                            float thresh_sigma,
                            st_star_t *out_stars,
                            int max_stars,
                            int *out_n)
{
    float bg, sigma, thresh;
    int32_t sp;                    /* stack pointer */
    int32_t idx, x, y, dx, dy;
    int n_found = 0;
    st_status_t rc;

    if (!img || !ws || !out_stars || !out_n) return ST_ERR_NULL;
    if (max_stars <= 0) return ST_ERR_BAD_PARAM;
    *out_n = 0;

    rc = st_estimate_background(img, &bg, &sigma);
    if (rc != ST_OK) return rc;
    thresh = bg + thresh_sigma * sigma;

    /* Pass 1: binary mask of above-threshold pixels. */
    for (idx = 0; idx < ST_IMG_W * ST_IMG_H; ++idx)
        ws->mask[idx] = ((float)img[idx] > thresh) ? 1u : 0u;

    /* Pass 2: 8-connected flood fill from each unvisited bright pixel. */
    for (y = 0; y < ST_IMG_H; ++y) {
        for (x = 0; x < ST_IMG_W; ++x) {
            int32_t seed = y * ST_IMG_W + x;
            double sum_w = 0.0, sum_wx = 0.0, sum_wy = 0.0;
            uint32_t npix = 0;
            int overflowed = 0;

            if (!ws->mask[seed]) continue;

            sp = 0;
            ws->stack[sp++] = seed;
            ws->mask[seed] = 0u;      /* mark visited immediately */

            while (sp > 0) {
                int32_t p  = ws->stack[--sp];
                int32_t px = p % ST_IMG_W;
                int32_t py = p / ST_IMG_W;
                /* Weight is above-background intensity. Using the raw pixel
                 * value instead would bias the centroid toward the image
                 * center of the blob's bounding box rather than its light
                 * distribution. */
                double w = (double)img[p] - (double)bg;
                if (w < 0.0) w = 0.0;

                sum_w  += w;
                sum_wx += w * ((double)px + 0.5);
                sum_wy += w * ((double)py + 0.5);
                npix++;

                for (dy = -1; dy <= 1; ++dy) {
                    for (dx = -1; dx <= 1; ++dx) {
                        int32_t qx = px + dx, qy = py + dy, q;
                        if (dx == 0 && dy == 0) continue;
                        if (qx < 0 || qx >= ST_IMG_W) continue;
                        if (qy < 0 || qy >= ST_IMG_H) continue;
                        q = qy * ST_IMG_W + qx;
                        if (ws->mask[q]) {
                            ws->mask[q] = 0u;
                            if (sp < ST_FILL_STACK) {
                                ws->stack[sp++] = q;
                            } else {
                                /* Only reachable for a blob far larger than
                                 * ST_MAX_BLOB_PIXELS, which we reject anyway.
                                 * The pixel stays marked visited so we do not
                                 * re-seed inside the same region. */
                                overflowed = 1;
                            }
                        }
                    }
                }
            }

            /* Reject blobs that are too small (hot pixels, cosmic rays) or
             * too large (stray light, smear, a planet). */
            if (overflowed) continue;
            if (npix < ST_MIN_BLOB_PIXELS) continue;
            if (npix > ST_MAX_BLOB_PIXELS) continue;
            if (sum_w <= 0.0) continue;

            if (n_found >= max_stars) {
                *out_n = n_found;
                return ST_ERR_OVERFLOW;
            }

            out_stars[n_found].x        = (float)(sum_wx / sum_w);
            out_stars[n_found].y        = (float)(sum_wy / sum_w);
            out_stars[n_found].flux     = (float)sum_w;
            out_stars[n_found].n_pixels = (uint16_t)npix;
            n_found++;
        }
    }

    *out_n = n_found;
    return ST_OK;
}
