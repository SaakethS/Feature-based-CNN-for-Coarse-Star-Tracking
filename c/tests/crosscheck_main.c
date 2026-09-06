/* crosscheck_main.c — expose the C feature and network stages to Python.
 *
 * Reads a plain-text case from stdin:
 *   fx fy cx cy k1 k2
 *   n
 *   u0 v0 flux0
 *   ...
 * and writes, to stdout:
 *   HIST <ST_NUM_BINS values>
 *   PROB <ST_NUM_CELLS values>
 *
 * The point of this program is that python/crosscheck.py can feed the exact
 * same star list to both implementations and diff the numbers. If those
 * numbers drift apart, the model is being trained on a feature the flight
 * software does not produce, and no amount of validation accuracy is real.
 */
#include <stdio.h>
#include <stdlib.h>
#include "st_types.h"
#include "st_camera.h"
#include "st_feature.h"
#include "st_mlp.h"
#include "model_params.h"

int main(void)
{
    st_camera_t cam;
    st_bins_t   bins;
    st_model_t  model;
    static st_star_t stars[ST_MAX_BLOBS];
    static st_vec3_t vecs[ST_MAX_STARS];
    static float hist[ST_NUM_BINS];
    static float prob[ST_NUM_CELLS];
    int n = 0, i, n_used;
    st_status_t rc;

    st_model_params_load(&model, &bins);

    if (scanf("%f %f %f %f %f %f",
              &cam.fx, &cam.fy, &cam.cx, &cam.cy, &cam.k1, &cam.k2) != 6) {
        fprintf(stderr, "bad camera line\n");
        return 2;
    }
    if (scanf("%d", &n) != 1 || n < 0 || n > ST_MAX_BLOBS) {
        fprintf(stderr, "bad star count\n");
        return 2;
    }
    for (i = 0; i < n; ++i) {
        if (scanf("%f %f %f", &stars[i].x, &stars[i].y, &stars[i].flux) != 3) {
            fprintf(stderr, "bad star %d\n", i);
            return 2;
        }
        stars[i].n_pixels = 9;
    }

    n_used = st_select_brightest(stars, n, ST_MAX_STARS);

    rc = st_pixels_to_vectors(&cam, stars, n_used, vecs);
    if (rc != ST_OK) { fprintf(stderr, "vector rc=%d\n", rc); return 3; }

    rc = st_histogram(vecs, n_used, &bins, hist);
    if (rc != ST_OK) { fprintf(stderr, "histogram rc=%d\n", rc); return 4; }

    rc = st_mlp_forward(&model, hist, prob);
    if (rc != ST_OK) { fprintf(stderr, "mlp rc=%d\n", rc); return 5; }

    printf("HIST");
    for (i = 0; i < ST_NUM_BINS; ++i) printf(" %.9g", (double)hist[i]);
    printf("\nPROB");
    for (i = 0; i < ST_NUM_CELLS; ++i) printf(" %.9g", (double)prob[i]);
    printf("\n");
    return 0;
}
