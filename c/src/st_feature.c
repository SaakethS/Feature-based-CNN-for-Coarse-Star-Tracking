#include "st_feature.h"
#include <string.h>

int st_select_brightest(st_star_t *stars, int n, int keep)
{
    int i, j;
    if (!stars || n <= 0) return 0;
    if (keep > n) keep = n;

    if (keep < 0) return 0;
    /* Stable descending insertion sort: ties preserve original input order. */
    for (i = 1; i < n; ++i) {
        st_star_t value = stars[i];
        j = i;
        while (j > 0 && stars[j - 1].flux < value.flux) {
            stars[j] = stars[j - 1];
            --j;
        }
        stars[j] = value;
    }
    return keep;
}

st_status_t st_histogram(const st_vec3_t *vecs, int n,
                         const st_bins_t *bins,
                         float *out_hist)
{
    int i, j, b;
    int in_range = 0;
    float inv;

    if (!vecs || !bins || !out_hist) return ST_ERR_NULL;
    memset(out_hist, 0, sizeof(float) * ST_NUM_BINS);
    if (n < 2) return ST_ERR_TOO_FEW;
    if (n > ST_MAX_STARS) n = ST_MAX_STARS;

    for (i = 0; i < n; ++i) {
        for (j = i + 1; j < n; ++j) {
            /* Dot product of two unit vectors = cos(angle between them).
             * No acos needed: we compare directly against cosine-space edges,
             * which are stored in DESCENDING order. */
            float d = vecs[i].x * vecs[j].x
                    + vecs[i].y * vecs[j].y
                    + vecs[i].z * vecs[j].z;

            /* Clamp: floating-point error can push an inner product a hair
             * outside [-1, 1] for nearly parallel vectors. */
            if (d >  1.0f) d =  1.0f;
            if (d < -1.0f) d = -1.0f;

            /* Outside the fitted range entirely -> discard the pair. */
            if (d > bins->cos_edges[0]) continue;
            if (d <= bins->cos_edges[ST_NUM_BINS]) continue;

            /* Linear scan over the configured edges. A binary search would be asymptotically
             * better but is slower in practice at this size and harder to audit.
             * Bin i holds pairs with cos_edges[i] >= d > cos_edges[i+1]. */
            for (b = 0; b < ST_NUM_BINS; ++b) {
                if (d > bins->cos_edges[b + 1]) {
                    out_hist[b] += 1.0f;
                    in_range++;
                    break;
                }
            }
        }
    }

    if (in_range == 0) return ST_ERR_TOO_FEW;

    /* L1-normalize so the feature is a distribution, not a count. Without this
     * the network would key on how many stars happened to be detected, which
     * varies with exposure and threshold and is not a property of the sky. */
    inv = 1.0f / (float)in_range;
    for (b = 0; b < ST_NUM_BINS; ++b) out_hist[b] *= inv;

    return ST_OK;
}
