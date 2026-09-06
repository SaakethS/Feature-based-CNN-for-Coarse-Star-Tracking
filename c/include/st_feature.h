/* st_feature.h — pairwise angular distance histogram.
 *
 * This is the feature the network sees. It is rotation-invariant: rolling the
 * camera about its boresight does not change the angle between any pair of
 * star directions, so the histogram is unchanged.
 *
 * COST NOTE: we never call acos(). The angle between two unit vectors a and b
 * is theta = acos(a . b). A histogram only needs to know WHICH BIN theta falls
 * in, and cos() is strictly decreasing on [0, pi], so comparing the raw dot
 * product against cosines of the bin edges gives exactly the same bin
 * assignment for zero transcendental calls. The bin edges are therefore stored
 * in COSINE space, in DESCENDING order (cos of the smallest angle first).
 */
#ifndef ST_FEATURE_H
#define ST_FEATURE_H

#include "st_types.h"

/* Bin edges in cosine space.
 *   cos_edges[0]              = cos(theta_min)   (largest cosine)
 *   cos_edges[ST_NUM_BINS]    = cos(theta_max)   (smallest cosine)
 * so there are ST_NUM_BINS+1 edges, strictly decreasing. A pair with dot
 * product d falls in bin i when cos_edges[i] >= d > cos_edges[i+1].
 * Pairs outside [cos_edges[NUM_BINS], cos_edges[0]] are discarded.
 *
 * These come from the ground-side quantile fit and live in model_params.h. */
typedef struct {
    float cos_edges[ST_NUM_BINS + 1];
} st_bins_t;

/* Sort stars by descending flux and keep the brightest `keep` of them.
 * Operates in place on the array. Returns the number retained. This uses an
 * insertion sort, which is the right choice here: n is at most a few hundred,
 * and it has no recursion and no scratch memory. */
int st_select_brightest(st_star_t *stars, int n, int keep);

/* Build the histogram feature from a set of direction vectors.
 *
 *   vecs      : unit vectors, n of them (n <= ST_MAX_STARS)
 *   bins      : cosine-space bin edges
 *   out_hist  : caller-provided array of ST_NUM_BINS floats
 *
 * The histogram is L1-normalized by the number of pairs that landed inside the
 * edge range, so the feature does not change scale with star count. If no pair
 * lands in range the function returns ST_ERR_TOO_FEW and out_hist is zeroed.
 *
 * Requires n >= 2, else ST_ERR_TOO_FEW.
 */
st_status_t st_histogram(const st_vec3_t *vecs, int n,
                         const st_bins_t *bins,
                         float *out_hist);

#endif /* ST_FEATURE_H */
