/* st_mlp.h — the trained network's forward pass.
 *
 * Architecture (fixed):
 *   input  ST_NUM_BINS
 *   -> dense ST_H1 + ReLU
 *   -> dense ST_H2 + ReLU
 *   -> dense ST_NUM_CELLS + sigmoid      (multi-label: several cells can be
 *                                         in the field of view at once, so
 *                                         this is NOT a softmax)
 *
 * Weight layout: W1 is row-major with shape [ST_H1][ST_NUM_BINS], i.e. the
 * weights for output neuron j are contiguous. That is the cache-friendly
 * layout for a matrix-vector product, and it is what the Python exporter
 * writes. Do not transpose one side without transposing the other.
 */
#ifndef ST_MLP_H
#define ST_MLP_H

#include "st_types.h"

typedef struct {
    const float *W1;  /* [ST_H1][ST_NUM_BINS]   */
    const float *b1;  /* [ST_H1]                */
    const float *W2;  /* [ST_H2][ST_H1]         */
    const float *b2;  /* [ST_H2]                */
    const float *W3;  /* [ST_NUM_CELLS][ST_H2]  */
    const float *b3;  /* [ST_NUM_CELLS]         */
    /* Input standardization, applied as (x - mean) / scale before layer 1.
     * Folding this into the model rather than the caller means the C and
     * Python paths cannot disagree about whether it was applied. */
    const float *in_mean;  /* [ST_NUM_BINS] */
    const float *in_scale; /* [ST_NUM_BINS] */
} st_model_t;

/* Run the forward pass.
 *   hist      : ST_NUM_BINS inputs (the raw histogram, un-standardized)
 *   out_prob  : ST_NUM_CELLS sigmoid outputs in [0,1]
 * Returns ST_OK or ST_ERR_NULL. */
st_status_t st_mlp_forward(const st_model_t *m,
                           const float *hist,
                           float *out_prob);

#endif /* ST_MLP_H */
