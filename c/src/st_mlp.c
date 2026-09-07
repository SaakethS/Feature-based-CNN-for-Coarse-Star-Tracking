#include "st_mlp.h"
#include <math.h>
#include <stddef.h>

/* Numerically stable logistic sigmoid.
 *   sigmoid(x) = 1 / (1 + e^-x)
 * Written directly, a large negative x makes e^-x overflow to +inf. The
 * branch below uses the algebraically identical form e^x / (1 + e^x) for
 * negative x, where the exponential can only underflow toward 0, which is
 * harmless. */
static float st_sigmoid(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

/* Dense layer: out[j] = sum_k W[j][k] * in[k] + b[j], optionally ReLU'd.
 * W is row-major [n_out][n_in], so the inner loop walks contiguous memory. */
static void st_dense(const float *W, const float *b,
                     const float *in, int n_in,
                     float *out, int n_out,
                     int relu)
{
    int j, k;
    for (j = 0; j < n_out; ++j) {
        const float *w = W + (size_t)j * (size_t)n_in;
        float acc = b[j];
        for (k = 0; k < n_in; ++k) acc += w[k] * in[k];
        if (relu && acc < 0.0f) acc = 0.0f;
        out[j] = acc;
    }
}

st_status_t st_mlp_forward(const st_model_t *m,
                           const float *hist,
                           float *out_prob)
{
    float x[ST_NUM_BINS];
    float h1[ST_H1];
    float h2[ST_H2];
    int i;

    if (!m || !hist || !out_prob) return ST_ERR_NULL;
    if (!m->W1 || !m->W2 || !m->W3 || !m->b1 || !m->b2 || !m->b3)
        return ST_ERR_NULL;
    for (i = 0; i < ST_NUM_BINS; ++i)
        if (!isfinite(hist[i])) return ST_ERR_BAD_PARAM;

    /* Standardize the input using the statistics captured at training time.
     * Keeping this inside the model is deliberate: it is the single most
     * common source of a silent train/flight mismatch. */
    for (i = 0; i < ST_NUM_BINS; ++i) {
        float s = m->in_scale ? m->in_scale[i] : 1.0f;
        float u = m->in_mean  ? m->in_mean[i]  : 0.0f;
        if (s == 0.0f) s = 1.0f;
        x[i] = (hist[i] - u) / s;
    }

    st_dense(m->W1, m->b1, x,  ST_NUM_BINS, h1, ST_H1, 1);
    st_dense(m->W2, m->b2, h1, ST_H1,       h2, ST_H2, 1);
    st_dense(m->W3, m->b3, h2, ST_H2, out_prob, ST_NUM_CELLS, 0);

    /* Sigmoid, not softmax: several cells are genuinely in the field of view
     * at once, so their probabilities must be free to sum to more than one. */
    for (i = 0; i < ST_NUM_CELLS; ++i)
        out_prob[i] = st_sigmoid(out_prob[i]);

    return ST_OK;
}
