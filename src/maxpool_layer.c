#include "uwnet.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width - 1) / l.stride + 1;
    int outh = (l.height - 1) / l.stride + 1;
    matrix out = make_matrix(in.rows, outw * outh * l.channels);

    int h, i, j, k, m, n, p, q;
    float curr_val;
    float max_val = 0;
    int padding = 1;
    if (l.width % (l.size * l.stride) == 0 && l.height % (l.size * l.stride) == 0) {
        padding = 0;
    }
    for (h = 0; h < in.rows; ++h) {
        for (k = 0; k < l.channels; ++k) {
            for (i = 0; i < outh; ++i) {
                for (j = 0; j < outw; ++j) {
                    for (m = 0; m < l.size; ++m) {
                        for (n = 0; n < l.size; ++n) {
                            if (padding) {
                                p = (i * l.stride - 1) + m;
                                q = (j * l.stride - 1) + n;
                                if (p < 0 || p >= l.height || q < 0 || q >= l.width) {
                                    curr_val = 0;
                                } else {
                                    curr_val = in.data[(h * l.channels * l.height * l.width) + (k * l.height * l.width)
                                        + (p * l.width) + q];
                                }
                            } else {
                                p = (i * l.stride) + m;
                                q = (j * l.stride) + n;
                                curr_val = in.data[(h * l.channels * l.height * l.width) + (k * l.height * l.width)
                                    + (p * l.width) + q];
                            }
                            if (curr_val > max_val) {
                                max_val = curr_val;
                            }
                        }
                    }
                    out.data[(h * l.channels * outh * outw) + (k * outh * outw) + (i * outw) + j] = max_val;
                    max_val = 0;
                }
            }
        }
    }

    /* printf("in size: %d\n", in.rows * in.cols); */
    /* printf("last in index: %d\n", ((h - 1) * l.channels * l.height * l.width) + ((k - 1) * l.height * l.width) + (p * l.width) + q); */

    /* printf("out size: %d\n", out.rows * out.cols); */
    /* printf("last out index: %d\n", ((h - 1) * l.channels * outh * outw) + ((k - 1) * outh * outw) + ((i - 1) * outw) + (j - 1)); */

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    matrix dx = make_matrix(dy.rows, l.width * l.height * l.channels);

    int outw = (l.width - 1) / l.stride + 1;
    int outh = (l.height - 1) / l.stride + 1;
    // 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int h, i, j, k, m, n, p, q;
    int max_p = 0, max_q = 0;
    float curr_val;
    float max_val = -999;
    float max_val_delta = 0;
    int padding = 1;
    if (l.width % (l.size * l.stride) == 0 && l.height % (l.size * l.stride) == 0) {
        padding = 0;
    }
    /* for (h = 0; h < dy.rows; ++h) { */
    for (h = 0; h < in.rows; ++h) {
        for (k = 0; k < l.channels; ++k) {
            for (i = 0; i < outh; ++i) {
                for (j = 0; j < outw; ++j) {
                    for (m = 0; m < l.size; ++m) {
                        for (n = 0; n < l.size; ++n) {
                            if (padding) {
                                p = (i * l.stride - 1) + m;
                                q = (j * l.stride - 1) + n;
                                if (p < 0 || p >= l.height || q < 0 || q >= l.width) {
                                    curr_val = 0;
                                } else {
                                    curr_val = in.data[(h * l.channels * l.height * l.width) + (k * l.height * l.width) + (p * l.width) + q];
                                }
                            } else {
                                p = (i * l.stride) + m;
                                q = (j * l.stride) + n;
                                curr_val = in.data[(h * l.channels * l.height * l.width) + (k * l.height * l.width) + (p * l.width) + q];
                            }
                            if (curr_val > max_val) {
                                max_val = curr_val;
                                max_val_delta = dy.data[(h * l.channels * outh * outw)
                                    + (k * outh * outw) + (i * outw) + j];
                                max_p = p;
                                max_q = q;
                            }
                        }
                    }
                    dx.data[(h * l.channels * l.height * l.width) + (k * l.height * l.width)
                        + (max_p * l.width) + max_q]
                        += max_val_delta;
                    max_val = -999;
                    max_val_delta = 0;
                }
            }
        }
    }

    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay) { }

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = { 0 };
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update = update_maxpool_layer;
    return l;
}
