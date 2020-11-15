#include "uwnet.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i, j;
    for (i = 0; i < y.rows; ++i) {
        for (j = 0; j < y.cols; ++j) {
            y.data[i * y.cols + j] += b.data[j / spatial];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i, j;
    for (i = 0; i < dy.rows; ++i) {
        for (j = 0; j < dy.cols; ++j) {
            db.data[j / spatial] += dy.data[i * dy.cols + j];
        }
    }
    return db;
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{
    int i, j, k;
    int outw = (im.w - 1) / stride + 1;
    int outh = (im.h - 1) / stride + 1;
    int rows = im.c * size * size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // Fill in the column matrix with patches from the image
    int m, n, p, q;
    int out_col, out_row;
    float val;
    int padding = 1;
    // if size * stride divides dimensions of image, we don't need padding
    if (im.w % (size * stride) == 0 && im.h % (size * stride) == 0) {
        padding = 0;
    }
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < outh; ++i) {
            for (j = 0; j < outw; ++j) {
                for (m = 0; m < size; ++m) {
                    for (n = 0; n < size; ++n) {
                        if (padding) {
                            p = (i * stride - 1) + m;
                            q = (j * stride - 1) + n;
                            if (p < 0 || p >= im.h || q < 0 || q >= im.w) {
                                val = 0;
                            } else {
                                val = get_pixel(im, q, p, k);
                            }
                        } else {
                            p = (i * stride) + m;
                            q = (j * stride) + n;
                            val = get_pixel(im, q, p, k);
                        }
                        out_col = i * outw + j;
                        out_row = size * m + n + size * size * k;
                        col.data[out_row * col.cols + out_col] = val;
                    }
                }
            }
        }
    }

    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size,
    int stride)
{
    int i, j, k;

    image im = make_image(width, height, channels);
    int outw = (im.w - 1) / stride + 1;
    int outh = (im.h - 1) / stride + 1;
    int rows = im.c * size * size;

    // 5.2
    // Add values into image im from the column matrix
    int m, n, p, q;
    int out_col, out_row;
    float old_val, new_val, addend;
    int padding = 1;
    int update_pixel = 1;
    // if size * stride divides dimensions of image, we don't need padding
    if (im.w % (size * stride) == 0 && im.h % (size * stride) == 0) {
        padding = 0;
    }
    for (k = 0; k < im.c; ++k) {
        for (i = 0; i < outh; ++i) {
            for (j = 0; j < outw; ++j) {
                for (m = 0; m < size; ++m) {
                    for (n = 0; n < size; ++n) {
                        out_col = i * outw + j;
                        out_row = size * m + n + size * size * k;
                        addend = col.data[out_row * col.cols + out_col];
                        if (padding) {
                            p = (i * stride - 1) + m;
                            q = (j * stride - 1) + n;
                            // if we're outside bounds of image, don't update pixel
                            if (p < 0 || p >= im.h || q < 0 || q >= im.w) {
                                update_pixel = 0;
                            } else {
                                update_pixel = 1;
                            }
                        } else {
                            p = (i * stride) + m;
                            q = (j * stride) + n;
                        }
                        if (update_pixel) {
                            new_val = get_pixel(im, q, p, k) + addend;
                            set_pixel(im, q, p, k, new_val);
                        }
                    }
                }
            }
        }
    }

    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width * l.height * l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width - 1) / l.stride + 1;
    int outh = (l.height - 1) / l.stride + 1;
    matrix out = make_matrix(in.rows, outw * outh * l.filters);
    for (i = 0; i < in.rows; ++i) {
        image example = float_to_image(in.data + i * in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for (j = 0; j < wx.rows * wx.cols; ++j) {
            out.data[i * out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width * l.height * l.channels);

    int i;
    int outw = (l.width - 1) / l.stride + 1;
    int outh = (l.height - 1) / l.stride + 1;

    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);

    matrix dx = make_matrix(dy.rows, l.width * l.height * l.channels);
    matrix wt = transpose_matrix(l.w);

    for (i = 0; i < in.rows; ++i) {
        image example = float_to_image(in.data + i * in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw * outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i * dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows * dy.cols;
    }
    free_matrix(wt);
    return dx;
}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum,
    float decay)
{
    // Currently l.dw and l.db store:
    // l.dw = momentum * l.dw_prev - dL/dw
    // l.db = momentum * l.db_prev - dL/db

    // For our weights we want to include weight decay:
    // l.dw = l.dw - decay * l.w
    axpy_matrix(-decay, l.w, l.dw);

    // Then for both weights and biases we want to apply the updates:
    // l.w = l.w + rate*l.dw
    // l.b = l.b + rate*l.db
    axpy_matrix(rate, l.dw, l.w);
    axpy_matrix(rate, l.db, l.b);

    // Finally, we want to scale dw and db by our momentum to prepare them for the next round
    // l.dw *= momentum
    // l.db *= momentum
    scal_matrix(momentum, l.dw);
    scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size,
    int stride)
{
    layer l = { 0 };
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w = random_matrix(filters, size * size * c, sqrtf(2.f / (size * size * c)));
    l.dw = make_matrix(filters, size * size * c);
    l.b = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    return l;
}
