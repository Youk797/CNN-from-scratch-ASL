from .numpy_cupy import xp


def softmax(logits, axis=1):
    m = xp.max(logits, axis=axis, keepdims=True)
    e = xp.exp(logits - m)
    return e / xp.sum(e, axis=axis, keepdims=True)


def im2col_nchw(x, kH, kW, stride=1, padding=0):
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    if padding > 0:
        x = xp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
        H_p, W_p = H + 2 * padding, W + 2 * padding
    else:
        H_p, W_p = H, W

    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    cols = xp.empty((N, C * kH * kW, out_h * out_w), dtype=x.dtype)

    idx = 0
    for c in range(C):
        for y in range(kH):
            y_max = y + stride * out_h
            for xk in range(kW):
                x_max = xk + stride * out_w
                patch = x[:, c, y:y_max:stride, xk:x_max:stride]  # (N, out_h, out_w)
                cols[:, idx, :] = patch.reshape(N, -1)
                idx += 1

    return cols, out_h, out_w


def col2im_nchw(cols, x_shape, kH, kW, stride=1, padding=0):
    # cols: (N, C*kH*kW, out_h*out_w)
    N, C, H, W = x_shape
    H_p, W_p = H + 2 * padding, W + 2 * padding
    out_h = (H_p - kH) // stride + 1
    out_w = (W_p - kW) // stride + 1

    x_padded = xp.zeros((N, C, H_p, W_p), dtype=cols.dtype)

    idx = 0
    for c in range(C):
        for y in range(kH):
            y_max = y + stride * out_h
            for xk in range(kW):
                x_max = xk + stride * out_w
                patch = cols[:, idx, :].reshape(N, out_h, out_w)
                x_padded[:, c, y:y_max:stride, xk:x_max:stride] += patch
                idx += 1

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded
