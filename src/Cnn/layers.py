from .numpy_cupy import xp
from .utils import im2col_nchw, col2im_nchw


class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, weight_scale=0.02):
        # Entrée / sortie
        self.in_channels = in_channels          # nb de canaux en entrée 
        self.out_channels = out_channels        # nb de filtres = nb de canaux en sortie

        # Taille du kernel
        self.kernel_h = kernel_size
        self.kernel_w = kernel_size

        # Déplacement du filtre et bordure
        self.stride = stride
        self.padding = padding

        # Poids + biais
        # W: (out_channels, in_channels, kH, kW)
        self.W = (weight_scale * xp.random.randn(out_channels, in_channels, self.kernel_h, self.kernel_w)).astype(xp.float32)
        # b: (out_channels,)
        self.b = xp.zeros((out_channels,), dtype=xp.float32)

        # Gradients (remplis pendant backward)
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)

        # Caches (infos gardées du forward pour le backward)
        self._input_shape = None     # forme de x
        self._patches = None         
        self._patches2d = None       

    def forward(self, x):
        # x: (N, C, H, W)
        self._input_shape = x.shape

        # (N, C*kH*kW, out_h*out_w)
        patches, out_h, out_w = im2col_nchw(x, self.kernel_h, self.kernel_w, stride=self.stride, padding=self.padding)

        # On met les patchs en 2D pour faire une multiplication de matrices
        # patches2d: (N*out_h*out_w, C*kH*kW)
        patches2d = patches.transpose(0, 2, 1).reshape(-1, self.in_channels * self.kernel_h * self.kernel_w)

        # On met les filtres en 2D 
        # filters2d: (out_channels, C*kH*kW)
        filters2d = self.W.reshape(self.out_channels, -1)

        # convolution -> multiplication + biais
        out_flat = patches2d @ filters2d.T + self.b  # (N*out_h*out_w, out_channels)

        # On remet en forme (N, out_channels, out_h, out_w)
        out = out_flat.reshape(x.shape[0], out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        # On garde pour backward
        self._patches = patches
        self._patches2d = patches2d
        return out

    def backward(self, dout):
        #  (N, out_channels, out_h, out_w)
        N, _, out_h, out_w = dout.shape

        # On met dout en 2D (plus simple pour les calculs)
        grad_out_2d = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        # Filtres en 2D
        filters2d = self.W.reshape(self.out_channels, -1)

        # Gradient du biais 
        self.db = xp.sum(grad_out_2d, axis=0).astype(xp.float32)

        # Gradient des poids 
        self.dW = (grad_out_2d.T @ self._patches2d).reshape(self.W.shape).astype(xp.float32)

       
        grad_patches2d = grad_out_2d @ filters2d  # (N*out_h*out_w, C*kH*kW)

        # On remet en forme comme col2im attend : (N, C*kH*kW, out_h*out_w)
        grad_patches = grad_patches2d.reshape(N, out_h * out_w, -1).transpose(0, 2, 1)

        # On reconstruit dx (gradient sur l'entrée x)
        dx = col2im_nchw(
            grad_patches,
            self._input_shape,
            self.kernel_h,
            self.kernel_w,
            stride=self.stride,
            padding=self.padding
        )
        return dx

    def params_and_grads(self):
        # Donne à l'optimiseur  (paramètre, gradient)
        yield self.W, self.dW
        yield self.b, self.db


class ReLU:
    def __init__(self):
        self.mask = None  # où x > 0

    def forward(self, x):
        # On retient où x est positif
        self.mask = x > 0
        # ReLU garde les positifs, met le reste à 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, pool=2, stride=2):
        self.pool = pool
        self.stride = stride

        
        self._x_shape = None
        self._max_pos = None   # position du max dans chaque petite fenêtre
        self._out_h = None
        self._out_w = None
        self._patches = None

    def forward(self, x):
        # x: (N, C, H, W)
        self._x_shape = x.shape
        N, C, H, W = x.shape

        # On découpe en petites fenêtres pool x pool
        patches, out_h, out_w = im2col_nchw(x, self.pool, self.pool, stride=self.stride, padding=0)
        # patches: (N, C*pool*pool, out_h*out_w)

        # On regroupe pour avoir une fenêtre par canal
        # (N, C, pool*pool, out_h*out_w)
        patches = patches.reshape(N, C, self.pool * self.pool, out_h * out_w)

        # Pour chaque fenêtre, on prend le max + on garde sa position
        max_pos = xp.argmax(patches, axis=2)  # (N, C, out_h*out_w)
        out = xp.max(patches, axis=2)         # (N, C, out_h*out_w)

        # On remet en (N, C, out_h, out_w)
        out = out.reshape(N, C, out_h, out_w)

        self._patches = patches
        self._max_pos = max_pos
        self._out_h, self._out_w = out_h, out_w
        return out

    def backward(self, dout):
        # dout: (N, C, out_h, out_w)
        N, C, out_h, out_w = dout.shape
        P = out_h * out_w                  # nb de fenêtres
        PP = self.pool * self.pool         # nb de valeurs par fenêtre

        # On remet le gradient uniquement là où il y avait le max
        grad_patches = xp.zeros((N, C, PP, P), dtype=dout.dtype)
        grads = dout.reshape(N, C, P)  # (N, C, P)

        if hasattr(xp, "put_along_axis"):
            idx = self._max_pos[:, :, None, :]  # (N, C, 1, P)
            xp.put_along_axis(grad_patches, idx, grads[:, :, None, :], axis=2)
        else:
            r = xp.arange(PP, dtype=self._max_pos.dtype)[None, None, :, None]  # (1,1,PP,1)
            mask = (r == self._max_pos[:, :, None, :])                         # (N,C,PP,P)
            grad_patches = mask * grads[:, :, None, :]

        # On remet en forme pour col2im  (N, C*PP, P)
        grad_patches = grad_patches.reshape(N, C * PP, P)

        dx = col2im_nchw(grad_patches, self._x_shape, self.pool, self.pool, stride=self.stride, padding=0)
        return dx


class Flatten:
    def __init__(self):
        self._input_shape = None  # forme d'origine

    def forward(self, x):
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        # Remet la forme d'origine
        return dout.reshape(self._input_shape)


class Dense:
    def __init__(self, in_dim, out_dim, weight_scale=0.02):
        # Poids + biais
        self.W = (weight_scale * xp.random.randn(in_dim, out_dim)).astype(xp.float32)
        self.b = xp.zeros((out_dim,), dtype=xp.float32)

        # Gradients
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)

        self._x_cache = None  # entrée gardée pour backward

    def forward(self, x):
        # y = xW + b
        self._x_cache = x
        return x @ self.W + self.b

    def backward(self, dout):
        # Gradient des poids et biais
        self.dW = (self._x_cache.T @ dout).astype(xp.float32)
        self.db = xp.sum(dout, axis=0).astype(xp.float32)

        # Gradient à renvoyer à la couche précédente
        dx = dout @ self.W.T
        return dx

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db
