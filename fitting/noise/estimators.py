import numpy as np
import scipy.special as sc
from scipy import signal
import scipy.fftpack as fp
import scipy.ndimage as sn 

from skimage.restoration import denoise_bilateral 


def get_kernel(kernel):
    if isinstance(kernel, int):
        return np.ones(shape=(kernel, kernel), dtype=np.float) / kernel**2
    elif kernel is None:
        return get_kernel(3)
    else:
        return kernel


def snr_correction(snr, a=0.25, b=-0.79, c=1.08):
    return a * ((snr / c + b) / np.sqrt(1 + (snr / c + b)**2) - 1)


def koay_correction(snr):
    # sigma_r**2 = koay(snr) * sigma_g**2
    return 2 + snr**2 - np.pi / 8 * np.exp(-snr**2 / 2) * ((2 + snr**2) *
                                                           sc.i0(snr**2 / 4) + snr**2 * sc.i1(snr**2 / 4))**2


def theta1_opt(snr):
    return np.where(snr <= 1.171, -np.sqrt(0.101 * snr**2 + 0.014) + 1.396, 1)


def theta2_opt(snr):
    nom = snr**4 - 6.606 * snr**3 + 16.109 * snr**2 - 15.093 * snr + 7.976
    denom = 2 * snr**4 - 12.930 * snr**3 + 29.777 * snr**2 - 19.172 * snr + 1
    return np.where(snr < 1.171, 0.467 * snr**2 + 0.901, nom / denom)

def matlab_style_gauss2D(shape, sigma=1):
    """ 
    same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    # https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def dct2(block):
    # Matlab's dct2 : https://stackoverflow.com/questions/40104377/issiue-with-implementation-of-2d-discrete-cosine-transform-in-python
    return fp.dct(fp.dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return fp.idct(fp.idct(block.T, norm='ortho').T, norm='ortho')


def lpf(img, sigma=3.4):
    # Alternative 1
    # kernel = get_kernel(3)
    # return signal.convolve2d(img, kernel, 'same')

    # Alternative 2
    # img_fre = fp.fftshift(fp.fft2((img).astype(float)))
    # gf = matlab_style_gauss2D(shape=img.shape, sigma=sigma)
    # gf = gf/gf.max()
    # img_filtered = img_fre*gf 
    # img_low =  fp.ifft2(fp.ifftshift(img_filtered)).real


    # Alternative 3 
    # img_low = sn.gaussian_filter(img, sigma=sigma)

    # Alternative 4
    gf = matlab_style_gauss2D(shape=np.asarray(img.shape)*2, sigma=sigma*2) # fspecial('gaussian',2.*[Mx,My],sigma.*2)
    gf = gf/gf.max()
    gf = gf[img.shape[0]:, img.shape[1]:]
    img_fre = dct2(img)  # lRnF=dct2(I)
    img_filtered = img_fre*gf  # lRnF2=lRnF.*h;
    img_low = idct2(img_filtered).real # If=real(idct2(lRnF2));
    
    return img_low 


def loc_moment(img, **kwargs):
    # local expectation/moment  
    method = kwargs.get('lm_method', 'bilateral')
    if method =='value':
        return kwargs['lm_value'] 
    elif method == 'average':
        kerne_size = kwargs.get('lm_kernel', 3) 
        kernel = get_kernel(kerne_size)
        return signal.convolve2d(img, kernel, 'same')
    elif method == 'bilateral':
        sigma = kwargs.get('lm_sigma', 110) 
        return denoise_bilateral(img, win_size=5, sigma_color=sigma, sigma_spatial=sigma)


def vst(img, sigma_0, snr_0):
    theta1 = theta1_opt(snr_0)
    theta2 = theta2_opt(snr_0)
    stab_img = sigma_0 * np.sqrt(np.maximum(theta1**2 * img**2 / sigma_0**2 - theta2, 0))
    return stab_img


def hmf_g(img, **kwargs):
    loc_first_moment = loc_moment(img, **kwargs)
    img_cent = np.abs(img - loc_first_moment)
    img_cent[img_cent==0] = 1e-3 
    img_lpf = lpf(np.log(img_cent))

    sigma = 2**0.5 * np.exp(img_lpf + np.euler_gamma / 2)
    return sigma


def hmf_r(img, snr_0, **kwargs):
    loc_first_moment = loc_moment(img, **kwargs)
    img_cent = np.abs(img - loc_first_moment)
    img_cent[img_cent==0] = 1e-3    
    img_lpf = lpf(np.log(img_cent)) 

    # WARNING: Paper (correction is done afterwards) and Code are not consistent here:
    # img_lpf = img_lpf-snr_correction(snr_0)
    # img_lpf = lpf(img_lpf, sigma=3.4+2) 
    
    sigma = 2**0.5 * np.exp(img_lpf + np.euler_gamma / 2 - snr_correction(snr_0) )
    return sigma


def ini0_ratio(x, n):
    # mask = x < 20
    # out = np.empty(x.shape)
    # out[mask] = sc.i1(x[mask]) / sc.i0(x[mask])
    # out[~mask] =(1-(4*n**2-1)/(8*x[~mask]))/(1-1/(8*x[~mask]))  # 1 - 4 * n**2 / (8 * x[~mask] + 1) 
    # return out

    # Code: http://www.lpi.tel.uva.es/node/671 
    thr = 1.5
    M = np.zeros(x.shape)

    K = x>=thr
    z8=8*x[K]
    Mn=1-3/z8-15/2/(z8)**2-(3*5*21)/6/(z8)**3
    Md=1+1/z8+9/2/(z8)**2+(25*9)/6/(z8)**3
    M[K]=Mn/Md

    K=x<thr
    M[K]=sc.i1(x[K])/sc.i0(x[K])

    K=x==0
    M[K]=0
    return M

def expectation_maximization(img, n_iter=10, **kwargs):
    loc_second_moment = loc_moment(img**2, **kwargs)
    loc_fourth_moment = loc_moment(img**4, **kwargs)
    A_k = (np.maximum(2 * loc_second_moment**2 - loc_fourth_moment, 0))**0.25
    var_k = 0.5 * np.maximum(loc_second_moment - A_k**2, 1e-4)

    for n in range(n_iter):
        A_k_bes = A_k * img / var_k
        A_k_in = ini0_ratio(A_k_bes, 1) * img
        A_k_mean = loc_moment(A_k_in)
        A_k = np.maximum(A_k_mean, 0)
        var_k = np.maximum(1 / 2 * loc_second_moment - A_k**2 / 2, 1e-4)

    return A_k, np.sqrt(var_k)


def lmmse(img, sigma, eps=1e-8):
    loc_second_moment = loc_moment(img**2)
    loc_fourth_moment = loc_moment(img**4)

    K = 1 - (4 * sigma**2 * (loc_second_moment - sigma**2)) / (loc_fourth_moment - loc_second_moment**2 + eps)
    K = np.maximum(K, 0)
    img_cor = loc_second_moment - 2 * sigma**2 + K * (img**2 - loc_second_moment)
    img_cor = np.maximum(img_cor, 0)
    return np.sqrt(img_cor)


def rlmmse(img, sigma, r=50, eps=1e-8):
    for n in range(r):
        img = lmmse(img, sigma)
        # Option 1:
        sigma = hmf_g(img)
        # Option 2:
        # A_0, sigma_0 = expectation_maximization(img)
        # snr_0 = A_0 / sigma_0
        # img_vst = vst(img, sigma_0, snr_0)
        # sigma = hmf_g(img_vst)
        # Option 3:
        # sigma = hmf_r(img, snr_0 )
        # Option 4:
        # bg = get_background(img)
        # sigma = np.sum(bg) / (bg.size) * np.sqrt(2 / np.pi)
        # sigma = np.ones(img.shape) * sigma
    return img


def noise_corrected_exp(signal, sigma, eps=1e-8):
    sigma = sigma + eps
    signal = signal + eps
    if signal.ndim == 2:
        signal = signal.T

    alpha = (signal / (2 * sigma))**2

    def cor_func(alpha, sigma):
        if alpha < 8:
            a = np.sqrt(np.pi * sigma**2 / 2) * np.exp(-alpha)
            b = (1 + 2 * alpha) * sc.i0(alpha) + 2 * alpha * sc.i1(alpha)
        else:
            # Bessel Approximation exp(-x)*I(x)= ...
            a = 0.5 * np.sqrt(sigma**2 / alpha)
            b = (0.5 + 1 / (8 * alpha) + 4 * alpha)
        return a * b

    cor_func = np.vectorize(cor_func)
    result = cor_func(alpha, sigma)

    if result.ndim == 2:
        result = result.T

    return result


if __name__ == "__main__":
    s = np.array([[1000, 500, 300, 200, 100], [1000, 500, 300, 200, 100]]) / 50
    sigma = np.array([10, 5])
    a = noise_corrected_exp(s, sigma)
    print("Signal", s)
    print("Corrected", a)
