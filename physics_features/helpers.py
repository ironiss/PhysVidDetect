import cv2
import numpy as np



def safe_div(a, b, eps=1e-9):
    return float(a) / float(b + eps)


def sobel_mag_theta(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy), np.arctan2(gy, gx)


def make_ring(mask_bool, r=3):
    m= mask_bool.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
    dil = cv2.dilate(m, k, iterations=1).astype(bool)
    ero= cv2.erode(m, k, iterations=1).astype(bool)
    return dil & (~ero)


def resize_mask(mask_bool, gray):
    if mask_bool.shape == gray.shape[:2]:
        return mask_bool
    
    return cv2.resize(mask_bool.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def lap_var(gray, mask):
    vals= cv2.Laplacian(gray, cv2.CV_32F, ksize=3)[mask]
    return float(vals.var()) if vals.size>=10 else float("nan")


def iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    uni= np.logical_or(m1, m2).sum()
    if uni:
        return float(inter) / float(uni)
    else:
        return float("nan")



def circ_mean_R(theta, weights=None):
    if theta.size<10:
        return float("nan"), float("nan")
    
    w = np.ones_like(theta, dtype=np.float64) if weights is None else weights.astype(np.float64)
    C= np.sum(w*np.cos(theta))
    S = np.sum(w * np.sin(theta))
    mu = float(np.arctan2(S, C))
    R = float(np.sqrt(C*C +S*S) / (np.sum(w)+1e-9))
    return mu, R


def circ_diff(a, b):
    d= a - b
    return (d + np.pi) % (2*np.pi) - np.pi


def nanstd(x):
    x = np.asarray(x, dtype=float)
    s= np.nanstd(x)
    return float(s) if np.isfinite(s) else float("nan")


def nanmedian(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    return float(m) if np.isfinite(m) else float("nan")


def zscore(x):
    x= np.asarray(x, dtype=float)
    mu, sig = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(sig) or sig<1e-8:
        return np.full_like(x, np.nan)
    return (x - mu) /sig


def basic_stats(x):
    if x.size==0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
        "median": float(np.nanmedian(x)),
        "p95": float(np.nanpercentile(x, 95)),
    }


def diff_stats(x):
    if x.size<3:
        return {"diff_mean": float("nan"), "diff_std": float("nan"), "diff_abs_mean": float("nan")}
    d= np.diff(x)
    return {
        "diff_mean": float(np.nanmean(d)),
        "diff_std": float(np.nanstd(d)),
        "diff_abs_mean": float(np.nanmean(np.abs(d))),
    }


def highpass(gray, ksize=7):
    blur= cv2.GaussianBlur(gray, (ksize, ksize), 0)
    return gray - blur


def crop_bbox(arr, mask, pad=2):
    ys, xs = np.where(mask)
    if xs.size==0:
        return None, None
    
    x1, x2= max(0, xs.min()-pad), min(arr.shape[1]-1, xs.max()+pad)
    y1, y2 = max(0, ys.min()-pad), min(arr.shape[0]-1, ys.max()+pad)
    return arr[y1:y2+1, x1:x2+1], mask[y1:y2+1, x1:x2+1]


def radial_psd_slope(img, mask=None, min_size=32, max_size=128, f_low=0.08, f_high=0.45):
    if img is None:
        return float("nan")
    
    if mask is not None:
        crop_img, crop_mask = crop_bbox(img, mask)
        if crop_img is None or crop_mask.sum()<20:
            return float("nan")
        
        m = crop_mask.astype(np.float32)
        mean_val= float(np.sum(crop_img*m) / (m.sum()+1e-9))
        crop_img = crop_img.copy()
        crop_img[m<0.5] = mean_val
        img= crop_img

    h, w = img.shape
    if h<min_size or w<min_size:
        return float("nan")
    
    scale = min(max_size/max(h, w), 1.0)
    if scale<1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape

    F= np.fft.fftshift(np.fft.fft2(img - img.mean()))
    psd = np.abs(F) **2
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y -cy)**2 + (X-cx)**2)
    r_max= r.max()
    
    if r_max<2:
        return float("nan")

    nbins = min(64, int(r_max))
    bins= np.linspace(0, r_max, nbins+1)
    radii = 0.5*(bins[:-1] +bins[1:])
    psd_r = np.zeros(nbins)
    cnt= np.zeros(nbins)
    idx = np.clip(np.digitize(r.flatten(), bins)-1, 0, nbins-1)
    np.add.at(psd_r, idx, psd.flatten())
    np.add.at(cnt, idx, 1.0)
    psd_r /= (cnt+1e-12)

    r_norm = radii/(r_max +1e-12)
    band = (r_norm>=f_low) & (r_norm<=f_high) & (psd_r>0) & (radii>0)
    if band.sum()<3:
        return float("nan")

    log_r= np.log(radii[band])
    log_p = np.log(psd_r[band])
    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope= np.linalg.lstsq(A, log_p, rcond=None)[0][0]
    return float(slope) if np.isfinite(slope) else float("nan")
