import numpy as np
import math
import cv2

def calculate_uicm(img, alphaL, alphaR):

    # Calculate RG and YB values for the whole image
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    rg = r - g
    rg = rg.flatten()
    rg.sort()

    yb = (r + g) // 2 - b
    yb = yb.flatten()
    yb.sort()

    # Calculate the mean values for RG and YB within the thresholded range
    TalphaL = math.ceil(len(rg) * alphaL)
    TalphaR = math.floor(len(rg) * alphaR)

    meanRG = np.mean(rg[TalphaL:-TalphaL])
    meanYB = np.mean(yb[TalphaL:-TalphaL])

    varianceRG = np.var(rg[TalphaL:-TalphaL])
    varianceYB = np.var(yb[TalphaL:-TalphaL])

    # Calculate UICM
    result = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * math.sqrt(varianceRG + varianceYB)
    print(f"UICM: {result}")
    return result
