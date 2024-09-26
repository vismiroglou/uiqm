import numpy as np

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
    TalphaL = int(np.ceil(len(rg) * alphaL))
    TalphaR = int(np.floor(len(rg) * alphaR))

    meanRG = np.mean(rg[TalphaL:-TalphaR])
    meanYB = np.mean(yb[TalphaL:-TalphaR])

    varianceRG = np.var(rg[TalphaL:-TalphaR])
    varianceYB = np.var(yb[TalphaL:-TalphaR])

    # Calculate UICM
    result = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(varianceRG + varianceYB)
    print(f"UICM: {result}")
    return result
