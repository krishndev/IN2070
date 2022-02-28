from imageio import imread, imwrite
import numpy as np
import matplotlib.pyplot as plt


# Generell implementasjon av konvolusjon
def generellkonvolusjon(image, filter):
    M,N = image.shape
    O,P = filter.shape

    # Rotere filteret 180 grader
    filter = np.rot90(np.rot90(filter))

    # m = O = 2a + 1
    a = int((O-1)/2)
    # n = P = 2b + 1
    b = int((P-1)/2)

    overlapp = np.pad(image,((a,a),(b,b)), mode='edge')
    f_out = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
            verdi = overlapp[i:(i+2*a+1),j:(j+2*b+1)]*filter
            f_out[i,j] = np.sum(verdi)

    return f_out

# Gauss filteret
def gaussf(sigma):
    # filterets dimensjoner
    dim = round(1 + 8*sigma)
    N, M = dim, dim
    f_out = np.zeros((N, M))
    # Senterpikselen
    senter = int((dim-1)/2)
    sum = 0
    for i in range(-senter, senter+1):
        for j in range(-senter, senter+1):
            a = np.exp(-((i ** 2 + j ** 2) / (2.0 * sigma ** 2))) * (1 / (2.0 * np.pi * sigma**2))
            f_out[senter + i, senter + j] = a
            sum+=a

    return f_out/sum

# Finne gradient retning og magnitude
def gradient_estimering(bilde):
    # Symmetrisk 1D-operator
    h_x = np.array([[0,1,0],
                   [0,0,0],
                   [0,-1,0]])

    h_y = np.array([[0,0,0],
                   [1,0,-1],
                   [0,0,0]])

    # Finner de horisontale kantene
    g_x = generellkonvolusjon(bilde, h_x)
    # Finner de vertikale kantene
    g_y = generellkonvolusjon(bilde, h_y)

    # Gradient-magnitude
    magnitude = np.sqrt(g_x**2+g_y**2)
    max = np.max(magnitude)
    min = np.min(magnitude)
    magnitude = ((255 - 0) / (max - min)) * (magnitude - min)


    # Gradientretning
    retning = np.arctan2(g_y,g_x)
    # radianer til grader
    retning = retning*(180/np.pi)

    return magnitude, retning

# Kant-tynning
def kant_tynning(M,theta):
    m,n = theta.shape
    f_out = np.zeros((m, n), dtype=np.int32)


    for i in range(1, m-1):
        for j in range(1, n-1):
            retningen = theta[i, j]
            q = 255
            r = 255

            # Horisontal
            if (0 <= retningen < 22.5) or (retningen < 157.5):
                q = M[i, j + 1]
                r = M[i, j - 1]
            # 45
            elif (22.5 < retningen <= 67.5) or (-112.5 > retningen >= -157.5):
                q = M[i + 1, j - 1]
                r = M[i - 1, j + 1]
            # Vertikal
            elif (67.5 < retningen <= 112.5):
                q = M[i + 1, j]
                r = M[i - 1, j]
            # -45
            elif (112.5 < retningen <= 157.5) or (-22.5 > retningen >= -67.5):
                q = M[i - 1, j - 1]
                r = M[i + 1, j + 1]

            # Gjor ingenting dersom naboene ikke har storre gradientmagnitude
            if (M[i, j] >= q) and (M[i, j] >= r):
                f_out[i, j] = M[i, j]
            # Dersom de har storre gradientmagnitude, settes verdien til 0
            else:
                f_out[i, j] = 0
    return f_out


# Hystereseterskling
def hystereseterskling(g_N, T_l, T_h):
    M, N = g_N.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            # Sjekker om piksel er lavterskel
            if (g_N[i, j] == T_l):
                # Sjekker om noen av naboene er lik hoyterskel
                if ((g_N[i + 1, j - 1] == T_h) or (g_N[i + 1, j] == T_h) or (g_N[i + 1, j + 1] == T_h)
                    or (g_N[i, j - 1] == T_h) or (g_N[i, j + 1] == T_h)
                    or (g_N[i - 1, j - 1] == T_h) or (g_N[i - 1, j] == T_h) or (g_N[i - 1, j + 1] == T_h)):
                    # Dersom det er slik at minst en av naboene er lik hoyterskel, settes pikselen lik hoyterskel
                    g_N[i, j] = T_h
                else:
                    g_N[i, j] = 0
    return g_N

def canny():
    # testmatriser
    # filter = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9],[1/9, 1/9, 1/9]])
    # innbilde = np.array([[1, 3, 2, 1], [5, 4, 5, 3], [4, 1, 1, 2], [2, 3, 2, 6]])


    sigma = 3
    Tl = 30
    Th = 85

    cellekjerner = imread("cellekjerner.png", as_gray = True)

    # Finner gaussfilteret
    filter = gaussf(sigma)

    # Legger paa gaussfilteret
    filtrert_img = generellkonvolusjon(cellekjerner, filter)
    print("forste steg ferdig")

    # Finner magnitude og retning
    magnitude,retning = gradient_estimering(filtrert_img)
    print("andre steg ferdig")

    # Kant-tynning
    tynnet_M = kant_tynning(magnitude,retning)
    print("tredje steg ferdig")

    # Hystereseterskling
    tersklet = hystereseterskling(tynnet_M,Tl,Th)
    print("siste steg ferdig")

    plt.figure()
    plt.imshow(tersklet, cmap='gray', vmin=0, vmax=255)
    plt.title('resultat bildet')

    plt.figure()
    plt.imshow(cellekjerner, cmap='gray', vmin=0, vmax=255)
    plt.title('original')

canny()

plt.show()

