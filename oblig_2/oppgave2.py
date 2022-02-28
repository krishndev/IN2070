from scipy import signal
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

# Henter ut bildet:
filnavn = 'uio.png'
q = np.array([0.1, 0.5, 2, 8, 32])

# Steg 1 og 2
# "hovedmetode"
def kompresjon(filnavn, q):
    bilde = imread(filnavn, as_gray=True)
    # Steg 5 - Q
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    qQ = np.multiply(q, Q)
    O, P = bilde.shape
    bildet = bilde
    bildet -= 128

    O_b = int(O / 8)
    P_b = int(P / 8)
    blokker = np.zeros((O_b, P_b), dtype=np.ndarray)
    rekonstruerte_blokker = np.zeros((O_b, P_b), dtype=np.ndarray)

    for i in range(0, O, 8):
        for j in range(0, P, 8):
            blokker[int(i / 8)][int(j / 8)] = transformer(bildet[i:i + 8, j:j + 8])
            blokker[int(i / 8)][int(j / 8)] = np.round(np.divide(blokker[int(i / 8)][int(j / 8)], qQ))

    # Resultat (Steg 6)
    entropien = beregn_entropi(utvid_blokker(blokker, bildet))
    print('Entropien for bilde med kompresjonsrate', q, '=', entropien)

    for i in range(O_b):
        for j in range(P_b):
            blokker[i][j] = np.round(np.multiply(blokker[i][j], qQ))
            rekonstruerte_blokker[i][j] = inv_transformer(blokker[i][j])

    rekonstruerte_blokker += 128

    rekonstruert = utvid_blokker(rekonstruerte_blokker, bilde)

    plt.imsave('rekonstruert_uio_q={}.png'.format(q), rekonstruert, cmap='gray')
    print("--lagret bilde--")

# Steg 3 - transformere
def transformer(f):
    F = np.zeros((8, 8))
    for v in range(8):
        for u in range(8):
            # Forste del av uttrykket
            del1 = (1 / 4) * c(u) * c(v)
            # Andre del av uttrykket
            del2 = 0
            # Folger formelen
            for y in range(8):
                for x in range(8):
                    del2 += f[y][x] * np.cos(((2 * y + 1) * u * np.pi) / 16) * np.cos(((2 * x + 1) * v * np.pi) / 16)

            F[u][v] = del1 * del2
    return F

# Steg 4 - rekonstruere, invers transform
def inv_transformer(F):
    f = np.zeros((8, 8))
    for y in range(8):
        for x in range(8):
            summen = 0
            for v in range(8):
                for u in range(8):

                    # Folger formelen
                    summen += c(u) * c(v) * F[u][v] * np.cos((2 * x + 1) * v * np.pi / 16) * np.cos((2 * y + 1) * u * np.pi / 16)
            f[y][x] = round(1/4 * summen)
    return f

# Hjelpemetode som bestemmer c(a)
def c(a):
    if (a == 0):
        c_uv = 1 / np.sqrt(2)
        return c_uv
    else:
        return 1

# Metode for aa regne ut entropien (Steg 6)
def beregn_entropi(A):
    # Finner histogram
    intensities = np.arange(256)
    p = np.zeros(256)
    for i in range(256):
        p[i] = np.sum(A == intensities[i])
    # Det normaliserte histogrammet
    p /= A.size

    # Finner entropien
    h_entropi = 0
    for i in range(256):
        p_i = p[i]
        if p_i > 1E-11:
            h_entropi += p_i * np.log2(p_i)
    print("entropi funnet")
    return -h_entropi

# Hjelpemetode
def utvid_blokker(blokker, bilde):
    O_b, P_b = bilde.shape
    resultat = np.zeros((O_b, P_b))

    for i in range(0, O_b, 8):
        for j in range(0, P_b, 8):
            resultat[i:i + 8, j:j + 8] = blokker[int(i / 8)][int(j / 8)]
    return resultat

for i in q:
    kompresjon(filnavn, i)


