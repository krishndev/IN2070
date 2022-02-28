from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import math

# Henter ut bildet:
filename = 'portrett.png'
f = imread(filename, as_gray=True)
# Bildet for transformasjonen
plt.imshow(f,cmap='gray')
intensities = np.arange(256)

# Finne a og b forst, ogsaa utfore graatransformasjon etter det
# For aa finne a og b maa vi finne My og o. Har allerede My_T og o_T fra oppgaveteksten

def finnHistogram():
    # Finner det normaliserte histogrammet til bildet
    p = np.zeros(256)
    for i in range(256):
        p[i] = np.sum(f == intensities[i])

    # Det normaliserte histogrammet
    p /= f.size
    return p

def finnMiddelverdi():
    # My tilsvarer summen av det normaliserte histogrammet ganget med itensiteten
    p = finnHistogram()
    My = sum(intensities*p)
    print("My: ", My)
    return My

def finnStandardavvik():
    # Skal naa finne o
    p = finnHistogram()
    ledd1 = sum((intensities**2)*p)
    ledd2 = (sum(intensities*p))**2

    #varians
    o_2 = ledd1 - ledd2

    # Finner o, standardavvik
    o = math.sqrt(o_2)
    print("o:", o)
    return o

def finnGraaTransformasjon():
    # Har alle variablene som trengs
    o_T = 64
    My_T = 127
    o = finnStandardavvik()
    My = finnMiddelverdi()

    # finner a
    a = o_T/o
    print("a: ", a)
    # finner b
    b_ledd = a*My
    b = My_T-b_ledd
    print("b: ", b)

    # Har naa a og b, kan naa finne graatonetransform
    print(f.shape)

    N,M = f.shape
    f_graa = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            f_graa[i,j] = a*f[i,j]+b

    return f_graa


# Finner graatonetransformasjon
f_graa = finnGraaTransformasjon()

plt.figure()
plt.imshow(f_graa,cmap='gray',vmin=0,vmax=255)
plt.title('Mellomresultat - graatransform')


# Skal nå finne den affine transformasjonen

# Maa finne transformmasjonsmatrisen forst (koeffisientene)

# Finner koeffisientene
def finne_a_b(G, d):
    G_T = np.transpose(G)
    ledd1 = np.linalg.inv(G_T @ G)
    a_b = ledd1 @ G_T @ d

    return a_b

#Har brukt imshow() og funnet tre punkter (begge oynene og nesetipp)
G = np.array([[86.6, 82.8, 1],
              [66.1, 118, 1],
              [97.2, 116.3, 1]])
d_x = np.array([[255],
                [257],
                [385]])
d_y = np.array([[166],
                [338],
                [256]])

a_koeffisienter = finne_a_b(G, d_x)
b_koeffisienter = finne_a_b(G, d_y)
print("a:", a_koeffisienter)
print("b: ", b_koeffisienter)


# FORLENGSMAPPING
def forlengstransformasjon(f_in, f_geometrimatrise, matrise):
    # Storrelsen til originale
    N, M = f_in.shape
    # Storrelsen til den nye
    O, P = f_geometrimatrise.shape
    # Bildet etter resultatet
    f_out = np.zeros((O, P))

    # Folger pseudo-koden
    for i in range(N):
        for j in range(M):
            vec_in = np.array([i, j, 1])
            vec_out = matrise @ vec_in
            # Utforer transform paa hver av pikslene
            x = int(vec_out[0])
            y = int(vec_out[1])
            if (x in range(O) and y in range(P)):
                f_out[round(x), round(y)] = f_in[i, j]
    return f_out

# BAKLENGS MAPPING
# narmeste nabo - interpolasjon
def narmesteNabo(f_in, f_geometrimaske, matrise):
    O, P = f_geometrimaske.shape

    f_out = np.zeros((O, P))

    matrise_inv = np.linalg.inv(matrise)

    for i in range(O):
        for j in range(P):
            vec_in = np.array([i, j, 1])
            vec_out = matrise_inv @ vec_in

            x = int(vec_out[0])
            y = int(vec_out[1])

            if (x in range(O) and y in range(P)):
                f_out[i, j] = f_in[round(x), round(y)]
    return f_out

# bilinear - interpolasjon
def bilinear(f_in, f_geometrimaske, matrise):
    O, P = f_geometrimaske.shape

    f_out = np.zeros((O, P))

    matrise_inv = np.linalg.inv(matrise)

    for i in range(O):
        for j in range(P):
            vec_in = np.array([i, j, 1])
            vec_out = matrise_inv @ vec_in

            x = int(vec_out[0])
            y = int(vec_out[1])

            if (x in range(O) and y in range(P)):
                x_0 = math.floor(x)
                y_0 = math.floor(y)
                x_1 = math.ceil(x)
                y_1 = math.ceil(y)
                d_x = x - x_0
                d_y = y - y_0
                p = f_in[x_0, y_0] + (f_in[x_1, y_0] - f_in[x_0, y_0])*d_x
                q = f_in[x_0, y_1] + (f_in[x_1, y_1] - f_in[x_0, y_1])*d_y
                f_out[i, j] = p + ((q-p)*d_y)
    return f_out



#Transformmatrisen
matrise = np.array([[(4.25429534), (2.53446177), (-323.27541114)],
                   [(-2.44746997), (3.4609905), (91.38088634)],
                   [0, 0, 1]])
print(np.linalg.inv(matrise))


# Geometrimasken:
filename = 'geometrimaske.png'
f_geometrimaske = imread(filename, as_gray=True)

# Forlengsmapping
f_forlengs = forlengstransformasjon(f_graa, f_geometrimaske, matrise)
# Baklengsmapping (narmeste nabo)
f_narmestenabo = narmesteNabo(f_graa, f_geometrimaske, matrise)
# Baklengsmapping (bilinear)
f_bilinear = bilinear(f_graa, f_geometrimaske, matrise)

plt.figure()
plt.imshow(f_geometrimaske,cmap='gray',vmin=0,vmax=255)
plt.title('Geometrimaske')

plt.figure()
plt.imshow(f_forlengs,cmap='gray',vmin=0,vmax=255)
plt.title('Forlengstransform')

plt.figure()
plt.imshow(f_narmestenabo,cmap='gray',vmin=0,vmax=255)
plt.title('Baklengs - Naermeste Nabo')

plt.figure()
plt.imshow(f_bilinear,cmap='gray',vmin=0,vmax=255)
plt.title('Baklengs - Bilinear')


plt.show()





