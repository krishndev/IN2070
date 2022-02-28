import time
from scipy import signal
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt


# OPPGAVE 1.1
# Henter ut bildet:
filename = 'cow.png'
bilde = imread(filename, as_gray=True)

# Lager middelverdifilteret
dim = 15
middelverdifilter = np.zeros((dim,dim))
totaldim = dim*dim
for i in range(dim):
    for j in range(dim):
        middelverdifilter[i][j] = 1/totaldim

# Romlig konvolusjon ved hjelp av conv2 - funksjonen
def romligKonvolusjon(bilde, filter):
    resultat = signal.convolve2d(bilde,filter,'same')

    return resultat

def fourierk(bilde, filter):
    # Nullutvide filteret
    nullutvid = np.zeros(bilde.shape)
    #print(nullutvid)
    nullutvid[:filter.shape[0], :filter.shape[1]] = filter
    #print(nullutvid)

    # Fourier av filteret
    f_filter = np.fft.fft2(nullutvid)
    # Fourier av bildet
    f_bilde = np.fft.fft2(bilde)
    # Resultat
    resultat = np.fft.ifft2(f_bilde*f_filter)
    resultat = np.real(resultat)

    return resultat


konv = romligKonvolusjon(bilde, middelverdifilter)
fourier = fourierk(bilde, middelverdifilter)

# OPPGAVE 1.3
def taTiden(bilde, filterdim):
    # Lager nytt middelverdifilter
    dim = filterdim
    nyttmiddelverdifilter = np.zeros((dim, dim))
    totaldim = dim * dim
    for i in range(dim):
        for j in range(dim):
            nyttmiddelverdifilter[i][j] = 1 / totaldim

    # Tar tiden for romlig Konvolusjon
    start_tid_konv = time.time()
    romn = romligKonvolusjon(bilde, nyttmiddelverdifilter)
    stopp_tid_konv = time.time() - start_tid_konv

    # Tar tiden for fourier
    start_tid_fourier = time.time()
    fouriern = fourierk(bilde, nyttmiddelverdifilter)
    stopp_tid_fourier = time.time() - start_tid_fourier

    return (stopp_tid_konv, stopp_tid_fourier)

def plotter(bilde):
    # Tidene
    romlig_tider = []
    fourier_tider = []
    # Filterdimensjon
    n = []

    # Sjekker for filterdimensjoner mellom 5 og 50 med et intervall paa 5
    for i in range(5, 50, 5):
        n.append(i)
        romlig_tider.append(taTiden(bilde, i)[0])
        fourier_tider.append(taTiden(bilde, i)[1])

    # Plotter resultat
    plt.figure()
    plt.plot(n, romlig_tider, 'b', label='Romlig Kovolusjon')
    plt.plot(n, fourier_tider, 'r', label='Fourier')
    plt.xlabel('Filterstørrelse')
    plt.ylabel('Antall sekunder')
    plt.title('Kjøretiden i sekunder')
    plt.legend()

plotter(bilde)

plt.figure()
plt.imshow(bilde,cmap='gray',vmin=0,vmax=255)
plt.title('Originalbildet')

plt.figure()
plt.imshow(konv,cmap='gray',vmin=0,vmax=255)
plt.title('Romlig Konvolusjon')

plt.figure()
plt.imshow(fourier,cmap='gray',vmin=0,vmax=255)
plt.title('Konvolusjon i frekvensdomenet')

plt.show()