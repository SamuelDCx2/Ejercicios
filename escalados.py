import numpy as np
import matplotlib.pyplot as plt



edad = np.array([17, 18, 18, 19, 20,
                21, 21, 40, 41, 42,
                42, 43, 44, 44, 45,
                50, 54, 60, 61, 67,
                67, 68, 76, 79, 83])

estudio = np.array([6, 6, 7, 7, 8,
                    8, 9, 10, 10, 10, 
                    11, 11, 11, 11, 12,
                    14, 16, 16, 16, 16,
                    16, 17, 19, 19, 22])

ingreso = np.array([11230, 9624, 13798, 3215, 19169, 
                    14982, 15419, 17286, 14232, 18092,
                    18318, 16260, 22347, 16710, 28949,
                    27309, 32779, 29743, 30341, 34088,
                    32435, 33909, 32263, 42067, 42295])

datos = np.stack((edad, estudio, ingreso), axis=1)



fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection="3d")

ax.scatter(datos.T[0], datos.T[1], datos.T[2],
        marker="o", c="purple", s=600, alpha=0.25)
ax.scatter(datos.T[0], datos.T[1], datos.T[2],
        marker=".", c="black", s=100)

ax.set_xlabel("Edad", fontsize=15, color="blue")
ax.set_ylabel("Años de estudio", fontsize=15, color="blue")
ax.set_zlabel("Ingreso ($)", fontsize=15, color="blue")

plt.show()

from sklearn import preprocessing

escalador = preprocessing.MinMaxScaler()
escalados = escalador.fit_transform(datos)



fig = plt.figure(figsize=(12, 12))

ax = fig.add_subplot(1, 2, 1, projection="3d")

ax.scatter(escalados.T[0], 
        escalados.T[1], 
        escalados.T[2],
        marker="o", c="purple", s=600, alpha=0.65)

ax.set_title("Datos Escalados", fontsize=15)
ax.set_xlabel("Edad", fontsize=10, color="blue")
ax.set_ylabel("Años de estudio", fontsize=10, color="blue")
ax.set_zlabel("Ingreso ($)", fontsize=10, color="blue")

ax = fig.add_subplot(1, 2, 2, projection="3d")

ax.scatter(datos.T[0], 
        datos.T[1], 
        datos.T[2],
        marker="o", c="darkblue", s=600, alpha=0.65)

ax.set_title("Datos Originales", fontsize=15)
ax.set_xlabel("Edad", fontsize=10, color="blue")
ax.set_ylabel("Años de estudio", fontsize=10, color="blue")
ax.set_zlabel("Ingreso ($)", fontsize=10, color="blue")



plt.show()

originales = escalados

faltantes = escalados.copy()
faltantes[[2, 7, 12, 17, 22], 2] = np.nan
faltantes

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights="uniform")
imputados_uni = imputer.fit_transform(faltantes)

imputer = KNNImputer(n_neighbors=5, weights="distance")
imputados_dis = imputer.fit_transform(faltantes)

np.stack((imputados_uni, imputados_dis), axis=1)

fig = plt.figure(figsize=(8, 8))

filtro = ~np.isnan(faltantes.T[2])

ax = fig.add_subplot(1, 1, 1, projection="3d")

ax.scatter(escalados.T[0][filtro], 
           escalados.T[1][filtro], 
           escalados.T[2][filtro],
           label="Originales",
           marker="o", c="blue", s=600, alpha=0.15)

ax.scatter(escalados.T[0][~filtro], 
           escalados.T[1][~filtro], 
           escalados.T[2][~filtro],
           label="Quitados",
           marker="*", c="cyan", s=800, alpha=0.65)

ax.scatter(imputados_uni.T[0][~filtro], 
           imputados_uni.T[1][~filtro], 
           imputados_uni.T[2][~filtro],
           label="Imp uniforme",
           marker="D", c="deeppink", s=600)

ax.scatter(imputados_dis.T[0][~filtro], 
           imputados_dis.T[1][~filtro], 
           imputados_dis.T[2][~filtro],
           label="Imp distancia",
           marker="s", c="darkorange", s=600)

ax.set_xlabel("Edad", fontsize=15, color="blue")
ax.set_ylabel("Años de estudio", fontsize=15, color="blue")
ax.set_zlabel("Ingreso ($)", fontsize=15, color="blue")
ax.legend(ncol=4)

plt.show()