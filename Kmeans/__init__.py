
import numpy as np
import matplotlib.pyplot as plt


def kmeanR1():
 

    plt.style.use('dark_background')

    # Make data

    x1 = np.random.standard_normal(50) * 0.6 + np.ones(50)

    x2 = np.random.standard_normal(50) * 0.5 - np.ones(50)

    X = np.concatenate((x1, x2), axis=0)



    num_clusters = 2

    # Inicializar los centroides de forma aleatoria
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]

    # Criterio de parada
    max_iterations = 150
    tolerance = 1e-4

    # Inicializar array old_centroids que iniciara en 0

    old_centroids = np.zeros_like(centroids)

    for _ in range(max_iterations):

        distances = np.abs(X[:, np.newaxis] - centroids)

        clusters = np.argmin(distances, axis=1)

        # Actualizar los centroides
        old_centroids = centroids.copy()

        for i in range(num_clusters):
            centroids[i] = np.mean(X[clusters == i])

            plt.plot(X, np.zeros_like(X), 'w.', markersize=8, label='dataPoints')
            plt.plot(centroids, np.zeros_like(centroids), 'rx', markersize=17, label='Centroides')
            plt.legend()
            plt.show()

            # Verificar la convergencia

        if np.mean(np.abs(centroids - old_centroids)) <= tolerance:
            centro1 = centroids[0]
            centro2 = centroids[1]
            break



def kmeanR1K3(n):
  

    plt.style.use('dark_background')

    # Make data

    x1 = np.random.standard_normal(n) * 0.6 + np.ones(n)

    x2 = np.random.standard_normal(n) * 0.5 - np.ones(n)

    x3 = np.random.standard_normal(n) * 0.4 + np.ones(n) * 2.5

    X = np.concatenate((x1, x2, x3), axis=0)

    num_clusters = 3

    # Inicializar los centroides de forma aleatoria
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]

    # Criterio de parada
    max_iterations = 250
    tolerance = 1e-4

    # Inicializar array old_centroids que iniciara en 0

    old_centroids = np.zeros_like(centroids)

    for _ in range(max_iterations):

        distances = np.abs(X[:, np.newaxis] - centroids)

        clusters = np.argmin(distances, axis=1)

        # Actualizar los centroides
        old_centroids = centroids.copy()

        for i in range(num_clusters):
            centroids[i] = np.mean(X[clusters == i])

            plt.plot(X, np.zeros_like(X), 'w.', markersize=8, label='dataPoints')
            plt.plot(centroids, np.zeros_like(centroids), 'rx', markersize=17, label='Centroides')
            plt.legend()
            plt.show()

            # Verificar la convergencia

        if np.mean(np.abs(centroids - old_centroids)) <= tolerance:
            centro1 = centroids[0]
            centro2 = centroids[1]
            centro3 = centroids[2]
            break
            


def kmeanR2():
    # Definimos variables
    plt.style.use('dark_background')

    x1 = np.random.standard_normal((150, 2)) * 0.6 + np.ones((150, 2))

    x2 = np.random.standard_normal((150, 2)) * 0.5 - np.ones((150, 2))
    X = np.concatenate((x1, x2), axis=0)

    num_clusters = 2
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]
    max_range = 150
    tolerance = 1e-4

    old_centroids = np.zeros_like(centroids)

    for _ in range(max_range):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)

        clusters = np.argmin(distances, axis=1)

        old_centroids = np.copy(centroids)
        for i in range(num_clusters):
            centroids[i] = X[clusters == i].mean(axis=0)
            plt.plot(X[:, 0], X[:, 1], 'w.', label='Puntos')
            plt.plot(centroids[:, 0], centroids[:, 1], 'rx', markersize=10, label='Centroides')
            plt.legend()
            plt.show()
            # Verificar la convergencia
        if np.linalg.norm(centroids - old_centroids) < tolerance:
            centro1 = centroids[0]
            centro2 = centroids[1]
            break


kmeanR2()
