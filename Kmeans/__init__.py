'''
 Author: JuanEstebanLG
'''

'Importamos las librerias necesarios'
import numpy as np 'numpy nos ayudara con la matematica y operaciones entre arrays del proyecto'
import matplotlib.pyplot as plt 'matplotlib nos ayudara a graficar los puntos y los centroids'

'Algoritmo para R1, con k = 2'
def kmeanR1():
    plt.style.use('dark_background')

    # Inicializamos las variables

    x1 = np.random.standard_normal(50) * 0.6 + np.ones(50)

    x2 = np.random.standard_normal(50) * 0.5 - np.ones(50)

    X = np.concatenate((x1, x2), axis=0)


    #El numero de clusters representara la cantidad de grupos de datos
    num_clusters = 2

    # Inicializar los centroides de forma aleatoria
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]
    print(centroids)
    # Criterio de parada
    max_iterations = 150
    tolerance = 1e-4

    # Inicializar array old_centroids que iniciara en 0

    old_centroids = np.zeros_like(centroids)

    for _ in range(max_iterations):
        
        #Calculamos las distancias en cada iteracion como la el valor absoluto de la diferencia entre los datos en X y los centroides
        distances = np.abs(X[:, np.newaxis] - centroids)

        #Asignamos los clusters como el punto con menor distancia
        clusters = np.argmin(distances, axis=1)

        # Actualizar los centroides
        old_centroids = centroids.copy()

        for i in range(num_clusters):
            #Se re asigna cada centroid como la media de su respectivo cluster
            centroids[i] = np.mean(X[clusters == i])

            #Grafica
            plt.plot(X, np.zeros_like(X), 'w.', markersize=8, label='dataPoints')
            plt.plot(centroids, np.zeros_like(centroids), 'rx', markersize=17, label='Centroides')
            plt.legend()
            plt.show()

            # Verificar la convergencia, para este caso, si el cambio en los centroids es menor o igual a 0.0001 se rompe el ciclo

        if np.mean(np.abs(centroids - old_centroids)) <= tolerance:
            break



def kmeanR2():
    # Definimos variables
    plt.style.use('dark_background')

    x1 = np.random.standard_normal((150, 2)) * 0.6 + np.ones((150, 2))

    x2 = np.random.standard_normal((150, 2)) * 0.5 - np.ones((150, 2))
    X = np.concatenate((x1, x2), axis=0)

    num_clusters = 2

    #Se asigna de forma al azar cada centroid 
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]
    max_range = 150
    tolerance = 1e-4

    old_centroids = np.zeros_like(X)

    for _ in range(max_range):
        #Se calcula la distancia utilizando la distancia euclidiana entre vectores
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)

        #Se asigna cada cluster como el punto con distancia menor en el eje Y
        clusters = np.argmin(distances, axis=1)

        #Se guarda una copia de los centroides 
        old_centroids = np.copy(centroids)
        for i in range(num_clusters):
            #Se reasignan los centroides como la media de los puntos en su cluster a lo largo del eje X
            centroids[i] = X[clusters == i].mean(axis=0)

            #Grafica
            plt.plot(X[:, 0], X[:, 1], 'w.', label='Puntos')
            plt.plot(centroids[:, 0], centroids[:, 1], 'rx', markersize=10, label='Centroides')
            plt.legend()
            plt.show()
            # Verificar la convergencia
        if np.linalg.norm(centroids - old_centroids) < tolerance:
            break


kmeanR1()
