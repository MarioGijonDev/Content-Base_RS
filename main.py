
# Para ejecutar el codigo:
# Instalar pandas, numpy y scikit-learn
#   - conda install pandas numpy
# Asegurarse que se encuentra el directorio rs-movie-cour con los siguientes ficheros_
#   - movie-tags.csv
#   - movie-titles.csv
#   - ratings.csv
#   - rating.users.csv
# Iniciar con "python3 main.py" en la terminal

# PASOS A SEGUIR
"""
1. Acceso a los datos.
  a. Crear los objetos para almacenar los datos proporcionados en los ficheros (usuarios, productos, tags y ratings)
  b. Crear las interfaces que acceden a los ficheros para obtener los datos de los ficheros y así crear los objetos para usuarios, productos, tags y ratings.
  c. Comprobar que el acceso a los datos es correcto

2. Implementación del Sistema de Recomendación Basado en Contenido: dicho sistema debe seguir los siguientes pasos
  a. Construcción del modelo de recomendación basado en la técnica TF-IDF.
    i. Implementación del cálculo de la matriz de valores IDF
    ii. Implementación del cálculo del perfil de producto, TF-IDF
  b. Recomendación
    i. Cálculo del perfil del usuario activo.
    ii. Calcular el conjunto de productos no valorados del usuario activo.
    iii. Similitud entre vectores de producto no valorados y de usuario, coseno.

3. Funcionamiento del sistema.
  a. La interfaz de interacción con el sistema consistirá en solicitar el Iduser del usuario activo al que se quiere hacer una recomendación y devolverá las 10 primeras recomendaciones con mayor valor de predicción.

4. Prueba y evaluación del sistema
  a. Para comprobar el funcionamiento del sistema, los valores a recomendar que se
    deberían obtener son los siguientes:

    Top 10 recommendations for user 1:

    Seven (a.k.a. Se7en) (1995): 0,1117
    Star Wars: Episode IV - A New Hope (1977): 0,0818
    Star Wars: Episode VI - Return of the Jedi (1983): 0,0816
    Catch Me If You Can (2002): 0,0809
    The Departed (2006): 0,0654
    Star Wars: Episode V - The Empire Strikes Back (1980): 0,0634
    Braveheart (1995): 0,0557
    Fight Club (1999): 0,0533
    Snatch (2000): 0,0459
    Memento (2000): 0,0383

    --------------------------------

    Top 10 recommendations for user 500:

    Twelve Monkeys (a.k.a. 12 Monkeys) (1995): 0,1127
    E.T. the Extra-Terrestrial (1982): 0,1037
    The Incredibles (2004): 0,0937
    X-Men (2000): 0,0935
    X2: X-Men United (2003): 0,0911
    Donnie Darko (2001): 0,0450
    Monsters Inc. (2001): 0,0429
    V for Vendetta (2006): 0,0378
    Apollo 13 (1995): 0,0343
    Finding Nemo (2003): 0,0333
"""
    

# IMPORTS
import pandas as pd
import numpy as np
import re
from math import log

def srbc():

  # Obtenemos el id del usuario activo
  idUser = requestIdUser()

  # Obtenemos los datos de los CSV
  arrayDfs = getDataframesFromDatabase()

  # Comprobar el acceso correcto a los datos
  # checkDfs(arrayDfs, 10)

  # Destructuración del array para obtener los dataframes de los datos obtenidos
  movieTitlesDf, movieTagsDf, usersDf, ratingsDf = arrayDfs

  # Creamos la matriz TF-IDF cono dataframe (es más facil de operar que como matriz dispersa)
  # tfidf = createTfidf(movieTagsDf)
  tfidf = createTfidf(movieTagsDf)

  # Normalización l2
    #   (La similitud del coseno entre dos vectores es su producto escalar)
    #   (divide cada vector por la norma Euclidiana, es decir, la raíz cuadrada de la suma de los cuadrados de los valores de sus elementos.)

  # IMPORTANTE: La biblioteca sklearn ya te da la opción de normalizarlo con l1 o l2 (en este caso, l2)

  # Normalizamos la matriz para que todos tengan modulo 1
  #   Para evitar que items con muchas etiquetas (Ej. populares) tengan mayor fuerza de ponderación
  # tfidf = normalizeTfidf(tfidf)

  # Obtenemos una lista con los idItem de los items valorados por el usuario
  itemsRated = getRatedItemsByUser(idUser, ratingsDf, movieTitlesDf)

  # Obtenemos la resta entre la valoración del usuario al item y la valoración media del usuario
  # Esto se hará por cada item
  ratingAndMeanRatingDifference = getRatingAndMeanRatingDifference(idUser, ratingsDf)

  # Obtenemos el perfil del usuario a partir de los items valroados, la matriz TF-IDF y la diferencia anterior
  activeUserProfile = getActiveUserProfile(itemsRated, tfidf, ratingAndMeanRatingDifference)

  # Obtenemos la matriz de vectores de los items que el usuario no ha valorado
  unratedItemsMatrix = getUnratedItemsMatrix(itemsRated, tfidf)

  # Obtenemos un dataframe con el idItem y la similitud del coseno respecto al perfil del usuario (de cada item)
  # Ordenado de mayor a menor similitud
  topCosineSimilarity = getCosineSimilarity(activeUserProfile, unratedItemsMatrix)
  
  # Formateamos el dataframe para añadir el nombre de las peliculas según su respectivo idItem
  topCosineSimilarity['nameItem'] = topCosineSimilarity.index
  topCosineSimilarity['nameItem'] = topCosineSimilarity['nameItem'].apply(lambda id: getItemName(id, movieTitlesDf))

  # Creamos una lista con las top 10 recomendaciones, en el formato solicitado
  # nameItem: cosineSimilarity
  topRecommendations = formattedResult(topCosineSimilarity)
  
  # Y mostramos en la terminal las top 10 recomendaciones
  displayResult(idUser, topRecommendations)



def displayResult(idUser, topRecommendations):
  print(f'\nTOP 10 RECOMMENDATIONS FOR USER {idUser}\n')
  for item in topRecommendations:
    print(item)
  print('\n')

def formattedResult(topCosineSimilarity):
  topRecommendations = []
  count = 0
  for _, row in topCosineSimilarity.iterrows():
      topRecommendations.append(f"{row['nameItem']}:  {round(row['cosineSimilarity'], 4)}")
      count += 1
      if count == 10:
        break
  return topRecommendations

def getCosineSimilarity(activeUserProfile, unratedItemsMatrix):

  # Similitud del coseno = producto escalar entre dos vectoes / producto de las normas de los dos vectores = (u · v) / (||u|| * ||v||)

  # Calculamos el producto escalar entre el perfil del usuario activo y los vectores de los items no valorados
  # Esto nos devolverá una matriz con el resultado del producto escalar de cada item
  dotProduct = np.dot(activeUserProfile, unratedItemsMatrix.T)

  # Calculamos el producto entre las normas del perfil de usuario activo y los vectores de los items no valorados
  # Esto nos devolverá una matriz con el resultado del producto por cada item
  normProduct = np.linalg.norm(activeUserProfile) * np.linalg.norm(unratedItemsMatrix, axis=1)

  # Calcular la similitud del coseno dividiendo el producto escalar por el producto de las normas, por cada item
  # Cuanto mayor es el resultado, mayor es la similitud del item con el usuario y por ende, más recomendado
  similarity = dotProduct / normProduct

  # Creamos un dataframe con los valores de similitud obtenidos y sus respectivos items
  similarityDf = pd.DataFrame(similarity, index=unratedItemsMatrix.index, columns=['cosineSimilarity'])

  # Ordenamos los items según la similitud con el usuario de mayor a menor
  topsimilarity = similarityDf.sort_values(by='cosineSimilarity', ascending=False)

  # Devolvemos el dataframe
  return topsimilarity

def getUnratedItemsMatrix(itemsRated, tfidf):
  return tfidf.drop(itemsRated, axis=0)

def getActiveUserProfile(itemsRated, tfidf, ratingAndMeanRatingDifference):
  
  # Recogemos las filas correspondientes a los items valorados por el usuario, de la matriz TF-IDF
  activeUserProfile = tfidf.loc[itemsRated,:]

  # Creamos un diccionario con los valores de diferencia entre la valoración y la valoración media para cada item
  diffDict = dict(zip(ratingAndMeanRatingDifference['idItem'], ratingAndMeanRatingDifference['diferencia']))

  # Agregamos una columna 'idItem' al perfil del usuario activo con los índices actuales y poder operar con ellos
  activeUserProfile['idItem'] = activeUserProfile.index

  activeUserProfile.iloc[:, 0:] = activeUserProfile.iloc[:, 0:].multiply(activeUserProfile['idItem'].map(diffDict), axis=0)
  # Realizamos la multiplicación de los valores en el perfil del usuario activo con los valores de diferencia correspondientes

  # Aliminamos la columna 'idItem' del perfil del usuario activo, ya no es necesaria
  activeUserProfile = activeUserProfile.drop('idItem', axis=1)

  # Calculamos la media de los valores en el perfil del usuario activo para obtener el vector que describirá los gustos del usuario
  activeUserProfile = activeUserProfile.mean()
  # También se puede utilizar la suma en lugar de la media
  #activeUserProfile = activeUserProfile.sum()

  # Devolvemos el perfil del usuario activo
  return activeUserProfile

def getRatingAndMeanRatingDifference(idUser, ratingsDf):

  # Filtrar el DataFrame de ratings por el usuario con id = 1
  ratingsUsuario = ratingsDf[ratingsDf['idUser'] == idUser]

  # Calcular la media de los ratings del usuario
  mediaRatingsUsuario = ratingsUsuario['rating'].mean()

  # Creamos un dataframe que refleje la diferencia, junto con el item correspondiente
  ratingAndMeanRatingDifference = pd.DataFrame({
      'idItem': ratingsUsuario['idItem'],
      'diferencia': ratingsUsuario['rating'] - mediaRatingsUsuario
    })

  # Devolvemos el dataframe
  return ratingAndMeanRatingDifference

def getRatedItemsByUser(idUser, ratingsDf, movieTitlesDf):
  itemsRated = ratingsDf[ratingsDf['idUser'] == idUser][['idItem']]
  return np.array(itemsRated).flatten().tolist()

def createTfidf(movieTagsDf):
  # Reemplazamos los espacos por guiones bajos para evitar separar un mismo tag por tener un espacio
  # Ej: "Tag de ejemplo" -> "Tag_de_ejemplo"
  movieTagsDf['tag'] = movieTagsDf['tag'].apply(lambda x: x.replace(' ', '_'))

  # Agrupamos los tags por idItem
  movieTagsDf = movieTagsDf.groupby('idItem')['tag'].apply(lambda x: ' '.join(x)).reset_index(name='tags')

  # Creamos un diccionario de tags únicos y su frecuencia en todo el conjunto de datos
  tagFrequency = {}
    # Iteramos cada fila (cada item) 
  for row in movieTagsDf.itertuples():
    # Iteramos cada tag (row[2] = Posición de la columna tags en el dataframe) haciendo uso del metodo split para separar los tags por espacios
    for tag in row[2].split():
      if tag not in tagFrequency:
        # Si el tag no se encuentra en el diccionario, significa que es un tag nuevo y se almacena en el diccionario con valor 1
        tagFrequency[tag] = 1
      else:
        # Si el tag se encuentra en el diccionario, aumentamos en 1 el número de ocurrencias en el conjunto de items
        tagFrequency[tag] += 1 

  # Calcular la matriz TF
  # Creamos una matriz compuesta de 0 con tamaño (nºItems x nºTagsUnicos)
  # Almacenará el TF de cada tag en su respectivo idItem
  tfMatrix = np.zeros((len(movieTagsDf), len(tagFrequency)))
  # Iteramos cada fila del dataframe, cada fila contiene el idItem y los tags asociados respectivamente
  for i, row in enumerate(movieTagsDf.itertuples()):
    # Iteramos sobre cada tag del tagFrequency (contiene los tags unicos)
    for j, tag in enumerate(tagFrequency.keys()):
      # Contamos cuantas veces aparece el tag en los tags asociados al item
      # Ingresamos el resultado en su correspondiente posición de la matriz TF
      tfMatrix[i,j] = row[2].split().count(tag)

  # Calcular el IDF para cada tag única
  # Creamos un vector con ceros inicialmente, con tamaño = nºtags únicas, que almacenará el IDF de cada tag
  idfVector = np.zeros(len(tagFrequency))
  # Iteramos sobre cada tag
  for j, tag in enumerate(tagFrequency.keys()):
    # Calculamos según la formula del IDF
    #   log(N/N_t)
    #     -> N = Número de items
    #     -> N_t = Número de items que contienen el tag

    # 1. Filtramos el dataframe original para obtener solo las filas que contienen el tag
    itemsWithTag = movieTagsDf[movieTagsDf['tags'].apply(lambda x: tag in x.split())]
    # 2. Calculamos la cantidad de filas que contienen el tag
    numItemsWithTag = len(itemsWithTag)
    # 3. Calculamos la proporción inversa de la cantidad anterior respecto al número de items
    inverseProportion = len(movieTagsDf) / numItemsWithTag
    # 4. Calculamos el logaritmo de la proporción inversa para obtener el IDF de la etiqueta
    itemTdf = log(inverseProportion)
    # Añadimos el valor tdf del item al vector
    idfVector[j] = itemTdf
    
  # Calcular la matriz TF-IDF (TF-IDF = TF * IDF)
  tfidfMatrix = tfMatrix * idfVector

  # Creamos un dataframe con los valores de las matrices asociados a sus respecitov sidItem y tags 
  tfidf = pd.DataFrame(tfidfMatrix, index=movieTagsDf['idItem'], columns=list(tagFrequency.keys()))

  # Calculamos las normas de los vectores de cada item 
  norms = np.linalg.norm(tfidf.values, axis=1, ord=2)
  # Dividimos el vector de cada item por su respectiva norma para obtener el TF-IDF normalizado
  # de manera que los modulos de los vectores de cada items sea 1
  normalizeTfidf = tfidf.div(norms, axis=0)

  # Devolvemos el dataframe de la matriz TF-IDF normalizada
  return normalizeTfidf

def checkDfs(arrayDfs, numRegisters):
  print('-'*55)
  for i in range(len(arrayDfs)):
    print(arrayDfs[i].head(numRegisters))
    print('-'*55)

def getDataframesFromDatabase():
  # movie-titles.csv
  movieTitlesCsvRoot = './rs-movie-cour/movie-titles.csv'
  movieTitlesDfNames = ['idItem', 'title']
  movieTitlesDf = getDataFromCSV(movieTitlesCsvRoot, movieTitlesDfNames)
  # movie-tag.csv
  movieTagsCsvRoot = './rs-movie-cour/movie-tags.csv'
  movieTagsDfNames = ['idItem', 'tag']
  movieTagsDf = getDataFromCSV(movieTagsCsvRoot, movieTagsDfNames)
  # movie-titles.csv
  usersCsvRoot = './rs-movie-cour/users.csv'
  usersDfNames = ['idUser', 'desc']
  usersDf = getDataFromCSV(usersCsvRoot, usersDfNames)
  # movie-titles.csv
  ratingsDfNames = ['idUser', 'idItem', 'rating']
  ratingsCsvRoot = './rs-movie-cour/ratings.csv'
  ratingsDf = getDataFromCSV(ratingsCsvRoot, ratingsDfNames)
  return [movieTitlesDf, movieTagsDf, usersDf, ratingsDf]

def getDataFromCSV(root, dfNames):
  return pd.read_csv(root, header=None, names=dfNames, encoding='latin1')

def getItemName(itemId, movieTitlesDf):
  return movieTitlesDf[movieTitlesDf['idItem'].isin([itemId])]['title'].values[0]

def requestIdUser():
  idUser = input('\nEnter the user id: ')
  if(idUser == None or not idUser.isdigit() or int(idUser) < 1 or int(idUser) > 5564):
    print('\nInvalid user id, please enter a valid user (1 - 5564)')
    return requestIdUser()
  print(f'\nGetting top 10 recommendations for user {idUser}...')
  return int(idUser)
  
if __name__ == "__main__":
  srbc()
