#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importo le librerie che mi serviranno
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# In[2]:


df = pd.read_csv("movies.csv")


# In[3]:


#Stampo le prime dieci righe del dataset
df.head(10)


# In[4]:


# Esploro il dataset
df.info()
df.describe()
df.isnull().sum()


# In[5]:


# Visualizzo le colonne del dataset
df.columns


# In[6]:


# Per esempio, se abbiamo una colonna 'genres', possiamo usare il TF-IDF per trasformare i dati di testo in vettori
from sklearn.feature_extraction.text import TfidfVectorizer

# Eseguo TF-IDF sui generi
tfidf = TfidfVectorizer(stop_words='english')
df['genres'] = df['genres'].fillna('')  # Riempi i valori nulli con stringhe vuote
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Visualizzo la forma della matrice TF-IDF
tfidf_matrix.shape


# In[7]:


# Utilizzo NearestNeighbors per calcolare le similarità
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=11)
knn.fit(tfidf_matrix)

# Creo una serie indicizzata dai titoli dei film
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Funzione di raccomandazione con gestione degli errori
def get_recommendations(title, model=knn, data=tfidf_matrix, indices=indices):
    # Verifico se il titolo è presente nel dataset
    if title not in indices:
        # Ricerca parziale nel dataset
        partial_matches = df[df['title'].str.contains(title, case=False, na=False)]['title']
        if partial_matches.empty:
            return f"Il film '{title}' non è presente nel dataset."
        else:
            # Prendo il primo risultato della ricerca parziale
            title = partial_matches.iloc[0]
    
    # Ottengo l'indice del film dato il suo titolo
    idx = indices[title]
    
    # Trovo i 10 vicini più vicini (11 inclusi se stesso)
    distances, indices = model.kneighbors(data[idx], n_neighbors=11)
    
    # Ottengo gli indici dei film consigliati
    movie_indices = indices.flatten()[1:]  # Ignora il primo che è il film stesso
    
    # Restituisco i titoli dei film consigliati
    return df['title'].iloc[movie_indices]


# In[8]:


# Controllo alcuni titoli di film nel dataset
df['title'].head(20)


# In[9]:


# Testo il sistema di raccomandazione con vari titoli di film
print(get_recommendations('The Godfather'))
print(get_recommendations('Toy Story'))
print(get_recommendations('Un film non presente'))


# In[10]:


# Ricerca parziale per "Godfather"
godfather_titles = df[df['title'].str.contains("Godfather", case=False, na=False)]['title']
print(godfather_titles)

# Ricerca parziale per "Toy Story"
toy_story_titles = df[df['title'].str.contains("Toy Story", case=False, na=False)]['title']
print(toy_story_titles)


# In[11]:


# Ricerca parziale per "The Social Network"
toy_story_titles = df[df['title'].str.contains("Social Network", case=False, na=False)]['title']
print(toy_story_titles)


# In[12]:


print(get_recommendations("Casino"))


# In[13]:


print(get_recommendations("Social Network, The"))


# In[14]:


# Ricerca parziale per "Forza Maggiore" in originale "Turist"
toy_story_titles = df[df['title'].str.contains("Turist", case=False, na=False)]['title']
print(toy_story_titles)


# In[15]:


print(get_recommendations("Force Majeure"))


# In[16]:


# Ricerca parziale per "Apocalypse Now"
toy_story_titles = df[df['title'].str.contains("Apocalypse now", case=False, na=False)]['title']
print(toy_story_titles)


# In[17]:


print(get_recommendations("Apocalypse Now"))


# In[19]:


print(get_recommendations("Children of Men"))


# In[ ]:




