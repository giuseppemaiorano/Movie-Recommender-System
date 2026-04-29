# Movie Recommender System

This project implements a **movie recommender system** using Python and machine learning libraries. The goal is to suggest movies similar to a title provided by the user, based on a movie dataset.

## Features

- Loads a movie dataset containing titles, genres, and other relevant information.
- Uses content-based similarity techniques to generate movie recommendations.
- Extracts movie information to provide relevant suggestions based on a given title.
- Displays recommended movies in response to the user's query.

## Notebook Structure

1. **Importing the required libraries**  
   The notebook uses libraries such as `pandas`, `scikit-learn`, and other tools for data preprocessing and similarity computation.

2. **Loading and preprocessing the data**  
   The dataset is loaded from a CSV file containing columns such as `movieId`, `title`, and `genres`.  
   The data is cleaned and preprocessed before being used by the recommendation system.

3. **Building the recommendation system**  
   Cosine similarity is used to calculate how similar movies are to each other based on their genres.  
   The project implements a `get_recommendations()` function that takes a movie title as input and returns a list of recommended movies.

4. **Running the system**  
   The notebook includes examples using movie titles such as `"Apocalypse Now"` and `"Children of Men"`, with the recommendations displayed in a table format.

## Requirements

To run this notebook, you need the following Python packages:

- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib` optional, if you want to visualize the data

You can install them using:


```bash
pip install pandas scikit-learn numpy matplotlib
```

# Movie Recommender System

Questo progetto implementa un **sistema di raccomandazione di film** utilizzando Python e librerie di machine learning. L'obiettivo è consigliare film simili a quelli specificati dall'utente basandosi su un insieme di dati di film.

## Funzionalità

- Caricamento di un dataset di film che include titoli, generi e altre informazioni rilevanti.
- Utilizzo di tecniche di similarità dei contenuti per generare raccomandazioni di film simili.
- Estrazione delle informazioni sui film per generare consigli pertinenti basati su titoli dati.
- Visualizzazione dei film consigliati in risposta alla query dell'utente.

## Struttura del Notebook

1. **Importazione delle librerie necessarie**: Include l'uso di `pandas`, `scikit-learn` e altre librerie per il preprocessing dei dati e il calcolo della similarità.
   
2. **Caricamento e preprocessamento dei dati**: 
   - Il notebook carica i dati dei film da un file CSV, che include colonne come `movieId`, `title` e `genres`.
   - I dati vengono puliti e preprocessati per essere utilizzati nel sistema di raccomandazione.

3. **Costruzione del sistema di raccomandazione**: 
   - Viene utilizzato un algoritmo di similarità del coseno per calcolare quanto i film siano simili tra loro basandosi sui generi.
   - Implementazione di una funzione `get_recommendations()` che prende in input un titolo di film e restituisce una lista di film consigliati.

4. **Esecuzione del sistema**: 
   - Sono mostrati esempi di esecuzioni che includono richieste per film specifici come "Apocalypse Now" e "Children of Men", con i risultati visualizzati in formato tabellare.

## Requisiti

Per eseguire questo notebook, sono necessari i seguenti pacchetti Python:

- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib` (facoltativo, se desideri visualizzare i dati)

Puoi installarli utilizzando il comando:
```bash
pip install pandas scikit-learn numpy matplotlib

