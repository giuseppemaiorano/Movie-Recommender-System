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
