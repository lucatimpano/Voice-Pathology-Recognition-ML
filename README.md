
# Analisi e Sviluppo di Tecniche di Machine Learning per il Riconoscimento di Patologie da Segnali Vocali

Progetto di Tesi di Laurea.

**Candidato:** Luca Timpano (Mat. 240236)
**Relatore:** Prof. Fabio Fassetti
**Correlatore:** Ing. Giuseppe Timpano
**Università:** Università della Calabria

---

## 🎯 Obiettivo del Progetto

L'obiettivo principale del lavoro di tesi è stato quello di **analizzare e sviluppare tecniche di Machine Learning (ML) per la classificazione automatica di malattie a partire da segnali vocali**.

Lo studio si è concentrato in particolare su **condizioni di tipo neurodegenerativo**, come il Morbo di Parkinson e la Sclerosi Laterale Amiotrofica (SLA).

Lo scopo è stato quello di valutare come i modelli si comportano nella distinzione tra segnali vocali sani e patologici, analizzando come la tipologia di input (vocali, frasi complete, sillabe) o la lunghezza dell’audio influenzassero la scelta delle etichette.

## 🧪 Metodologia e Materiali

### 📊 Dataset Utilizzati

Sono stati utilizzati diversi dataset contenenti registrazioni audio di soggetti sani e patologici. Nello specifico:

1.  **TORGO Database:** Contiene registrazioni di pazienti affetti da Paralisi Cerebrale (PC) o Sclerosi Laterale Amiotrofica (SLA), patologie che causano disartria.
2.  **Italian Parkinson’s Voice and Speech:** Contiene dati di pazienti affetti da Morbo di Parkinson (PD) in lingua italiana.
3.  **Dataset Multiclasse:** È stato creato un dataset unificando i dati precedenti (TORGO e Italian Parkinson), prelevando esclusivamente audio contenenti vocali, consonanti e sillabe per mantenere omogeneità e aumentare la dimensione del *training set*.

### 🛠️ Pre-processing Audio

Prima di alimentare gli algoritmi di machine learning, il segnale audio è stato sottoposto a una fase di pre-elaborazione per migliorare la qualità e la coerenza dei dati. La pipeline di pre-processing include:

1.  **Caricamento dati** (TORGO, Italian Parkinson).
2.  **Riduzione rumore** (utilizzando la sottrazione spettrale).
3.  **Filtro passa-alto** (filtro di Butterworth per il passaggio delle frequenze al di sopra di una soglia).
4.  **Normalizzazione** (uniformazione dell’ampiezza dei segnali nell'intervallo `[-1, 1]`).
5.  **Trimming silenzi** (rimozione di silenzi marginali per conservare solo il contenuto informativo).

I risultati della pre-elaborazione hanno mostrato che il segnale originale (waveform e spettrogramma) è stato reso più pulito e coerente.

### 💻 Modelli di Machine Learning

Per la classificazione sono stati utilizzati e confrontati due modelli di Deep Learning:

| Modello | Architettura | Tipo di Input | Caratteristiche Principali |
| :--- | :--- | :--- | :--- |
| **AST** (Audio Spectrogram Transformer) | Basato interamente sull'architettura **Transformer** e sul meccanismo di *self-attention*. | **Spettrogrammi** (Log-Mel). | Rappresenta lo stato dell'arte e supera modelli ibridi e basati solo su CNN. |
| **Wav2Vec2** | Modello **ibrido** (Convoluzione + Transformer, self-supervised). | Segnale vocale **grezzo** (waveform). | Apprende direttamente dalla forma d'onda audio e utilizza l'apprendimento contrastivo. |

Entrambi i modelli sono stati **pre-addestrati** su grandi dataset e successivamente adattati al dominio specifico tramite una fase di **fine-tuning limitato** (poche epoche per raggiungere la convergenza).

## 📊 Risultati Sperimentali

I risultati ottenuti hanno dimostrato la **fendibilità dell'applicazione** di queste architetture su dataset di segnali vocali patologici.

### Performance
*   Entrambi i modelli hanno raggiunto **prestazioni elevate** sui singoli dataset (TORGO e Parkinson), con valori di *Accuracy* e *ROC AUC* prossimi a 1.
*   Nel task **multiclasse**, i modelli si sono comportati bene ma hanno mostrato **difficoltà nel classificare la classe minoritaria**. Questa difficoltà evidenzia la necessità di un dataset bilanciato.

### Insight Chiave
*   **Indipendenza Linguistica:** L'utilizzo di sole vocali o sillabe, confrontato con l'uso di frasi complete, non ha comportato un incremento o riduzione significativa delle prestazioni. Ciò suggerisce che **le reti riescono ad estrarre feature discriminanti indipendentemente dal contenuto linguistico** (frasi, sillabe, vocali).
*   **Overfitting e Training:** Durante il *training* si è notato che il *validation loss* decresce rapidamente, e non si è riscontrata la presenza di *overfitting*.
*   **Limiti:** I modelli hanno mostrato scarse performance in esperimenti di generalizzazione incrociata (*cross-dataset*), indicando una forte dipendenza dalle caratteristiche del dataset e della malattia analizzata.

### Esempio di Confronto sui Dataset

| Dataset | Modello | Accuracy | Macro F1 | ROC AUC |
| :--- | :--- | :--- | :--- | :--- |
| **TORGO** | MIT-AST | 0.95 | 0.94 | 1.00 |
| | Wav2Vec2 | 0.99 | 0.99 | 1.00 |
| **Parkinson** | MIT-AST | 1.00 | 1.00 | 1.00 |
| | Wav2Vec2 | 0.93 | 0.93 | 0.97 |
| **Multiclasse** | MIT-AST | 0.92 | 0.85 | 0.99 |
| | Wav2Vec2 | 0.97 | 0.90 | 1.00 |

---

## 🛠️ Struttura del Codice e Implementazione

L'intero processo metodologico è stato sviluppato in un **notebook Jupyter** con una pipeline robusta e modulare.

### Ambiente Tecnologico e Librerie

Il codice si basa sull'ecosistema di Deep Learning moderno, in particolare utilizzando:

*   **Huggingface Transformers Library:** Utilizzata per il *fine-tuning* e l'inferenza, fornendo interfacce ad alto livello per gestire i modelli AST e Wav2Vec2.
*   **Librerie di Audio Processing:** `librosa` (per ricampionamento, normalizzazione e trimming).
*   **Datasets Library:** Utilizzata per la gestione e la suddivisione dei campioni audio (*training*, *validation*, *test*).

### Pipeline dell'Algoritmo

Il codice segue i seguenti passaggi principali:

1.  **Caricamento Dataset:** Creazione di un file CSV strutturato (`path` e `label`) a partire dalle cartelle dei dataset, facilitando il caricamento tramite `Dataset.from_pandas`.
2.  **Pre-elaborazione Audio:** Applicazione della sequenza di filtri (riduzione rumore, filtro passa-alto, normalizzazione, trimming).
3.  **Encoding dei Dati:** Definizione di una `preprocess_function` che carica l'audio, effettua il *resampling* a 16 kHz e trasforma i dati in *tensori* utilizzando il `feature_extractor` specifico per il modello (spettrogrammi per AST o waveform per Wav2Vec2).
4.  **Inizializzazione del Modello:** Caricamento dei modelli pre-addestrati (`AutoModelForAudioClassification.from_pretrained`) e configurazione dei parametri di *training*.
5.  **Addestramento (Fine-tuning):** Utilizzo del `Trainer` di Huggingface con parametri come `learning_rate` (es. $3 \times 10^{-5}$), *batch size* simulato (es. 32 tramite `gradient_accumulation_steps=4`), e un numero limitato di epoche (es. 10).
6.  **Valutazione:** Calcolo automatico delle metriche (Accuracy, F1-score, Recall, Precision, ROC-AUC) tramite la funzione `compute_metrics(pred)` sul set di validazione e sul *test set*.
