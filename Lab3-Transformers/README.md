# Lab 3: Sentiment Analysis con Transformers

Classificazione binaria del sentiment su recensioni cinematografiche: confronto tra un classificatore tradizionale su feature congelate (SVM), fine tuning completo di DistilBERT e fine tuning parameter efficient con LoRA.

## Contenuti

* `Lab3-Transformers-Completed.ipynb`: notebook principale con tutti gli esercizi, codice, output ed analisi

## Obiettivi del Laboratorio

1. Dataset Exploration: analisi del Cornell Rotten Tomatoes dataset
2. Feature Extraction: DistilBERT come estrattore di feature per un classificatore SVM
3. Fine tuning End to End: adattamento completo del transformer al task specifico
4. Parameter Efficient Fine Tuning: LoRA per ottimizzare i parametri allenati

## Setup Ambiente

Il notebook è pensato per Google Colab (GPU consigliata: Runtime, Change runtime type, GPU).

La prima cella installa tutte le dipendenze necessarie:

```bash
pip install -q -U transformers datasets accelerate peft scikit-learn
```

Nota: Colab preinstalla spesso `torchao` in una versione incompatibile con `peft`. La stessa cella di setup lo disinstalla automaticamente, dato che qui non serve (viene usato solo per la quantizzazione).

## Risultati Sperimentali

### Exercise 1: Sentiment Analysis Warm up

#### 1.1 Dataset Exploration: Cornell Rotten Tomatoes

| Split | Esempi | Bilanciamento |
|---|---|---|
| Train | 8,530 | 50% neg / 50% pos |
| Validation | 1,066 | 50% neg / 50% pos |
| Test | 1,066 | 50% neg / 50% pos |

Dataset perfettamente bilanciato, nessuno split manuale necessario.

#### 1.2 DistilBERT Pre trained Model

* Parametri: circa 67M
* Layers: 6 transformer block
* Hidden size: 768
* Nessun `pooler_output` (a differenza di BERT): la rappresentazione di frase va estratta manualmente dal token `[CLS]` dell'ultimo layer

#### 1.3 Stable Baseline (DistilBERT frozen + SVM)

| Split | Accuracy | F1 |
|---|---|---|
| Validation | 81.9% | 0.809 |
| Test | 79.6% | 0.791 |

Baseline discreta ma non ottimizzata per il task: il vettore `[CLS]` pre addestrato solo con masked language modeling non è mai stato ottimizzato per riassumere il significato dell'intera frase.

### Exercise 2: Fine tuning DistilBERT

#### 2.3 Fine tuning Results

| Epoca | Training Loss | Validation Loss | Val. Accuracy |
|---|---|---|---|
| 1 | 0.362 | 0.412 | 81.9% |
| 2 | 0.241 | 0.382 | 85.1% |
| 3 | 0.160 | 0.472 | 85.8% |

Performance finali (test set):
* Accuracy: 83.4%
* F1: 0.837
* Miglioramento vs baseline: +3.8 punti percentuali

Osservazioni: la validation loss risale nell'ultima epoca mentre l'accuracy continua a salire, un segnale di overfitting più sottile del solito (il modello diventa via via più overconfident sulle proprie predizioni, anche quelle sbagliate), probabile anticipazione di un plateau più netto con qualche epoca in più.

### Exercise 3.1: Parameter Efficient Fine tuning (LoRA)

Applicato su `q_lin`/`v_lin` (i moduli di proiezione dell'attenzione in DistilBERT), `r=8`, `lora_alpha=16`.

| Epoca | Training Loss | Validation Loss | Val. Accuracy |
|---|---|---|---|
| 1 | 0.395 | 0.376 | 83.4% |
| 2 | 0.323 | 0.364 | 83.7% |
| 3 | 0.250 | 0.376 | 85.1% |

Performance finali (test set):
* Accuracy: 84.3%
* F1: 0.843
* Tempo di training: 48.3s

Efficienza LoRA:
* Parametri trainable: 739,586 su 67,694,596 totali (1.09%)
* Riduzione parametri: 90.5 volte meno parametri da addestrare
* Risparmio memoria stimato (stati dell'ottimizzatore): 98.9%

A differenza del full fine tuning, la validation loss resta stabile lungo le tre epoche: con molti meno parametri allenabili, LoRA ha meno capacità di adattarsi eccessivamente al training set e agisce come regolarizzatore implicito.

## Confronto Completo degli Approcci

| Approccio | Tipo Addestramento | Test Accuracy | Test F1 | Parametri Trainable | Tempo Training |
|---|---|---|---|---|---|
| Baseline (SVM) | Feature extraction + classificatore | 79.6% | 0.791 | 0 (DistilBERT congelato) | circa 20s |
| Full Fine tuning | End to end su tutto il modello | 83.4% | 0.837 | 66.9M (100%) | 86.2s |
| LoRA Fine tuning | Parameter efficient con adapter | 84.3% | 0.843 | 0.7M (1.09%) | 48.3s |

## Conclusioni Chiave

### 1. Il fine tuning batte la feature extraction

Permettere a tutti i pesi del transformer di adattarsi al task produce un miglioramento netto rispetto alle feature `[CLS]` congelate (+3.8 punti di accuracy), ma al costo di aggiornare tutti i circa 67M di parametri, e con i primi segnali di overfitting già dopo 2 3 epoche su un dataset di queste dimensioni.

### 2. LoRA non è solo un compromesso, qui vince su tutta la linea

Con appena l'1.09% dei parametri allenabili, LoRA non si limita ad avvicinarsi al full fine tuning: lo supera (84.3% contro 83.4%), probabilmente perché la capacità ridotta lo rende meno soggetto a overfittare su un training set piccolo. In più, è quasi 2 volte più veloce e usa una frazione minima della memoria per gli stati dell'ottimizzatore.

### 3. Un limite onesto da segnalare

Questi numeri vengono da un singolo run (un solo seed). La testa di classificazione è inizializzata casualmente ad ogni esecuzione, quindi il gap tra full fine tuning e LoRA (qui circa un punto percentuale) è plausibilmente in parte dentro il rumore statistico di run to run. Per una conclusione più solida servirebbero più run con seed diversi.

## Esecuzione

1. Apri il notebook in Google Colab
2. (Consigliato) Attiva la GPU: Runtime, Change runtime type, GPU
3. Esegui la prima cella di setup (installazione dipendenze e fix di compatibilità)
4. Esegui le celle in sequenza dall'alto verso il basso

Tempo stimato: circa 20 30 minuti su GPU T4 (1 2 ore su CPU).

## Dataset

* Cornell Rotten Tomatoes: scaricato automaticamente tramite HuggingFace Datasets
* Preprocessing: tokenizzazione con il tokenizer di DistilBERT (`Dataset.map`, padding dinamico via `DataCollatorWithPadding`)
* Split: train, validation, test predefiniti

## Note Tecniche

* GPU raccomandata per il training (LoRA funziona comunque anche su CPU, più lentamente)
* Il notebook rileva a runtime se `transformers` richiede `eval_strategy` o `evaluation_strategy` in `TrainingArguments`, per restare compatibile con versioni diverse della libreria
* `torchao`, se preinstallato in una versione incompatibile con `peft`, viene rimosso automaticamente nella cella di setup. Se il problema persiste, riavviare il runtime e rilanciare dall'inizio

Corso: Deep Learning Applications
