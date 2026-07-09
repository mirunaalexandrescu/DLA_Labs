# Laboratory 4: Adversarial Learning e OOD Detection

## Obiettivo

In questo laboratorio ho costruito una pipeline di Out of Distribution detection su CIFAR 10, ho misurato le sue prestazioni con metriche standard, e ho sperimentato con attacchi adversariali (FGSM) sia per capire quanto sono vulnerabili i modelli sia per provare a renderli più robusti tramite adversarial training. Come esercizio a scelta ho approfondito gli attacchi targeted (Esercizio 3.3).

## Setup

| Elemento | Dettaglio |
|---|---|
| Dataset ID | CIFAR 10 (50000 train, 10000 test), scaricato da Hugging Face |
| Dataset OOD | FakeData di torchvision, 1000 immagini di rumore sintetico (3x32x32) |
| Modello base | CNN con 5 layer convoluzionali (32, 64, 128, 128, 256 canali) più 3 fully connected, dropout p 0.5 prima degli ultimi due fully connected |
| Pesi | pretrained, 40 epoche, caricati senza riallenare |
| Accuratezza test set | 70.73% |

## Esercizio 1: OOD Detection e valutazione delle prestazioni

### 1.1 Pipeline di OOD detection

Ho implementato due approcci per assegnare un punteggio di "quanto è OOD" un campione.

1. Max logit: il valore massimo dell'output del classificatore prima della softmax. Su dati familiari (CIFAR 10) il modello dovrebbe produrre un logit massimo alto, su rumore puro (FakeData) dovrebbe essere più incerto e quindi più basso.
2. Autoencoder convoluzionale: allenato solo su CIFAR 10 a ricostruire le immagini. Il punteggio è l'errore di ricostruzione. Su immagini che il modello non ha mai visto in training, l'errore dovrebbe essere sistematicamente più alto.

### 1.2 Metriche

| Metodo | AUC ROC |
|---|---|
| Max logit | 0.67 |
| Autoencoder | 0.94 |

L'autoencoder separa ID e OOD molto meglio del max logit su questo modello. La spiegazione più probabile è che il dropout rende il classificatore meno overconfident: i suoi punteggi massimi sono meno estremi rispetto a un modello standard, e quindi si sovrappongono di più con quelli calcolati su FakeData. L'autoencoder invece non dipende dal dropout del classificatore, quindi non risente di questo effetto, e mantiene una separazione quasi perfetta tra le due distribuzioni.

## Esercizio 2: Robustezza agli attacchi adversariali

### 2.1 FGSM

Ho implementato il Fast Gradient Sign Method, sia in versione untargeted (per confondere il modello) sia targeted (per forzare una classificazione specifica).

| Test | Risultato |
|---|---|
| Attacco targeted singolo, verso "deer" | successo con budget 15 su 255 |

Guardando la mappa della perturbazione si vede chiaramente che FGSM non aggiunge rumore casuale: concentra le modifiche in punti specifici dell'immagine, quelli che pesano di più sul gradiente della loss.

### 2.2 Adversarial training

Ho allenato un secondo modello (stessa architettura) mescolando ad ogni batch immagini pulite e immagini avversarie generate al volo con FGSM (epsilon 8 su 255), per 10 epoche.

| Epoca | Loss |
|---|---|
| 1 | 2.03 |
| 2 | 1.79 |
| 3 | 1.68 |
| 4 | 1.62 |
| 5 | 1.57 |
| 6 | 1.52 |
| 7 | 1.47 |
| 8 | 1.43 |
| 9 | 1.39 |
| 10 | 1.34 |

La loss scende in modo regolare, segno che il modello sta effettivamente imparando a classificare bene anche gli esempi perturbati.

Confrontando il modello robusto con quello standard sui livelli di attacco crescenti (valori letti dal grafico, quindi indicativi):

| Perturbazione (livelli su 255) | Accuratezza standard | Accuratezza robusta |
|---|---|---|
| 0 | 0.71 | 0.63 |
| 1 | 0.60 | 0.60 |
| 3 | 0.43 | 0.54 |
| 6 | 0.27 | 0.46 |
| 12 | 0.16 | 0.32 |
| 20 | 0.10 | 0.19 |

Il modello robusto perde un po' di accuratezza sui dati puliti (0.71 contro 0.63), ma resta molto più stabile all'aumentare della perturbazione, mantenendo quasi il doppio dell'accuratezza del modello standard agli attacchi più forti.

Impatto sull'OOD detection (max logit):

| Modello | AUC ROC |
|---|---|
| Standard | 0.6714 |
| Robusto (adversarial training) | 0.7105 |

L'adversarial training non peggiora la capacità di distinguere ID da OOD, anzi l'AUC ROC sale leggermente (0.6714, 0.7105). Il miglioramento va nella direzione opposta a quanto spesso riportato in letteratura, dove l'adversarial training tende a peggiorare l'OOD detection, ma con un solo seed e un delta di questa dimensione non è possibile dire se sia un effetto reale o rumore statistico tra run.

## Esercizio 3.3: Attacchi targeted

Success rate di un attacco targeted verso diverse classi, a epsilon fisso (0.025):

| Classe target | Success rate |
|---|---|
| Automobile | 12.58% |
| Ship | 13.23% |
| Airplane | 16.90% |
| Frog | 19.80% |
| Deer | 26.70% |

Automobile è la classe più difficile da imitare, deer la più facile. Questo suggerisce che alcune classi hanno rappresentazioni interne più vicine ad altre nello spazio delle feature del modello, e quindi sono più facili da raggiungere con una piccola perturbazione.

Confronto tra attacco targeted (verso "ship") e attacco untargeted, sullo stesso range di epsilon:

| Epsilon | Targeted (ship) | Untargeted |
|---|---|---|
| 0.010 | 6.58% | 53.68% |
| 0.020 | 11.29% | 69.12% |
| 0.025 | 13.23% | 73.97% |
| 0.030 | 15.23% | 77.72% |
| 0.040 | 18.19% | 82.36% |
| 0.050 | 20.40% | 84.94% |

Il gap tra le due strategie resta ampio a tutti i livelli di epsilon testati, confermando che forzare una classificazione verso una classe precisa è molto più difficile del semplice obiettivo di confondere il modello.

## Conclusioni

Il laboratorio conferma alcune intuizioni di base sull'adversarial learning. I modelli neurali, anche quando raggiungono un'accuratezza discreta, restano vulnerabili a perturbazioni piccole e mirate. L'adversarial training è un modo efficace per aumentare la robustezza a questi attacchi, e nel mio caso non ha penalizzato l'OOD detection, anzi l'AUC ROC è leggermente salito, anche se con un solo seed non posso dire con certezza che sia un effetto reale e non rumore statistico. Gli attacchi targeted restano molto più difficili degli attacchi untargeted, con differenze di successo che superano i sessanta punti percentuali a parità di budget di perturbazione.

Un risultato che non mi aspettavo è la differenza netta tra i due metodi di OOD detection: l'autoencoder (AUC 0.94) supera nettamente il max logit (AUC 0.67) sul modello con dropout. Probabilmente perché il dropout riduce l'overconfidence del classificatore e quindi penalizza proprio i metodi che si basano sulla sua fiducia nelle predizioni, mentre l'autoencoder lavora su un compito indipendente (la ricostruzione) e non risente di questo effetto.

## Possibili miglioramenti

1. La consegna chiede esplicitamente di ricavare uno split di validazione dal training set, separato dal test set, da usare per le valutazioni. In questo lavoro ho usato direttamente il test set ufficiale di CIFAR 10 sia per l'accuratezza sia per l'OOD detection.
2. La dipendenza da epsilon della qualità degli esempi avversari, richiesta nell'Esercizio 2.1, è coperta più diffusamente negli Esercizi 2.2 e 3.3 piuttosto che con un test dedicato subito nell'Esercizio 2.1.
3. Sarebbe interessante ripetere l'Esercizio 1 anche con il modello senza dropout, per verificare se davvero è il dropout a spiegare la differenza di prestazioni tra max logit e autoencoder osservata qui.
