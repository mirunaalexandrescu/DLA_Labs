# Lab 1: CNN e Connessioni Residuali

Questo laboratorio riproduce su piccola scala l'esperimento centrale del paper di ResNet (He et al., Deep Residual Learning for Image Recognition, CVPR 2016): verificare che una rete più profonda non è automaticamente più capace di ridurre l'errore di training, e che questa degradazione non è dovuta a overfitting ma a un problema di ottimizzazione legato al modo in cui il gradiente si propaga attraverso molti layer. La verifica viene condotta prima su un MLP con MNIST, poi su una CNN con CIFAR 10, per mostrare che il fenomeno non è specifico di un'architettura. L'ultima parte usa il modello CNN allenato per un esercizio di interpretabilità con le Class Activation Maps.

## Contenuti

* `Lab1_CNNs_solution.ipynb`: notebook principale con tutti gli esercizi, codice, output ed analisi

## Obiettivi del Laboratorio

1. Warm up con MLP: verifica del problema di degradazione descritto nel paper di ResNet
2. Aggiunta di connessioni residuali e confronto con la versione plain su più profondità
3. Stessa verifica con CNN in stile CIFAR ResNet su CIFAR 10
4. Class Activation Maps per spiegare le predizioni della CNN allenata

## Setup Ambiente

Il notebook è pensato per Google Colab (GPU consigliata: Runtime, Change runtime type, GPU). Non servono installazioni aggiuntive oltre alle librerie già presenti su Colab (PyTorch, torchvision, matplotlib). I dataset MNIST e CIFAR 10 vengono scaricati automaticamente al primo utilizzo tramite `torchvision.datasets`.

## Risultati Sperimentali

### Exercise 1.1: MLP di base

Un MLP con due blocchi nascosti da 128 unità ciascuno, allenato su MNIST con early stopping basato sulla validation loss.

| Metrica | Valore |
|---|---|
| Epoca di stop | 17 (early stopping, pazienza 5) |
| Miglior validation loss | 0.0946 |
| Test loss | 0.0931 |
| Test accuracy | 97.65% |

MNIST è un task sufficientemente semplice da rendere questo risultato prossimo al limite superiore raggiungibile da un'architettura fully connected di queste dimensioni: già con due soli blocchi nascosti il modello satura gran parte della capacità discriminativa che la rappresentazione del pixel grezzo consente. È proprio per questo motivo che il confronto tra rete plain e rete residuale nell'esercizio successivo richiede un compito più impegnativo, altrimenti l'effetto della degradazione resterebbe mascherato dalla facilità del problema.

### Exercise 1.2: MLP residuale e sweep di profondità

Confronto tra MLP plain e MLP residuale, dove le due architetture condividono la stessa classe, la stessa inizializzazione e lo stesso seed, e differiscono solo per la presenza della somma di scarto (skip connection) all'interno di ogni blocco. Lo sweep copre profondità di 1, 2, 4, 8, 16 e 32 blocchi.

| Profondità (blocchi) | MLP plain | MLP residuale |
|---|---|---|
| 1, 2, 4, 8 | circa 97% di accuracy | circa 97 98% di accuracy |
| 16, 32 | crolla al livello del caso (circa 10%) | resta circa 98% |

Fino a 8 blocchi le due architetture si comportano in modo pressoché identico. Oltre questa soglia la rete plain smette letteralmente di apprendere: il loss di training converge a ln(10), il valore atteso per un modello che assegna probabilità uniforme a tutte e 10 le classi, segno che i pesi non vengono più aggiornati in modo significativo. Non si tratta quindi di un degrado graduale delle prestazioni, ma di un collasso completo del processo di ottimizzazione.

Per capire l'origine del fenomeno, ho misurato la norma del gradiente sul layer di input e sul primo blocco su reti appena inizializzate, prima di qualunque aggiornamento dei pesi. Questa scelta è deliberata: misurare i gradienti su reti già allenate confonderebbe l'effetto strutturale dell'architettura con quanto ciascuna rete si è già avvicinata a un minimo locale.

| Profondità | Norma gradiente, MLP plain | Norma gradiente, MLP residuale |
|---|---|---|
| 1 | circa 0.5 | circa 1 |
| 4 | circa 1e minus 3 | circa 1 |
| 8 | circa 1e minus 6 | circa 1 |
| 16 | circa 1e minus 12 | circa 1 |
| 32 | 0.0 esatto (sottoflow numerico in float32) | circa 1 |

Per la rete plain, la regola della catena moltiplica in sequenza lo jacobiano di ogni blocco, ciascuno approssimativamente della forma ReLU'(x) per W. Con l'inizializzazione standard questi fattori non conservano esattamente la norma, quindi il loro prodotto tende a decadere in modo geometrico con la profondità, fino ad annullarsi numericamente. Per la rete residuale, ogni blocco calcola x più f(x), quindi il suo jacobiano è la somma tra la matrice identità e lo jacobiano di f. La componente identità garantisce che esista sempre almeno un percorso, quello attraverso le sole connessioni di scarto, lungo il quale il gradiente si propaga dalla loss fino al primo layer senza alcuna attenuazione, indipendentemente da quanti blocchi si attraversano. Le connessioni residuali non eliminano il vanishing gradient all'interno di f, ma offrono una via alternativa che lo aggira completamente.

### Exercise 1.3: CNN in stile CIFAR ResNet

Stessa verifica con le CNN, su CIFAR 10. Ho reimplementato l'architettura descritta nella sezione 4.2 del paper originale (stem convoluzionale 3x3, tre stage da 16, 32 e 64 canali con n blocchi residuali ciascuno, downsampling con stride 2 tra uno stage e l'altro), non il ResNet generico di `torchvision` pensato per immagini ImageNet a risoluzione ben più alta. Questo permette di riprodurre fedelmente l'esperimento di Figura 6 del paper.

| Profondità (layer) | CNN plain | CNN residuale |
|---|---|---|
| 20 | 82.14% | 85.19% |
| 32 | 78.86% | 85.69% |
| 44 | 71.22% | 85.18% |
| 56 | 73.76% | 85.79% |

La rete plain perde progressivamente accuracy con la profondità, passando da 82% a circa 71 74%, mentre la residuale resta stabile intorno all'85 86% a ogni configurazione testata. A differenza dell'MLP, qui il degrado è graduale e non un collasso totale: la presenza della batch normalization dopo ogni convoluzione stabilizza la distribuzione delle attivazioni tra i layer e attenua in parte il vanishing gradient, ma non lo risolve del tutto. La rete plain resta comunque più difficile da ottimizzare con lo stesso budget di epoche, un effetto che il paper di ResNet attribuisce non alla capacità rappresentativa della rete (identica a parità di profondità) ma alla difficoltà di trovare, tramite discesa del gradiente, una soluzione altrettanto buona quanto quella raggiungibile da una rete meno profonda.

### Exercise 2.3: Class Activation Maps

Ho scelto questo esercizio tra i tre proposti perché la CNN dell'esercizio 1.3 termina già con la sequenza global average pooling seguita da un singolo layer lineare, l'architettura esatta richiesta dal CAM classico di Zhou et al. (Learning Deep Features for Discriminative Localization, CVPR 2016). Grazie a questa struttura, la mappa di attivazione per una data classe si calcola come combinazione lineare delle feature map dell'ultimo layer convoluzionale, pesate dalla riga del classificatore finale corrispondente a quella classe, senza bisogno di calcolare alcun gradiente aggiuntivo rispetto all'input. Applicato al modello residuale allenato su CIFAR 10, il metodo mostra che la rete concentra l'evidenza discriminativa sulla regione centrale dell'oggetto principale dell'immagine, in modo coerente con la classe effettivamente predetta.

## Confronto Completo degli Approcci

| Architettura | Profondità testate | Test accuracy | Comportamento con la profondità |
|---|---|---|---|
| MLP plain | 1 a 32 blocchi | crolla al 10% oltre 8 blocchi | collasso totale dell'ottimizzazione |
| MLP residuale | 1 a 32 blocchi | stabile 97 98% | nessun degrado |
| CNN plain | 20 a 56 layer | da 82% a 71 74% | degradazione progressiva |
| CNN residuale | 20 a 56 layer | stabile 85 86% | nessun degrado |

## Conclusioni Chiave

### 1. Le connessioni residuali risolvono un problema di ottimizzazione, non di capacità

A parità di profondità, la rete plain e quella residuale hanno esattamente lo stesso numero di parametri, quindi la stessa capacità rappresentativa in linea di principio. Il divario di accuracy osservato non nasce quindi da un limite del modello, ma dalla difficoltà crescente, con la profondità, di trovare tramite discesa del gradiente una configurazione dei pesi che sfrutti effettivamente quella capacità. Il termine identità introdotto dalla skip connection restituisce all'ottimizzatore un percorso privilegiato lungo cui il segnale di errore rimane informativo indipendentemente dal numero di layer attraversati.

### 2. L'effetto è più drammatico sull'MLP che sulla CNN

Sull'MLP il collasso è netto: oltre gli 8 blocchi la rete plain smette del tutto di allenarsi, passando da circa 97% a livello del caso in un salto praticamente discontinuo. Sulla CNN, che include batch normalization dopo ogni convoluzione, il degrado è più graduale e la rete continua comunque ad apprendere, solo con prestazioni via via peggiori all'aumentare della profondità. Questo suggerisce che la batch normalization attenui, senza eliminarlo, il problema descritto dal paper di ResNet, e conferma che la questione non è specifica di un'architettura ma di come il gradiente si propaga attraverso composizioni sempre più lunghe di trasformazioni non lineari.

### 3. Un limite onesto da segnalare

Questi numeri vengono da un singolo run per configurazione, con un solo seed. I valori riportati per lo sweep dell'MLP sono letti in modo approssimato dai grafici del notebook, dato che le accuracy esatte non vengono stampate come testo per quell'esperimento. Lo sweep sulla CNN copre inoltre solo quattro profondità (20, 32, 44 e 56 layer), contro le profondità fino a 110 layer esplorate nel paper originale, per limiti di tempo di training disponibile.

## Esecuzione

1. Apri il notebook in Google Colab
2. (Consigliato) Attiva la GPU: Runtime, Change runtime type, GPU
3. Esegui le celle in sequenza dall'alto verso il basso

## Dataset

* MNIST: scaricato automaticamente tramite `torchvision.datasets.MNIST`
* CIFAR 10: scaricato automaticamente tramite `torchvision.datasets.CIFAR10`
* Split: train, validation (5000 esempi estratti dal training set), test predefiniti

## Note Tecniche

* GPU raccomandata per l'esercizio 1.3, il training delle CNN su CIFAR 10 è lento su CPU
* L'analisi dei gradienti nell'esercizio 1.2 è fatta su reti appena inizializzate, non su reti già allenate, per isolare l'effetto strutturale della skip connection dalla convergenza del training
* Nell'esercizio 2.3, la cella CAM su un ResNet 18 pre allenato su ImageNet richiede di impostare manualmente `IMAGE_PATH` con il percorso di un'immagine locale, altrimenti viene saltata

Corso: Deep Learning Applications
