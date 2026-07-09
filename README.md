# Deep Learning Applications 2025–2026

Questo repository raccoglie il lavoro svolto per tre dei laboratori del corso Deep Learning Applications: connessioni residuali nelle reti profonde, fine-tuning di transformer per sentiment analysis, e detection Out-of-Distribution con robustezza adversariale. Ogni laboratorio affronta un problema diverso ma tutti condividono lo stesso impianto: verificare empiricamente un fenomeno descritto in letteratura, non limitarsi a implementare un'architettura.

## Lab 1 — CNN e Connessioni Residuali

Riproduzione su piccola scala dell'esperimento centrale di He et al. (Deep Residual Learning for Image Recognition, CVPR 2016): una rete più profonda non riduce automaticamente l'errore di training, e la causa non è overfitting ma un problema di ottimizzazione legato alla propagazione del gradiente. La verifica è condotta su un MLP con MNIST e su una CNN in stile CIFAR-ResNet con CIFAR-10, per mostrare che il fenomeno non è specifico di un'architettura. L'ultima parte usa il modello CNN allenato per un esercizio di interpretabilità con Class Activation Maps.

Il punto centrale del laboratorio è la misura della norma del gradiente su reti appena inizializzate, prima di qualsiasi training: per l'MLP plain il gradiente decade in modo geometrico con la profondità fino ad annullarsi numericamente oltre gli 8 blocchi, mentre per la versione residuale resta stabile a ogni profondità testata, grazie al percorso identità che la skip connection garantisce indipendentemente dal numero di layer attraversati.

Dettagli, tabelle e discussione completa nel [README del laboratorio](Lab1-CNNs/README.md).

## Lab 3 — Sentiment Analysis con Transformers

Confronto tra tre approcci sullo stesso task di classificazione binaria del sentiment (Cornell Rotten Tomatoes): un classificatore SVM su feature `[CLS]` estratte da DistilBERT congelato, fine-tuning end-to-end del transformer, e fine-tuning parameter-efficient con LoRA applicato ai moduli di proiezione dell'attenzione.

Il risultato che vale la pena sottolineare è che LoRA, con l'1.09% dei parametri allenabili rispetto al full fine-tuning, non si limita ad avvicinarsi alle sue prestazioni ma le supera, probabilmente perché la capacità ridotta lo rende meno soggetto a overfitting su un training set di dimensioni contenute. È anche circa due volte più veloce e usa una frazione minima della memoria per gli stati dell'ottimizzatore.

Dettagli, tabelle e discussione completa nel [README del laboratorio](Lab3-Transformers/README.md).

## Lab 4 — Adversarial Learning e OOD Detection

Costruzione di una pipeline di Out-of-Distribution detection su CIFAR-10 confrontando due approcci (punteggio di massimo logit e errore di ricostruzione di un autoencoder convoluzionale), seguita da un'analisi degli attacchi adversariali FGSM, sia untargeted che targeted, e da un esperimento di adversarial training per valutarne l'effetto sulla robustezza e, di conseguenza, sulla capacità di OOD detection.

L'autoencoder separa distribuzione in-domain e out-of-domain molto meglio del massimo logit su un modello con dropout, verosimilmente perché il dropout rende il classificatore meno overconfident e quindi meno distinguibile su quella metrica, mentre l'autoencoder lavora su un compito indipendente. L'adversarial training raddoppia circa l'accuratezza sotto attacco forte rispetto al modello standard, senza penalizzare l'OOD detection, un risultato che va nella direzione opposta a quanto spesso riportato in letteratura e che viene discusso con la dovuta cautela nel README specifico.

Dettagli, tabelle e discussione completa nel [README del laboratorio](Lab4-OOD/README.md).

## Struttura del repository

```
DLA_Labs/
├── README.md
├── environment-dla.yml
├── environment-transformers.yml
├── requirements.txt
├── Lab1-CNNs/
│   ├── Lab1_CNNs_Miruna.ipynb
│   └── README.md
├── Lab3-Transformers/
│   ├── Lab3_Transformers_Miruna.ipynb
│   └── README.md
└── Lab4-OOD/
    ├── Lab4_Miruna.ipynb
    └── README.md
```

## Ambiente

Lab 1 richiede solo PyTorch, torchvision e le librerie scientifiche standard. Lab 3 e Lab 4 aggiungono `transformers`, `datasets`, `accelerate` e `peft`, per questo l'ambiente è separato in due file:

```
conda env create -f environment-dla.yml            # Lab 1
conda env create -f environment-transformers.yml    # Lab 3 e Lab 4
```

In alternativa, `pip install -r requirements.txt` copre entrambi gli ambienti in un'unica installazione.

I dataset (MNIST, CIFAR-10, Cornell Rotten Tomatoes) non sono inclusi nel repository e vengono scaricati automaticamente al primo utilizzo tramite `torchvision.datasets` e HuggingFace `datasets`.

## Limiti metodologici

Tutti e tre i laboratori riportano risultati da un singolo seed, non da run ripetuti: le differenze più contenute tra configurazioni (ad esempio il gap tra full fine-tuning e LoRA nel Lab 3, o l'incremento di AUC-ROC dopo adversarial training nel Lab 4) vanno lette con questa cautela. Limiti più specifici a ciascun esperimento, dove rilevanti, sono discussi nel README del laboratorio corrispondente.

---

Corso: Deep Learning Applications
Autore: Miruna Alexandrescu
Anno Accademico: 2025–2026
