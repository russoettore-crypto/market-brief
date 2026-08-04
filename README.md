Market Brief
Report giornaliero automatico: notizie finanziarie + sentiment + variazione prezzi,
per un gruppo di indici/titoli/crypto che scegli tu. Gira da solo ogni mattina nel
cloud (GitHub Actions, gratuito) e pubblica una dashboard su GitHub Pages (gratuito).
Notizie e prezzi non richiedono alcuna registrazione; il sentiment usa FinBERT
chiamato online (Hugging Face Inference API, gratuita) — nessun modello scaricato
su questa macchina, ma serve un token HF gratuito (nessuna carta di credito, 2 minuti,
spiegato sotto).
Include anche un track record (quanto hanno azzeccato i segnali passati, misurato
sul serio nel tempo) e un calcolatore di size (quanto investire, calcolato sui
parametri di rischio che scegli tu).
Non è consulenza finanziaria. Il tool mostra sentiment delle notizie e movimento
prezzo, non ordini di acquisto/vendita: la decisione resta sempre tua. Vedi "Limiti"
in fondo.
Cosa fa ogni esecuzione
Legge `watchlist.json` (indici, titoli, crypto, temi macro, parametri di rischio) —
e si ferma con un errore leggibile se manca qualcosa, invece di fallire in silenzio
Cerca le notizie delle ultime 24h per ciascuno (Google News RSS)
Calcola un punteggio di sentiment sui titoli delle notizie, chiamando FinBERT
online tramite la Hugging Face Inference API (nessun modello scaricato)
Scarica prezzo, variazione % e volatilità recente/ATR (Yahoo Finance via `yfinance`)
Segnala eventuali divergenze (es. prezzo in calo ma notizie con tono positivo)
e il trend del sentiment (in miglioramento/peggioramento rispetto ai giorni scorsi)
Salva un'istantanea dei segnali di oggi, e confronta quella di N esecuzioni fa con
il prezzo di oggi: così il track record cresce da solo, giorno dopo giorno
Calcola, per ogni titolo, quanto varrebbe una posizione secondo i TUOI parametri
di rischio — con uno stop loss calibrato sulla volatilità reale del singolo
titolo (ATR) invece di un numero identico per tutti (calcolatore di size)
Genera `docs/index.html` (la dashboard) e la salva anche in `docs/archive/`
Un secondo workflow (`validate.yml`) controlla `watchlist.json` e `main.py` ad ogni
modifica che carichi su GitHub, così un errore di battitura si vede subito come una ✗
rossa sul commit — non alle 6 del mattino dopo, quando il report semplicemente non parte.
Setup (una tantum, ~10 minuti)
Crea un repository su GitHub (consigliato: pubblico → minuti Actions illimitati
e Pages gratuito. I dati mostrati sono comunque tutti pubblici, quindi non c'è
nulla di sensibile da nascondere).
Carica tutti questi file nel repository (via `git push` o trascinandoli
dall'interfaccia web di GitHub).
Crea un token Hugging Face gratuito (serve solo per il sentiment, chiamato
online): registrati su huggingface.co (gratis,
nessuna carta), poi Settings → Access Tokens → "New token", tipo "Read". Copialo.
Salvalo come secret nel repository: Settings del repo → Secrets and variables
→ Actions → "New repository secret" → nome `HF_TOKEN`, valore il token appena
creato. Non finisce mai nel codice, resta privato.
Abilita GitHub Pages: Settings → Pages → Source: "Deploy from a branch" →
branch `main`, cartella `/docs` → Save.
Verifica i permessi delle Actions: Settings → Actions → General → "Workflow
permissions" → seleziona "Read and write permissions" (serve perché il workflow
deve poter salvare il report nel repo).
Test manuale: tab "Actions" → "Daily Market Brief" → "Run workflow". Dopo
1-2 minuti, controlla che sia comparso un commit nuovo con `docs/index.html`.
La tua dashboard sarà visibile su `https://<tuo-utente>.github.io/<nome-repo>/`
(trovi l'URL esatto in Settings → Pages dopo il primo deploy).
Da qui in poi non devi fare più nulla: il workflow gira da solo ogni giorno feriale
alle 06:00 UTC (le 7-8 del mattino in Italia, a seconda dell'ora legale).
Personalizzare
Tutto si modifica editando `watchlist.json` (non serve toccare il codice):
`watchlist`: aggiungi/rimuovi titoli. Usa il ticker esatto di Yahoo Finance
(es. `ENEL.MI` per Enel a Milano, `BTC-USD` per Bitcoin) e una `query` di ricerca
in linguaggio naturale per le notizie.
`indices`: gli indici/asset mostrati nella striscia in alto.
`macro_queries`: i temi macro (Fed, BCE, geopolitica...) da monitorare.
`news_per_item`: quante notizie considerare per ciascun titolo (default 5).
`track_record_horizon_runs`: dopo quante esecuzioni valutare se un segnale ha
azzeccato la direzione (default 5, cioè circa una settimana lavorativa dopo).
`risk_settings`: i parametri con cui il calcolatore di size fa i conti — sono
placeholder, cambiali con i tuoi numeri reali:
`account_size`: il capitale che consideri per il calcolo (non deve essere
per forza tutto il tuo portafoglio).
`risk_per_trade_pct`: quanto sei disposto a perdere su una singola operazione
se lo stop loss viene colpito, in % del capitale. 1-2% è un valore spesso
citato nella letteratura di risk management, ma la scelta resta tua.
`stop_loss_mode`: `"atr"` (default, consigliato) calcola uno stop diverso per
ogni titolo in base alla sua volatilità reale delle ultime settimane; `"fixed"`
usa sempre `default_stop_loss_pct` per tutti, identico.
`atr_multiplier`: quante volte l'ATR usare come distanza dello stop (default 2×,
un valore comune; più alto = stop più largo = size più piccola a parità di rischio).
`default_stop_loss_pct`: usato solo se `stop_loss_mode` è `"fixed"`, oppure come
ripiego automatico quando l'ATR non è calcolabile (es. titolo appena quotato).
Per cambiare orario di esecuzione, modifica la riga `cron` in
`.github/workflows/daily.yml` (formato minuto-ora-giorno mese-giorno settimana, UTC).
Testare in locale (facoltativo)
```bash
pip install -r requirements.txt
python3 main.py
# poi apri docs/index.html nel browser
```
Limiti — leggere prima di fidarsi dei numeri
Non è un consiglio di investimento. È un riassunto automatico di tono delle
notizie, movimento prezzo, e aritmetica sui parametri che imposti tu. Non conosce
i fondamentali dell'azienda, il resto del tuo portafoglio, né la tua fiscalità.
Il track record non è un backtest storico. Non esiste un archivio gratuito di
notizie del passato, quindi non si può simulare "cosa avrebbe detto il sentiment
6 mesi fa". Quello che il tool fa è onesto ma diverso: registra ogni segnale reale
da oggi in poi e, dopo N esecuzioni, controlla cosa ha fatto davvero il prezzo.
Serve tempo — con meno di ~30 segnali valutati i numeri sono indicativi, non
statisticamente affidabili.
Il calcolatore di size fa solo aritmetica, con la formula standard
rischio-per-operazione (position size = rischio in valuta ÷ % di stop loss). Lo
stop "ATR" è una misura di volatilità storica delle ultime settimane, non una
previsione: non garantisce che il prezzo si fermi lì. Il calcolo non sa nulla di
correlazione tra le tue posizioni, leva, tassazione o liquidità. Se i parametri
di rischio non sono i tuoi numeri reali, il risultato non significa nulla per
la tua situazione.
Il trend del sentiment confronta oggi con la media delle ultime esecuzioni:
utile per notare un cambio di tono, ma con poche esecuzioni salvate (i primi
giorni) è rumoroso — diventa più utile dopo 2-3 settimane di storico.
Il sentiment dipende da un servizio esterno gratuito (Hugging Face). FinBERT è
molto più accurato di un analizzatore lessicale generico sul linguaggio finanziario,
ma la chiamata online ha un costo in affidabilità: la prima chiamata dopo un po' di
inattività può essere lenta (il modello si "risveglia"), e il livello gratuito ha
limiti di frequenza che possono cambiare nel tempo. Il codice ritenta automaticamente
e, se proprio non risponde, assegna sentiment neutro a quel titolo invece di bloccare
l'intero report — quindi un valore "Misto" isolato può voler dire "nessuna notizia
interessante" oppure "l'API non ha risposto": in caso di dubbio, controlla i log
dell'esecuzione in Actions.
Le fonti gratuite hanno dei limiti: `yfinance` non è un'API ufficiale e può
occasionalmente fallire o essere rallentata; Google News RSS può mancare notizie
o duplicarle. Il codice salta gli errori singoli senza bloccare l'intero report,
ma controlla sempre la fonte originale (il link è sempre incluso) prima di agire.
Copertura, non esaustività: "tutti i mercati" in pratica significa la
watchlist che hai configurato — puoi ampliarla quanto vuoi, ma nessuna lista
coprirà davvero "tutto" in automatico senza dati a pagamento.
