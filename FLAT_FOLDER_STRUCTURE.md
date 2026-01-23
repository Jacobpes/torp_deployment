# Flat Folder Structure - Allt i samma mapp

## Översikt

När executable-filen körs, sparas nu **alla filer i samma mapp** som där `torp_report_generator.exe` ligger. Ingen mappstruktur skapas.

## Filstruktur på Windows

När du kör executable-filen ska följande filer finnas i samma mapp:

```
C:\torp_deployment\
├── torp_report_generator.exe          ← Executable-filen
├── id_ed25519                         ← SSH-nyckel (för nedladdning)
├── Leveransfrekvens.csv               ← Parameterfil
├── Beställningsfrekvens.csv           ← Parameterfil
│
├── [Nedladdade CSV-filer]             ← Sparas här direkt
│   ├── product_sales_items_*.csv
│   ├── visitor_report_*.csv
│   ├── user_report_*.csv
│   └── stock_report_*.csv
│
└── [Output-filer]                     ← Sparas här direkt
    ├── [Butiknamn].csv                ← Plocklistor (en per butik)
    ├── Orderlista_[Leverantör].csv    ← Orderlistor (en per leverantör)
    ├── Plocklista_[Butik].pdf         ← Grafiska plocklistor (en per butik)
    ├── picking_list_results.csv        ← Om den skapas
    ├── torp_report_generator.log      ← Loggfil
    └── torp_report_generator_errors.log ← Fellogg
```

## Viktiga ändringar

### 1. Nedladdade filer
- **Före:** Sparades i `data/downloads/`
- **Nu:** Sparas direkt i samma mapp som .exe-filen

### 2. Plocklistor
- **Före:** Sparades i `plocklistor/` mapp
- **Nu:** Sparas direkt i samma mapp som .exe-filen
- **Filnamn:** `[Butiknamn].csv` (t.ex. `Torp kiosk.csv`)

### 3. Orderlistor
- **Före:** Sparades i `orderlistor/` mapp
- **Nu:** Sparas direkt i samma mapp som .exe-filen
- **Filnamn:** `Orderlista_[Leverantör].csv` (t.ex. `Orderlista_Arla.csv`)

### 4. Grafiska plocklistor
- **Före:** Sparades i `picking_list_graphical/` mapp
- **Nu:** Sparas direkt i samma mapp som .exe-filen
- **Filnamn:** `Plocklista_[Butik].pdf` (t.ex. `Plocklista_Torp kiosk.pdf`)

### 5. Parameterfiler
- **Före:** Låg i `data/parametrar/`
- **Nu:** Skulle ligga direkt i samma mapp som .exe-filen
- **Filnamn:** `Leveransfrekvens.csv` och `Beställningsfrekvens.csv`

## Förberedelse för deployment

### Steg 1: Kopiera filer till Windows-datorn

Skapa en mapp (t.ex. `C:\torp_deployment\`) och kopiera dit:

1. ✅ `torp_report_generator.exe` - Executable-filen
2. ✅ `id_ed25519` - SSH-nyckel
3. ✅ `Leveransfrekvens.csv` - Parameterfil (kopiera från `data/parametrar/`)
4. ✅ `Beställningsfrekvens.csv` - Parameterfil (kopiera från `data/parametrar/`)

### Steg 2: Kör executable-filen

1. Dubbelklicka på `torp_report_generator.exe`
2. Alla filer kommer sparas i samma mapp

### Steg 3: Hitta resultat

Efter körning hittar du:
- **Nedladdade filer:** I samma mapp (CSV-filer med datum i namnet)
- **Plocklistor:** `[Butiknamn].csv` filer
- **Orderlistor:** `Orderlista_[Leverantör].csv` filer
- **Grafiska plocklistor:** `Plocklista_[Butik].pdf` filer
- **Loggar:** `torp_report_generator.log` och `torp_report_generator_errors.log`

## Fördelar med platt struktur

✅ **Enklare** - Allt på ett ställe  
✅ **Ingen mappstruktur** - Lättare att hitta filer  
✅ **Portabel** - Hela mappen kan kopieras enkelt  
✅ **Tydligare** - Alla filer syns direkt  

## Obs!

- **Filstorlek:** Mappen kan bli stor med många nedladdade filer
- **Organisering:** Överväg att rensa gamla filer regelbundet
- **Backup:** Backa upp hela mappen regelbundet

## Om du vill organisera filer

Om du vill organisera filerna efteråt kan du:
1. Skapa undermappar manuellt (t.ex. `downloads/`, `output/`)
2. Flytta filer dit efter körning
3. Eller använda batch-script för att automatisera detta
