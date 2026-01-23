# Komplett Guide: Bygga Windows Executable på Windows-dator

## Steg 1: Installera Python på Windows

### 1.1 Ladda ner Python
1. Gå till https://www.python.org/downloads/
2. Klicka på "Download Python 3.x.x" (senaste versionen, minst 3.8)
3. Installer-filen laddas ner (t.ex. `python-3.11.5-amd64.exe`)

### 1.2 Installera Python
1. **Dubbelklicka** på installer-filen
2. ✅ **VIKTIGT:** Kryssa i **"Add Python to PATH"** längst ner i installationsfönstret
3. Klicka på **"Install Now"**
4. Vänta tills installationen är klar
5. Klicka på **"Close"**

### 1.3 Verifiera installation
1. Öppna **Command Prompt** (cmd):
   - Tryck `Windows + R`
   - Skriv `cmd` och tryck Enter
2. Skriv följande kommando:
   ```cmd
   python --version
   ```
3. Du bör se något som: `Python 3.11.5`
4. Om du får felmeddelandet "python is not recognized":
   - Python är inte i PATH
   - Starta om datorn och försök igen
   - Eller installera om Python och se till att kryssa i "Add Python to PATH"

## Steg 2: Förbered projektet på USB-minnet

### 2.1 Kontrollera att alla filer finns
Se till att USB-minnet innehåller hela projektmappen med:
- ✅ `build_executable.bat` - Build-scriptet
- ✅ `build_windows_executable.py` - Python build-script
- ✅ `torp_report_generator.spec` - PyInstaller konfiguration
- ✅ `main.py` - Huvudprogrammet
- ✅ `requirements_executable.txt` - Beroenden
- ✅ `scripts/` - Mapp med Python-script
- ✅ `data/parametrar/` - Parameterfiler
- ✅ `id_ed25519` - SSH-nyckel (om den behövs)

## Steg 3: Kopiera projektet till Windows-datorn

### 3.1 Kopiera från USB
1. Sätt in USB-minnet i Windows-datorn
2. Öppna **File Explorer**
3. Navigera till USB-minnet (t.ex. `E:\` eller `D:\`)
4. Kopiera hela projektmappen (t.ex. `torp`) till Windows-datorn
5. Rekommenderad plats: `C:\torp\` eller `C:\Users\[DittAnvändarnamn]\Desktop\torp\`

### 3.2 Verifiera filstruktur
Öppna projektmappen och kontrollera att den innehåller:
```
torp/
├── build_executable.bat          ← Detta ska du köra!
├── build_windows_executable.py
├── torp_report_generator.spec
├── main.py
├── requirements_executable.txt
├── id_ed25519
├── scripts/
│   ├── 1download.py
│   ├── 2generate_picking_list.py
│   ├── 3generate_order_list.py
│   └── 4generate_graphical_picking_list.py
└── data/
    └── parametrar/
        ├── Beställningsfrekvens.csv
        └── Leveransfrekvens.csv
```

## Steg 4: Bygg executable-filen

### 4.1 Kör build-scriptet
**Metod 1: Dubbelklicka (Enklast)**
1. Öppna projektmappen i File Explorer
2. **Dubbelklicka** på `build_executable.bat`
3. Ett konsolfönster öppnas och visar förloppet

**Metod 2: Från Command Prompt**
1. Öppna **Command Prompt** (cmd)
2. Navigera till projektmappen:
   ```cmd
   cd C:\torp
   ```
   (eller var du nu kopierade projektmappen)
3. Kör build-scriptet:
   ```cmd
   build_executable.bat
   ```

### 4.2 Vad händer nu?
Build-processen gör följande automatiskt:
1. ✅ Kontrollerar att Python är installerat
2. ✅ Installerar alla nödvändiga Python-paket (pandas, numpy, sklearn, matplotlib, paramiko, pyinstaller)
3. ✅ Bygger executable-filen med PyInstaller
4. ✅ Skapar `torp_report_generator.exe` i `dist/` mappen

**Detta kan ta 5-15 minuter** beroende på datorns hastighet.

### 4.3 När build är klar
Du kommer se meddelandet:
```
========================================
Build Complete!
========================================

Executable location: dist\torp_report_generator.exe
```

## Steg 5: Hitta executable-filen

### 5.1 Lokalisera filen
Executable-filen finns i:
```
torp/
└── dist/
    └── torp_report_generator.exe    ← Här är den!
```

### 5.2 Kontrollera filstorlek
- Executable-filen bör vara cirka **80-150 MB** stor
- Om den är mycket mindre (t.ex. < 10 MB) så gick något fel

## Steg 6: Testa executable-filen

### 6.1 Kör executable-filen
1. Navigera till `dist/` mappen
2. Dubbelklicka på `torp_report_generator.exe`
3. Ett konsolfönster öppnas och programmet körs

### 6.2 Om det inte fungerar
- Kontrollera att `id_ed25519` finns i samma mapp som `.exe`-filen
- Eller kopiera den till `dist/` mappen
- Se till att `data/parametrar/` finns i rätt plats

## Steg 7: Förbered för distribution

### 7.1 Skapa deployment-mapp
Skapa en ny mapp för distribution, t.ex. `torp_deployment/`:

```
torp_deployment/
├── torp_report_generator.exe       ← Kopiera från dist/
├── id_ed25519                      ← Kopiera från projektroten
└── data/
    └── parametrar/
        ├── Beställningsfrekvens.csv
        └── Leveransfrekvens.csv
```

### 7.2 Kopiera filer
Kopiera följande filer till deployment-mappen:
1. `dist/torp_report_generator.exe` → `torp_deployment/torp_report_generator.exe`
2. `id_ed25519` → `torp_deployment/id_ed25519`
3. `data/parametrar/Beställningsfrekvens.csv` → `torp_deployment/data/parametrar/Beställningsfrekvens.csv`
4. `data/parametrar/Leveransfrekvens.csv` → `torp_deployment/data/parametrar/Leveransfrekvens.csv`

### 7.3 Testa deployment-mappen
1. Kopiera hela `torp_deployment/` mappen till en annan plats
2. Kör `torp_report_generator.exe` därifrån
3. Om det fungerar är du klar!

## Felsökning

### Problem: "Python is not recognized"
**Lösning:**
- Python är inte installerat eller inte i PATH
- Installera Python igen och se till att kryssa i "Add Python to PATH"
- Starta om datorn efter installation

### Problem: "pip is not recognized"
**Lösning:**
- Python-installationen är ofullständig
- Installera om Python och välj "Install pip" under installationen

### Problem: Build tar för lång tid eller hänger sig
**Lösning:**
- Detta är normalt - build kan ta 10-15 minuter
- Vänta tills det är klart
- Om det hänger sig mer än 30 minuter, avbryt och försök igen

### Problem: "Failed to install dependencies"
**Lösning:**
- Kontrollera internetanslutning
- Försök installera manuellt:
  ```cmd
  pip install pandas numpy scikit-learn matplotlib paramiko pyinstaller
  ```

### Problem: Executable startar men stängs direkt
**Lösning:**
1. Öppna Command Prompt i mappen där `.exe`-filen finns
2. Kör:
   ```cmd
   torp_report_generator.exe
   ```
3. Detta visar eventuella felmeddelanden i konsolen

### Problem: "MSVCR140.dll is missing"
**Lösning:**
- Installera Visual C++ Redistributable:
  - Ladda ner från: https://aka.ms/vs/17/release/vc_redist.x64.exe
  - Installera och starta om datorn

## Tips

✅ **Använd SSD:** Om möjligt, bygg på en SSD för snabbare build-tid  
✅ **Stäng antivirus:** Vissa antivirusprogram kan störa build-processen  
✅ **Tillräckligt med diskutrymme:** Se till att du har minst 2 GB ledigt utrymme  
✅ **Stabil internetanslutning:** Behövs för att ladda ner Python-paket  

## Nästa steg

När executable-filen är klar:
1. Kopiera `torp_deployment/` mappen till USB-minnet
2. Den kan nu köras på vilken Windows-dator som helst **utan att installera Python**
3. Se `WINDOWS_INSTALLATION.md` för instruktioner för slutanvändare
