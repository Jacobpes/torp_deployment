# Windows Installation Guide

## ❌ Python behövs INTE installeras!

Om executable-filen (`torp_report_generator.exe`) är korrekt byggd med PyInstaller, så innehåller den allt som behövs - inklusive Python-interpretatorn och alla bibliotek.

## Snabbinstallation på Windows

### Steg 1: Kopiera filer till Windows-datorn

Kopiera hela `torp_deployment`-mappen från USB-minnet till Windows-datorn, t.ex. till:
```
C:\torp_report_generator\
```

### Steg 2: Kontrollera filstruktur

Mappen ska innehålla:
```
torp_report_generator/
├── torp_report_generator.exe    ← Huvudprogrammet
├── id_ed25519                    ← SSH-nyckel (för datahämtning)
└── data/
    └── parametrar/
        ├── Beställningsfrekvens.csv
        └── Leveransfrekvens.csv
```

### Steg 3: Kör programmet

1. Öppna File Explorer och navigera till mappen
2. Dubbelklicka på `torp_report_generator.exe`
3. Ett konsolfönster öppnas och visar förloppet
4. När programmet är klart skapas resultatfiler i samma mapp

## Om Python skulle behövas (endast om executable inte fungerar)

Om executable-filen av någon anledning inte fungerar, kan du installera Python och köra källkoden direkt:

### Installera Python på Windows

1. **Ladda ner Python:**
   - Gå till https://www.python.org/downloads/
   - Ladda ner Python 3.10 eller senare (Windows installer)

2. **Installera Python:**
   - Kör installer-filen
   - ✅ **VIKTIGT:** Kryssa i "Add Python to PATH" under installationen
   - Klicka "Install Now"

3. **Verifiera installation:**
   - Öppna Command Prompt (cmd)
   - Skriv: `python --version`
   - Du bör se Python-versionen, t.ex. "Python 3.11.5"

4. **Installera beroenden:**
   ```cmd
   pip install pandas numpy scikit-learn matplotlib paramiko
   ```

5. **Kör programmet:**
   ```cmd
   python main.py
   ```

## Felsökning

### "Windows cannot access the specified device, path, or file"
- Kontrollera att filen inte är blockerad av Windows Defender
- Högerklicka på `.exe`-filen → Properties → Unblock (om möjligt)
- Kör som administratör: Högerklicka → "Run as administrator"

### "The program can't start because MSVCR140.dll is missing"
- Installera Visual C++ Redistributable:
  - Ladda ner från: https://aka.ms/vs/17/release/vc_redist.x64.exe
  - Installera och starta om datorn

### "Python is not recognized"
- Python är inte installerat eller inte i PATH
- Se instruktioner ovan för att installera Python
- Eller använd executable-filen istället (behöver inte Python)

### Executable startar men stängs direkt
- Öppna Command Prompt i mappen
- Kör: `torp_report_generator.exe`
- Detta visar eventuella felmeddelanden

## Fördelar med Executable

✅ **Ingen installation behövs** - bara kopiera och kör  
✅ **Allt ingår** - Python och alla bibliotek är inkluderade  
✅ **Enklare distribution** - bara en fil att dela  
✅ **Inga konflikter** - påverkar inte annan Python-installation  

## Kontakt

Om executable-filen inte fungerar, kontakta utvecklaren för att få en ny version byggd.
