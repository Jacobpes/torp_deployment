# Windows Executable - Felsökning

## Problem: Nedladdningen säger att den lyckades men inga filer syns

### Steg 1: Kontrollera var filerna sparas

När executable-filen körs, ska filerna sparas i:
```
[Var executable-filen ligger]\data\downloads\
```

**Exempel:**
- Om executable-filen ligger i `C:\torp_deployment\torp_report_generator.exe`
- Så sparas filerna i `C:\torp_deployment\data\downloads\`

### Steg 2: Kontrollera log-filer

Executable-filen skapar två log-filer i samma mapp som executable-filen:

1. **`torp_report_generator.log`** - Fullständig logg med all information
2. **`torp_report_generator_errors.log`** - Endast fel och varningar

**Öppna log-filerna och leta efter:**
- `Download destination: [sökväg]` - Detta visar var filerna sparas
- `Download complete! X file(s) downloaded.` - Detta visar hur många filer som laddades ner
- Eventuella felmeddelanden

### Steg 3: Kontrollera konsol-output

När executable-filen körs, öppnas ett konsolfönster. Leta efter:
- `Download destination: [sökväg]` - Detta visar var filerna sparas
- `Downloading: [filnamn] ([storlek] bytes)...` - Detta visar att filer laddas ner
- `✓ Saved to: [sökväg]` - Detta bekräftar att varje fil sparades

### Steg 4: Kontrollera att SSH-nyckeln finns

Executable-filen behöver `id_ed25519` filen i samma mapp:
```
torp_report_generator.exe
id_ed25519          ← Denna fil måste finnas här!
data/
```

### Steg 5: Testa manuellt

1. Öppna **Command Prompt** (cmd) i mappen där executable-filen ligger
2. Kör:
   ```cmd
   torp_report_generator.exe
   ```
3. Detta visar alla meddelanden i konsolen och du kan se exakt vad som händer

### Steg 6: Kontrollera filsystemet

1. Öppna **File Explorer**
2. Navigera till mappen där executable-filen ligger
3. Kontrollera att `data\downloads\` mappen finns
4. Om mappen är tom, kontrollera log-filerna för felmeddelanden

## Vanliga problem och lösningar

### Problem: "SSH key not found"
**Lösning:**
- Se till att `id_ed25519` filen finns i samma mapp som executable-filen
- Kontrollera att filnamnet är exakt `id_ed25519` (ingen filändelse)

### Problem: "Download destination" visar fel sökväg
**Lösning:**
- Detta kan hända om executable-filen körs från en annan plats
- Kör executable-filen från mappen där den ligger, inte från en genväg
- Eller kopiera hela mappen till en ny plats och kör därifrån

### Problem: Nedladdningen tar bara 1 sekund
**Möjliga orsaker:**
1. **Ingen anslutning till servern** - Kontrollera internetanslutning
2. **SSH-nyckeln är fel** - Kontrollera att `id_ed25519` är korrekt
3. **Servern är nere** - Kontrollera att servern `64.226.94.227` är tillgänglig
4. **Filer sparas på fel plats** - Kontrollera log-filerna för exakt sökväg

### Problem: "No files found in remote directory"
**Lösning:**
- Detta betyder att anslutningen fungerar men serverns mapp är tom
- Kontakta serveradministratören för att kontrollera `/prod/export` mappen

## Debugging-steg

### 1. Aktivera detaljerad output

Öppna Command Prompt och kör:
```cmd
cd C:\torp_deployment
torp_report_generator.exe
```

Detta visar all output direkt i konsolen.

### 2. Kontrollera nätverksanslutning

Testa om du kan nå servern:
```cmd
ping 64.226.94.227
```

### 3. Testa SSH-anslutning manuellt (om du har SSH-klient)

```cmd
ssh -i id_ed25519 torpshop_dl@64.226.94.227
```

### 4. Kontrollera filrättigheter

Se till att du har skrivrättigheter i mappen där executable-filen ligger:
- Högerklicka på mappen → Properties → Security
- Kontrollera att din användare har "Write" behörighet

## Kontakta support

Om inget av ovanstående hjälper, samla in följande information:

1. **Log-filer:**
   - `torp_report_generator.log`
   - `torp_report_generator_errors.log`

2. **Konsol-output:**
   - Kopiera all text från konsolfönstret

3. **Systeminformation:**
   - Windows-version
   - Var executable-filen ligger
   - Var filerna förväntas sparas (enligt log-filerna)

4. **Testresultat:**
   - Resultat från `ping 64.226.94.227`
   - Om `data\downloads\` mappen finns och är tom eller inte
