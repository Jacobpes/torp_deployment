# Torp Report Generator - Windows Executable

This is a Windows executable that automates the entire report generation process for Torp shop.

## Quick Start

1. **Download the executable**: `torp_report_generator.exe`
2. **Place required files** in the same directory:
   - `id_ed25519` (SSH key for SFTP download)
   - `data/parametrar/Beställningsfrekvens.csv` (supplier order frequencies)
   - `data/parametrar/Leveransfrekvens.csv` 
3. **Run** `torp_report_generator.exe`

## What It Does

The executable automatically:

1. **Downloads** latest data from SFTP server (if SSH key is available)
2. **Generates** `picking_list_results.csv` - picking list per store
3. **Generates** order lists in `orderlistor/` folder - one CSV file per supplier
4. **Generates** graphical picking lists in `picking_list_graphical/` folder - PDF files per store

## Output Files

After running, you'll find:

- `picking_list_results.csv` - Main picking list with all stores and products
- `orderlistor/Orderlista_[Supplier].csv` - Order lists grouped by supplier
- `picking_list_graphical/Plocklista_[Store].pdf` - Graphical picking lists per store

## Required Files

### 1. SSH Key (`id_ed25519`)
- Required for SFTP download
- Place in the same directory as the executable
- If missing, the program will skip download and use existing data files

### 2. `parametrar.csv`
Required format:
```csv
Butik;Leveransfrekvens_dagar
Torp kiosk;7
Bosund kiosk;7
...
```

### 3. `data/parametrar/Beställningsfrekvens.csv`
- Supplier order frequencies
- Must be in `data/` subdirectory

## Troubleshooting

### "Script not found" errors
- Make sure all Python scripts are in the same directory as the executable
- Or rebuild the executable with updated paths

### SFTP download fails
- Check that `id_ed25519` file exists
- Verify network connectivity
- The program will continue with existing data files if download fails

### Missing data files
- Ensure `data/parametrar/Beställningsfrekvens.csv` exists
- Ensure `parametrar.csv` exists
- The program will create `data/downloads/` automatically

### No output files generated
- Check console output for error messages
- Verify input data files exist and are readable
- Ensure you have write permissions in the directory

## Building Your Own Executable

See `BUILD_INSTRUCTIONS.md` for detailed instructions on building the executable from source.

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all required files are present
3. Ensure data files are in correct format
4. Try running the Python scripts directly: `python main.py`








