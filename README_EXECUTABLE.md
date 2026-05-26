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
2. **Generates** picking lists per store (Excel + PDF) in `output/plocklistor/`
3. **Generates** order lists per supplier (Excel) in `output/orderlistor/`
4. **Keeps** machine-readable CSV copies in `system_data/` for the pipeline
   itself - these are not needed by end users

## Output Files

After running, you'll find two top-level folders:

### `output/` - what the user opens

- `output/plocklistor/Plocklista_[Store].pdf` - Graphical picking lists per store
- `output/plocklistor/[Store].xlsx` - Picking list per store (Excel)
- `output/plocklistor/Plocklistor_sammanställning.xlsx` - Combined picking list (Excel)
- `output/orderlistor/Orderlista_[Supplier].xlsx` - Order lists grouped by supplier

### `system_data/` - working data the scripts read between steps

- `system_data/plocklistor/[Store].csv` - Per-store picking list (raw CSV)
- `system_data/plocklistor/picking_list_results.csv` - Combined picking list (raw CSV); read by the PDF step
- `system_data/orderlistor/Orderlista_[Supplier].csv` - Per-supplier order list (raw CSV)

End users normally only need the files in `output/`. The `system_data/`
folder exists so the pipeline has a clean place to keep its intermediate
CSV files without cluttering the user's view.

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








