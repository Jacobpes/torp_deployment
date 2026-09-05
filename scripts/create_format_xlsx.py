"""One-off helper to create the initial format.xlsx from the template list."""
import sys
from pathlib import Path

_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))

from _common import _write_format_workbook, normalize_name  # noqa: E402

RED = 'FF0000'
BLACK = '000000'
BLUE = '00B0F0'

ROWS = [
    ('Torp Jordgubbslemonad 500ml', '6429811349308', RED),
    ('Torp JORDGUBBAR 250g inhemsk', '2400013958929', RED),
    ('Torp tomat II KLASS LÖS', '10', RED),
    ('Torp tomat BAMANO ask 250g (pärl gul)', '2400098994980', RED),
    ('Torp tomat Mix 250g / Grön, Gul, Röd 250g', '2400070950461', RED),
    ('Torp tomat DELISHER 250g ask (röd pärl)', '2400094082889', RED),
    ('Torp tomat GOURAMI 250g ask (grön)', '2400050416567', RED),
    ('Ramlösa citron 330 ml', '7310070009906', BLACK),
    ('Wild & Berg WHITE chocolate Cranberry Proteinrich', '', BLACK),
    ('Fyllsnack', '6429810997050', BLACK),
    ('Essegg ÄGG M 30 st/bricka', '6430066260027', BLACK),
    ('Essegg ÄGG 15 st/ask', '6429830045403', BLACK),
    ('Närpes Drottninggräddglass 480 ml', '6429810833532', BLUE),
    ('Närpes Vaniljgräddglass', '6429810833006', BLUE),
    ('Närpes Kaffegräddglass 480ml', '6429810833020', BLUE),
    ('Närpes Chokladgräddglass 480 ml', '6429810833037', BLUE),
    ('Närpes Vit choklad-jordgubbsglass 480 ml', '6429810833051', BLUE),
    ('Närpes Salmiac gräddglass 480 ml', '6429810833082', BLUE),
    ('Närpes Saltkinuskigräddglass', '6429810833150', BLUE),
    ('Närpes Hallonsorbet-Salmiakgräddglass 480ml', '6429810833167', BLUE),
]


def main():
    entries = [
        {
            'name': name,
            'norm': normalize_name(name).lower(),
            'code': code,
            'color': color,
        }
        for name, code, color in ROWS
    ]
    root = _script_dir.parent
    targets = [
        root / 'data' / 'parametrar' / 'format.xlsx',
        root / 'Prognosautomation' / 'data' / 'parametrar' / 'format.xlsx',
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_format_workbook(target, entries)
        print(f'Created {target}')


if __name__ == '__main__':
    main()
