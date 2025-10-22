from pathlib import Path
path = Path('step3_example_plus_mask.json')
text = path.read_text()
path.write_text(text, encoding='utf-8')
