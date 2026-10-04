from pathlib import Path

path = Path('webapp/static/style.css')
source = path.read_text(encoding='utf-8')
source = source.replace('{', ' {\n').replace('}', '\n}\n').replace(';', ';\n')
lines = []
indent = 0
for line in source.splitlines():
    line = line.strip()
    if not line:
        continue
    if line == '}':
        indent -= 1
    lines.append('  ' * indent + line)
    if line.endswith('{'):
        indent += 1
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
