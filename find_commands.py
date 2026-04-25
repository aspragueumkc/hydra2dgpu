import ras_commander as rc
import os
root = rc.__path__[0]
matches = []
for dirpath, dirnames, filenames in os.walk(root):
    for fn in filenames:
        if fn.endswith('.py'):
            path = os.path.join(dirpath, fn)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                txt = f.read()
            if '<Command Type=' in txt:
                matches.append(path)
print('files with Command Type examples:', matches[:10])
for p in matches[:5]:
    print('---', p)
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if '<Command Type=' in l:
                print(i+1, l.strip())
