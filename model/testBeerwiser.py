from pathlib import Path
from core.trbs import TheResponsibleBusinessSimulator

# Read the data and build the case
path = Path.cwd() / 'data'
file_format = 'xlsx'
name = 'beerwiser'
case = TheResponsibleBusinessSimulator(path, file_format, name)
case.build()

case.visualize('network', 'dependencies', **{'case': name, 'node': 'Accidental redu ction'})
