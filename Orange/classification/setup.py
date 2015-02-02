import subprocess
import os

# TODO: use distutils for this
os.chdir('Orange/classification')
subprocess.check_output('gcc -Wall -O2 -fPIC --shared -o _tree.so _tree.c'.split())
os.chdir('../..')
