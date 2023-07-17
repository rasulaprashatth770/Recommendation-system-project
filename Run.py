import sys
import subprocess
import os
# implement pip as a subprocess:
wd = os.getcwd()
print(wd)
os.chdir(wd)

subprocess.check_call([sys.executable, '-m', 'pip', 'install',
'streamlit'])


subprocess.check_call(['streamlit','run','model.py'])