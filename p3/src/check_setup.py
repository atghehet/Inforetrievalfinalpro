
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("P3 folder:", project_folder)
print("P3 folder exists:", os.path.isdir(project_folder))
