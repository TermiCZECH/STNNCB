import os
import shutil

# Define the source files and directories
source_files = [
    'config.json',
    'server.py',
    'output.json',
    'output-copy.json',
    'default_config.json',
    'config.py'
]

# Define the target directory for the built application
target_directory = 'app'

# Create the target directory
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Copy the source files to the target directory
for file in source_files:
    shutil.copy(file, target_directory)

# Copy the requirements.txt file to the target directory
shutil.copy('requirements.txt', target_directory)

# Copy the website folder to the target directory
shutil.copytree('website', f'{target_directory}/website')

# Optionally, you can install the required packages from requirements.txt using pip
os.system(f"pip install -r {target_directory}/requirements.txt")

print("Build completed successfully.")

