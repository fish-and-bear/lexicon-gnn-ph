# === ML Pipeline Script for Colab ===
#
# Instructions for use in Google Colab:
# 1. Upload your entire 'fil-relex' project folder to the root of your Google Drive.
# 2. Create a new Colab notebook.
# 3. Copy the code blocks below into separate cells in your Colab notebook.
# 4. !! IMPORTANT !! In the cell marked "## 2. Change Directory", update the 'project_path'
#    variable to the correct path of your 'fil-relex' folder on Google Drive.
# 5. Run the cells sequentially.

# === Cell 1: Mount Google Drive ===
print("--- Running Cell 1: Mount Google Drive ---")
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except ImportError:
    print("This script expects to be run in Google Colab. 'google.colab' not found.")
except Exception as e:
    print(f"An error occurred during drive mounting: {e}")
print("-" * 30)

# === Cell 2: Change Directory ===
print("--- Running Cell 2: Change Directory ---")
import os

# --- !!! UPDATE THIS PATH !!! ---
# Set this to the path where your 'fil-relex' folder is located in Google Drive
project_path = '/content/drive/MyDrive/fil-relex' # <-- EXAMPLE, CHANGE AS NEEDED
# --------------------------------

print(f"Attempting to change directory to: {project_path}")
try:
    os.chdir(project_path)
    print(f"Successfully changed directory to: {os.getcwd()}")
except FileNotFoundError:
    print(f"ERROR: Directory not found: {project_path}")
    print("Please ensure:")
    print("  1. You uploaded the 'fil-relex' folder to the root of your Google Drive.")
    print(f"  2. The 'project_path' variable above is set to the correct path (e.g., '/content/drive/MyDrive/fil-relex')")
except Exception as e:
    print(f"An error occurred changing directory: {e}")
print("-" * 30)


# === Cell 3: Install Dependencies ===
print("--- Running Cell 3: Install Dependencies ---")
import subprocess
import sys

requirements_file = 'ml/requirements.txt'
print(f"Checking if {requirements_file} exists...")
if os.path.exists(requirements_file):
    print(f"Attempting to install dependencies from {requirements_file}...")
    # Using subprocess to capture output/errors better in a script context
    process = subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file], capture_output=True, text=True)
    if process.stdout:
        print("PIP STDOUT:\n", process.stdout)
    if process.stderr:
        print("PIP STDERR:\n", process.stderr)

    if process.returncode == 0:
        print("Dependency installation command finished successfully.")
    else:
        print(f"Dependency installation command failed with exit code {process.returncode}.")

    # Optional PyG install - uncomment the following lines in Colab cell if needed
    # print("Attempting to install PyTorch Geometric (optional)...")
    # pyg_command_parts = [
    #     sys.executable, '-m', 'pip', 'install', 'torch_geometric', 'pyg_lib',
    #     'torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv',
    #     '-f', 'https://data.pyg.org/whl/torch-$(python -c \'import torch; print(torch.__version__)\')+cpu.html'
    # ]
    # pyg_process = subprocess.run(pyg_command_parts, capture_output=True, text=True)
    # if pyg_process.stdout:
    #     print("PyG PIP STDOUT:\n", pyg_process.stdout)
    # if pyg_process.stderr:
    #     print("PyG PIP STDERR:\n", pyg_process.stderr)
    # if pyg_process.returncode == 0:
    #      print("PyG installation command finished successfully.")
    # else:
    #      print(f"PyG installation command failed with exit code {pyg_process.returncode}.")

else:
    print(f"ERROR: {requirements_file} not found in the current directory ({os.getcwd()}).")
    print("Ensure you have changed to the correct project directory in Cell 2.")
print("-" * 30)


# === Cell 4: Run ML Pipeline ===
print("--- Running Cell 4: Run ML Pipeline ---")
pipeline_script = 'ml/run_pipeline.py'
config_file = 'config/default_config.json'
db_config_file = 'my_db_config.json'
output_dir = 'output/pipeline_run_colab'

print(f"Checking if script '{pipeline_script}' and configs exist...")
if not os.path.exists(pipeline_script):
     print(f"ERROR: Pipeline script not found: {pipeline_script}")
elif not os.path.exists(config_file):
     print(f"ERROR: Main config not found: {config_file}")
elif not os.path.exists(db_config_file):
     print(f"ERROR: DB config not found: {db_config_file}")
else:
    print(f"Attempting to run: python {pipeline_script} ...")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    # Construct command
    command = [
        sys.executable,
        pipeline_script,
        '--config', config_file,
        '--db-config', db_config_file,
        '--output-dir', output_dir,
        '--debug'
    ]

    # Run the pipeline script using subprocess
    process = subprocess.run(command, capture_output=True, text=True)

    if process.stdout:
        print("Pipeline STDOUT:\n", process.stdout)
    if process.stderr:
        print("Pipeline STDERR:\n", process.stderr)

    if process.returncode == 0:
        print("Pipeline script finished successfully.")
    else:
        print(f"Pipeline script failed with exit code {process.returncode}.")

print("-" * 30)
print("--- Script Finished ---") 