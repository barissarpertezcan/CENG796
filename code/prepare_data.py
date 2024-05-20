import os
import shutil
import sys

def organize_files(source_dir):
    # Ensure the provided path is an absolute path
    source_dir = os.path.abspath(source_dir)
    
    # Verify the source directory exists
    if not os.path.isdir(source_dir):
        print(f"Error: The directory {source_dir} does not exist.")
        return

    # Change to the source directory
    os.chdir(source_dir)

    # Get a list of all files in the directory
    files = os.listdir(source_dir)

    # Extract unique base names (without extensions) to create directories
    base_names = set()
    for file in files:
        base_name = file.split('.')[0]
        base_names.add(base_name)

    # Create directories and move files
    for base_name in base_names:
        # Create a directory for the base name
        os.makedirs(base_name, exist_ok=True)
        
        # Move each corresponding file into the newly created directory
        for ext in ['txt', 'json', 'jpg']:
            file_name = f"{base_name}.{ext}"
            if file_name in files:
                shutil.move(file_name, os.path.join(base_name, file_name))

    print("Files have been organized into directories.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python organize_files.py <path_to_your_files>")
    else:
        source_dir = sys.argv[1]
        organize_files(source_dir)
