import os
import shutil

def symlink_specific_files(old_dir, new_dir, filenames):
    """
    Create symbolic links for specific files from old_dir to new_dir, preserving the directory structure.
    
    Parameters:
    - old_dir (str): The root directory to search for files.
    - new_dir (str): The root directory where symbolic links will be created.
    - filenames (set): A set of filenames to search for and create symbolic links for.
    """
    # Ensure the new directory exists
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Walk through the old directory
    for root, dirs, files in os.walk(old_dir):
        # Calculate the relative path from the old directory
        relative_path = os.path.relpath(root, old_dir)
        # Define the new directory path
        new_root = os.path.join(new_dir, relative_path)
        
        # Ensure the corresponding directory exists in the new directory
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        
        # Create symbolic links for specific files
        for file_name in files:
            if file_name in filenames:
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(new_root, file_name)
                
                # Remove the destination file if it already exists
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                
                # Create a symbolic link
                os.symlink(old_file_path, new_file_path)
                print(f'Created symlink: {new_file_path} -> {old_file_path}')


def copy_specific_files(old_dir, new_dir, filenames):
    """
    Copy specific files from old_dir to new_dir, preserving the directory structure.
    
    Parameters:
    - old_dir (str): The root directory to search for files.
    - new_dir (str): The root directory where files will be copied to.
    - filenames (set): A set of filenames to search for and copy.
    """
    # Ensure the new directory exists
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    # Walk through the old directory
    for root, dirs, files in os.walk(old_dir):
        # Calculate the relative path from the old directory
        relative_path = os.path.relpath(root, old_dir)
        # Define the new directory path
        new_root = os.path.join(new_dir, relative_path)
        
        # Ensure the corresponding directory exists in the new directory
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        
        # Copy specific files
        for file_name in files:
            if file_name in filenames:
                old_file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(new_root, file_name)
                shutil.copy2(old_file_path, new_file_path)
                print(f'Copied: {old_file_path} to {new_file_path}')

def check_files_exist(old_dir, filenames):
    """
    Copy specific files from old_dir to new_dir, preserving the directory structure.
    
    Parameters:
    - old_dir (str): The root directory to search for files.
    - filenames (set): A set of filenames to search for and copy.
    """

    # Walk through the old directory
    for root, dirs, files in os.walk(old_dir):
        # Calculate the relative path from the old directory
        relative_path = os.path.relpath(root, old_dir)

        # Copy specific files
        for file_name in files:
            if file_name not in filenames:
                print('Missing file {file_name} for sim {old_file_path}')

# Example usage
old_directory = '/home/x-dbartlett/scratch_dir/cmass/quijotelike/pinocchio/L1000-N512'
new_directory = '/home/x-dbartlett/scratch_dir/cmass/L1000-N512_halos'
file_names_to_symlink = {'halos.h5', 'nbody.h5'}

#symlink_specific_files(old_directory, new_directory, file_names_to_symlink)
#copy_specific_files(old_directory, new_directory, file_names_to_symlink)
check_files_exist(new_directory, file_names_to_symlink)

