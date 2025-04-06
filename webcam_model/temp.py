import os

# Path to the folder containing your files
folder_path = "D:\\projects\\postureproject\\dataset\\labels\\train\\bad Posture"

# List all files in the folder
files = os.listdir(folder_path)

# Sort files to ensure they are renamed in order
files.sort()

# Start renaming
for i, file_name in enumerate(files):
    # Extract file extension
    file_extension = os.path.splitext(file_name)[1]
    
    # New file name with numbers from 100-199
    new_name = f"training_pic {100 + i}{file_extension}"
    
    # Rename the file
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_name))

print("Renaming complete!")
