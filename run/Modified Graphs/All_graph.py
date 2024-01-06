import os

def write_lines_to_csv(folder_path):
    # Get the full path for the output file
    output_file_path = os.path.join(folder_path, "all_graph.csv")

    # Open the output file for writing
    with open(output_file_path, 'w') as output_file:
        # List all files in the specified folder
        files = os.listdir(folder_path)

        print("hello!")

        # Iterate through each file in the folder
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            # Check if the item in the folder is a file (not a subdirectory)
            if os.path.isfile(file_path):
                cytokine = file_name[0:file_name.index("_")]
                if file_name == "All_graph.py":
                    continue
                with open(file_path, 'r') as file:
                    # Read and write each line to the output file
                    for line in file:
                        parts = line.split(",")
                        output_file.write(parts[0] + "," + cytokine + "\n")
                        output_file.write(cytokine + "," + parts[1] + "\n")

# Get the folder path from the script's directory
folder_path = os.path.dirname(__file__)

# Call the function to write lines to the "all_graph.csv" file
write_lines_to_csv(folder_path)
