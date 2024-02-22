import os

def list_files_in_directory():
    # Get the current directory path
    current_directory = os.path.dirname(os.path.realpath(__file__))

    # List all files in the current directory
    for filename in os.listdir(current_directory):
        if os.path.isfile(os.path.join(current_directory, filename)):
            if(filename == "__init__.py" or filename== "All_graph.py"):
                continue
            under_index = filename.index("_")
            name = (filename[0:under_index])
            new_name = name + "_diverse_graph.csv"
            os.rename("run/Graphs/" + filename,"run/Graphs/" + new_name)

list_files_in_directory()
