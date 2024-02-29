import os

def write_file_contents_to_csv(directory, output_file):
    line_set = set()
    with open(output_file, 'w') as csv_file:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            # Check if it's a file and not a directory
            if os.path.isfile(file_path):
                if(filename == "All_graph.py" or filename == "__init__.py"):
                    continue

                
                under_index = filename.index("_")
                cyto_name = filename[0:4]
                
                with open(file_path, 'r') as file:
                    for line in file:
                        if(len(line) == 0):
                            continue
                        parts = line.split(",")
                        print(filename)
                        line1 = parts[0] + "," + cyto_name + "\n"
                        line2 = cyto_name + "," + parts[1]
                        if(not (line1 in line_set)):
                            csv_file.write(line1)
                            line_set.add(line1)
                        
                        if(not (line2 in line_set)):
                            csv_file.write(line2)
                            line_set.add(line2)                          

current_directory = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_directory, "full_diverse.csv")
write_file_contents_to_csv(current_directory, output_file)
