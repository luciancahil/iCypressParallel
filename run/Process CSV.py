def read_and_write_file(input_file_name, output_file_name):
    try:
        # Open the input file for reading
        with open(input_file_name, 'r') as input_file:
            # Open the output file for writing
            with open(output_file_name, 'w') as output_file:
                # Read each line from the input file
                lines = input_file.readlines()
                for index, line in enumerate(lines):
                    line = line.strip()
                    parts = line.split(",")
                    # Check if the second part is 'Y', then write to output file
                    if (len(parts )> 1) and (parts[1] == "Y" or parts[1] == "N"):
                        output_file.write(parts[0] + '\n')
                    else:
                        output_file.write(line + "\n")
    except FileNotFoundError:
        print(f"File '{input_file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'run/GenesToTissues.csv' with the path to your file if it's in a different directory
input_file_name = 'run/GenesToTissues.csv'
output_file_name = 'run/copy2.csv'

# Call the function
read_and_write_file(input_file_name, output_file_name)
