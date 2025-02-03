import os

def save_code(file_name, new_dir):
    current_directory = os.getcwd()
    main_path = os.path.join(current_directory, file_name)
    with open(main_path, 'r') as file:
        content = file.readlines()
    new_file_path = os.path.join(current_directory, new_dir)
    with open(new_file_path, 'w') as file:
        file.writelines(content)

def get_lines(file_name, start_line, end_line):
    start = 0
    end = 0
    with open(file_name, 'r') as file:
        content = file.readlines()
        for line in content:
            if line.startswith(start_line):
                start = content.index(line)
            if line.startswith(end_line):
                end = content.index(line)
    return "".join(content[start:end])
