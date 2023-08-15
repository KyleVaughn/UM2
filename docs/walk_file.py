import os

file_list:list[str]=[]

def find_directories_and_header_files(directory:str):
    for root, dirs, files in os.walk(directory):
        global file_list
        for file in files:
            if file.endswith('.h') or file.endswith('.hpp') or file.endswith('inl'):
                file_list.append(os.path.join(root, file))
        for subdir in dirs:
            find_directories_and_header_files(subdir)


include_dir = '../include'
find_directories_and_header_files(include_dir)
file_list = list(filter(lambda x: not isinstance(x, list), file_list))
file_string = ' '.join(file_list)