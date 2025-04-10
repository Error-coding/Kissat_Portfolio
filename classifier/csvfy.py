import csv
import re
import os

def parse_line(line):
    instance_pattern = re.compile(r'Instance /nfs/share/instances/gbd/cnf/([a-f0-9]{32})-.*finished after ([\d.]+) seconds')
    timeout_pattern = re.compile(r'TIMEOUT: Instance /nfs/share/instances/gbd/cnf/([a-f0-9]{32})-.*punishment ([\d.]+) seconds')
    
    instance_match = instance_pattern.match(line)
    timeout_match = timeout_pattern.match(line)
    
    if instance_match:
        return instance_match.group(1), float(instance_match.group(2))
    elif timeout_match:
        return timeout_match.group(1), float(timeout_match.group(2))
    else:
        return None

def parse_configuration(line):
    try:
        config = eval(line.strip())
        if isinstance(config, dict):
            return config
        else:
            return "Default"
    except:
        return "Default"

def append_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as csvfile:
        fieldnames = ['key', 'time', 'configuration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)

def main(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    configuration = None
    if lines:
        configuration = parse_configuration(lines[0])
    
    for line in lines[1:]:
        parsed_data = parse_line(line)
        if parsed_data:
            key, time = parsed_data
            data = {'key': key, 'time': time, 'configuration': configuration}
            append_to_csv(output_file, data)

def check_time(file_path, key, configuration):

    if not os.path.isfile(file_path):
        return -1
    
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['key'] == key and row['configuration'] == str(configuration):
                print(configuration)
                return float(row['time'])
    
    return -1

if __name__ == "__main__":
    input_file = './data/anni2.out'
    output_file = 'data/res.csv'
    #print(check_time(output_file, "ffab5112bbf8e8831bc954e9211a6b27", "Default"))
    main(input_file, output_file)


