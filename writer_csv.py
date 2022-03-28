import csv
import os


def writer_list(file_path, parameter_run, name='Configuration_runs.csv', mode='a'):
    file_path = os.path.join(file_path, name)
    if mode == 'a':
        with open(file_path, 'r') as fi:
            reader = csv.reader(fi, delimiter=',')
            heads = next(reader)
            line_count = sum(1 for row in reader)
            print('File contains {} lines'.format(line_count))

        # add new results check whether list dimensions fits row count in file
        if len(heads) != len(parameter_run[0]):
            raise NameError('Given parameter list does not match number of rows in chosen file')

        # add empty line to better distinguish between connected runs
        with open(file_path, mode=mode) as fi:
            writer = csv.writer(fi, delimiter=',')
            for entry in range(len(parameter_run)):
                writer.writerow(parameter_run[entry])

    elif mode == 'w':
        with open(file_path, mode=mode) as wi:
            # first element should be list of Headers/ Keys
            heads = parameter_run[0]
            writer = csv.writer(wi, delimiter=',')
            writer.writerow(heads)
            for entry in range(1, len(parameter_run)):
                writer.writerow(parameter_run[entry])


def writer_dict(file_path, parameter_run, name='Configuration_runs.csv', mode='a'):
    file_path = os.path.join(file_path, name)
    if mode == 'a':
        with open(file_path, mode='r') as fi:
            reader = csv.DictReader(fi)
            fieldnames = reader.fieldnames
            line_count = sum(1 for row in reader)
            print('File contains {} lines'.format(line_count))

        if len(parameter_run[0]) != len(fieldnames):
            raise NameError('Given parameter list does not match number of rows in chosen file')

        with open(file_path, mode=mode) as wi:
            writer = csv.DictWriter(wi, fieldnames=fieldnames)
            for ii in range(len(parameter_run)):
                writer.writerow(parameter_run[ii])
    elif mode == 'w':
        fieldnames = parameter_run[0].keys()
        with open(file_path, mode=mode) as wi:
            writer = csv.DictWriter(wi, fieldnames=fieldnames)
            writer.writeheader()
            for ii in range(len(parameter_run)):
                writer.writerow(parameter_run[ii])


def write_csv_file(file_path, file_name, parameter_run, version='dict', mode='a'):
    if not isinstance(parameter_run, list):
        raise TypeError('The given data has a wrong type. Please provide your results in a list directly or '
                        'in list containing dictionaries')
    if not parameter_run:
        raise RuntimeError('The given data list is empty')
    if not (mode == 'a' or mode == 'w'):
        raise NameError('Mode is not supported, it has to be either "a" to extend an existing file or "w" '
                        'for creating a new file')

    if version == 'list':
        writer_list(file_path, parameter_run, file_name, mode=mode)
    elif version == 'dict':
        writer_dict(file_path, parameter_run, file_name, mode=mode)
    else:
        raise NameError('Versions supported are either "list" or "dict".\n'
                        'Version default is dict.\n'
                        'Please make sure that your parameters are passed in a fitting form (list/dict)')


def to_csv(pth, data, filename):
    if os.path.exists(os.path.join(pth, filename)):
        mode = 'a'
    else:
        mode = 'w'
    write_csv_file(pth, filename, data, version='dict', mode=mode)
