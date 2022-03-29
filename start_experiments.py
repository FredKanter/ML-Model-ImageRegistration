import sys
import yaml

from experiments import do_experiments

filenames = sys.argv[1:] if len(sys.argv) > 1 else ['parameters.yml']

for fn in filenames:
    print(f"Processing parameter file: '{fn}'")
    with open(fn, 'r') as stream:
        try:
            parameters = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            raise exc

        do_experiments(parameters)
