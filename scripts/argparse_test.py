"""
Testing functionality of argparse for https://stackoverflow.com/questions/37367331/is-it-possible-to-use-argparse-to-capture-an-arbitrary-set-of-optional-arguments and https://stackoverflow.com/questions/21920989/parse-non-pre-defined-argument.
"""

import argparse
from addict import Dict

class StoreNameValuePair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print(values)
        n, v = values.split('=')
        setattr(namespace, n, v)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model.')
    #parser.add_argument('--hi', type=str)
    
    parsed, unknown = parser.parse_known_args() #this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments
    
    d = Dict()

    for arg in unknown:
        if arg.startswith(("--")):
            #you can pass any arguments to add_argument
            #print(arg)
            arg, value = arg.split('=')
            arg = arg.split('-')[-1]
            print(arg, value)
            parser.add_argument(arg, action=StoreNameValuePair)


    args = parser.parse_args()

    return args
#def parse_arguments():
#    parser = argparse.ArgumentParser(description='Train a model.')##

#    args = parser.add_argument("", action=StoreNameValuePair)#

"""def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a model.')
    # put in the config path
    DEFAULT_CONFIG_PATH = 'configs/midi2params-1.yml'
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                        help='Path to the train config file')

    args = parser.parse_args()

    return args
"""

if __name__ == '__main__':
    args = parse_arguments()
    
    print(args)