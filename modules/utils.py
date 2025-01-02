import sys
import torch
import json
import argparse

def detect_platform(cuda_num):
    if torch.cuda.is_available():
        print("cuda is available")
        return f'cuda:{cuda_num}'
    elif torch.backends.mps.is_available():
        print("mps is available")
        return 'mps'
    else:
        print("cpu is available")
        return 'cpu'

class CLIParser():
    arqs_dict = None
    parser = None
    def __init__(self):
        ...

    def get_json_defaults(self, config_json):
        with open(config_json, "r") as f:
            self.arqs_dict = json.load(f)
    
    def get_argument_parser(self):
        if self.parser:
            return self.parser 

        parser = argparse.ArgumentParser(description="Debugging the CLIParser")

        # parser.add_argument("config", default="config.json")
        # print("abcabc", parser.parse_args().config)
        if len(sys.argv) > 0 and sys.argv[1].endswith(".json"):
            self.get_json_defaults(sys.argv[1])
        else:
            self.get_json_defaults("config.json")

        parser.add_argument("config", default=sys.argv[1])
        for elem in self.arqs_dict:
            if type(self.arqs_dict[elem]) == dict:
                for elem2 in self.arqs_dict[elem]:
                    parser.add_argument(f"--{elem}.{elem2}", type=type(self.arqs_dict[elem][elem2]), default=self.arqs_dict[elem][elem2])
                continue
            parser.add_argument(f"--{elem}", type=type(self.arqs_dict[elem]), default=self.arqs_dict[elem])
        return parser

if __name__ == "__main__": 
    # Testing the CLIParser
    cp = CLIParser()
    argument_parser = cp.get_argument_parser()
    my_args = argument_parser.parse_known_args()
    print("Overriden Args: ", my_args)