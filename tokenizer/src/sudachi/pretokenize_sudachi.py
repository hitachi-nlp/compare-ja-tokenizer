from sudachipy import tokenizer
from sudachipy import dictionary
import re

def main(args):
    input_file = args.input_path
    output_file = args.output_path
    sudachi = dictionary.Dictionary().create()
    input_f = open(input_file, "r")
    output_f = open(output_file, "w")
    whole_file = input_f.readlines()
    for i, text in enumerate(whole_file):
        try:
            output_f.write(" ".join([token.surface() for token in sudachi.tokenize(text)]))
        except:
            pass
        
    input_f.close()
    output_f.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Tokenize text file with Sudachi."
    )
    parser.add_argument(
        "--input_path", 
        help="(str) Path to a input file path.", 
        required=True
    )
    parser.add_argument(
        "--output_path", 
        help="(str) Path to a output file path.", 
        required=True
    )

    args = parser.parse_args()
    main(args)
