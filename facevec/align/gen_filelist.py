import argparse
import os
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('bounding_boxes_filename', type=str, help='bounding boxes filename.')
    return parser.parse_args(argv)


def main(args):
    with open(args.bounding_boxes_filename, 'r') as fr:
        with open(os.path.join(os.path.dirname(args.bounding_boxes_filename), "filelist.txt"), 'w') as fw:
            line = fr.readline()
            type_set = set()
            while line:
                line = fr.readline()
                tmp = line.split(' ')
                if len(tmp) > 1:
                    filename = tmp[0]
                    typename = os.path.dirname(filename)
                    if typename not in type_set:
                        type_set.add(typename)
                    # fw.write("%s %d\n" % (filename.replace("../../data/", ""), len(type_set) - 1))
                    fw.write("%s %d\n" % (filename, len(type_set) - 1))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

