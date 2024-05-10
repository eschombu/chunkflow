import argparse

from chunkflow.scripting.utils import parse_file, print_section_header, print_statement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to file containing commands')
    parser.add_argument('--dryrun', action='store_true', help='Print commands without running')
    args = parser.parse_args()

    cmd_seqs = parse_file(args.file)
    for i, cmd_seq in enumerate(cmd_seqs):
        print_section_header(f'Chunkflow command sequence {i + 1}')
        cmd_seq.run(dryrun=args.dryrun)


if __name__ == '__main__':
    main()
