import argparse
import os
from datetime import datetime
from glob import glob

from chunkflow.scripting.utils import parse_file, print_section_header


def get_timestamp(for_filename=False) -> str:
    if for_filename:
        return datetime.now().strftime('%Y%m%dT%H%M%S')
    else:
        return datetime.now().isoformat()


def get_latest_log_path() -> str:
    log_files = glob('chunkflow.*.log')
    if log_files:
        return sorted(log_files)[-1]
    else:
        return get_new_log_path()


def get_new_log_path() -> str:
    return f'chunkflow.{get_timestamp(for_filename=True)}.log'


def write_msg_to_log(log, msg):
    if log is not None:
        with open(log, 'a') as f:
            f.write(msg + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to file containing commands')
    parser.add_argument('--args', type=str, default=None, help='Arguments to pass to the script')
    parser.add_argument('--dryrun', action='store_true', help='Print commands without running')
    parser.add_argument('--log', action='store_true', help='Append commands to log file')
    parser.add_argument('--log-path', type=str, default=None, help='Path to log file')
    parser.add_argument('--log-restart', action='store_true', help='Create a new log file')
    args = parser.parse_args()

    if args.log and int(os.environ.get('DISBATCH_REPEAT_INDEX', '-1')) <= 0:
        if args.log_restart:
            log = get_new_log_path()
        elif args.log_path:
            log = args.log_path
        else:
            log = get_latest_log_path()
        print(f'Logging to {log}')
    else:
        log = None

    cmd_seqs = parse_file(args.file, args=args.args)
    sep_string = '=' * 80
    write_msg_to_log(log, sep_string + '\n')
    cmd_str = args.file
    if args.args:
        cmd_str += f' {args.args}'
    write_msg_to_log(log, f'[{get_timestamp()}] Running commands: {cmd_str} >>>\n')
    for i, cmd_seq in enumerate(cmd_seqs):
        print_section_header(f'Chunkflow command sequence {i + 1}')
        write_msg_to_log(log, cmd_seq.get_command())
        try:
            cmd_seq.run(dryrun=args.dryrun)
            write_msg_to_log(log, '\n<<< SUCCEEDED\n')
        except:
            write_msg_to_log(log, '\n<<< FAILED\n')
            raise


if __name__ == '__main__':
    main()
