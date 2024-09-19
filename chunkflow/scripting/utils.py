import os
from typing import List

from chunkflow.scripting.command_sequence import ChunkflowCommandSequence, Command, CommandBase


def split_lines(lines: str) -> List[str]:
    lines = [line.split('#', maxsplit=1)[0].rstrip() for line in lines.split('\n')]
    lines = [line for line in lines if line]
    if lines and lines[-1].endswith('\\'):
        lines[-1] = lines[-1][:-1]
    return lines


def parse_sh_file(file_path: str) -> List[Command]:
    with open(file_path, 'r') as f:
        lines = split_lines(f.read())

    # Replace `source <file>` with the contents of the file
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('source '):
            source_file = line.split(' ', maxsplit=1)[1]
            with open(source_file, 'r') as f:
                source_lines = split_lines(f.read())
            lines = lines[:i] + source_lines + lines[(i + 1):]
        i += 1

    return [Command('\n'.join(lines))]


def split_commands(lines: str) -> List[str]:
    out = []
    lines = lines.replace('\\\n', ' ')
    for line in lines.strip().split('\n'):
        line = line.split('#')[0].strip()  # Remove comments
        if line:
            out.append(' '.join(line.split()).strip())
    return out


def parse_cf_file(file_path: str) -> List[ChunkflowCommandSequence]:
    with open(file_path, 'r') as f:
        lines = split_commands(f.read())

    # Replace `source <file>` with the contents of the file
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('source '):
            source_file = line.split(' ', maxsplit=1)[1]
            with open(source_file, 'r') as f:
                source_lines = split_commands(f.read())
            lines = lines[:i] + source_lines + lines[(i + 1):]
        i += 1

    if not any(line.startswith('chunkflow') for line in lines):
        lines = ['chunkflow'] + lines

    variables = {}
    chunkflow_seqs = []
    current_cf_cmd = None
    for line in lines:
        if line.startswith('chunkflow'):
            if current_cf_cmd is not None:
                chunkflow_seqs.append(ChunkflowCommandSequence(current_cf_cmd, variables))
            current_cf_cmd = [line]
        elif current_cf_cmd is not None:
            current_cf_cmd.append(line)
        else:
            if current_cf_cmd is not None:
                raise RuntimeError('All variable assignments must come before the first chunkflow command.')
            name, value = line.split('=')
            variables[name.strip()] = value.strip()
    if current_cf_cmd is not None:
        chunkflow_seqs.append(ChunkflowCommandSequence(current_cf_cmd, variables))

    return chunkflow_seqs


def parse_file(file_path: str, args: str = None) -> List[CommandBase]:
    if os.path.splitext(file_path)[-1] in {'.sh'}:
        if args:
            raise ValueError("Arguments are not supported for shell scripts")
        return parse_sh_file(file_path)
    elif os.path.splitext(file_path)[-1] in {'.cf', '.chunkflow'}:
        if args:
            raise ValueError("Arguments are not supported for chunkflow scripts")
        return parse_cf_file(file_path)
    elif os.path.splitext(file_path)[-1] in {'.py'}:
        command = f'python {file_path}'
        if args:
            command += f' {args}'
        return [Command(command)]
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def print_section_header(header: str):
    sep_line = '========================================'
    s = f'\n{sep_line}\n  {header}\n{sep_line}\n'
    print(s, flush=True)


def print_statement(*parts: str, sep=' '):
    print(f">>>>> {sep.join(str(s) for s in parts)}")
