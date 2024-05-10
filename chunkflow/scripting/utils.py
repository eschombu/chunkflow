from typing import List

from chunkflow.scripting.command_sequence import ChunkflowCommandSequence


def split_commands(lines: str) -> List[str]:
    out = []
    lines = lines.replace('\\\n', ' ')
    for line in lines.strip().split('\n'):
        line = line.split('#')[0].strip()  # Remove comments
        if line:
            out.append(' '.join(line.split()).strip())
    return out


def parse_file(file_path: str) -> List[ChunkflowCommandSequence]:
    with open(file_path, 'r') as f:
        lines = split_commands(f.read())
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


def print_section_header(header: str):
    sep_line = '========================================'
    s = f'\n{sep_line}\n  {header}\n{sep_line}\n'
    print(s)


def print_statement(*parts: str, sep=' '):
    print(f">>>>> {sep.join(str(s) for s in parts)}")
