import subprocess
from typing import Iterable, Optional, Union


class ChunkflowCommandSequence:
    def __init__(self, commands: Union[str, Iterable[str]], variables: Optional[dict] = None):
        if isinstance(commands, str):
            commands = [commands]
        if not isinstance(commands, list):
            commands = list(commands)
        assert len(commands) > 0
        if not commands[0].startswith('chunkflow'):
            commands = ['chunkflow'] + commands
        self.commands = commands
        self.variables = variables or {}

    def run(self, print_cmd=True, dryrun=False) -> None:
        var_assignments = '\n'.join(f'{k}={v}' for k, v in self.variables.items())
        command = ' \\\n    '.join(self.commands)
        if var_assignments:
            command = f'{var_assignments}\n{command}'
        if print_cmd:
            print(command)
            print('\n---------------- OUTPUT ----------------\n')
        if dryrun:
            print('*** DRY RUN ***')
        else:
            subprocess.run(command, shell=True, check=True)
        print()
