from abc import ABC, abstractmethod
import subprocess
from typing import Iterable, Optional, Union


class CommandBase(ABC):
    @abstractmethod
    def get_command(self) -> str:
        pass

    def run(self, print_cmd=True, dryrun=False) -> None:
        command = self.get_command()
        if print_cmd:
            print(command, flush=True)
            print('\n---------------- OUTPUT ----------------\n', flush=True)
        if dryrun:
            print('*** DRY RUN ***', flush=True)
        else:
            subprocess.run(command, shell=True, check=True)
        print()


class Command(CommandBase):
    def __init__(self, command: str):
        self.command = command

    def get_command(self) -> str:
        return self.command


class ChunkflowCommandSequence(CommandBase):
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

    def _filter_relevant_vars(self) -> dict:
        commands_str = '\n'.join(self.commands)
        return {k: v for k, v in self.variables.items() if k in commands_str}

    def get_command(self) -> str:
        var_assignments = '\n'.join(f'{k}={v}' for k, v in self._filter_relevant_vars().items())
        command = ' \\\n    '.join(self.commands)
        if var_assignments:
            command = f'{var_assignments}\n{command}'
        return command
