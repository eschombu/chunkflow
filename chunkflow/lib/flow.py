import os
import pdb
import sys
import traceback
from typing import Union

from functools import update_wrapper, wraps

import click
from .cartesian_coordinate import Cartesian


class CartesianParamType(click.ParamType):
    name = 'Cartesian'
    
    def convert(self, value: Union[list, tuple], param, ctx):
        assert len(value) == 3
        return Cartesian.from_collection(value)        

CartesianParam = CartesianParamType()

# global dict to hold the operators and parameters
state = {'operators': {}}
DEFAULT_CHUNK_NAME = 'chunk'
DEFAULT_SYNAPSES_NAME = 'syns'
DEFAULT_SKELETON_NAME = 'skels'


def get_initial_task():
    return {'log': {'timer': {}}}


def default_none(ctx, _, value):
    """
    click currently can not use None with tuple type
    it will return an empty tuple if the default=None details:
    https://github.com/pallets/click/issues/789
    """
    if not value:
        return None
    else:
        return value


# the code design is based on:
# https://github.com/pallets/click/blob/master/examples/imagepipe/imagepipe.py
@click.group(chain=True)
@click.option('--mip', '-m',
              type=click.INT, default=0,
              help='default mip level of chunks.')
@click.option('--dry-run/--real-run', default=False,
              help='dry run or real run. default is real run.')
@click.option('--verbose/--quiet', default=False, 
    help='show more information or not. default is False.')
@click.option('--debug/--no-debug', default=False,
    help='drop into pdb.postmortem upon exception.')
def main(mip, dry_run, verbose, debug):
    """Compose operators and create your own pipeline."""
    
    state['mip'] = mip
    state['dry_run'] = dry_run
    state['verbose'] = verbose
    state['debug'] = debug if 'SLURM_JOB_ID' not in os.environ else False

    if dry_run:
        print('\nYou are using dry-run mode, will not do the work!')


@main.result_callback()
def process_commands(operators, mip, dry_run, verbose, debug):
    """This result callback is invoked with an iterable of all 
    the chained subcommands. As in this example each subcommand 
    returns a function we can chain them together to feed one 
    into the other, similar to how a pipe on unix works.
    """
    try:
        # It turns out that a tuple will not work correctly!
        stream = [get_initial_task(), ]

        # Pipe it through all stream operators.
        for operator in operators:
            stream = operator(stream)
            # task = next(stream)

        # Evaluate the stream and throw away the items.
        for _ in stream:
            pass
    except (KeyboardInterrupt, pdb.bdb.BdbQuit):
        sys.exit(1)
    except Exception:
        if state.get('debug'):
            traceback.print_exc()
            pdb.post_mortem()
        else:
            raise


def operator(f):
    """
    Help decorator to rewrite a function so that
    it returns another function from it.
    """
    def new_func(*args, **kwargs):
        def op(stream):
            return f(stream, *args, **kwargs)

        return update_wrapper(op, f)

    return update_wrapper(new_func, f)


def generator(f):
    """Similar to the :func:`operator` but passes through old values unchanged 
    and does not pass through the values as parameter.
    """
    def new_func(stream, *args, **kwargs):
        # yield from stream
        yield from f(*args, **kwargs)

    return update_wrapper(operator(update_wrapper(new_func, f)), f)
