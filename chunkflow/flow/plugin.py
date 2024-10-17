# -*- coding: utf-8 -*-

import os
from typing import Callable, Union

import numpy as np

from chunkflow.lib.cartesian_coordinate import Cartesian
from chunkflow.lib.utils import str_to_dict

from .base import OperatorBase

from chunkflow.lib import load_source
from chunkflow.chunk import Chunk
from chunkflow.point_cloud import PointCloud


def array_to_chunk(arr: Union[np.ndarray, Chunk], voxel_offset: Cartesian, 
        voxel_size: Cartesian, shape: tuple):
    if isinstance(arr, np.ndarray):
        # in case the plugin did some symmetric cropping
        offset = tuple(vo + (ins - outs)//2 for vo, ins, outs in zip(voxel_offset, shape[-3:], arr.shape[-3:]) )
        return Chunk(arr, voxel_offset=offset, voxel_size=voxel_size)
    else:
        return arr


class Plugin(OperatorBase):
    r"""
    Chunk operation using a custom python file.
    """
    def __init__(self, f: Callable, name: str = 'plugin-1'):
        super().__init__(name=name)
        self.execute = f

    @classmethod
    def from_file(cls, plugin_file_name: str, name: str = 'plugin-1'):
        r"""
        Loads a custom python file specified in `plugin_file_name`, which
        should contain a callable f such that a call of `f(chunk, args)`
        operates on the chunk. The callable's name can be specified using
        the format <plugin_file_name>::<callable>, with 'execute' as the
        default name.
        """
        file_name, func_name = plugin_file_name.rsplit('::', maxsplit=1)
        if func_name is None:
            func_name = 'execute'

        if not file_name.endswith('.py'):
            file_name += '.py'

        plugin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../plugins')
        plugin_dir1 =os.path.join(plugin_dir, 'chunkflow-plugins/chunkflowplugins')

        plugin_dirs = ['./', plugin_dir, plugin_dir1]
        if 'CHUNKFLOW_PLUGIN_DIR' in os.environ:
            plugin_dirs.append(os.environ['CHUNKFLOW_PLUGIN_DIR'])

        for plugin_dir in plugin_dirs:
            fname = os.path.join(plugin_dir, file_name)
            if os.path.exists(fname):
                print(f'loading plugin {fname}::{func_name}')
                program = load_source(fname)
                # NOTE: assuming this is a func / static functor for now, maybe make it a class?
                assert hasattr(program, func_name), f'plugin {file_name} does not have a function named {func_name}'
                plugin_exec = getattr(program, func_name)
                return cls(plugin_exec, name)

        # if we reach here, the plugin was not found
        raise RuntimeError(f'did not find plugin: {file_name}')

    def __call__(self, inputs: list, args: str = None):
        voxel_offset = None
        voxel_size = None
        shape = None
        for inp in inputs:
            if isinstance(inp, Chunk):
                voxel_offset = inp.voxel_offset
                voxel_size = inp.voxel_size
                shape = inp.shape
                break
        if args is not None and '=' in args:
            args = str_to_dict(args)

        if len(inputs) == 0 and args is None:
            outputs = self.execute()
        elif len(inputs) == 0 and args is not None:
            if isinstance(args, str):
                outputs = self.execute(args=args)
            elif isinstance(args, dict):
                outputs = self.execute(**args)
            else:
                raise ValueError(f'unsupported argument: {args}')
        elif len(inputs) > 0 and args is None:
            outputs = self.execute(*inputs)
        else:
            if isinstance(args, str):
                outputs = self.execute(*inputs, args=args)
            elif isinstance(args, dict):
                outputs = self.execute(*inputs, **args)
            else:
                raise ValueError(f'unsupported argument: {args}')
                
        # assert isinstance(outputs, list) or isinstance(outputs, tuple) or outputs is None
        if isinstance(outputs, tuple):
            outputs = [*outputs] 
        # automatically convert the ndarrays to Chunks
        if voxel_offset is not None and outputs is not None:
            assert shape is not None
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                for idx, output in enumerate(outputs):
                    outputs[idx] = array_to_chunk(output, voxel_offset, voxel_size, shape)
            elif isinstance(outputs, PointCloud):
                pass
            elif isinstance(outputs, np.ndarray):
                    outputs = array_to_chunk(outputs, voxel_offset, voxel_size, shape)
            elif isinstance(outputs, Chunk):
                # this is good
                pass
            else:
                # will return outputs as is
                pass
       
        return outputs
