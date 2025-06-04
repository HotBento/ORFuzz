import os
from argparse import ArgumentParser
import yaml
import json

from dataclasses import dataclass, field
from typing import Optional, Tuple

def add_arg(parser:ArgumentParser, *args, **kwargs):
    if args[0] not in parser._option_string_actions:
        parser.add_argument(*args, **kwargs)
    return parser

@dataclass
class BaseConfig():
    log_path:str            = "log"
    debug:bool              = False
    hf_token:str              = ""
    def update(self, args:dict):
        for key, value in args.items():
            if hasattr(self, key) and value != None and not callable(getattr(self, key)):
                setattr(self, key, value)
    
    def save_yaml(self, path:str):
        with open(path, "w") as f:
            yaml.dump(vars(self), f)

    @classmethod
    def get_config(cls, args:Optional[dict]=None):
        config = cls()
        if args == None:
            args = config.parse_args()
        config.update(config._from_yaml(args.get("config_path", None)))
        config.update(args)
        return config
    
    def parse_args(self):
        parser = ArgumentParser()
        
        parser = self.add_args(parser)
        
        args = parser.parse_args()
        return vars(args)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser)->ArgumentParser:
        parser = add_arg(parser, "--config_path", type=str, default=None)
        parser = add_arg(parser, "--log_path", type=str, default=None)
        parser = add_arg(parser, "--debug", action="store_true")
        parser = add_arg(parser, "--hf_token", type=str, default=None)
        
        return parser
    
    def _from_yaml(self, path:str)->dict:
        if path == None:
            return dict()
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @classmethod
    def from_yaml(cls, path:str):
        config = cls()
        config.update(config._from_yaml(path))
        return config
    
    @classmethod
    def from_json(cls, path:str):
        config = cls()
        with open(path, "r") as f:
            config.update(json.load(f))
        return config

@dataclass
class FullGenConfig(BaseConfig):
    dataset_path:str    = "dataset/full.csv"
    gen_model:str       = "qwen-ds"
    eval_model:str      = "model_finetuning/lora/qwen"
    batch_size:int      = 10
    result_path:str     = "result/result_gendata"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    n_memory:int        = 1
    stream:bool         = False
    server:str          = ""
    model_list:list[str] = field(default_factory=list)
    modified_path:str   = "result/result_tuning/llama3-0"
    finetuning:bool     = False
    if_modify:bool      = False
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--batch_size", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--model_list", nargs="+", type=str, default=["llama3", "gemma", "mistral", "phi", "qwen"])
        parser = add_arg(parser, "--modified_path", type=str, default=None)
        parser = add_arg(parser, "--finetuning", action="store_true")
        parser = add_arg(parser, "--if_modify", action="store_true")
        
        return parser

class ServerConfig(BaseConfig):
    host:str            = "127.0.0.1"
    port:int            = 4000
    model_path:str      = "model_finetuning/lora/qwen"
    
    @classmethod
    def add_args(cls, parser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--host", type=str, default=None)
        parser = add_arg(parser, "--port", type=int, default=None)
        parser = add_arg(parser, "--model_path", type=str, default=None)
        
        return parser

@dataclass
class ORFuzzConfig(BaseConfig):
    dataset_path:str    = "dataset/full.csv"
    sample_num:int      = 3
    gen_model:str       = "deepseek"
    eval_model:str      = "model_finetuning/lora/qwen"
    total_round:int     = 50
    result_path:str     = "result/result_gendata"
    is_peft:bool        = False
    base_model:str      = "Qwen/Qwen2.5-14B-Instruct"
    th:float            = 0.5
    div_rate:float      = 0.5
    n_memory:int        = 5
    mutator_manager:str = "ucb"
    selection_type:str  = "mcts_explore"
    reconstruction:bool = False
    num_seed:int        = 5
    n_mutator:int       = 3
    refiner:str         = "CoT"
    stream:bool         = False
    server:str          = ""
    model_list:list[str] = field(default_factory=list)
    
    @classmethod
    def add_args(cls, parser:ArgumentParser):
        parser = super().add_args(parser)
        parser = add_arg(parser, "--dataset_path", type=str, default=None)
        parser = add_arg(parser, "--sample_num", type=int, default=None)
        parser = add_arg(parser, "--gen_model", type=str, default=None)
        parser = add_arg(parser, "--eval_model", type=str, default=None)
        parser = add_arg(parser, "--total_round", type=int, default=None)
        parser = add_arg(parser, "--result_path", type=str, default=None)
        parser = add_arg(parser, "--is_peft", action="store_true")
        parser = add_arg(parser, "--base_model", type=str, default=None)
        parser = add_arg(parser, "--th", type=float, default=None)
        parser = add_arg(parser, "--div_rate", type=float, default=None)
        parser = add_arg(parser, "--n_memory", type=int, default=None)
        parser = add_arg(parser, "--mutator_manager", type=str, default=None)
        parser = add_arg(parser, "--selection_type", type=str, default=None)
        parser = add_arg(parser, "--reconstruction", action="store_true")
        parser = add_arg(parser, "--refiner", type=str, default=None)
        parser = add_arg(parser, "--stream", action="store_true")
        parser = add_arg(parser, "--server", type=str, default=None)
        parser = add_arg(parser, "--model_list", nargs="+", type=str, default=["llama3", "gemma", "mistral", "phi", "qwen"])
        
        return parser
