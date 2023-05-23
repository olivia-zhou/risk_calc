"""
Main function:
installs requirements, runs program

input: none
output: program output
"""

from model_wrapper import wrapper
import json

config = {}
with open("config.json", 'r') as config_file:
    config = json.loads(config_file.read())
print(config)
model_wrapper = wrapper(**config)
model_wrapper.train()