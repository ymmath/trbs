# Building a dependency tree with the root node as the Internal Variable (investment in pig farms) and terminating at any key output.
from pathlib import Path
from core.trbs import TheResponsibleBusinessSimulator
import numpy as np
import pandas as pd

class TreeNode:
    def __init__(self, value, category):
        self.value = value
        self.category = category
        self.operator = []
        self.operand = []
        self.child = []

    def __str__(self, depth=0):
        indent = "  " * depth
        result = f"{indent}Value: {self.value}, Category: {self.category}\n"

        if self.operator:
            for i, op in enumerate(self.operator):
                result += f"{indent}Operator {i + 1}: {op}, Operand {i + 1}: {self.operand[i]}\n"

        if self.child:
            result += f"{indent}Child:\n"
            for c in self.child:
                result += c.__str__(depth + 1)

        return result


def build_tree(value):
    category = determine_category(value)
    root = TreeNode(value, category)

    # Base case
    if category == "Key output":
        return root

    root_indices = np.where(input_dict["argument_1"] == value)
    for root_index in np.nditer(root_indices):
        root.operator.append(input_dict["operator"][root_index])
        root.operand.append(input_dict["argument_2"][root_index])

        # Recursively build the child node
        destination = input_dict["destination"][root_index]
        root.child.append(build_tree(destination))

    return root


def determine_category(value):
    if value in input_dict["key_outputs"]:
        return "Key output"
    elif value in input_dict["internal_variable_inputs"]:
        return "Internal input"
    elif value in input_dict["external_variable_inputs"]:
        return "External input"
    elif value in input_dict["fixed_inputs"]:
        return "Fixed input"
    else:
        return "Intermediate"

# Specify your case and format.
path = Path.cwd() / 'data'
file_format = 'xlsx'
name = 'FinalTemplate'

case = TheResponsibleBusinessSimulator(path, file_format, name)
case.build()

input_dict = case.input_dict

# Root nodes are all the decision maker options
root_nodes = input_dict["internal_variable_inputs"]
trees = []

# Build a tree for each of the internal variables
for root_value in root_nodes:
    tree = build_tree(root_value)
    trees.append(tree)

# Example usage to print the tree structure
for tree in trees:
    print(tree)
    print()