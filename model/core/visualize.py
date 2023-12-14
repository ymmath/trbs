"""
This file contains the Visualize class that deals with the creation of all graphs and tables
"""
import re
import pandas as pd
import numpy as np
from random import choices
from collections import defaultdict
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from core.utils import round_all_dict_values, number_formatter, get_values_from_target, check_list_content
from difflib import get_close_matches


class VisualizationError(Exception):
    """
    This class deals with the error handling of the Visualize class
    """

    def __init__(self, message):  # ignore warning about super-init | pylint: disable=W0231
        self.message = message

    def __str__(self):
        return f"Visualization Error: {self.message}"


class Visualize:
    """This class deals with the creation of all graphs and tables"""

    def __init__(self, input_dict, outcomes, options):
        # for visualization purposes two digits is sufficient
        self.input_dict = input_dict
        self.outcomes = round_all_dict_values(outcomes)
        self.options = options
        self.colors = ["#D04A02", "#EB8C00", "#FFB600", "#295477", "#299D8F"]
        self.available_visuals = {
            "table": self._create_table,
            "barchart": self._create_barchart,
            "network": self._create_network
        }
        self.available_outputs = [
            "key_outputs",
            "appreciations",
            "dependencies",
            "weighted_appreciations",
            "decision_makers_option_appreciation",
        ]
        self.available_kwargs = [
            "scenario",
            "decision_makers_option",
            "stacked",
            "show_legend",
            "case",
            "node"
        ]

    def _validate_kwargs(self, **kwargs) -> None:
        """
        This function validates whether the provided additional arguments are relevant.
        :param **kwargs: the additional arguments provided
        :return: None if all **kwargs are available, else raise an VisualizationError.
        """
        for argument, _ in kwargs.items():
            if argument not in self.available_kwargs:
                raise VisualizationError(f"Invalid argument '{argument}'")

    def _find_dimension_level(self, my_dict: dict, target_key: str, level: int = 1) -> int or None:
        """
        This recursive function returns the dimension level (level of nesting) for a given dictionary and target key.
        For example in dictionary {A: {B: {C: 1.23, ..}, ..}, ..}. 'A' is nested at level 1, B at level 2 & C level 3.
        :param my_dict: input dictionary
        :param target_key: name of the key
        :param level: current dimension level
        :return: the dimension level (if target_key exists)
        """
        if target_key in my_dict:
            return level

        for _, value in my_dict.items():
            if isinstance(value, dict):
                found_level = self._find_dimension_level(value, target_key, level + 1)
                if found_level is not None:
                    return found_level
        return None

    @staticmethod
    def _str_snake_case_to_text(snake_case_str: str) -> str:
        """
        This styling function returns a formatted string where the snake case is replaced by spaces.
        :param snake_case_str: string to reformat
        :result: string where _ is replaced by spaces
        """
        words = snake_case_str.split("_")
        return " ".join(words)

    @staticmethod
    def _truncate_title_list(title_list: list) -> list:
        """
        This function truncates a list of strings to ensure that in total they have no more than 85 chars. For example,
        two strings could have 42 chars whereas, 10 strings can only use 8 chars.
        :param title_list: list of title names to be truncated
        :return: a truncated (where necessary) list of titles
        """
        max_char_length = int(85 / len(title_list))
        truncated_list = [
            f"{item[:(max_char_length - 2)]}.." if len(item) > max_char_length else item for item in title_list
        ]
        return truncated_list

    @staticmethod
    def _table_styler(styler: pd.DataFrame.style, table_name: str) -> pd.DataFrame.style:
        """
        This function adds a coherent style for all generated tables
        :param styler: a Pandas styler object
        :param table_name: name of the styled table
        :return: an updated styler object
        """
        # set color palette
        cmap = mpl.cm.Blues(np.linspace(0, 1, 30))
        cmap = mpl.colors.ListedColormap(cmap[:8, :-1])

        # style
        styler.format(number_formatter)
        styler.set_caption(table_name)
        styler.background_gradient(cmap=cmap, axis=1)
        return styler

    def _graph_styler(self, axis: mpl.axis, title: str, show_legend: bool) -> mpl.axis:
        """
        This function adds a coherent style for all generated graphs.
        :param axis: a matplotlib axis object containing the graph
        :param title: title of the graph
        :return: formatted matplotlib axis object
        """
        xtick_formatted = self._truncate_title_list([label.get_text() for label in axis.get_xticklabels()])
        axis.set_title(title, color="#777777", fontsize=12)
        axis.set_ylim(0, 100)
        axis.set_xticklabels(xtick_formatted, rotation=0, fontweight="bold")
        axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
        if not show_legend:
            axis.legend_ = None
        return axis

    def _format_data_for_visual(self, key_data: str) -> pd.DataFrame:
        """
        This function formats the values of the given key into a dataframe.
        :param key_data: name of the key that needs formatting
        :return: a pd.DataFrame with decision maker options as row index, key outputs as columns and weighted
        appreciations as values.
        """
        dim_level = self._find_dimension_level(self.outcomes, key_data)
        dim_names = ["scenario", "decision_makers_option"]

        # iterate until we are at the level where the 'key_data' can be found
        dict_for_iteration = self.outcomes.copy()
        formatted_data = pd.DataFrame(index=range(self.options))
        while dim_level > 1:
            dim_values = []
            tmp_dict = {}
            for key, value in dict_for_iteration.items():
                dim_values.append(key)
                # ensure keys are unique, so we don't lose any values
                tmp_dict.update({f"@{key}@{tmp_key}": tmp_value for tmp_key, tmp_value in value.items()})

            # for higher dimensions entries will be duplicated in the columns
            replicate_n = int(self.options / len(dim_values))
            # using a re(gex) expression the entries to ensure uniqueness in the for-loop are removed again.
            formatted_data[dim_names.pop(0)] = [
                re.sub(r"@.*?@", "", value) for value in dim_values for _ in range(replicate_n)
            ]
            dict_for_iteration = tmp_dict
            dim_level -= 1

        # add the values from 'key_data' | this can be either a list of dictionaries or a list of values
        key_data_list = get_values_from_target(self.outcomes, key_data)
        key_data_list_content = check_list_content(key_data_list)

        if key_data_list_content == "numeric":
            # if the final dimension is numeric we initialised too many rows
            formatted_data = formatted_data.drop_duplicates(ignore_index=True)
            formatted_data["value"] = key_data_list
        elif key_data_list_content == "dictionaries":
            formatted_data[key_data] = [key for dictionary in key_data_list for key, _ in dictionary.items()]
            formatted_data["value"] = [value for dictionary in key_data_list for _, value in dictionary.items()]
        return formatted_data

    @staticmethod
    def _apply_filters(dataframe: pd.DataFrame, drop_used: bool = False, **kwargs) -> pd.DataFrame and str:
        """
        This function applies filters, based on **kwargs arguments, on the dataframe and generates a corresponding
        name for the visual.
        :param dataframe: dataframe that needs filtering
        :param drop_used: indicator whether the used filter column should be dropped
        :return: a filtered dataframe with corresponding name
        """
        name_str = ""
        for arg, value in kwargs.items():
            if arg not in dataframe.columns:
                continue
            dataframe = dataframe[dataframe[arg] == value]
            if drop_used:
                dataframe = dataframe.drop(columns=[arg], axis=1)
            name_str += f" | {value}"

        # Check if dataframe is empty after applying filters
        if dataframe.empty:
            raise VisualizationError("No data for given selection. Are your arguments correct?")

        return dataframe, name_str

    def _create_table(self, key: str, **kwargs) -> pd.DataFrame.style:
        """
        This function creates a 2- or 3-dimensional table depending on the key.
        :param key: key of the values for the table
        :return: a styled table
        """
        table_data = self._format_data_for_visual(key)

        # Filter the data based on potentially provided arguments by the user.
        table_data, name_str = self._apply_filters(table_data, **kwargs)
        table_data = (
            table_data.set_index(["scenario", key])
            .pivot(columns="decision_makers_option", values="value")
            .rename_axis((None, None))
            .rename_axis(None, axis=1)
        )
        table_name = f"Values of {self._str_snake_case_to_text(key)}{name_str}"
        return self._table_styler(table_data.style, table_name)

    def _create_barchart(self, key: str, **kwargs) -> None:
        """
        This function creates and shows a barchart for a given data key.
        :param key: name of values of interest
        :return: a plotted barchart
        """
        dims = self._find_dimension_level(self.outcomes, key)
        if dims > 2 and "scenario" not in kwargs:
            raise VisualizationError(f"Too many dimensions ({dims}). Please specify a scenario")
        stacked = kwargs["stacked"] if "stacked" in kwargs else True
        show_legend = kwargs["show_legend"] if "show_legend" in kwargs else True

        appreciations = self._format_data_for_visual(key)
        bar_data, name_str = self._apply_filters(appreciations, drop_used=True, **kwargs)
        rest_cols = [col for col in bar_data.columns if col not in ["decision_makers_option", "value"]]
        bar_data = bar_data.pivot(index="decision_makers_option", columns=rest_cols, values="value").reset_index()
        axis = bar_data.plot.bar(x="decision_makers_option", stacked=stacked, color=self.colors, figsize=(10, 5))
        self._graph_styler(axis, f"Values of {self._str_snake_case_to_text(key)}{name_str}", show_legend)

        plt.show()

    def _create_legend_for_network(self):
        legend_dict = {
            'Fixed input': 'yellow',
            'Internal input': 'turquoise',
            'External input': 'orange',
            'Output': 'green',
            'Intermediate': 'grey'
        }
        handles = []
        labels = []
        for category, color in legend_dict.items():
            handles.append(plt.Line2D([0], [0], color=color, marker='o', markersize=10, linewidth=0))
            labels.append(f'{category}')
        return handles, labels

    def _determine_category_for_node(self, node):
        if node in self.input_dict["key_outputs"]:
            return "key_output"
        elif node in self.input_dict["internal_variable_inputs"]:
            return "internal_input"
        elif node in self.input_dict["external_variable_inputs"]:
            return "external_input"
        elif node in self.input_dict["fixed_inputs"]:
            return "fixed_input"
        else:
            return "intermediate"

    # Functions below this point are related to network graph
    def _create_network(self, key: str, **kwargs):
        """
        This function creates a network graph for a given graph type. Default graph type
        is the 'dependency-tree' for which we have the sub-graph option to view the dependency tree
        for only the network associated with the given node.
        :param subgraph: boolean to display a sub-graph of the network associated with the provided node
        """
        # TODO: If subgraph and no node provided, throw error
        # TODO: Handle nodes that are missing in the `dependencies` sheet
        # TODO: Questions for Thom-Ivar:
        #  How to access the input_dict?
        #  Discuss function signature
        #  Check for other cases in the open source version regarding the numeric nodes
        print("COWABUNGA!")

        if 'case' not in kwargs:
            print('Please specify the name of the case. Otherwise we cannot label graph.')
        else:
            case_name = kwargs.get('case')

            # Create the graph structure object
            G = nx.DiGraph()

            # Restructure the data into a pandas DataFrame
            data = pd.DataFrame({
                'argument_1': self.input_dict['argument_1'],
                'argument_2': self.input_dict['argument_2'],
                'operator': self.input_dict['operator'],
                'destination': self.input_dict['destination']
            })

            data['dep_color'] = self._determine_edge_color_for_network(data)

            # Iterate over the data and add nodes/edge to nx graph
            for index, row in data.iterrows():
                destination = row['destination']
                argument_1 = row['argument_1']
                argument_2 = row['argument_2']
                operator = row['operator']
                # row_level = row['hierarchy']
                edge_color = row['dep_color']

                # Add nodes for destination
                G.add_node(destination, color=self._determine_category_color_for_network(destination))

                # TODO: For cases with an integer in the arguments
                if pd.isna(argument_1):
                    G.add_node(argument_2, color=self._determine_category_color_for_network(argument_2))
                    G.add_edge(argument_2, destination, label='squeezed', color=edge_color, weight=1)
                elif pd.isna(argument_2):
                    G.add_node(argument_1, color=self._determine_category_color_for_network(argument_1))
                    G.add_edge(argument_1, destination, label='squeezed', color=edge_color, weight=1)
                else:
                    G.add_node(argument_1, color=self._determine_category_color_for_network(argument_1))
                    G.add_node(argument_2, color=self._determine_category_color_for_network(argument_2))
                    G.add_edge(argument_1, destination, label=self._label_operators(operator, 'arg1'), color=edge_color,
                               weight=3)
                    G.add_edge(argument_2, destination, label=self._label_operators(operator, 'arg2'), color=edge_color,
                               weight=3)

            # Determine positions of each of the nodes
            pos = self._determine_node_positions(G)

            # Plot the graph
            edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
            edge_colors = [G[u][v]['color'] for u, v in G.edges()]
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            node_colors = [G.nodes[node]["color"] for node in G.nodes()]

            if 'node' in kwargs:
                self.name_of_node = kwargs.get('node')
                if self.name_of_node == '':
                    print('Please enter the name of the node you wish to see!')
                else:
                    print(self.name_of_node)
                    self._select_subset(G, case_name + '_' + self.name_of_node, data, self.name_of_node)
            else:
                self._draw_network(G, case_name, pos, edge_labels, edge_colors, edge_weights, node_colors, False)
                return edge_labels, edge_colors, edge_weights, node_colors

    def _select_subset(self, graph, case, data, name_of_node):
        # Match to the closest node
        name_of_node = "".join(get_close_matches(word=name_of_node, possibilities=list(graph), n=1))
        # TODO: Assert name_of_node is not empty
        nodes_subset = list(nx.ancestors(graph, name_of_node))
        nodes_subset.append(name_of_node)
        nodes_subset += nx.descendants(graph, name_of_node)
        #     print(nodes_subset)
        #     print(nx.ancestors(G, name_of_node))
        #     print(nx.descendants(G, name_of_node))
        # nx.draw(ancestors, pos, node_color=node_colors, with_labels=False, node_size=1200, font_size=10)
        self._make_subgraph(case, data, nodes_subset)

    def _make_subgraph(self, case, data, nodes):
        tuples = []

        for index, row in data.iterrows():
            destination = row['destination']
            argument_1 = row['argument_1']
            argument_2 = row['argument_2']
            operator = row['operator']
            # row_level = row['hierarchy']
            edge_color = row['dep_color']

            if ((argument_1 in nodes) or (argument_2 in nodes)) and (destination in nodes):
                t = (argument_1, argument_2, destination, operator, edge_color)
                tuples.append(t)

        # Make the graph and add the nodes
        G = nx.DiGraph()

        for t in tuples:
            G.add_node(t[2], color=self._determine_category_color_for_network(t[2]))

            # Add edges from argument nodes to the destination node with levels
            if pd.isna(t[0]):
                G.add_node(t[1], color=self._determine_category_color_for_network(t[1]))
                G.add_edge(t[1], t[2], label='squeezed', color='black', weight=1)
            elif pd.isna(t[1]):
                G.add_node(t[0], color=self._determine_category_color_for_network(t[0]))
                G.add_edge(t[0], t[2], label='squeezed', color='black', weight=1)
            else:
                G.add_node(t[0], color=self._determine_category_color_for_network(t[0]))
                G.add_node(t[1], color=self._determine_category_color_for_network(t[1]))
                G.add_edge(t[0], t[2], label=self._label_operators(t[3], "arg1"), color=t[4], weight=3)
                G.add_edge(t[1], t[2], label=self._label_operators(t[3], "arg2"), color=t[4], weight=3)

        # Determine positions of each of the nodes
        pos = self._determine_node_positions(G)

        # Plot the graph
        edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        node_colors = [G.nodes[node]["color"] for node in G.nodes()]

        self._draw_network(G, case, pos, edge_labels, edge_colors, edge_weights, node_colors, True)
        return edge_labels, edge_colors, edge_weights, node_colors

    def _draw_network(self, graph, case, pos, edge_labels, edge_colors, edge_weights, node_colors, is_subgraph):
        # This is a function to visualize the plot
        if is_subgraph:
            plt.figure(figsize=(18, 24))
        else:
            plt.figure(figsize=(45, 60))

        # with_labels=False because of rotating the node labels
        nx.draw(graph, pos,
                node_color=node_colors,
                edge_color=edge_colors,
                width=edge_weights,
                with_labels=False,
                node_size=2500,
                font_size=10)
        text = nx.draw_networkx_labels(graph, pos, font_weight='bold')

        # Rotate the node labels
        for _, t in text.items():
            t.set_rotation(45)

        nx.draw_networkx_edge_labels(graph, pos,
                                     edge_labels=edge_labels,
                                     label_pos=0.5,
                                     font_color='black',
                                     font_size=18,
                                     font_weight='bold')

        # Create and add legend
        legend_handles, legend_labels = self._create_legend_for_network()

        plt.title("Hierarchical Graph Based on Node Levels")
        plt.legend(legend_handles, legend_labels, loc='upper right')
        plt.axis('off')  # Turn off axis for cleaner display
        plt.savefig('C:/Users/achatterje195/Documents/trbs/model/plots/' + case + '.png')
        plt.show()

    def create_visual(self, visual_request: str, key: str, **kwargs):
        """
        This function redirects the visual_request based on the requested format to the correct helper function.
        Validation checks are performed on the available visuals and outputs.See 'available_outputs' and
        'available_kwargs' for all possible options.
        :param visual_request: type of visual that is requested
        :param key: key in output_dict containing the values to be visualised
        :param **kwargs: any additional arguments provide by the user
        :return: requested visual
        """
        self._validate_kwargs(**kwargs)
        if visual_request not in self.available_visuals:
            raise VisualizationError(f"'{visual_request}' is not a valid chart type")
        if key not in self.available_outputs:
            raise VisualizationError(f"'{key}' is not a valid option")
        return self.available_visuals[visual_request](key, **kwargs)

    def _determine_category_color_for_network(self, node):
        category_is = self._determine_category_for_node(node)
        if category_is == 'fixed_input':
            return 'yellow'
        elif category_is == 'external_input':
            return 'orange'
        elif category_is == 'internal_input':
            return 'turquoise'
        elif category_is == 'key_output':
            return 'green'
        else:
            # intermediaries and other unaccounted categories
            return 'grey'

    def _determine_node_positions(self, G):
        # Create a dictionary to store nodes at each level
        nodes_by_level = defaultdict(list)
        for i, node_list in enumerate(nx.topological_generations(G)):
            for node in node_list:
                level = i
                nodes_by_level[level].append(node)

        x_spacing = 1.0  # Adjust this value for spacing between levels
        y_spacing = 1.0  # Adjust this value for vertical spacing between nodes at the same level
        pos = {}
        for level, nodes in nodes_by_level.items():
            num_nodes = len(nodes)
            x_value = level * x_spacing
            y_values = [(i * (level + 2) - (num_nodes - 1) / 2) * y_spacing for i in range(num_nodes)]
            pos.update((node, (x_value, y_value)) for node, y_value in zip(nodes, y_values))
        return pos

    def _determine_edge_color_for_network(self, data):
        number_of_colors = len(data)
        return choices(self.colors, k=number_of_colors)

    def _label_operators(self, operator, arg_type):
        label_map = {
            '/': {'arg1': '/_arg1', 'arg2': '/_arg2'},
            '-': {'arg1': '-_arg1', 'arg2': '-_arg2'},
        }
        return label_map.get(operator, {}).get(arg_type, operator)
