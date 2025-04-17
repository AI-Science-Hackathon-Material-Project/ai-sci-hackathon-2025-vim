import json
import pandas as pd
import os
import networkx as nx

def load_data_from_file(filename):
    """
    Load a dictionary of graphs from JSON file.
    """
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict)

def load_data_from_string(json_string):
    """
    Load a dictionary of graphs from JSON string.
    """
    string_dict = json.loads(json_string)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict):
	result_dict = {}
	for key in string_dict:
		graph = nx.node_link_graph(string_dict[key], edges="edges")
		result_dict[key] = graph
	return result_dict

def write_data_to_json_string(graph_dict, **kwargs):
    """
    Write dictionary of graphs to JSON string.
    """
    json_string = json.dumps(graph_dict, default=nx.node_link_data, **kwargs)
    return json_string


def write_data_to_json_file(graph_dict, filename, **kwargs):
    """
    Write dictionary of graphs to JSON file.
    """
    with open(filename, "w") as file_handle:
        file_handle.write(write_data_to_json_string(graph_dict, **kwargs))
         
def load_data_frame_from_file(file_name):
    res = load_data_from_file(file_name) 
    dat = []

    for i, (smile, graph) in enumerate(res.items()):
        molecule_name = "UNASSIGNED" 
        molecule_type = "UNCATEGORIZED"
        
        dat.append({
            'smile': smile,
            'graph': graph,
            'name': molecule_name,
            'type': molecule_type,
            })
        
    df = pd.DataFrame(dat)
    return df

if __name__ == "__main__":
 
    file_path = '/home/ivan/Notes/material_characterization/ai-sci-hackathon-2025/material_characterize_project/graph_data.json'

    test = load_data_frame_from_file(file_path)

    print(test)
    
