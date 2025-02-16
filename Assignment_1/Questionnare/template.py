import json
import itertools
import math
from collections import defaultdict


########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################




class Inference:
    def __init__(self, data):
        """
        Initialize the Inference class with the input data.
        
        Parameters:
        -----------
        data : dict
            The input data containing the graphical model details, such as variables, cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques, potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation, and message passing.
        
        Refer to the sample test case for the structure of the input data.
        """
        self.num_vars = data['VariablesCount']
        self.k = data['k value (in top k)']
        self.edges = set()
        self.potentials = {}
        self.graph = defaultdict(set)
        for clique_data in data['Cliques and Potentials']:
            clique = tuple(clique_data['cliques'])
            self.potentials[clique] = clique_data['potentials']
            for u,v in itertools.combinations(clique,2):
                self.graph[u].add(v)
                self.graph[v].add(u)
        self.junction_tree = {}
        self.messages = {}

    def triangulate_and_get_cliques(self):
        """
        Triangulate the undirected graph and extract the maximal cliques.
        
        What to do here:
        ----------------
        - Implement the triangulation algorithm to make the graph chordal.
        - Extract the maximal cliques from the triangulated graph.
        - Store the cliques for later use in junction tree creation.

        Refer to the problem statement for details on triangulation and clique extraction.
        """
        # create a copy of graph to work on
        graph = {node : set(neighbors) for node, neighbors in self.graph.items()}
        fill_edges = set()
        # Triangulation:Minimum degree greedy heuristic
        while graph:
            # pick node with smallest degree
            node = min(graph, key = lambda x: len(graph[x]))  
            neighbors = graph[node]
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[i] not in graph[neighbors[j]]:
                        graph[neighbors[i]].add(neighbors[j])
                        graph[neighbors[j]].add(neighbors[i])
                        fill_edges.add((neighbors[i], neighbors[j]))
            del graph[node]
            for n in neighbors:
                graph[n].remove(node)
        
        self.triangulated_cliques = []
        visited = set()
        for node in self.graph:
            clique = {node} | self.graph[node]
            if tuple(clique) not in visited:
                self.triangulated_cliques.append(clique)
                visited.add(tuple(clique))


    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        adjacency_list = defaultdict(set)
        for i,c1 in enumerate(self.triangulated_cliques):
            for j,c2 in enumerate(self.triangulated_cliques):
                if i != j and c1 & c2:
                    adjacency_list[c1].add(c2)
                    adjacency_list[c2].add(c1)
        self.junction_tree = adjacency_list

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        self.assigned_potentials = {}
        for clique in self.triangulated_cliques:
            if clique in self.potentials:
                self.assigned_potentials[clique] = self.potentials[clique]
            else:
                self.assigned_potentials[clique] = [1] * (2 ** len(clique))
                for existing_clique, potential in self.potentials.items():
                    if set(existing_clique).issubset(set(clique)):
                        self.assigned_potentials[clique] = potential
                        break

    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        What to do here:
        ----------------
        - Implement the message passing algorithm to compute the partition function (Z value).
        - The Z value is the normalization constant for the probability distribution.
        
        Refer to the problem statement for details on computing the partition function.
        """
        
        pass

    def compute_marginals(self):
        """
        Compute the marginal probabilities for all variables in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to compute the marginal probabilities for each variable.
        - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
        Refer to the sample test case for the expected format of the marginals.
        """
        pass

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model.
        
        What to do here:
        ----------------
        - Use the message passing algorithm to find the top-k assignments with the highest probabilities.
        - Return the assignments along with their probabilities in the specified format.
        
        Refer to the sample test case for the expected format of the top-k assignments.
        """
        pass



########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)
    
    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)


if __name__ == '__main__':
    print("Hello")
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    # evaluator.write_output('Sample_Testcase_Output.json') 

    
