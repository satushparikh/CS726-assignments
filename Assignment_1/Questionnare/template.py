import json
import itertools
import math
from collections import defaultdict
import heapq



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
    # def assign_potentials_to_cliques(self):
    #     """
    #     Assign potentials to the cliques in the junction tree.
        
    #     - Map the given potentials from the input data to the corresponding cliques in the junction tree.
    #     - If multiple factors belong to the same clique, multiply them element-wise.
    #     - If a clique has no assigned factor, initialize its potential to [1] (neutral element for multiplication).
    #     """
    #     # Initialize clique potentials
    #     self.clique_potentials = {}
        
    #     for clique in self.junction_tree:
    #         # Start with a neutral potential of [1] (list form for element-wise multiplication)
    #         potential = [1]
            
    #         # Multiply all potentials assigned to this clique (element-wise)
    #         for assigned_clique, assigned_potential in self.potentials.items():
    #             if set(assigned_clique).issubset(clique):
    #                 if len(potential) == 1:
    #                     potential = assigned_potential  # Initialize from first potential
    #                 else:
    #                     potential = [a * b for a, b in zip(potential, assigned_potential)]
            
    #         self.clique_potentials[clique] = potential

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
            # convert set to list for indexing
            neighbors = list(graph[node])
            
            # Connect neighbors to form a clique
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u,v = neighbors[i], neighbors[j]
                    if v not in graph[u]:
                        graph[u].add(v)
                        graph[v].add(u)
                        fill_edges.add((u, v))
            # Remove the node from the graph
            del graph[node]
            for n in neighbors:
                graph[n].remove(node)
        #  Construct triangulated graph = original graph + fill_edges
        triangulated_graph = {node: set(neighbors) for node, neighbors in self.graph.items()}
        for u,v in fill_edges: 
            triangulated_graph[u].add(v)
            triangulated_graph[v].add(u)
        
        def bron_kerbosch(R, P, X, cliques):
            # If there are no more candidates and no excluded nodes,     R is maximal.
            if not P and not X:
                cliques.append(R)
                return
            # Iterate over a static list of candidates (P) to allow     modifications during iteration.
            for v in list(P):
                # Recurse with v added to R, and restrict P and X to neighbors of v.
                bron_kerbosch(R | {v}, P & triangulated_graph[v],     X & triangulated_graph[v], cliques)
            # Move v from the candidate set P to the excluded set X.
                P.remove(v)
                X.add(v)     
                
        cliques = []
        # Start with an empty clique (R), all nodes as potential (P), and no excluded nodes (X)
        bron_kerbosch(set(), set(triangulated_graph.keys()), set(), cliques)   
        self.triangulated_cliques = cliques

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
        edges = []
        #  Sets are mutable and thus not hashable and cannot be used as dictionary key. So converting to frozensets
        self.triangulated_cliques = [frozenset(c) for c in self.triangulated_cliques]
        # creating weighted clique-graph where edge weights are size of separator sets
        for i, c1 in enumerate(self.triangulated_cliques):
            for j in range(i+1, len(self.triangulated_cliques)):
                c2 = self.triangulated_cliques[j]
                separator = set(c1) & set(c2)
                if separator:
                    weight = len(separator)
                    edges.append((weight, c1, c2))
                    adjacency_list[c1].add((weight, c2))
                    adjacency_list[c2].add((weight, c1))
        
        self.junction_tree = defaultdict(set)
        if not edges:
            raise  ValueError("Edges list coming out empty.")
        # creating a minimum weight spanning tree using prim's method for clique graph
        start_clique = self.triangulated_cliques[0]
        pq = [(-weight, start_clique, neighbor) for weight, neighbor in adjacency_list[start_clique]]
        heapq.heapify(pq)
        visited = {start_clique}
        # loop runs until all nodes are added to junction tree
        while pq:
            weight, c1, c2 = heapq.heappop(pq)
            if c2 not in visited:
                visited.add(c2)
                self.junction_tree[c1].add(c2)
                self.junction_tree[c2].add(c1)
                for w, neighbor in adjacency_list[c2]:
                    if neighbor not in visited:
                        heapq.heappush(pq, (-w, c2, neighbor))
        self.check_running_intersection_property()
        
    def check_running_intersection_property(self):
        """
        Check that the junction tree satisfies the Running Intersection Property (RIP):
        For every variable, the cliques containing that variable should form a connected subgraph.
        """
        from collections import defaultdict, deque

        # Build a mapping: variable -> set of cliques that contain the variable.
        variable_clique_map = defaultdict(set)
        # Iterate over cliques in the junction tree (keys of the junction_tree dictionary)
        for clique in self.junction_tree.keys():
            for var in clique:
                variable_clique_map[var].add(clique)
        
        # For each variable, check if the induced subgraph is connected.
        for var, cliques in variable_clique_map.items():
            # If there's only one clique, it's trivially connected.
            if len(cliques) <= 1:
                continue

            # Start BFS from an arbitrary clique that contains the variable.
            start = next(iter(cliques))
            visited = set()
            queue = deque([start])
            
            while queue:
                current = queue.popleft()
                visited.add(current)
                # Only traverse neighbors that also contain var.
                for neighbor in self.junction_tree[current]:
                    if neighbor in cliques and neighbor not in visited:
                        queue.append(neighbor)
            
            if visited != cliques:
                raise ValueError(f"Running Intersection Property violated for variable: {var}")
            
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
    evaluator = Get_Input_and_Check_Output('Assignment_1\Questionnare\Sample_Testcase.json')
    # evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    # evaluator.write_output('Sample_Testcase_Output.json') 

    
