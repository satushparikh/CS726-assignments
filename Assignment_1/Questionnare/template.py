import json
import itertools
import math
from collections import defaultdict
import heapq
import os


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
            clique_potential = clique_data['potentials']
            if clique in self.potentials :
                # Element wise multiplication of previously stored and current 'clique_potnetial'
                self.potentials[clique] = [
                    x*y for x,y in zip(self.potentials[clique],clique_potential)
                ]
            else:
                self.potentials[clique] = clique_potential
                
            # Update the graph edges based on cliques stored
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
        if len(self.triangulated_cliques) == 1:
            self.junction_tree[self.triangulated_cliques[0]] = set()
            return

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
        # creating a Maximum weight spanning tree using prim's method for clique graph
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
        """
        For each triangulated clique, build a clique potential by iterating over all
        2**n binary assignments (n = number of nodes in the clique). For each clique,
        multiply in the contribution from each factor (from self.potentials) that is
        completely contained in the clique and that has not yet contributed to any other clique.
        
        The resulting clique potential is stored as a pair:
        (clique_vars, potential_table)
        where:
        - clique_vars is a tuple (in sorted order) of the clique’s nodes, and
        - potential_table is a dictionary mapping each assignment (a tuple of 0's and 1's)
            to its computed potential value.
        """
        # 1. Initialize a set to keep track of factors that have already been assigned.
        assigned_factors = set()  # We store each factor as a canonical (sorted) tuple.
        
        # 2. Prepare a dictionary to hold the final potential for each clique.
        self.clique_potentials = {}  # Each key will be a clique (e.g., a frozenset), mapping to its (clique_vars, potential_table)
        
        # 3. Iterate over each triangulated clique.
        #    Assume self.triangulated_cliques is a list of cliques, each represented as a set of variables.
        for clique in self.triangulated_cliques:
            # a. Convert the clique into a canonical, sorted tuple.
            clique_vars = tuple(sorted(clique))
            n = len(clique_vars)  # Number of variables (nodes) in the clique.
            
            # b. Initialize the clique's potential table.
            #    Since each variable is binary, there are 2**n possible assignments.
            potential_table = {}
            for assignment in itertools.product(range(2), repeat=n):
                potential_table[assignment] = 1  # Start with 1 (multiplicative identity).
            
            # c. For each factor in the provided potentials...
            for factor, flat_list in self.potentials.items():
                # Convert the factor to its canonical (sorted) form.
                canonical_factor = tuple(sorted(factor))
                # Check if this factor's variables are all in the current clique and it hasn't been used yet.
                if set(factor).issubset(clique) and canonical_factor not in assigned_factors:
                    # Mark this factor as assigned so it won't be used in another clique.
                    assigned_factors.add(canonical_factor)
                    
                    # Convert the flat list of potential values into a dictionary keyed by assignments.
                    factor_table = self.factor_from_list(canonical_factor, flat_list)
                    
                    # Determine, for each variable in the factor, its index within the clique_vars.
                    indices = [clique_vars.index(var) for var in canonical_factor]
                    
                    # d. For every assignment in the clique's table, multiply in the factor's contribution.
                    for clique_assignment in potential_table:
                        # Extract the sub-assignment corresponding to the factor's variables.
                        sub_assignment = tuple(clique_assignment[i] for i in indices)
                        # Multiply the factor's value into the current clique potential.
                        potential_table[clique_assignment] *= factor_table[sub_assignment]
            
            # e. Save the computed potential for this clique.
            self.clique_potentials[clique] = (clique_vars, potential_table)
            
        unassigned_factors = set(self.potentials.keys()) - assigned_factors
        if unassigned_factors:
            raise ValueError(f"Some factors were not assigned to any clique: {unassigned_factors}")

    def factor_from_list(self, factor_vars, flat_list):
        """
        Convert a flat list representation of a factor into a dictionary mapping assignments to values.
        
        Parameters:
        - factor_vars: A tuple (in canonical order) of variables in the factor.
        - flat_list: A flat list of potential values (assumed to be in lexicographic order
                    over all assignments to factor_vars, with each variable binary).
        
        Returns:
        A dictionary mapping each assignment (a tuple of 0's and 1's) to its potential value.
        
        For example, if factor_vars is (1,2) then there will be 2**2 = 4 entries.
        """
        m = len(factor_vars)
        # Create all possible binary assignments for m variables.
        assignments = list(itertools.product(range(2), repeat=m))
        if len(assignments) != len(flat_list):
            raise ValueError("Number of assignments does not match length of potential list.")
        return dict(zip(assignments, flat_list))
    def get_z_value(self):
        """
        Compute the partition function (Z value) of the graphical model.
        
        This method uses message passing (the sum-product algorithm) on the junction tree.
        It selects an arbitrary root clique and passes messages upward (from the leaves to the root).
        The final belief at the root (its potential multiplied by all incoming messages) is summed
        over all assignments to obtain Z.
        
        Assumes that:
          - self.junction_tree is a dictionary mapping each clique (a frozenset) to its neighbor cliques.
          - self.clique_potentials is a dictionary mapping each clique (a frozenset) to a tuple:
                (clique_vars, potential_table)
            where:
                * clique_vars is a tuple (in sorted order) listing the clique's variables.
                * potential_table is a dictionary mapping assignments (tuples of 0's/1's of length len(clique_vars))
                  to their potential value.
        """
        
        # This dictionary will hold messages between cliques.
        # A message from clique A to clique B is stored as messages[(A, B)]
        messages = {}
        
        # Helper: given an ordering (clique_vars) and a subset of variables,
        # return the list of indices (in order) at which those variables appear.Used to return nodes of factors which are subset of cliques
        def get_indices(clique_vars, subset):
            return [i for i, var in enumerate(clique_vars) if var in subset]
        
        # Helper: given an assignment (tuple) and a list of indices,
        # return the sub-assignment (tuple) corresponding to those indices.
        def project_assignment(assignment, indices):
            return tuple(assignment[i] for i in indices)
        
        # Compute the message from clique 'from_clique' to clique 'to_clique'
        def send_message(from_clique, to_clique):
            # Get the potential for the sending clique.
            clique_vars, pot_table = self.clique_potentials[from_clique]
            
            # Start with a copy of the original potential.
            product_factor = {}
            for assignment, value in pot_table.items():
                product_factor[assignment] = value
            
            # Multiply in all incoming messages to 'from_clique' (except from 'to_clique').
            for neighbor in self.junction_tree[from_clique]:
                if neighbor == to_clique:
                    continue
                # Expect that a message from neighbor to from_clique has already been computed.
                if (neighbor, from_clique) in messages:
                    msg = messages[(neighbor, from_clique)]
                    # The message is defined on the separator S = from_clique ∩ neighbor.
                    separator = from_clique & neighbor
                    indices = get_indices(clique_vars, separator)
                    # For every assignment of the full clique, multiply by the appropriate message value.
                    for assignment in product_factor:
                        proj = project_assignment(assignment, indices)
                        product_factor[assignment] *= msg[proj]
            
            # Now marginalize out the variables that are in 'from_clique' but not in the separator S = from_clique ∩ to_clique.
            separator = from_clique & to_clique
            indices_to_keep = get_indices(clique_vars, separator)
            msg_result = {}
            for assignment, value in product_factor.items():
                proj = project_assignment(assignment, indices_to_keep)
                if proj in msg_result:
                    msg_result[proj] += value
                else:
                    msg_result[proj] = value
            return msg_result
        
        # Upward pass: recursively pass messages from leaves to the root.
        def upward_pass(clique, parent=None):
            for neighbor in self.junction_tree[clique]:
                if neighbor == parent:
                    continue
                upward_pass(neighbor, clique)
                # By calling upward_pass recursively(DFS way) w ensure that every leaf node(which has no children) is processed first. After processing a neighbor's subtree, we compute a message from that neighbor(child) to the current clique(parent) using send_message function
                messages[(neighbor, clique)] = send_message(neighbor, clique)
        def backward_pass(clique,parent=None):
            for neighbor in self.junction_tree[clique]:
                if neighbor == parent: 
                    continue 
                messages[(clique,neighbor)] = send_message(clique,neighbor)
                backward_pass(neighbor,clique)
        # Select an arbitrary clique as the root.
        root = next(iter(self.junction_tree))
        upward_pass(root)
        backward_pass(root)
        
        # At the root, compute the final belief by multiplying the clique's own potential with all incoming messages.
        root_vars, root_pot = self.clique_potentials[root]
        belief = {}
        for assignment, value in root_pot.items():
            belief[assignment] = value
        for neighbor in self.junction_tree[root]:
            if (neighbor, root) in messages:
                msg = messages[(neighbor, root)]
                separator = root & neighbor  # Separator between root and this neighbor.
                indices = get_indices(root_vars, separator)
                # Multiply the message into the belief for every assignment of the root clique.
                for assignment in list(belief.keys()):
                    proj = project_assignment(assignment, indices)
                    belief[assignment] *= msg[proj]
        
        # The partition function Z is the sum of the belief over all assignments.
        Z = sum(belief.values())
        self.Z = Z
        self.messages = messages.copy()
        return Z

    def compute_marginals(self):
        """
        Compute the marginal probability of each variable in the graphical model
        using the computed messages.
        
        Returns:
            marginals: A dictionary mapping each variable to its marginal distribution.
        """
        marginals = {}
        def get_indices(clique_vars, subset):
            return [i for i, var in enumerate(clique_vars) if var in subset]
        
        # Helper: given an assignment (tuple) and a list of indices,
        # return the sub-assignment (tuple) corresponding to those indices.
        def project_assignment(assignment, indices):
            return tuple(assignment[i] for i in indices)    
        # Compute beliefs for each clique
        clique_beliefs = {}
        
        for clique, (clique_vars, potential_table) in self.clique_potentials.items():
            belief = potential_table.copy()
            
            # Multiply all incoming messages to compute belief
            for neighbor in self.junction_tree[clique]:
                msg = self.messages[(neighbor, clique)]
                separator = clique & neighbor
                indices = get_indices(clique_vars, separator)

                for assignment in belief:
                    proj = project_assignment(assignment, indices)
                    belief[assignment] *= msg[proj]

            # Normalize the belief
            Z = sum(belief.values())
            for assignment in belief:
                belief[assignment] /= Z  # Normalize to make it a probability distribution

            clique_beliefs[clique] = belief

        # Compute marginals for each variable
        all_vars = {var for clique_vars, _ in self.clique_potentials.values() for var in clique_vars}
        
        for var in all_vars:
            # Find a clique containing this variable
            for clique, (clique_vars, _) in self.clique_potentials.items():
                if var in clique_vars:
                    break
            
            # Marginalize belief over all other variables
            indices = get_indices(clique_vars, {var})
            marginal_dist = {}

            for assignment, prob in clique_beliefs[clique].items():
                proj = project_assignment(assignment, indices)
                if proj in marginal_dist:
                    marginal_dist[proj] += prob
                else:
                    marginal_dist[proj] = prob

            marginals[var] = marginal_dist

        return marginals

    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model using a max-product 
        (k-best) message passing algorithm on the junction tree.

        Instead of summing over assignments (as in the sum-product algorithm), this method 
        uses maximization and, at each clique, retains the top-k candidate assignments along 
        with backpointers. After an upward pass from the leaves to the root, a downward 
        backtracking pass reconstructs the full assignments.

        Returns:
            A list of tuples [(assignment_dict, joint_probability), ...] for the top k assignments,
            sorted in descending order of (unnormalized) joint probability.
        """
        k = self.k
        # --- Helper functions ---
        def get_indices(clique_vars, subset):
            # Returns the list of indices at which variables from 'subset' appear in clique_vars.
            return [i for i, var in enumerate(clique_vars) if var in subset]

        def project_assignment(assignment, indices):
            # Given an assignment tuple and a list of indices, return the sub-assignment.
            return tuple(assignment[i] for i in indices)

        # --- Upward pass ---
        # For each clique, we compute candidates for the joint assignment on that clique’s variables 
        # (combined with all its subtree) and then “send a message” to its parent by marginalizing 
        # out the variables not in the separator.
        # Each candidate is a triple (score, assignment, backpointers) where:
        #   - score is the joint potential (product of factors) up to that clique,
        #   - assignment is a tuple giving the assignment to the clique's variables,
        #   - backpointers is a dict mapping a child clique to the candidate (assignment, bp) chosen there.
        def upward_pass_top_k(clique, parent=None):
            # Get the children (neighbors except the parent)
            children = [nbr for nbr in self.junction_tree[clique] if nbr != parent]
            clique_vars, pot_table = self.clique_potentials[clique]
            
            # Initialize candidates for this clique using its own potential.
            candidates = []
            for assignment, value in pot_table.items():
                candidates.append((value, assignment, {}))
            
            # Incorporate each child's message.
            for child in children:
                # Recursively compute the child's message (a dict mapping separator assignments
                # to a list of candidate tuples from the child's subtree).
                child_msg = upward_pass_top_k(child, clique)
                # The separator is the intersection of the current clique and the child.
                separator = clique & child
                indices = get_indices(clique_vars, separator)
                new_candidates = []
                for cand in candidates:
                    score, assignment, bp = cand
                    proj = project_assignment(assignment, indices)
                    # Only if the child sent a message for this separator assignment do we combine.
                    if proj in child_msg:
                        # For each candidate coming from the child, update the score and store backpointer.
                        for child_cand in child_msg[proj]:
                            child_score, child_assignment, child_bp = child_cand
                            combined_score = score * child_score  # Multiply the potentials.
                            new_bp = bp.copy()
                            new_bp[child] = (child_assignment, child_bp)
                            new_candidates.append((combined_score, assignment, new_bp))
                # Retain only the top-k candidates (sorted in descending order of score).
                candidates = sorted(new_candidates, key=lambda x: x[0], reverse=True)[:k]

            # If there is a parent, send a message to it by marginalizing out variables not in the separator.
            if parent is not None:
                separator = clique & parent
                indices = get_indices(clique_vars, separator)
                msg = {}
                for cand in candidates:
                    score, assignment, bp = cand
                    proj = project_assignment(assignment, indices)
                    entry = (score, assignment, bp)
                    msg.setdefault(proj, []).append(entry)
                # For each projection (separator assignment), keep only the top-k candidates.
                for proj in msg:
                    msg[proj] = sorted(msg[proj], key=lambda x: x[0], reverse=True)[:k]
                return msg
            else:
                # At the root, return the list of candidates.
                return sorted(candidates, key=lambda x: x[0], reverse=True)[:k]

        # --- Downward (backtracking) pass ---
        # Given a candidate for a clique, use its stored backpointers to recursively reconstruct 
        # the full assignment over all variables.
        def reconstruct_assignment(clique, candidate, parent=None):
            # candidate is a tuple (assignment, backpointers)
            full_assignment = {}
            clique_vars, _ = self.clique_potentials[clique]
            assignment = candidate[0]
            # Record the assignment for the clique's variables.
            for var, val in zip(clique_vars, assignment):
                full_assignment[var] = val
            # Recurse into children.
            for child in self.junction_tree[clique]:
                if child == parent:
                    continue
                if child in candidate[1]:
                    child_candidate = candidate[1][child]  # (child_assignment, child_backpointers)
                    child_full_assignment = reconstruct_assignment(child, child_candidate, parent=clique)
                    full_assignment.update(child_full_assignment)
            return full_assignment

        # --- Run the algorithm ---
        # Select an arbitrary clique as the root.
        root = next(iter(self.junction_tree))
        # Perform the upward pass to compute top-k candidates at the root.
        top_candidates = upward_pass_top_k(root, parent=None)
        # Each candidate is (score, assignment, backpointers) for the root clique.
        results = []
        for score, assignment, bp in top_candidates:
            full_assign = reconstruct_assignment(root, (assignment, bp), parent=None)
            results.append((full_assign, score))
        # Sort the results (they should already be sorted, but we sort to be sure).
        results = sorted(results, key=lambda x: x[1], reverse=True)
        # print type and contents of results
        print(f"Type of results: {type(results)}")
        print("Contents of results:")
        for result in results:
            print(result)
        top_k_assignments = []
        for assignment_dict, count in results:
            # Convert the assignment dict into a list, sorted by key.
            # This ensures that the assignment list is in the order of keys 0,1,2,...
            assignment_list = [assignment_dict[k] for k in sorted(assignment_dict.keys())]
            probability = count / self.Z
            # print(assignment_list)
            # print(probability)
            print({
                "assignment": assignment_list,
                "probability": probability
            })
            top_k_assignments.append({
                "assignment": assignment_list,
                "probability": probability
            })
        return top_k_assignments


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
    # evaluator = Get_Input_and_Check_Output('Assignment_1\Questionnare\Sample_Testcase.json')
    evaluator = Get_Input_and_Check_Output('samp_test2.json')
    # evaluator = Get_Input_and_Check_Output(os.getcwd())
    # evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json') 

