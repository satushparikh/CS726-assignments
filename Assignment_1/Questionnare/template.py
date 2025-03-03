import json



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
        pass

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
        pass

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
        pass

    def assign_potentials_to_cliques(self):
        """
        Assign potentials to the cliques in the junction tree.
        
        What to do here:
        ----------------
        - Map the given potentials (from the input data) to the corresponding cliques in the junction tree.
        - Ensure the potentials are correctly associated with the cliques for message passing.
        
        Refer to the sample test case for how potentials are associated with cliques.
        """
        pass

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
<<<<<<< HEAD
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
            marginals[var] = [marginal_dist[(0,)], marginal_dist[(1,)]]

        marginals_list = [marginals[var] for var in sorted(marginals.keys())]
        return marginals_list
=======
        pass
>>>>>>> 04661199ae3b017dd78e28a6b3f6769c59f8c43e

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
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')