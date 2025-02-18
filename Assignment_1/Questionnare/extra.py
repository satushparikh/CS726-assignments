    # def compute_marginals(self):
    #     """
    #     Compute the marginal probabilities for all variables.
        
    #     - Uses message passing (sum-product algorithm) to compute marginals.
    #     - Each variable's marginal is obtained by summing over all assignments in relevant cliques.
    #     """
    #     marginals = {var: [0, 0] for var in range(self.num_vars)}  # Assuming binary variables (0,1)
        
    #     # Compute beliefs for each clique
    #     clique_beliefs = {}
    #     for clique in self.triangulated_cliques:
    #         belief = self.assigned_potentials[clique]
    #         for neighbor in self.junction_tree[clique]:
    #             if (neighbor, clique) in self.messages:
    #                 belief = [p1 * p2 for p1, p2 in zip(belief, self.messages[(neighbor, clique)])]
    #         clique_beliefs[clique] = belief
        
    #     # Compute marginals for each variable
    #     for var in range(self.num_vars):
    #         marginal = [0, 0]
    #         for clique, belief in clique_beliefs.items():
    #             if var in clique:
    #                 marginal[0] += belief[0]  # Probability of 0
    #                 marginal[1] += belief[1]  # Probability of 1
            
    #         # Normalize
    #         total = marginal[0] + marginal[1]
    #         if total > 0:
    #             marginal[0] /= total
    #             marginal[1] /= total
            
    #         marginals[var] = marginal
        
    #     return [marginals[i] for i in range(self.num_vars)]
    # def compute_marginals(self):
    #             """
    #     Compute the marginal probabilities for all variables in the graphical model.
        
    #     What to do here:
    #     ----------------
    #     - Use the message passing algorithm to compute the marginal probabilities for each variable.
    #     - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
        
    #     Refer to the sample test case for the expected format of the marginals.
    #     """
    #     """
    #     Compute the marginal probabilities for all variables in the graphical model.
        
    #     Assumes:
    #       - self.clique_potentials is a dict mapping each clique (a frozenset) to a tuple:
    #             (clique_vars, potential_table)
    #         where:
    #             * clique_vars is a tuple (in sorted order) listing the clique's variables.
    #             * potential_table is a dictionary mapping assignments (tuples of 0's/1's) to values.
    #       - self.junction_tree maps each clique to its neighbor cliques.
    #       - self.messages is a dict containing messages from one clique to another as computed by
    #         your message passing algorithm.
        
    #     Returns:
    #       A list of marginals (one per variable, in order from variable 0 to num_vars-1),
    #       where each marginal is a list of two numbers (the probability of 0 and 1, respectively).
    #     """
    #     # First, compute the "belief" for each clique by multiplying its assigned potential
    #     # with all incoming messages (i.e. messages from neighbors directed to this clique).
    #     clique_beliefs = {}  # Will map clique -> (clique_vars, belief_table)
    #     for clique in self.triangulated_cliques:
    #         # Get the original potential for this clique.
    #         clique_vars, pot = self.clique_potentials[clique]
    #         # Make a copy of the potential table (belief will be updated by multiplying messages).
    #         belief = {assignment: value for assignment, value in pot.items()}
            
    #         # Multiply in all incoming messages (from every neighbor of 'clique').
    #         for neighbor in self.junction_tree[clique]:
    #             # Message should be from neighbor to clique.
    #             if (neighbor, clique) in self.messages:
    #                 msg = self.messages[(neighbor, clique)]
    #                 # The separator is the intersection of the two cliques.
    #                 separator = clique & neighbor
    #                 # Determine the positions (indices) in clique_vars corresponding to the separator.
    #                 indices = [i for i, var in enumerate(clique_vars) if var in separator]
                    
    #                 # For each full assignment in the clique's belief, project to the separator
    #                 # and multiply in the message value.
    #                 for assignment in belief:
    #                     # Project assignment onto separator (preserving the order from clique_vars).
    #                     proj = tuple(assignment[i] for i in indices)
    #                     belief[assignment] *= msg[proj]
    #         clique_beliefs[clique] = (clique_vars, belief)
        
    #     # Now compute the marginal for each variable.
    #     # Because of the running intersection property, we can pick any clique that contains the variable.
    #     marginals = {}
    #     for var in range(self.num_vars):
    #         marginal_counts = {0: 0, 1: 0}
    #         found = False
    #         # Iterate over cliques until one is found that contains 'var'
    #         for clique, (clique_vars, belief) in clique_beliefs.items():
    #             if var in clique:
    #                 # Determine the index of var in the clique's ordering.
    #                 var_index = clique_vars.index(var)
    #                 # Sum over all assignments in the belief that assign 0 or 1 to var.
    #                 for assignment, value in belief.items():
    #                     # assignment is a tuple of 0's and 1's for all variables in the clique.
    #                     marginal_counts[assignment[var_index]] += value
    #                 found = True
    #                 break
    #         if not found:
    #             raise ValueError(f"Variable {var} not found in any clique.")
    #         # Normalize the marginal so that the probabilities sum to 1.
    #         total = marginal_counts[0] + marginal_counts[1]
    #         if total > 0:
    #             marginal = [marginal_counts[0] / total, marginal_counts[1] / total]
    #         else:
    #             marginal = [0.0, 0.0]
    #         marginals[var] = marginal
        
    #     # Return the marginals as a list (ordered by variable number).
    #     return [marginals[i] for i in range(self.num_vars)]
# """     def compute_marginals(self):
#             """         
#             Compute the marginal probabilities for all variables in the graphical model.
            
#             What to do here:
#             ----------------
#             - Use the message passing algorithm to compute the marginal probabilities for each variable.
#             - Return the marginals as a list of lists, where each inner list contains the probabilities for a variable.
            
#             Refer to the sample test case for the expected format of the marginals.
#             """ 
#             from itertools import product
#             # Initialize messages: messages[clique1][clique2] stores the message from clique1 to clique2
#             messages = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
#             # Initialize beliefs with clique potentials
#             beliefs = {clique: self.clique_potentials[clique].copy() for clique in self.cliques}
            
#             # Perform message passing until convergence (assume a fixed number of iterations)
#             max_iters = 10  # Adjust based on the problem size
#             for _ in range(max_iters):
#                 new_messages = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
#                 for clique1 in self.junction_tree:
#                     for clique2 in self.junction_tree[clique1]:
#                         # Compute message from clique1 to clique2
#                         marginal_vars = tuple(set(clique1) - set(clique2))
                        
#                         # Sum over marginalized variables
#                         summed_out = defaultdict(float)
#                         for assignment in product([0, 1], repeat=len(clique1)):
#                             assignment_dict = dict(zip(clique1, assignment))
#                             value = beliefs[clique1][tuple(assignment_dict[v] for v in clique1)]
#                             for neighbor in self.junction_tree[clique1]:
#                                 if neighbor != clique2:
#                                     value *= messages[neighbor][clique1][tuple(assignment_dict[v] for v in neighbor)]
                            
#                             reduced_assignment = tuple(assignment_dict[v] for v in clique2)
#                             summed_out[reduced_assignment] += value
                        
#                         # Normalize message
#                         norm_factor = sum(summed_out.values())
#                         for key in summed_out:
#                             new_messages[clique1][clique2][key] = summed_out[key] / norm_factor if norm_factor > 0 else 0
                
#                 messages = new_messages
            
#             # Compute final beliefs by multiplying incoming messages
#             for clique in beliefs:
#                 for assignment in product([0, 1], repeat=len(clique)):
#                     assignment_dict = dict(zip(clique, assignment))
#                     for neighbor in self.junction_tree[clique]:
#                         beliefs[clique][tuple(assignment_dict[v] for v in clique)] *= messages[neighbor][clique][tuple(assignment_dict[v] for v in neighbor)]
                    
#             # Compute marginal distributions for each variable
#             marginals = {var: [0, 0] for var in self.variables}
#             for clique in beliefs:
#                 for assignment in product([0, 1], repeat=len(clique)):
#                     assignment_dict = dict(zip(clique, assignment))
#                     for var in clique:
#                         marginals[var][assignment_dict[var]] += beliefs[clique][tuple(assignment_dict[v] for v in clique)]
            
#             # Normalize marginals
#             for var in marginals:
#                 norm_factor = sum(marginals[var])
#                 marginals[var] = [x / norm_factor if norm_factor > 0 else 0 for x in marginals[var]]
            
#             return [marginals[var] for var in sorted(self.variables)] """

    # def get_z_value(self):
  
    #     """
    #     Compute the partition function (Z value) of the graphical model.
        
    #     What to do here:
    #     ----------------
    #     - Implement the message passing algorithm to compute the partition function (Z value).
    #     - The Z value is the normalization constant for the probability distribution.
        
    #     Refer to the problem statement for details on computing the partition function.
    #     """
      
    #     """
    #     Compute the partition function (Z value) of the graphical model.
        
    #     - Uses message passing (sum-product algorithm) on the junction tree to compute Z.
    #     - Selects a root clique and passes messages bottom-up.
    #     - The final belief at the root clique gives the partition function.
    #     """
    #     root = next(iter(self.junction_tree))  # Pick an arbitrary root
    #     messages = {}

    #     def send_message(from_clique, to_clique):
    #         """Compute the message from one clique to another."""
    #         separator = from_clique & to_clique  # Find separator set
    #         incoming_potential = self.assigned_potentials[from_clique]
            
    #         # If there are incoming messages, multiply them
    #         for neighbor in self.junction_tree[from_clique]:
    #             if neighbor != to_clique and (neighbor, from_clique) in messages:
    #                 incoming_potential = [
    #                     p1 * p2 for p1, p2 in zip(incoming_potential, messages[(neighbor, from_clique)])
    #                 ]
            
    #         # Marginalize out non-separator variables
    #         marginalized_potential = [sum(incoming_potential)]  # Sum over all variables
    #         messages[(from_clique, to_clique)] = marginalized_potential

    #     # Perform upward pass (rooted at an arbitrary clique)
    #     visited = set()

    #     def upward_pass(clique, parent=None):
    #         """Recursive function to pass messages from leaves to root."""
    #         visited.add(clique)
    #         for neighbor in self.junction_tree[clique]:
    #             if neighbor not in visited:
    #                 upward_pass(neighbor, clique)
    #                 send_message(neighbor, clique)

    #     upward_pass(root)

    #     # Z value is the sum of beliefs at the root
    #     root_potential = self.assigned_potentials[root]
    #     for neighbor in self.junction_tree[root]:
    #         root_potential = [p1 * p2 for p1, p2 in zip(root_potential, messages[(neighbor, root)])]

    #     self.Z = sum(root_potential)
    #     return self.Z