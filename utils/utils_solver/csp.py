from utils.utils_solver.csp_errors import DomainError, UnknownVariable
from copy import deepcopy
import numpy as np
import time


class CSP:
    def __init__(self, variables, domains, constraints):
        """
        Args:
            variables (list[Variable])
            domains (Domain)
            constraints (list[Constraint])
        """
        self.final_solution = None
        self.time_ac3 = 0
        self.variables = variables
        self.constraints = {}
        self.constraints_list = constraints
        self.domains = domains.dict  # domain of each variable : dict[Variable, Domain]
        for variable in self.variables:
            self.constraints[variable.name] = []
            if variable.name not in self.domains.keys():
                raise DomainError(variable)

        for constraint in constraints:
            for variable in constraint.variables:
                if variable not in self.variables:
                    raise UnknownVariable(variable)
                else:
                    self.constraints[variable.name].append(constraint)

    def is_consistent(self, variable_name, instantiation):
        """
            For a given variable and instantiation, tell if the variable is consistent
        """
        for constraint in self.constraints[variable_name]:
            if not constraint.is_satisfied(instantiation):
                return False
        return True

    def backtracking(self, instantiation, domains, mode_var_heuristic=1, args_var_selection=(), mode_val_heuristic=1,
                     starting_time=0, time_limit=np.inf, n_branching=0, forward_check=True):
        """
        Backtracking algorithm

        Args:
            instantiation (dict[Variable, int]): partial instantiation of the CSP

        return:
            solution_found : Bool
            termination_status : indicates whether the research terminated in the given time
            execution_time : execution_time
            n_visited_nodes : number of nodes visited during the research
        """
        if time.time() - starting_time > time_limit:
            return False, False, time.time() - starting_time, n_branching
            
        for variable_name in instantiation:
            if not self.is_consistent(variable_name, instantiation):
                return False, True, time.time() - starting_time, n_branching

        # If the instantiation is full
        if len(instantiation) == len(self.variables):
            self.final_solution = instantiation
            return True, True, time.time() - starting_time, n_branching

        # We pick a non-instantiated variable
        # var_name = self.heuristic_variable_choice_1(instantiation, domains)
        var_name = self.heuristic_variable_selector(mode_var_heuristic, instantiation, domains, args=args_var_selection)
 
        # We extract its possible values
        if not domains[var_name]:  # empty list
            return False, True, time.time() - starting_time, n_branching

        # values = self.heuristic_values_choice_1(var_name, domains)

        values = self.heuristic_values_selector(mode_val_heuristic, instantiation, domains, var_name)
        # print(var_name, values, domains)

        # print(f"variable {var_name} with instantiation {instantiation}")

        for v in values:  # for all values in the domain of var
            n_branching += 1
            local_instantiation = instantiation.copy()
            local_instantiation[var_name] = v
            # Forward checking
            if forward_check:
                local_domains = self.forward_checking(instantiation, domains, var_name, v)

            else:
                local_domains = deepcopy(domains)
                local_domains[var_name] = [v]

            sons_result = self.backtracking(local_instantiation, local_domains, mode_var_heuristic, args_var_selection, mode_val_heuristic, starting_time, time_limit, n_branching, forward_check)
            n_branching = sons_result[3]
            if sons_result[0]: 
                return True, True, time.time() - starting_time, n_branching
            if not sons_result[1]: 
                return False, False, time.time() - starting_time, n_branching

        return False, True, time.time() - starting_time, n_branching

    # Heuristic for variable choice (first variable allowed)
    def heuristic_variable_choice_1(self, instantiation, domains):
        """ Naive approach : take the first possible variable in the list"""
        for i in range(len(self.variables)):
            if self.variables[i].name not in instantiation:
                return self.variables[i].name

    # Heuristic for variable choice (smaller domain)
    def heuristic_variable_choice_2(self, instantiation, domains):
        """ Second approach : take the variable with the smaller remaining domain"""
        return min([k for k in domains.keys() if k not in instantiation], key=lambda x : len(domains[x]))

    # Heuristic for variable choice (most constrainted)
    def heuristic_variable_choice_3(self, instantiation, domains, list_var_sorted_by_nconstraints):
        """ Third approach : take the variable involved in the highest number of constraints"""
        for i in range(len(self.variables)):
            if list_var_sorted_by_nconstraints[i] not in instantiation:
                return list_var_sorted_by_nconstraints[i]
    
    def compute_list_heuristic_var_3(self):
        dict_var_nconstr = {}
        for var in self.variables:
            dict_var_nconstr[var.name] = len(self.constraints[var.name])
        return sorted(dict_var_nconstr.keys(), key=lambda x : dict_var_nconstr[x], reverse=True)

    # Heuristic for variable choice (most linked to instance)
    def heuristic_variable_choice_4(self, instantiation, domains):
        """ Fourth approach : take the variable involved in the highest number of constraints"""
        dict_var_n_linked_constraints = {}
        for var in self.variables:
            if var.name not in instantiation:
                dict_var_n_linked_constraints[var.name] = 0
                for constraint in self.constraints[var.name]:
                    y = None
                    for u in constraint.variables:
                        if u.name != var.name:
                            y = u.name
                    if y in instantiation:
                        dict_var_n_linked_constraints[var.name] += 1
        return max(dict_var_n_linked_constraints.items(), key=lambda a: a[1])[0]

    def heuristic_variable_selector(self, i, instantiation, domains, args=()):
        if i == 1:
            return self.heuristic_variable_choice_1(instantiation, domains)
        elif i == 2:
            return self.heuristic_variable_choice_2(instantiation, domains)
        elif i == 3:
            return self.heuristic_variable_choice_3(instantiation, domains, *args)
        else:
            return self.heuristic_variable_choice_4(instantiation, domains)

    # Heuristic for values choice (first variable allowed)
    def heuristic_values_choice_1(self, variable_name, domains):
        """ Naive approach : take the domain in its defaults order"""
        return domains[variable_name]

    def heuristic_values_choice_2(self, instantiation, domains, variable_name):
        """most suppported value"""
        dict_supporting_tuples = {u: 0 for u in domains[variable_name]}
        for constraint in self.constraints[variable_name]:
            if constraint.variables[0].name == variable_name:
                idx_var = 0
            else:
                idx_var = 1
            if constraint.variables[~idx_var].name not in instantiation:
                for tuples in constraint.tuples:
                    if tuples[idx_var] in domains[variable_name]:
                        dict_supporting_tuples[tuples[idx_var]] += 1                
        return sorted(dict_supporting_tuples.keys(), key=lambda x : dict_supporting_tuples[x], reverse=True)

    def heuristic_values_selector(self, i, instantiation, domains, variable_name):
        if i == 1:
            return self.heuristic_values_choice_1(variable_name, domains)
        elif i == 2:
            return self.heuristic_values_choice_2(instantiation, domains, variable_name)

    # consistence algorithm
    def ac3(self, verbose):
        if verbose:
            print("\nAC results :")
        to_test = list()
        for constraint in self.constraints_list:
            var = constraint.variables
            to_test.append(var)
            to_test.append((var[1], var[0]))
        while len(to_test) > 0:
            (x, y) = to_test.pop()
            instantiation = {x.name: None,
                             y.name: None}
            for x_value in self.domains[x.name]:
                # next 8 lines : check if x_value has a support
                instantiation[x.name] = x_value
                is_supported = False
                for y_value in self.domains[y.name]:
                    # print(f"{x.name} : {x_value} ; {y.name} : {y_value}")
                    instantiation[y.name] = y_value
                    # if x_value is supported by at least 1 value of y, then it's ok
                    if self.is_consistent(y.name, instantiation):
                        is_supported = True
                if not is_supported:
                    if verbose:
                        print("- value " + str(x_value) + " not supported for var " + str(x.name))
                    self.domains[x.name].remove(x_value)
                    for constraint in self.constraints[x.name]:
                        for z in constraint.variables:
                            if z.name != x.name:
                                to_test += [(z, x)]

    def forward_checking(self, instantiation, domains, x, a):  # change with .name
        new_domains = deepcopy(domains)
        new_domains[x] = [a]
        for constraint in self.constraints[x]:
            y = None
            for variable in constraint.variables:
                if variable.name != x:
                    y = variable.name
            if y not in instantiation:
                for b in new_domains[y]:
                    instantiation = {x: a, y: b}
                    if not constraint.is_satisfied(instantiation):
                        new_domains[y].remove(b)
        return new_domains

    def main(self, instantiation, start, mode_var_heuristic=2, mode_val_heuristic=1, time_limit=np.inf,
             forward_check=True, arc_consistence=True, ac3_verbose=False):

        starting_time = time.time()    
        if arc_consistence:
            self.ac3(verbose=ac3_verbose)
            for var in self.domains:
                if not self.domains[var]:
                    return False, True, 0, 0
            if ac3_verbose:
                print("END OF AC3")
        end = time.time()
        self.time_ac3 = end - start

        args_var_selection = ()
        if mode_var_heuristic == 3:
            list_var_sorted_by_nconstraints = self.compute_list_heuristic_var_3()
            args_var_selection = (list_var_sorted_by_nconstraints,)

        return self.backtracking(instantiation, self.domains, mode_var_heuristic=mode_var_heuristic, args_var_selection=args_var_selection, mode_val_heuristic=mode_val_heuristic, starting_time=starting_time, time_limit=time_limit, forward_check=forward_check)
