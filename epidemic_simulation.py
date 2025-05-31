# Copyright (c) 2025 Michael T.M. Emmerich
# Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

"""
This module implements an efficient stochastic simulation of epidemics in networks.
It uses a Gillespie-style algorithm for exact stochastic simulation.
"""

import numpy as np
import networkx as nx
import random

class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0, model=1):  
        """
        Initialize epidemic simulation graph
        
        Args:
            infection_rate: Rate at which infection spreads between connected nodes
            recovery_rate: Rate at which infected nodes recover
            model: Type of epidemic model (0: SI, 1: SIS, 2: SIR)
        """
        self.G = nx.Graph()
        self.model = model 
        self.infection_rate, self.recovery_rate = infection_rate, recovery_rate
        self.infected_nodes = []
        self.total_infection_rate, self.total_recovery_rate = 0, 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False,
                        recovered=False, vaccinated=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_step(self):
        if not self.infected_nodes:
            return float('inf')
        if (self.model != 0):    
           total_rate = self.total_infection_rate + self.total_recovery_rate
        else:
           total_rate = self.total_infection_rate
        if (total_rate < 0.0001): 
            return 0
        wait_time = random.expovariate(total_rate)
        
        r = random.uniform(0, total_rate)
        if r < self.total_infection_rate:  # Infection event
            target = random.uniform(0, self.total_infection_rate)
            cumulative = 0
            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]['sum_of_weights_i']
                if cumulative > target:
                    self.infect_neighbor(node)
                    break
        else:  # Recovery event
            target = random.uniform(0, self.total_recovery_rate)
            cumulative = self.total_infection_rate
            for node in self.infected_nodes:
                cumulative += self.recovery_rate
                if cumulative > target:
                    self.recover_node(node)
                    break
        return wait_time

    def recover_node(self, node):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.model == 1:  # SIS
            for neighbor in self.G.neighbors(node):
                self.G.nodes[neighbor]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
            self.total_infection_rate += self.G.nodes[node]['sum_of_weights_i']
            self.G.nodes[node]['infected'] = False
        elif self.model == 2:  # SIR
            self.G.nodes[node]['recovered'], self.G.nodes[node]['infected'] = True, False

    def infect_neighbor(self, node):
        neighbors = [n for n in self.G.neighbors(node) if
                     
                     not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if neighbors:
            weights = np.array([self.G[node][n]['weight'] for n in neighbors])
            cumulative, target = 0, random.uniform(0, np.sum(weights))
            for i, weight in enumerate(weights):
                cumulative += weight
                if cumulative > target:
                    self.infect_node(neighbors[i])
                    break

    def infect_node(self, node):
        if self.G.nodes[node]['vaccinated']:
            return 
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        for neighbor in self.G.neighbors(node):   # the infection rate becomes now relevant
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_infection_rate += self.G[node][neighbor]['weight']
            elif self.G.nodes[neighbor]['infected'] and (neighbor !=node):
                # reduce the rate for the infected neighbor, as it has one less infected neighbor    
                w = self.G[node][neighbor]['weight']
                self.G.nodes[neighbor]['sum_of_weights_i'] = self.G.nodes[neighbor]['sum_of_weights_i']-w
                self.total_infection_rate -= w

    def vaccinate_nodes(self, vaccinated_nodes):
        for node in vaccinated_nodes:
            self.G.nodes[node]['vaccinated'] = True

def run_simulation(x_vector, G, repeats=5):
    """
    Simulates epidemic spread in a network with a given vaccination strategy.
    
    Args:
        x_vector: Binary vector where 1=vaccinated, 0=not vaccinated
        G: NetworkX graph representing the contact network
        repeats: Number of simulation repeats for averaging
    
    Returns:
        float: Average peak infection (maximum number of simultaneously infected nodes)
    """
    peak_infections = []
    for _ in range(repeats):
        # Initialize new epidemic network with SIR model (model=2)
        graph = EpidemicGraph(infection_rate=0.2, recovery_rate=0.8, model=2)
        
        # Copy network structure
        for node in G.nodes():
            graph.add_node(node)
        for edge in G.edges():
            graph.add_edge(edge[0], edge[1], 1.0)

        # Vaccinate selected nodes
        vaccinated_nodes = [i for i, xi in enumerate(x_vector) if xi == 1]
        graph.vaccinate_nodes(vaccinated_nodes)

        # Start infection from a random non-vaccinated node
        start_candidates = [i for i, xi in enumerate(x_vector) if xi == 0]
        start_node = random.choice(start_candidates)
        graph.infect_node(start_node)

        # Run simulation and track the peak
        max_infected = 0
        for _ in range(1000):
            infected_now = len([n for n in graph.G.nodes if graph.G.nodes[n]['infected']])
            max_infected = max(max_infected, infected_now)
            graph.simulate_step()
        peak_infections.append(max_infected)
    return np.mean(peak_infections)

