"""
This python file contains useful methods and classes for processing data
"""

import math
import os
import pickle
import random
from collections import defaultdict

# Visualization libraries primarily for statistics of GraphOT_Factory
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="white")


class GraphOT:
    """
    A class for graph objects that are used in GraphOT operations.
    """

    def __init__(
        self,
        graph: nx.Graph,
        prob_method="degree",
        cost_method="shortest_path",
        info="",
        max_dist=1000,
        filter_lcc=False,
    ):
        """
        Initialize a GraphOT instance.

        Parameters
        ----------
        graph : nx.Graph
            The networkx object representing the

        prob_method : str
            The method to use for endowing the nodes with a probability
            distibution. Acceptable strings are "uniform" and "degree".
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value
            proportional to how many neighbors it has in the graph, normalized.

        cost_method : str
            The method to use for computing the relational cost in the graph
            space. Acceptable strings are "adjacency" and "shortest_path".

        info : str
            A string containing any additional information about the graph

        max_dist : float
            The bound on the cost matrix in the case that the cost method
            has infinite values (i.g. shortest_path configuration can result
            in infinite distance if the graph is not strongly connected).

        filter_lcc : bool
            Whether to filter the graph to only include the largest connected
            component. This is intepreted as the weakly component in the case
            of directed graphs.
        """
        if filter_lcc:
            if nx.is_directed(graph):
                subset = max(nx.weakly_connected_components(graph), key=len)
                self.graph = nx.convert_node_labels_to_integers(
                    graph.subgraph(subset), label_attribute="name"
                )
            else:
                subset = max(nx.connected_components(graph), key=len)
                self.graph = nx.convert_node_labels_to_integers(
                    graph.subgraph(subset), label_attribute="name"
                )
        else:
            self.graph = nx.convert_node_labels_to_integers(
                graph, label_attribute="name"
            )
        self.labels = list(nx.get_node_attributes(self.graph, "name").values())
        self.max_dist = max_dist
        self.node_dist = self.compute_prob(prob_method)
        self.cost = self.compute_cost(cost_method)
        self.info = info

    def compute_prob(self, prob_method: str) -> np.ndarray:
        """
        Compute the probability distribution according to outlined method

        Parameters
        ----------
        prob_method : string
            The method to use for endowing the nodes with a probability
            distibution. Acceptable strings are "uniform" and "degree".
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value
            proportional to how many neighbors it has in the graph, normalized.

        Returns
        ----------
        node_dist : ndarray (n)
            Probability distribution corresponding to the nodes of the graph.
        """
        n = self.graph.number_of_nodes()
        node_dist = np.zeros(n)

        if prob_method == "uniform":
            node_dist += 1
            node_dist /= np.sum(node_dist)
            return node_dist

        elif prob_method == "degree":
            # traverse through all connections
            for edge in self.graph.edges:
                src = edge[0]
                dst = edge[1]
                # add weights if present, otherwise just add 1's
                if nx.is_weighted(self.graph):
                    node_dist[src] += self.graph[src][dst]["weight"]
                    node_dist[dst] += self.graph[src][dst]["weight"]
                else:
                    node_dist[src] += 1
                    node_dist[dst] += 1
            # normalize
            node_dist /= np.sum(node_dist)
            return node_dist

        # if `prob_method` is not anticipated, return the uniform node distribution
        print("Warning: Non-identifiable probability method.")
        return self.compute_prob("uniform")

    def compute_cost(
        self, cost_method: str, check_metric=False, no_inf=True
    ) -> np.ndarray:
        """
        Compute the relational cost matrix according to the outlined method

        Parameters
        ----------
        cost_method : string
            The method to use for computing relational distance between
            the nodes in the graph. Acceptable strings are "adjacency"
            and "shortest_path".

        check_metric : boolean (optional)
            Ensures that the provided method results in a metric over the
            graph space. If not, throw an exception.

        Returns
        ----------
        cost_matrix : ndarray (n, n)
            Relational cost matrix where entry (i, j) represents the
            distance between node (i) and node (j) in the graph.
        """

        if check_metric:
            metrics = ["shortest_path"]
            assert cost_method in metrics, "Non-metric for relational cost matrix"

        if cost_method == "adjacency":
            # extract the dense representation of the graph
            return nx.to_numpy_array(self.graph)

        elif cost_method == "shortest_path":
            # floyd_warshall is a shortest path method that
            # works on graphs with negative edges
            c = nx.floyd_warshall_numpy(self.graph)
            if no_inf:
                c[c == np.inf] = self.max_dist
            return c

        # if `cost_method` is not anticipated, return the shortest_path method result
        print("Warning: Non-identifiable cost method.")
        return self.compute_cost("shortest_path", no_inf)

    def get_node_dist(self):
        """
        Returns the graph's node distribution
        """
        return np.copy(self.node_dist)

    def get_cost(self):
        """
        Returns the graph's cost matrix
        """
        return np.copy(self.cost)

    def extract_info(self):
        """
        Returns the graph's node distribution AND cost matrix
        """
        return np.copy(self.node_dist), np.copy(self.cost)

    def get_size(self):
        """
        Returns the number of nodes of this GraphOT
        """
        return self.graph.number_of_nodes()


# def gromov_wasserstein(graph1: GraphOT, graph2: GraphOT):
#     """
#     Computes the gromov wasserstein distance between two graphs.

#     Inputs:
#         - graph1: GraphOT object representing the first graph
#         - graph2: GraphOT object representing the second graph

#     Returns: The gromov wasserstein distance between the two graphs
#     """
#     # extract the geometry and probability distributions of the graphs
#     # based on the specification used in the graph class.
#     cost1 = graph1.get_cost()
#     cost2 = graph2.get_cost()
#     p1 = graph1.get_node_dist()
#     p2 = graph2.get_node_dist()
#     # compute the GW distance using deterministic solver
#     _, log_info = ot.gromov.gromov_wasserstein(
#         cost1, cost2, p1, p2, "square_loss", log=True
#     )
#     return log_info["gw_dist"]


class GraphOT_Factory:
    """
    Generates and maintains a set of GraphOT objects. Contains
    neat operations such as compute the GW_Barycenter of a set of OT graphs
    """

    def __init__(
        self,
        name2graph: dict,
        prob_method="uniform",
        cost_method="shortest_path",
        max_dist=1000,
        filter_lcc=False,
    ):
        """
        Initialzies an instance of GraphOT factory.

        Parameters
        ----------
        name2graph : dict : str -> nx.Graph
            A mapping from graph names to the networkX objects

        prob_method : str
            The method to use for endowing the nodes with a probability
            distibution. Acceptable strings are "uniform" and "degree".
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value
            proportional to how many neighbors it has in the graph, normalized.

        cost_method : str
            The method to use for computing the relational cost in the graph
            space. Acceptable strings are "adjacency" and "shortest_path".

        max_dist : float
            The bound on the cost matrix in the case that the cost method
            has infinite values (i.g. shortest_path configuration can result
            in infinite distance if the graph is not strongly connected).

        largest_cc : bool
            Whether to filter all graphs to only include the largest connected
            component. This is intepreted as the weakly component in the case
            of directed graphs.

        filter_lcc : bool
            Whether to filter the graphs to only include the largest connected
            component.
        """
        self.factory = name2graph.copy()
        if filter_lcc:
            for name, nx_graph in self.factory.items():
                if nx.is_directed(nx_graph):
                    subset = max(nx.weakly_connected_components(nx_graph), key=len)
                    self.factory[name] = nx_graph.subgraph(subset)
                else:
                    subset = max(nx.connected_components(nx_graph), key=len)
                    self.factory[name] = nx_graph.subgraph(subset)
        self.ot_factory = self.make(name2graph, prob_method, cost_method, max_dist)
        self.names = list(name2graph.keys())
        self.names.sort()
        self.num_graphs = len(name2graph)
        self.max_dist = max_dist

    def make(self, name2graph, prob_method, cost_method, max_dist):
        """
        Make a dictionary of GraphOT objects

        Parameters
        ----------
        name2graph : dict : str -> nx.Graph
            A mapping from graph names to the networkX objects

        prob_method : str
            The method to use for endowing the nodes with a probability
            distibution. Acceptable strings are "uniform" and "degree".
            If "uniform", then every node will receive the same, normalized
            value. If "degree", then every node will received a value
            proportional to how many neighbors it has in the graph, normalized.

        cost_method : str
            The method to use for computing the relational cost in the graph
            space. Acceptable strings are "adjacency" and "shortest_path".
        """
        ot_factory = {}
        for name, nx_graph in name2graph.items():
            ot_factory[name] = GraphOT(nx_graph, prob_method, cost_method, max_dist)
        return ot_factory

    def save(self, save_path: str):
        """
        Save the current GraphOT_Factory

        Parameters
        ----------
        save_path : str
            The path for saving the current factory
            Path should end with ".pkl" for consistency
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(load_path: str):
        """
        Load a GraphOT object from the specified path.

        Parameters
        ----------
        load_path : str
            The path to a pickle file storing the GraphOT_Factory
        """
        with open(load_path, "rb") as f:
            graphOT_factory = pickle.load(f)
        return graphOT_factory

    def summary(self, visualize=True, save=False, save_path="", save_title="summary"):
        """
        Compute and output summary statistics of this graph factory

        Parameters
        ----------
        visualize: bool
            Whether to visualize the computed statistics
        save : bool
            Whether to save the computed instances of the
            summary call
        save_path : str
            The path for saving the computed statistics and mappings
        save_title : str
            The title of the save file

        Returns
        ----------
        info : dict
            A mapping containing all the computed variables of this function
        """
        # Initialize all the relevant variables
        total_nodes = 0.0  # the total number of nodes
        total_edges = 0.0  # the total number of edges
        max_nodes = -float("inf")  # the largest number of nodes
        max_name = ""  # the graph with the maximum nodes
        min_nodes = float("inf")  # the smallest number of nodes
        min_name = ""  # the graph with the minimal nodes
        graph_ct = 0.0  # the total number of graphs
        graph_with_cycles = 0  # the number of graphs with at least one cycle
        has_cycle = []  # the name of graphs with cycle(s)
        animal_freq = defaultdict(
            int
        )  # the mapping of animal to occurence over all graphs
        edge_distribution = defaultdict(
            int
        )  # the number of times X edges populated a graph
        node_distribution = defaultdict(
            int
        )  # the number of times X nodes populated a graph
        tot_deg_distribution = defaultdict(
            int
        )  # the mapping of degree to the number of nodes with that degree
        graph2deg_dist = {}  # the mapping of graph name to the degree distribution
        num_undirected = 0  # the number of undirected graphs

        # Iterate through the networkx graphs
        for name, nx_graph in self.factory.items():
            deg_distribution = defaultdict(int)
            # extract the number of nodes and edges in the current graph
            num_nodes = nx_graph.number_of_nodes()
            num_edges = nx_graph.number_of_edges()
            # compare to current min and max
            if max_nodes < num_nodes:
                max_nodes = num_nodes
                max_name = name
            if min_nodes > num_nodes:
                min_nodes = num_nodes
                min_name = name
            # keep track of the animal appearances and degrees
            for node in nx_graph.nodes:
                animal_freq[node] += 1
                degree = nx_graph.degree[node]
                deg_distribution[degree] += 1
                tot_deg_distribution[degree] += 1
            # check for cycles in the graph
            if nx.is_directed(nx_graph) and len(list(nx.simple_cycles(nx_graph))) > 0:
                graph_with_cycles += 1
                has_cycle.append(name)
            elif not nx.is_directed(nx_graph):
                num_undirected += 1
            # keep track of total edges
            total_nodes += num_nodes
            total_edges += num_edges
            # increment the edge and node distributions
            node_distribution[num_nodes] += 1
            edge_distribution[num_edges] += 1
            # store the degree distribution of the current graph
            graph2deg_dist[name] = deg_distribution
            # increment the graph_ct
            graph_ct += 1

        # calculate average node and edges
        average_node = total_nodes / float(graph_ct)
        average_edge = total_edges / float(graph_ct)

        # output
        if num_undirected > 0:
            print("WARNING: Cannot Detected Cycles in Undirected Graphs")
            print(f"Number of Undirected Graphs  : {num_undirected}\n\n")

        print("--------------- Summary ---------------")
        print(f"Total number of graphs       : {graph_ct}")
        print(f"Average number of nodes      : {average_node}")
        print(f"Average number of edges      : {average_edge}")
        print(f"Largest food web             : {max_name} with {max_nodes} animals")
        print(f"Smallest food web            : {min_name} with {min_nodes} animals")
        print(f"Number of graphs with cycles : {graph_with_cycles}")

        # set-up the 3-subplots for kernel density estimation of the respective distributions
        if visualize:
            _, ax = plt.subplots(3, 1, figsize=(8, 15))
            # plot node distribution
            # add the values number of key pairs in node dictionary to a list
            node_count = [key * value for key, value in node_distribution.items()]
            sns.kdeplot(node_count, ax=ax[0], color="green", fill=True)
            ax[0].set_title("Node Distribution")
            ax[0].set_xlabel("Number of Nodes")
            ax[0].set_ylabel("Frequency")
            # plot edge distribution
            edge_count = [key * value for key, value in edge_distribution.items()]
            sns.kdeplot(edge_count, ax=ax[1], color="red", fill=True)
            ax[1].set_title("Edge Distribution")
            ax[1].set_xlabel("Number of Edges")
            ax[1].set_ylabel("Frequency")
            # plot degree distribution
            degree_count = [key * value for key, value in tot_deg_distribution.items()]
            sns.kdeplot(degree_count, ax=ax[2], color="blue", fill=True)
            ax[2].set_title("Degree Distribution")
            ax[2].set_xlabel("Degree")
            ax[2].set_ylabel("Frequency")

        # store all the computed variables into the info mapping
        info = {}
        info["total_nodes"] = total_nodes
        info["total_edges"] = total_edges
        info["max_nodes"] = max_nodes
        info["max_name"] = max_name
        info["min_nodes"] = min_nodes
        info["min_name"] = min_name
        info["graph_ct"] = graph_ct
        info["graph_wl_cycles"] = graph_with_cycles
        info["has_cycle"] = has_cycle
        info["animal_freq"] = animal_freq
        info["edge_dist"] = edge_distribution
        info["node_dist"] = node_distribution
        info["tot_deg_dist"] = tot_deg_distribution
        info["graph2deg_dist"] = graph2deg_dist
        info["num_undirected"] = num_undirected

        # save the info if required
        if save:
            file_path = os.path.join(save_path, save_title + ".pkl")
            with open(file_path, "wb") as f:
                pickle.dump(info, f)

        return info

    def get_ot_graph(self, name: str):
        """
        Return the graphOT graph corresponding to the specified name
        """
        return self.ot_factory[name]

    def get_nx_graph(self, name: str):
        """
        Return the networkx graph corresponding to the specified name
        """
        return self.factory[name]

    def get_probs(self) -> list:
        """
        Returns the probability of each GraphOT object in the factory
        """
        probs = []
        for name in self.names:
            probs.append(self.ot_factory[name].get_node_dist())
        return probs

    def get_costs(self) -> list:
        """
        Returns the cost matrix of each GraphOT object in the factory
        """
        costs = []
        for name in self.names:
            costs.append(self.ot_factory[name].get_cost())
        return costs

    def random_sample(self, sample_size: int, seed: int = None):
        """
        Return a random subset of this GraphOT_Factory with the specified size

        Parameters
        ----------
        sample_size : int
            The size of the random subset of this GraphOT_Factory
        seed : int
            If provided, set to this seed for randomization to be reproducible
        """
        if seed is not None:
            random.seed(seed)
        random_dict = dict(random.sample(self.factory.items(), sample_size))
        random_factory = GraphOT_Factory(random_dict)
        return random_factory

    def to_list(self):
        """
        Return a list of the GraphOT_Factory OT_graphs with corresponding
        order to the names attribute
        """
        lst = []
        for name in self.names:
            lst.append(self.get_ot_graph(name))
        return lst

    def barycenter(
        self,
        size: int,
        mode="GW",
        weights=None,
        save=False,
        save_path=None,
        features=None,
    ):
        """
        Compute the barycenter of the current GraphOT_Factory

        Parameters
        ----------
        size : int
            The size of the barycenter graph
        epsilon : float
            The regularization constant for entropic-barycenter
            computations.
        mode : str
            The type of barycenter to compute. Valid strings are
            "GW" and "FGW" which correspond to gromov-wasserstein
            and fused gromov-wasserstein respectively
        save : bool
            Whether to save the computed instances of the
            summary call
        save_path : str
            The path for saving the computed barycenter

        Returns
        ----------
        barycenter : array-like, shape (`N`, `N`)
            The similarity matrix in the barycenter space.
        """

        if mode == "GW":
            probs = self.get_probs()
            costs = self.get_costs()
            if weights is None:
                weights = ot.unif(self.num_graphs)
            bary = ot.gromov.gromov_barycenters(
                size,
                costs,
                probs,
                ot.unif(size),
                weights,
                loss_fun="square_loss",
                random_state=0,
            )
        elif mode == "FGW":
            probs = self.get_probs()
            costs = self.get_costs()
            if weights is None:
                weights = ot.unif(self.num_graphs)
            bary = ot.gromov.fgw_barycenters(
                size,
                features,
                costs,
                probs,
                ot.unif(size),
                weights,
                loss_fun="square_loss",
                random_state=0,
            )
        if save and save_path is not None:
            with open(save_path, "wb") as f:
                pickle.dump(bary, f)
        return bary

    def compute_pairwise_mat(
        self,
        mode="GW",
        normalize=False,
        save=False,
        save_path=None,
        feature_mat=None,
        alpha=0.5,
        random_seed=42,
    ):
        """
        Compute the pairwise GW or FGW distance for graphs in this factory.

        Parameters
        ----------
        mode : str
            The pairwise distance type. Valid strings are
            "GW" and "FGW" which correspond to gromov-wasserstein
            and fused gromov-wasserstein respectively
        normalize: bool
            Whether to normalize the entires in the pairwise distance matrix.
        save : bool
            Whether to save the computed instances of the
            summary call
        save_path : str
            The path for saving the computed barycenter
        feature_mat: dict
            A dictionary containing name to species feature dissimilarity
            matrix for the Fused Gromov-Wasserstein computation
        alpha: float
            A float between 0 and 1 for the Fused Gromov-Wasserstein computation
        random_seed: int
            The random seed for the computation.

        Returns
        ----------
        pairwise_dist : dataframe, shape (`N`, `N`)
        pairwise_T : dict of transport matrices between graphs
        """
        # check for non-implemented distance modes
        if mode != "GW" and mode != "FGW":
            raise NotImplementedError("Only GW or FGW distance is currently supported")
        # set up the pairwise distance matrix
        pairwise_dist = pd.DataFrame(
            np.zeros((self.num_graphs, self.num_graphs)),
            columns=self.names,
            index=self.names,
        )
        pairwise_T = {}
        # initialize progress bar counters
        print("\nComputing pairwise distance...")
        total_processes = math.floor(self.num_graphs * self.num_graphs / 2)
        ten_percent = max(int(total_processes / 10), 1)
        counter = 0
        # compute the pairwise distance between each pair of graphs
        # and store the result in the pairwise distance matrix
        with tqdm(total=100) as pbar:
            for i in range(len(self.names)):
                for j in range(i + 1, len(self.names)):
                    name1 = self.names[i]
                    name2 = self.names[j]
                    graph1 = self.ot_factory[name1]
                    graph2 = self.ot_factory[name2]
                    if mode == "GW":
                        T, log = ot.gromov_wasserstein(
                            graph1.get_cost(),
                            graph2.get_cost(),
                            graph1.get_node_dist(),
                            graph2.get_node_dist(),
                            "square_loss",
                            log=True,
                            random_seed=random_seed,
                            random_state=random_seed,
                        )
                        # add the labels to the row and column of the transport matrix
                        T = pd.DataFrame(T, columns=graph2.labels, index=graph1.labels)
                        pairwise_T[(name1, name2)] = T
                        pairwise_T[(name2, name1)] = T.T
                        dist = log["gw_dist"]
                    elif mode == "FGW":
                        M = feature_mat[(name1, name2)]
                        M = M.loc[graph1.labels, graph2.labels]
                        T, log = ot.fused_gromov_wasserstein(
                            feature_mat[(name1, name2)].to_numpy(),
                            graph1.get_cost(),
                            graph2.get_cost(),
                            graph1.get_node_dist(),
                            graph2.get_node_dist(),
                            "square_loss",
                            log=True,
                            alpha=alpha,
                            random_seed=random_seed,
                            random_state=random_seed,
                        )
                        # add the labels to the row and column of the transport matrix
                        T = pd.DataFrame(T, columns=graph2.labels, index=graph1.labels)
                        pairwise_T[(name1, name2)] = T
                        pairwise_T[(name2, name1)] = T.T
                        dist = log["fgw_dist"]
                    pairwise_dist[name1][name2] = dist
                    pairwise_dist[name2][name1] = dist
                    # update progress bar if at some fraction of tenths
                    counter += 1
                    if counter % ten_percent == 0 or counter == total_processes:
                        pbar.update(10)

        # normalize the pairwise distance matrix if required
        if normalize:
            pairwise_dist = pairwise_dist / np.max(pairwise_dist)
        # save the pairwise distance matrix if required
        if save and save_path is not None:
            pairwise_dist.to_csv(save_path)
        return pairwise_dist, pairwise_T


# TODO: Implement a filter operation that takes into account
# some function by which to filter, some function by which to
# retrieve certain variable values from the GraphOT / NetworkX,
# the type of objects to filter by, and outputs a new GraphOT_Factory
