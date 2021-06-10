from .dissimilarity_calculator import DissimilarityCalculator
import networkx as nx
import warnings

class WithinThresholdDistance(DissimilarityCalculator):
    """
    Implements the DissimilarityCalculator as a calculator of the Hamming distance but with a specialty: The similarity
    depends on the agreement threshold 'at'. Features are same or perceived as the same when they are within the 'at'
    The class contains the attribute 'exclude' which accepts a list of strings with the names of agent features that
    should not be used to calculate between agent similarity.
    """

    def __init__(self, exclude=[]):
        self.exclude = exclude

    def calculate_dissimilarity(self, network: nx.Graph, agent1_id: int, agent2_id: int, **kwargs) -> float:
        """
        Computes the WithinThreshold Distance between two Agents, i.e. returns the proportion of features that
        the two agents have not in common and not within the threshold. 1 means therefore total dissimilarity, and 0 is
        total overlap. Only works with categorical but ordinal attributes.
        :param network: The network in which the agents exist.
        :param agent1_id: The index of the first agent.
        :param agent2_id: The index of the agent to compare with.
        :param kwargs: Additional parameters specific to the implementation of the WithinThresholdDistance.
            Possible parameters are the following:
        :param int=1 at: Agreement threshold as max. Distance to other attribute
        :returns a float value, representing the distance between the two agents
        """
        try:
            self.at = kwargs["agreement_threshold"]
        except KeyError:
            warnings.warn("agreement_threshold not specified, using default value 1")
            self.at = 1
        # todo: implement in such a way that only categorical attributes are considered, and others are ignored
        number_of_features = len(network.nodes[agent1_id])
        return len([k for k in network.nodes[agent1_id] if
                    abs(abs(network.nodes[agent1_id][k]) - abs(network.nodes[agent2_id][k])) <= self.at
                    and k not in self.exclude])/number_of_features

    def calculate_dissimilarity_networkwide(self, network: nx.Graph, **kwargs):
        """
        Calculates the distance from each agent to each other and sets that distance as an attribute on the edge
        between them.
        :param network: The network that is modified.
        """
        for agent in network.nodes():
            for neighbor in network.neighbors(agent):
                network.edges[agent, neighbor]['dist'] = self.calculate_dissimilarity(network,
                                                                                      agent,
                                                                                      neighbor,
                                                                                      **kwargs)