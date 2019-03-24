import random


class Selection:

    def __init__(self, num_agents: int):
        """

        :param num_agents: Upper boundary of num_agents range
        """
        self.num_agents = num_agents  # upper boundary of num_agents range

    def get_selected(self):
        raise NotImplementedError()


class RouletteWheelSelection(Selection):

    def __init__(self, num_agents: int, probability: float):
        super().__init__(num_agents)

    def get_selected(self):
        pass


class RankingSelection(Selection):

    def get_selected(self):
        pass


class TournamentSelection(Selection):
    """

    https://www.geeksforgeeks.org/tournament-selection-ga/
    """
    def get_selected(self):
        pass


class RandomSelection(Selection):

    def get_selected(self):
        return random.randint(0, self.num_agents)
