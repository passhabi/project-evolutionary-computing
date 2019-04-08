import numpy as np
import scipy.stats as st


class Selection:

    def __init__(self, population_size: int):
        """

        :param population_size: Upper boundary of num_agents range
        """
        self.num_agents = population_size  # upper boundary of num_agents range

    def get_selected(self):
        raise NotImplementedError()


class RouletteWheelSelection(Selection):
    roulette_wheel_made: bool  # did you roll the roulette wheel?

    def __init__(self, population_size: int):
        super().__init__(population_size)
        self.roulette_wheel_made = False

    def make_roulette_wheel(self, fitness_col):
        self.roulette_wheel_made = True

        nominator = 1 / (fitness_col + 1)  # make small fitnesses worthy.
        denominator = sum(nominator)

        probability = nominator / denominator  # Cumulative probability vector

        self.cdf = [probability[0]]  # add the the first element to the list.
        for i in range(1, len(probability)):
            self.cdf.append(self.cdf[i-1] + probability[i])  # Cumulative probability function

        self.cdf = np.array(self.cdf)

    def get_selected(self):
        if not self.roulette_wheel_made:
            raise Exception("get_selected method called before calling roll_the_wheel method.")

        indicator = np.random.rand()
        return np.array(np.where(indicator <= self.cdf)).flatten()[0]


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
        return np.random.randint(0, self.num_agents)
