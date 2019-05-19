import numpy as np


class RouletteWheelSelection:
    roulette_wheel_made: bool  # did you roll the roulette wheel?

    def __init__(self):
        self.roulette_wheel_made = False
        self.cdf = list()

    def make_roulette_wheel(self, cost_list):
        self.roulette_wheel_made = True

        nominator = 1 / (cost_list + 1)  # make small finesses worthy.
        denominator = sum(nominator)

        probability = nominator / denominator  # Cumulative probability vector.

        self.cdf = [probability[0]]  # add the the first element to the cdf.
        for i in range(1, len(probability)):
            self.cdf.append(self.cdf[i - 1] + probability[i])  # Cumulative probability function.

        self.cdf = np.array(self.cdf) # convert to array

    def roll(self):
        if not self.roulette_wheel_made:
            raise Exception("roll method called before calling make_roulette_wheel.")

        indicator = np.random.rand()
        return np.array(np.where(indicator <= self.cdf)).flatten()[0]
