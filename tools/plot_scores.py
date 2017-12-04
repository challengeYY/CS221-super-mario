import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import path


def smoothByAverage(scores, factor):
    newScores = []
    for i in range(factor, len(scores)):
        newScores.append(float(sum(scores[i - factor: i + 1])) / factor)
    return newScores


def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()

    # Game options
    parser.add_argument('--score_log_path', dest='score_log_path', action='store', default='',
                        help='Path to score log file to plot.')
    parser.add_argument('--smooth_factor', dest='smooth_factor', action='store', default=20,
                        help='number of points to average over')

    (options, args) = parser.parse_known_args()

    indices = []
    scores = []
    with open(options.score_log_path, mode='r') as score_log:
        for index, score in enumerate(score_log):
            indices.append(index)
            scores.append(int(score))

    scores = smoothByAverage(scores, options.smooth_factor)

    fig, ax = plt.subplots()
    ax.plot(indices[options.smooth_factor:], scores)

    ax.set(xlabel='game number', ylabel='distance',
           title='Average distance Mario end at during training')
    ax.grid()

    fig.savefig(path.dirname(options.score_log_path) + "/score_plot.png")
    plt.show()


if __name__ == "__main__":
    main()