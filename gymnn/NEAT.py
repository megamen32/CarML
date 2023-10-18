import functools
import os.path
import pickle
import traceback

import neat
from neat.parallel import ParallelEvaluator
import multiprocessing

from gymnn.racingenv import RacingEnv

GENOME_PKL = 'best_neat_genome2.pkl'

# Define the evaluation function for the NEAT algorithm
render=True


def eval_genomes(genomes, config):
    env=RacingEnv(render=render)
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        for _ in range(2):  # Run for 10 episodes to get an average performance
            state = env.reset()
            done = False
            while not done:
                #action = net.activate(state)
                action = [float(a) for a in net.activate(state)]
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if render:
                    env.render(mode='train')
        genome.fitness = total_reward / 2
def eval_genome(genome, config):
    env=RacingEnv(render=render)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0
    for _ in range(2):  # Run for 10 episodes to get an average performance
        state = env.reset()
        done = False
        while not done:
            action = [float(a) for a in net.activate(state)]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                env.render(mode='train')
    return total_reward / 2

def test_winner(winner_genome, config):
    env=RacingEnv(render=render)
    # Decode the winner genome into a neural network
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    total_reward = 0
    num_episodes = 10  # Number of episodes to test

    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = [float(a) for a in net.activate(state)]
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render(mode='human')

    average_reward = total_reward / num_episodes
    print(f"Average reward over {num_episodes} episodes: {average_reward}")


config_path = 'neat_config.txt'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

# Check for the latest checkpoint and load it if exists
checkpoint_path = max([f for f in os.listdir() if f.startswith('neat-checkpoint-')], default=None, key=os.path.getctime)
if checkpoint_path:
    p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
else:
    p = neat.Population(config)

# Add reporters for stdout and saving statistics
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

single_thread = False
if __name__=='__main__':

    if single_thread:
        # Run the NEAT algorithm for up to 300 generations
        winner = p.run(eval_genomes, 300)
    else:
        # Determine the number of workers (processes) to use. Here, we use the number of available CPU cores.
        num_workers = multiprocessing.cpu_count()-1
        evaluator = ParallelEvaluator(num_workers, eval_genome)
        winner = p.run(evaluator.evaluate, 300)

