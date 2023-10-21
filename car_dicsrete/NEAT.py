import os.path

import neat
import torch
from neat.parallel import ParallelEvaluator
import multiprocessing

from acrobotneat.vis import draw_net
from car_dicsrete.discreteracingenv import DiscreteRacingEnv
from gymnn.racingenv import RacingEnv

testing_times = 1

GENOME_PKL = 'best_neat_genome2.pkl'

# Define the evaluation function for the NEAT algorithm
render=True


def eval_genomes(genomes, config):
    env=RacingEnv(render=render)
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        for _ in range(testing_times):  # Run for 10 episodes to get an average performance
            state = env.reset()
            done = False
            while not done:
                #action = net.activate(state)
                action = [float(a) for a in net.activate(state)]
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if render:
                    env.render(mode='train')
        genome.fitness = total_reward / testing_times
def eval_genome(genome, config):
    env=DiscreteRacingEnv(render_mode='human' if render else 'train',patience=70)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    total_reward = 0
    for _ in range(testing_times):  # Run for 10 episodes to get an average performance
        state = env.reset()
        done = False
        while not done:
            prob = net.activate(state)
            action = prob.index(max(prob))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if render:
                draw_net(config,genome,env.screen,state,prob)
    return total_reward / testing_times

def test_winner(winner_genome, config):
    env=DiscreteRacingEnv(render_mode='human' if render else 'train')
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
        winner = p.run(eval_genomes, 3000)
    else:
        # Determine the number of workers (processes) to use. Here, we use the number of available CPU cores.
        num_workers = multiprocessing.cpu_count()-1
        evaluator = ParallelEvaluator(num_workers, eval_genome)
        winner = p.run(evaluator.evaluate, 3000)

