import multiprocessing
import os

import gym
import neat
from gym.envs.classic_control import AcrobotEnv
from neat import ParallelEvaluator
import neat

import vis

# Initialize environment
env = AcrobotEnv()

# Number of input nodes will be the number of observations from the environment
num_inputs = env.observation_space.shape[0]
# Number of output nodes will be the number of possible actions
num_outputs = env.action_space.n

# Define the fitness function
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    observation,_ = env.reset()
    cumulative_reward = 0
    done = False
    steps = 0
    MAX_STEPS=300

    while not done and steps < MAX_STEPS:

        actions = net.activate(observation)
        vis.draw_net(config, genome,env.screen,observation,actions)
        action = actions.index(max(actions))

        observation, reward, done, _, _ = env.step(action)
        env.render()

        cumulative_reward += reward
        steps += 1
    # Визуализация структуры лучшей нейронной сети




    return cumulative_reward
# Load the NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neat_config.txt')


if __name__ == '__main__':
    # Check for the latest checkpoint and load it if exists
    checkpoint_path = max([f for f in os.listdir() if f.startswith('neat-checkpoint-')], default=None, key=os.path.getctime)
    if checkpoint_path:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_path)

    else:
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))


# Используйте ParallelEvaluator для параллельной оценки всех геномов в популяции
    pe = ParallelEvaluator(num_workers=multiprocessing.cpu_count()-1, eval_function=eval_genome)

    # Запустите процесс оптимизации для заданного числа поколений
    winner = pop.run(pe.evaluate)

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))
