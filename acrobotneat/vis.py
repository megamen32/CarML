
import pygame

def draw_net(config, genome, screen,inputs,output_activations, node_radius=5, node_spacing=30):
    """ Рисует нейронную сеть с помощью pygame. """
    width, height = screen.get_size()
    font_size=15
    font = pygame.font.Font(None, font_size)

    # Получите ключи для входных, скрытых и выходных узлов
    input_keys = config.genome_config.input_keys
    hidden_keys = [k for k in genome.nodes.keys() if k not in input_keys and k not in config.genome_config.output_keys]
    output_keys = config.genome_config.output_keys



    translation=(100,100)
    # Вычислите позиции для входных, скрытых и выходных узлов
    input_positions = [(i * node_spacing+translation[0], node_spacing+translation[1]) for i in range(len(input_keys))]
    hidden_positions = [(i * node_spacing+translation[0], 2 * node_spacing+translation[1]) for i in range(len(hidden_keys))]
    output_positions = [(i * node_spacing+translation[0], 3 * node_spacing+translation[1]) for i in range(len(output_keys))]

    # Словарь для хранения позиций всех узлов
    node_positions = {}
    for k, pos in zip(input_keys, input_positions):
        node_positions[k] = pos
    for k, pos in zip(hidden_keys, hidden_positions):
        node_positions[k] = pos
    for k, pos in zip(output_keys, output_positions):
        node_positions[k] = pos

    # Соберите все активации в одном месте
    all_activations = {}
    for k, v in zip(input_keys, inputs):
        all_activations[k] = v
    for k, v in zip(output_keys, output_activations):
        all_activations[k] = v
    # Рисуем узлы и их активации
    for key, pos in node_positions.items():

        pygame.draw.circle(screen, (255, 0, 0), pos, node_radius)
        try:
            activation_text = font.render("{:.2f}".format(all_activations[key]), True, (0, 100, 0))
            screen.blit(activation_text, (pos[0] - node_radius // 2, pos[1] - font_size // 2))
        except:pass
    # Рисуем связи
    for cg in genome.connections.values():
        if cg.enabled:
            input, output = cg.key
            start_pos = node_positions[input]
            end_pos = node_positions[output]
            color = (0, 255, 0) if cg.weight > 0 else (255, 0, 0)
            width = int(1 + abs(cg.weight))
            pygame.draw.line(screen, color, start_pos, end_pos, width)




    pygame.display.flip()