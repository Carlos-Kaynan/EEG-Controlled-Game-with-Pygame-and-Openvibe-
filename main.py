import pygame
from classes.game import Game

if __name__ == "__main__":
    enemy_images = [
        r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\pngwing1.png',
        r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\pngwing2.png',
        r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\pngwing3.png'
    ]
    game = Game(1400, 700, r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\Paisagem.jfif', 
                r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\protagonista.png', 
                enemy_images, r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\robotNAO.png')
    game.run()