# classes/player.py
import pygame

class Player:
    def __init__(self, x, y, size, image_path):
        self.rect = pygame.Rect(x, y, size, size)
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (size, size))  # Redimensiona a imagem

    def move(self, dx, dy):
        self.rect.move_ip(dx, dy)

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)  # Desenha a imagem
