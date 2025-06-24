import pygame
import random

class Enemy:
    def __init__(self, x, y, size, image_path):
        self.rect = pygame.Rect(x, y, size, size)
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (size, size))
        self.speed = 5

    def update(self):
        self.rect.move_ip(self.speed, 0)
        if self.rect.left > 640:
            self.rect.right = 0
            self.rect.y = random.randint(0, 480 - self.rect.height)

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)
