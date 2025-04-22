import pygame

class Projectile:
    def __init__(self, x, y, size, direction):
        self.rect = pygame.Rect(x, y, size, size)
        self.speed = direction * 15

    def update(self):
        self.rect.move_ip(self.speed, 0)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 0, 0), self.rect)
