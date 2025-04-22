import pygame
# classes/big_boss.py
from classes.projectile import Projectile  # Importação correta


class BigBoss:
    def __init__(self, x, y, size, image_path):
        self.rect = pygame.Rect(x, y, size, size)
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (size, size))
        self.health = 100
        self.projectiles = []
        self.shoot_timer = pygame.time.get_ticks()
        self.shoot_delay = 1000
        self.speed = 2
        self.direction = 1

    def update(self):
        self.rect.y += self.speed * self.direction
        if self.rect.top <= 0 or self.rect.bottom >= 480:
            self.direction *= -1

        current_time = pygame.time.get_ticks()
        if current_time - self.shoot_timer >= self.shoot_delay:
            self.shoot_timer = current_time
            projectile = Projectile(self.rect.right, self.rect.centery + self.rect.height // 2, 5, 1)
            self.projectiles.append(projectile)

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)
        for projectile in self.projectiles:
            projectile.draw(surface)
