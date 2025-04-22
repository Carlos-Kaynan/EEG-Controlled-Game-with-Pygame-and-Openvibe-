import pygame
from classes.player import Player 
from classes.enemy import Enemy
from classes.projectile import Projectile
from classes.big_boss import BigBoss
import random


class Game:
    def __init__(self, width, height, background_image_path, player_image_path, enemy_images, boss_image_path):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.background_image = pygame.image.load(background_image_path)
        self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))
        self.x_pos = 0
        self.player = Player(width // 2, height // 2, 30, player_image_path)
        self.enemies = []
        self.enemy_spawn_times = [3000, 5000, 7000]
        self.enemy_spawned = [False, False, False]
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        self.enemy_images = enemy_images
        self.game_over = False
        self.projectiles = []
        self.last_shot_time = 0
        self.shot_delay = 300
        self.score = 0
        self.big_boss = None
        self.boss_image_path = boss_image_path
        self.game_started = False  # Controla se o jogo começou ou não
        self.restart_button_rect = None
        self.quit_button_rect = None

    def draw_start_screen(self):


        self.screen.fill((0, 0, 0))
        font = pygame.font.Font(None, 60)
        title_text = font.render("Space Adventure", True, (255, 255, 255))
        self.screen.blit(title_text, (self.width // 2 - title_text.get_width() // 2, self.height // 4))
        
        button_font = pygame.font.Font(None, 50)
        start_text = button_font.render("Clique para Começar", True, (0, 255, 0))
        start_button_rect = start_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(start_text, start_button_rect)

        tips_font = pygame.font.Font(None, 20)  # Menor tamanho de fonte para dicas
        tips_text = tips_font.render("Dica: Use W, A, S, D para se mover, e o mouse para atirar!", True, (255, 255, 255))
        self.screen.blit(tips_text, (self.width // 2 - tips_text.get_width() // 2, self.height // 2 + 200))

        pygame.display.flip()
        return start_button_rect

    def draw_game_over_screen(self):
        self.screen.fill((0, 0, 0))
        
        # Exibir o texto "Game Over"
        font = pygame.font.Font(None, 74)
        game_over_text = font.render("Fim de Jogo", True, (255, 0, 0))
        self.screen.blit(game_over_text, (self.width // 2 - game_over_text.get_width() // 2, self.height // 4))
        
        # Exibir o score
        score_font = pygame.font.Font(None, 50)
        score_text = score_font.render(f"Pontuação: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.width // 2 - score_text.get_width() // 2, self.height // 2 - 50))
        
        # Exibir o botão "Reiniciar"
        button_font = pygame.font.Font(None, 50)
        restart_text = button_font.render("Reiniciar", True, (255, 255, 255))
        self.restart_button_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 50))
        self.screen.blit(restart_text, self.restart_button_rect)
        
        # Exibir o botão "Sair"
        quit_text = button_font.render("Sair", True, (255, 255, 255))
        self.quit_button_rect = quit_text.get_rect(center=(self.width // 2, self.height // 2 + 150))
        self.screen.blit(quit_text, self.quit_button_rect)
        
        # Atualizar a tela
        pygame.display.flip()


    def reset_game(self):
        self.player = Player(self.width // 2, self.height // 2, 30, r'C:\Users\carlo\OneDrive\Área de Trabalho\PyGame\PyGame\png\protagonista.png')
        self.enemies.clear()
        self.projectiles.clear()
        self.enemy_spawned = [False, False, False]
        self.score = 0
        self.big_boss = None
        self.game_over = False
        self.start_time = pygame.time.get_ticks()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if not self.game_started:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        if self.start_button_rect.collidepoint(mouse_pos):
                            self.game_started = True
                            self.start_time = pygame.time.get_ticks()  # Reinicia o tempo quando o jogo começar

                elif self.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mouse_pos = pygame.mouse.get_pos()
                        if self.restart_button_rect.collidepoint(mouse_pos):
                            self.reset_game()  # Reinicia o jogo
                        elif self.quit_button_rect.collidepoint(mouse_pos):
                            running = False  # Sai do jogo

                else:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and pygame.time.get_ticks() - self.last_shot_time >= self.shot_delay:
                            self.projectiles.append(Projectile(self.player.rect.left, self.player.rect.centery, 5, -1))  # -1 para atirar para a esquerda
                            self.last_shot_time = pygame.time.get_ticks()

            if not self.game_started:
                self.start_button_rect = self.draw_start_screen()

            elif self.game_over:
                self.draw_game_over_screen()

            else:
                current_time = pygame.time.get_ticks() - self.start_time

                for i in range(len(self.enemy_spawn_times)):
                    if current_time > self.enemy_spawn_times[i] and not self.enemy_spawned[i]:
                        enemy_image = self.enemy_images[i]
                        self.enemies.append(Enemy(0, random.randint(0, self.height - 30), 30, enemy_image))
                        self.enemy_spawned[i] = True

                keys = pygame.key.get_pressed()
                if keys[pygame.K_w]:
                    self.player.move(0, -10)
                if keys[pygame.K_s]:
                    self.player.move(0, 10)
                if keys[pygame.K_a]:
                    self.player.move(-10, 0)
                if keys[pygame.K_d]:
                    self.player.move(10, 0)

                for enemy in self.enemies:
                    enemy.update()
                    if self.player.rect.colliderect(enemy.rect):
                        self.game_over = True

                for projectile in self.projectiles[:]:
                    projectile.update()
                    for enemy_index, enemy in enumerate(self.enemies[:]):
                        if projectile.rect.colliderect(enemy.rect):
                            self.enemies.remove(enemy)
                            self.projectiles.remove(projectile)
                            self.score += 1
                            self.enemies.append(Enemy(0, random.randint(0, self.height - 30), 30, self.enemy_images[enemy_index]))
                            break
                    if projectile.rect.left < 0:
                        self.projectiles.remove(projectile)

                if self.score >= 10 and self.big_boss is None:
                    self.big_boss = BigBoss(0, self.height // 2 - 50, 100, self.boss_image_path)

                if self.big_boss:
                    self.big_boss.update()
                    for projectile in self.projectiles[:]:
                        if projectile.rect.colliderect(self.big_boss.rect):
                            self.big_boss.health -= 1
                            self.projectiles.remove(projectile)
                            if self.big_boss.health <= 0:
                                self.big_boss = None
                            break
                    for boss_projectile in self.big_boss.projectiles[:]:
                        boss_projectile.update()
                        if boss_projectile.rect.colliderect(self.player.rect):
                            self.game_over = True
                        if boss_projectile.rect.right > self.width:
                            self.big_boss.projectiles.remove(boss_projectile)

                self.screen.fill((0, 0, 0))
                self.screen.blit(self.background_image, (self.x_pos, 0))
                self.screen.blit(self.background_image, (self.x_pos - self.width, 0))
                self.x_pos += 2
                if self.x_pos >= self.width:
                    self.x_pos = 0

                self.player.draw(self.screen)
                for enemy in self.enemies:
                    enemy.draw(self.screen)
                for projectile in self.projectiles:
                    projectile.draw(self.screen)
                if self.big_boss:
                    self.big_boss.draw(self.screen)
                    font = pygame.font.Font(None, 36)
                    boss_health_text = font.render(f'Boss Health: {self.big_boss.health}', True, (255, 0, 0))
                    self.screen.blit(boss_health_text, (10, 50))  # Ajuste a posição conforme necessário

                font = pygame.font.Font(None, 36)
                score_text = font.render(f'Score: {self.score}', True, (255, 0, 0))
                self.screen.blit(score_text, (10, 10))

                pygame.display.flip()
                self.clock.tick(60)

        pygame.quit()