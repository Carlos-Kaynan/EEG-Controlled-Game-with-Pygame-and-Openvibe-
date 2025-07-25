import pygame
import random
import sys

# --- Constantes ---
LARGURA_TELA = 800
ALTURA_TELA = 600
COR_FUNDO = (25, 25, 35)
COR_TEXTO = (240, 240, 240)
COR_BOTAO = (60, 60, 90)
COR_BOTAO_HOVER = (100, 100, 160)
COR_ACERTO = (100, 255, 100)
COR_ERRO = (255, 100, 100)
COR_TEMPO = (255, 165, 0)
FPS = 60
TEMPO_RODADA_MS = 10000

# --- Inicialização ---
pygame.init()
tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
pygame.display.set_caption("Jogo de Reflexo")
relogio = pygame.time.Clock()

# --- Fontes ---
fonte_grande = pygame.font.SysFont('Consolas', 72)
fonte_media = pygame.font.SysFont('Consolas', 48)
fonte_pequena = pygame.font.SysFont('Consolas', 36)

# --- Função botão ---
def desenhar_botao(texto, fonte, ret, mouse_pos):
    if ret.collidepoint(mouse_pos):
        cor = COR_BOTAO_HOVER
    else:
        cor = COR_BOTAO
    pygame.draw.rect(tela, cor, ret, border_radius=12)
    texto_render = fonte.render(texto, True, COR_TEXTO)
    texto_rect = texto_render.get_rect(center=ret.center)
    tela.blit(texto_render, texto_rect)

# --- Tela de Menu ---
def menu():
    while True:
        tela.fill(COR_FUNDO)
        mouse_pos = pygame.mouse.get_pos()

        titulo = fonte_grande.render("Menu Principal", True, COR_TEXTO)
        tela.blit(titulo, titulo.get_rect(center=(LARGURA_TELA / 2, 100)))

        botao_jogo = pygame.Rect(LARGURA_TELA / 2 - 150, 220, 300, 60)
        botao_treinamento = pygame.Rect(LARGURA_TELA / 2 - 150, 320, 300, 60)

        desenhar_botao("Jogo", fonte_media, botao_jogo, mouse_pos)
        desenhar_botao("Treinamento", fonte_media, botao_treinamento, mouse_pos)

        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if botao_jogo.collidepoint(mouse_pos):
                    return "jogo"
                elif botao_treinamento.collidepoint(mouse_pos):
                    return "treinamento"

        pygame.display.flip()
        relogio.tick(FPS)

# --- Função principal do jogo ---
def executar_jogo(modo="jogo"):
    placar_esquerda = 0
    placar_direita = 0
    placar_nulo = 0
    direcao_atual = random.choice(['esquerda', 'direita'])
    feedback_texto = ""
    feedback_cor = COR_TEXTO
    feedback_tempo_exibicao = 0

    NOVA_RODADA = pygame.USEREVENT + 1
    if modo == "jogo":
        pygame.time.set_timer(NOVA_RODADA, TEMPO_RODADA_MS)

    def nova_instrucao():
        nonlocal direcao_atual, feedback_texto
        direcao_atual = random.choice(['esquerda', 'direita'])
        feedback_texto = ""

    botao_voltar = pygame.Rect(20, ALTURA_TELA - 70, 150, 40)

    rodando = True
    while rodando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if evento.type == NOVA_RODADA and modo == "jogo":
                placar_nulo += 1
                feedback_texto = "Tempo Esgotado!"
                feedback_cor = COR_TEMPO
                feedback_tempo_exibicao = pygame.time.get_ticks() + 1000
                nova_instrucao()

            if evento.type == pygame.KEYDOWN:
                acertou = False
                if evento.key == pygame.K_LEFT and direcao_atual == 'esquerda':
                    placar_esquerda += 1
                    acertou = True
                elif evento.key == pygame.K_RIGHT and direcao_atual == 'direita':
                    placar_direita += 1
                    acertou = True

                if acertou:
                    feedback_texto = "Acertou!"
                    feedback_cor = COR_ACERTO
                    nova_instrucao()
                    if modo == "jogo":
                        pygame.time.set_timer(NOVA_RODADA, TEMPO_RODADA_MS)
                else:
                    feedback_texto = "Errou!"
                    feedback_cor = COR_ERRO
                    feedback_tempo_exibicao = pygame.time.get_ticks() + 500

            if evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if botao_voltar.collidepoint(pygame.mouse.get_pos()):
                    return  # <-- Volta para o menu

        if feedback_texto != "Acertou!" and feedback_tempo_exibicao > 0 and pygame.time.get_ticks() > feedback_tempo_exibicao:
            feedback_texto = ""
            feedback_tempo_exibicao = 0

        tela.fill(COR_FUNDO)

        # Texto da instrução
        texto_instrucao = fonte_grande.render("Esquerda" if direcao_atual == 'esquerda' else "Direita", True, COR_TEXTO)
        tela.blit(texto_instrucao, texto_instrucao.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2)))

        # Placar
        texto_placar_esq = fonte_pequena.render(f"Esquerda: {placar_esquerda}", True, COR_TEXTO)
        texto_placar_dir = fonte_pequena.render(f"Direita: {placar_direita}", True, COR_TEXTO)
        texto_placar_nulo = fonte_pequena.render(f"Nulo: {placar_nulo}", True, COR_TEXTO)
        tela.blit(texto_placar_esq, (20, 20))
        tela.blit(texto_placar_dir, texto_placar_dir.get_rect(topright=(LARGURA_TELA - 20, 20)))
        tela.blit(texto_placar_nulo, texto_placar_nulo.get_rect(center=(LARGURA_TELA / 2, 40)))

        # Feedback
        if feedback_texto:
            texto_feedback = fonte_media.render(feedback_texto, True, feedback_cor)
            tela.blit(texto_feedback, texto_feedback.get_rect(center=(LARGURA_TELA / 2, ALTURA_TELA / 2 + 100)))

        # Botão Voltar
        desenhar_botao("Voltar", fonte_pequena, botao_voltar, pygame.mouse.get_pos())

        pygame.display.flip()
        relogio.tick(FPS)

# --- Execução principal ---
while True:
    modo_escolhido = menu()
    executar_jogo(modo_escolhido)
