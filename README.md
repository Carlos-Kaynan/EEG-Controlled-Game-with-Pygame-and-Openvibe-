# 🧠 EEG-Controlled Game with Pygame 

## Interação Cérebro Máquina (BCI)

Este projeto é um **jogo controlado por sinais cerebrais (EEG)** utilizando Python e Pygame. Através da integração com o **OpenViBE**, o jogo interpreta sinais EEG em tempo real para controlar o personagem: movimentando-o para cima, para baixo e permitindo atirar. As ondas cerebrais **Alfa, Beta e Teta** são mapeadas para diferentes ações no jogo.

## 🎮 Como Funciona

- **Ondas Alfa**: Movimentam o personagem para **cima**.
- **Ondas Teta**: Movimentam o personagem para **baixo**.
- **Ondas Beta**: Comando de **atirar**.

O jogo foi desenvolvido para fins de pesquisa e experimentação com interfaces cérebro-máquina (Brain-Computer Interfaces - BCI).

## 🧰 Tecnologias Utilizadas

- [Python](https://www.python.org/)
- [Pygame](https://www.pygame.org/news)
- [OpenViBE](http://openvibe.inria.fr/) – para aquisição e envio dos sinais EEG

## 📦 Instalação

1. Clone este repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

