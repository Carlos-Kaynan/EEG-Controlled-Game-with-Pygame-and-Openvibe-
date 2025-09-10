# visualization.py

"""
Classe para gerenciar o feedback visual para o usuário.
Atualmente, implementado para feedback via console.
"""

class VisualizadorConsole:
    """Classe para gerenciar o feedback visual para o usuário via console."""
    def mostrar_prompt(self, texto: str):
        """Exibe um texto no console."""
        print(f"==> {texto}")
        
    def fechar(self):
        """Método vazio para manter a compatibilidade com interfaces que precisam ser fechadas."""
        pass