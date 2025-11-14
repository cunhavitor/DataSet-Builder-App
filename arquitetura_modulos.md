# Arquitetura de Módulos  
*(Dataset Builder / Calibrator)*

Este documento define a arquitetura lógica da aplicação, separando claramente:

- **Camada core** (lógica, sem UI)
- **Camada UI** (janelas e interação com o utilizador)

O objetivo é garantir um projeto **modular, testável e fácil de manter**.

---

# 1. Visão Geral da Arquitetura

A aplicação será dividida em duas camadas principais:

1. **Core (`core/`)**
   - Implementa toda a lógica de:
     - criação de sessões
     - processamento de folhas
     - deteção e crop das latas
     - testes de qualidade
     - gestão de metadados (`sessions.csv`, `cans.csv`)
     - exportação e calibração
   - Não conhece nada sobre UI (Qt, botões, janelas, etc.).

2. **UI (`ui/`)**
   - Implementa as janelas (PySide6, etc.).
   - Chama funções do core.
   - Responsável por mostrar imagens, grelhas, mensagens e recolher input do utilizador.

Regra de ouro:
> **UI → chama o core** (one-way).  
> O core nunca importa nada da UI.

---

# 2. Módulos do Core

Localização: `core/`

Cada módulo tem uma responsabilidade bem definida.

---

## 2.1. `capture.py`

**Responsabilidade:**  
Lidar com a origem das imagens de folha (por agora, pastas em disco).

### Funções principais (proposta)

```python
def list_sheet_images(input_folder: str) -> list[str]:
    """Devolve a lista de paths de imagens de folha presentes na pasta dada."""

def load_sheet_image(path: str):
    """Carrega imagem BGR (cv2.imread) e faz validações básicas."""
