## 🧱 Arquitetura dos Módulos (Core + UI) para Dataset Builder/Calibrator 🛠️
Este documento define a arquitetura lógica da aplicação, focando na separação de responsabilidades entre a Camada Core (lógica de negócio) e a Camada UI (interação com o utilizador). O objetivo é garantir um projeto modular, testável e fácil de manter.

## 1. Visão Geral da Arquitetura

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

## 2\. Módulos da Camada Core (`core/`)

### 2.2. `preprocess.py`

**Responsabilidade:** Aplicar o pipeline de normalização e retornar a folha normalizada.

```python
def normalize_sheet_image(img_bgr, normalization_profile: dict):
    """
    Aplica o pipeline de normalização:
    - gray world
    - CLAHE
    - resize/letterbox
    - máscara opcional
    - normalização brilho
    """
    # Retorna numpy.ndarray
```

### 2.3. `tests_quality.py`

**Responsabilidade:** Executar os testes de qualidade da folha, usando a imagem normalizada e o template.

**Exemplos de Testes:**

  * Diferença LAB vs template
  * Brilho global
  * Centragem/alinhamento
  * Rotação

<!-- end list -->

```python
from typing import Tuple, Dict, Any

def run_all_tests(sheet_img_norm, template_img_norm, config: dict) -> Tuple[bool, Dict[str, Any]]:
    """
    Corre todos os testes de qualidade definidos.
    Retorna:
      - tests_pass (bool)
      - details: dicionário com scores por teste:
        {
          "brightness": float,
          "lab_delta_L": float,
          "lab_delta_ab": float,
          "rotation": float,
          "centering": float,
          ...
        }
    """
```

### 2.4. `can_detector.py`

**Responsabilidade:** Detetar as 48 latas na folha, usando o modelo (ONNX, etc.).

```python
def detect_cans_on_sheet(sheet_img_norm, model_config: dict) -> list[dict]:
    """
    Deteta latas na imagem normalizada.
    Retorna uma lista de deteções:
    [
      {
        "can_id": int,
        "bbox": (x, y, w, h),
        "score": float
      },
      ...
    ]
    """
    # Futuro: suportar segmentação além de bounding boxes.
```

### 2.5. `cropper.py`

**Responsabilidade:** Criar as imagens das latas e aplicar a máscara, a partir das deteções.

```python
from typing import List, Dict

def crop_cans_from_sheet(sheet_img_norm, detections: List[Dict], mask_template=None) -> List[Dict]:
    """
    Faz crop de cada lata com base nas detecções e aplica máscara.
    Retorna uma lista de resultados:
    [
      {
        "can_id": int,
        "image": np.ndarray,  # imagem da lata (normalizada + máscara)
      },
      ...
    ]
    """
```

### 2.6. `dataset_manager.py`

**Responsabilidade:** Gestão de `sessions.csv` e `cans.csv`, criação de sessões, registo de folhas/latas, atualização de *labels* e exportação de dataset. (O **único** a manipular CSVs).

#### Funções Principais (Nível Alto)

```python
def create_session(sku: str, operator: str, notes: str = "") -> str:
    """Cria uma nova sessão no sessions.csv e devolve session_id (ex: 'session_0001')."""
    pass

def register_sheet(session_id: str, sheet_index: int, tests_pass: bool, tests_details: dict, raw_path: str, norm_path: str) -> str:
    """Regista uma folha associada a uma sessão. Retorna sheet_id (ex: 'sheet_0001')."""
    pass

def register_cans_for_sheet(session_id: str, sheet_id: str, cans_info: list[dict]):
    """
    Regista as 48 latas de uma folha no cans.csv.
    Cada elemento de cans_info pode conter:
    {
      "can_id": int,
      "image_path": str,
      "quality_ok": bool,
      "scores": {...}
    }
    """
    pass

# Atualização de labels
def update_can_label(session_id: str, sheet_id: str, can_id: int, new_label: str):
    """Atualiza a coluna 'label' de uma lata específica em cans.csv."""
    pass

# Exportação
def export_dataset(
    output_folder: str,
    label_filter: list[str],
    sessions_filter: list[str] | None = None,
    require_quality_ok: bool = True
) -> str:
    """
    Exporta o dataset para uma pasta de treino, copiando (ou linkando) imagens
    e gerando um manifest (CSV/JSON).
    Retorna o caminho do manifest.
    """
    pass
```

### 2.7. `calibration.py`

**Responsabilidade:** Lidar com ficheiros de calibração (JSON) do Colab.

```python
def import_calibration(json_path: str, name: str, model_type: str) -> dict:
    """
    Lê o ficheiro JSON, adiciona metadados (nome, data, modelo)
    e guarda em config/calibration/.
    Retorna o dicionário de calibração carregado.
    """
    pass

def list_calibrations() -> list[dict]:
    """Lista todos os perfis de calibração disponíveis em config/calibration/."""
    pass
```

-----

## 3\. Módulos da Camada UI (`ui/`) 🖼️

Localização: `ui/`

Cada janela implementa um dos fluxos principais da aplicação.

### 3.1. `main_window.py`

**Responsabilidade:** Janela principal, ponto de entrada.

**Elementos principais:**

  * Botão: Nova sessão e processar folhas (**Fluxo A**)
  * Botão: Rever dataset (**Fluxo B**)
  * Botão: Exportar dataset (**Fluxo C**)
  * Botão: Calibração (**Fluxo D**)
  * Área de *logs* / estado

### 3.2. `session_window.py` (ou `session_wizard.py`)

**Responsabilidade:** UI do **Fluxo A** – criar nova sessão e processar folhas.

**Ações Típicas:**

  * Inputs: `SKU`, `operador`, `notas`, pasta de entrada de folhas.
  * **Botão Processar** que executa a sequência de chamadas ao Core.

### 3.3. `review_window.py`

**Responsabilidade:** UI do **Fluxo B** – rever e etiquetar latas.

**Ações Típicas:**

  * Filtros: `sessão`, `folha`, `label`, `quality_ok`.
  * Grelha de *thumbnails*.
  * Botões (`good`, `defect`, etc.) que chamam `dataset_manager.update_can_label`.

### 3.4. `export_window.py`

**Responsabilidade:** UI do **Fluxo C** – exportar dataset.

**Ações Típicas:**

  * Selecionar: *labels* a incluir, sessões, e opção `apenas quality_ok == True`.
  * **Botão Exportar** que chama `dataset_manager.export_dataset`.

### 3.5. `calibration_window.py`

**Responsabilidade:** UI do **Fluxo D** – importar e listar calibrações.

**Ações Típicas:**

  * **Botão Importar calibração** que chama `calibration.import_calibration`.
  * Lista de calibrações existentes.

-----

## 4\. Fluxos → Módulos (Mapa) 🗺️

| Fluxo | UI | Módulos do Core Envolvidos |
| :--- | :--- | :--- |
| **Fluxo A – Criar sessão** | `session_window.py` | `capture`, `preprocess`, `tests_quality`, `can_detector`, `cropper`, `dataset_manager` |
| **Fluxo B – Rever dataset** | `review_window.py` | `dataset_manager` |
| **Fluxo C – Exportar dataset** | `export_window.py` | `dataset_manager` |
| **Fluxo D – Importar calibração** | `calibration_window.py` | `calibration.py` |

-----

## 5\. Regras de Dependência 🛡️

  * A UI nunca deve manipular CSVs diretamente. Apenas **`dataset_manager.py`** é autorizado a aceder a `sessions.csv` e `cans.csv`.
  * Os módulos Core devem ser **puros** (recebem e devolvem `numpy.ndarray`s ou `dict`s/`list`s).
  * I/O no disco deve ser restrito a:
      * `capture.py`
      * `dataset_manager.py`
      * `calibration.py`
