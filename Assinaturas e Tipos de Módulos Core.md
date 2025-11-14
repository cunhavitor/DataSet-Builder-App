## 📝 Assinaturas e Tipos de Módulos Core (Design)

Este documento define as **assinaturas rigorosas** das funções da camada Core do *Dataset Builder / Calibrator*, utilizando tipagem (`typing`, `TypedDict`) para garantir consistência e servir de base para a implementação em `core/`.

-----

## 1\. Convenções Gerais de Tipagem

```python
import numpy as np
from pathlib import Path
from typing import TypedDict, Literal, List, Dict, Tuple, Optional
```

### 1.1. Aliases Úteis

```python
ImageArray = np.ndarray  # Imagem BGR ou RGB dependendo do contexto
PathLike = str | Path
LabelType = Literal["good", "defect", "duvidosa", "excluded"]
```

### 1.2. Tipos Estruturados (`TypedDict`)

#### 1.2.1. Resultados dos Testes de Qualidade da Folha

```python
class SheetTestsDetails(TypedDict, total=False):
    brightness: float
    lab_delta_L: float
    lab_delta_ab: float
    rotation: float
    centering: float
    # podes adicionar mais métricas no futuro
```

```python
SheetTestsResult = Tuple[bool, SheetTestsDetails]
# Retorna: (tests_pass, details)
```

#### 1.2.2. Detecções das Latas

```python
class Detection(TypedDict):
    can_id: int                    # 1–48 (índice na folha, se aplicável)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) em px
    score: float                   # confiança do modelo (0–1)
```

#### 1.2.3. Resultado do Crop das Latas

```python
class CanCropResult(TypedDict):
    can_id: int
    image: ImageArray              # imagem da lata (ex: 256x256x3, já com máscara)
```

#### 1.2.4. Informação para Registo de Latas no CSV

```python
class CanRegisterInfo(TypedDict, total=False):
    can_id: int
    image_path: str
    quality_ok: bool
    brightness_score: float
    lab_delta_L: float
    lab_delta_ab: float
    rotation_score: float
    centering_score: float
    # Outros scores de testes de lata...
```

-----

## 2\. Módulo `capture.py`

**Responsabilidade:** Listar e carregar imagens de folhas.

```python
from typing import List

def list_sheet_images(input_folder: PathLike, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Lista os ficheiros de imagem de folha existentes na pasta dada.
    
    Args:
        input_folder: Pasta onde estão as imagens.
        extensions: Lista opcional de extensões aceites (ex: ['.png', '.jpg']).

    Returns:
        Lista de Paths (pathlib.Path) ordenada alfabeticamente.
    """
    ...

def load_sheet_image(path: PathLike) -> ImageArray:
    """
    Carrega uma imagem de folha em BGR (ex: usando cv2.imread).

    Args:
        path: Caminho do ficheiro de imagem.

    Returns:
        np.ndarray com shape (H, W, 3) em BGR.

    Raises:
        FileNotFoundError, ValueError (se imagem não for carregável/3 canais).
    """
    ...
```

-----

## 3\. Módulo `preprocess.py`

**Responsabilidade:** Aplicar o pipeline de normalização.

```python
def normalize_sheet_image(
    img_bgr: ImageArray,
    normalization_profile: dict,
) -> ImageArray:
    """
    Aplica o pipeline de normalização à imagem da folha (gray world, CLAHE, resize/letterbox, máscara, brilho).

    Args:
        img_bgr: Imagem BGR original da folha.
        normalization_profile: Dicionário com parâmetros de normalização.

    Returns:
        Imagem normalizada (consistente com o projeto, p.ex., BGR uint8).
    """
    ...
```

-----

## 4\. Módulo `tests_quality.py`

**Responsabilidade:** Correr os testes de qualidade da folha.

```python
def run_all_tests(
    sheet_img_norm: ImageArray,
    template_img_norm: ImageArray,
    config: dict,
) -> SheetTestsResult:
    """
    Corre todos os testes de qualidade definidos para a folha normalizada.

    Args:
        sheet_img_norm: Imagem da folha já normalizada.
        template_img_norm: Imagem do template (mesmo espaço/cor/tamanho).
        config: Dicionário com parâmetros (thresholds, pesos, etc.).

    Returns:
        (tests_pass, details): Resultado booleano e scores detalhados (SheetTestsDetails).
    """
    ...
```

-----

## 5\. Módulo `can_detector.py`

**Responsabilidade:** Detetar as latas na folha.

```python
from typing import List

def detect_cans_on_sheet(
    sheet_img_norm: ImageArray,
    model_config: dict,
) -> List[Detection]:
    """
    Deteta latas na imagem normalizada da folha.

    Args:
        sheet_img_norm: Imagem normalizada da folha.
        model_config: Configuração do modelo (paths, thresholds, etc.).

    Returns:
        Lista de deteções (Detection).
    """
    ...
```

-----

## 6\. Módulo `cropper.py`

**Responsabilidade:** Recortar as latas e aplicar a máscara.

```python
from typing import List

def crop_cans_from_sheet(
    sheet_img_norm: ImageArray,
    detections: List[Detection],
    mask_template: Optional[ImageArray] = None,
) -> List[CanCropResult]:
    """
    Faz crop de cada lata com base nas deteções e aplica uma máscara opcional.

    Args:
        sheet_img_norm: Imagem normalizada da folha.
        detections: Lista de detecções.
        mask_template: Imagem / máscara a aplicar (ex: 256x256).

    Returns:
        Lista de resultados (CanCropResult), contendo o ID da lata e a imagem recortada.
    """
    ...
```

-----

## 7\. Módulo `dataset_manager.py`

**Responsabilidade:** Gerir metadados e exportações (`sessions.csv`, `cans.csv`).

### Criação de Nova Sessão

```python
def create_session(
    sku: str,
    operator: str,
    notes: str = "",
    metadata_dir: PathLike = "data/metadata",
) -> str:
    """Cria uma nova sessão no sessions.csv. Retorna o session_id gerado (ex: 'session_0001')."""
    ...
```

### Registo de Folha

```python
def register_sheet(
    session_id: str,
    sheet_index: int,
    tests_pass: bool,
    tests_details: SheetTestsDetails,
    raw_path: Optional[str],
    norm_path: str,
    metadata_dir: PathLike = "data/metadata",
) -> str:
    """Regista uma folha associada a uma sessão. Retorna o sheet_id gerado (ex: 'sheet_0001')."""
    ...
```

### Registo de Latas de uma Folha

```python
from typing import List

def register_cans_for_sheet(
    session_id: str,
    sheet_id: str,
    cans_info: List[CanRegisterInfo],
    metadata_dir: PathLike = "data/metadata",
) -> None:
    """Regista as latas de uma folha no cans.csv. (Side effect: Adiciona linhas a cans.csv)."""
    ...
```

### Atualização de Label de uma Lata

```python
def update_can_label(
    session_id: str,
    sheet_id: str,
    can_id: int,
    new_label: LabelType,
    metadata_dir: PathLike = "data/metadata",
) -> None:
    """Atualiza a label de uma lata específica em cans.csv. (Side effect: Modifica cans.csv)."""
    ...
```

### Exportação de Dataset

```python
from typing import List, Optional

def export_dataset(
    output_folder: PathLike,
    label_filter: List[LabelType],
    sessions_filter: Optional[List[str]] = None,
    require_quality_ok: bool = True,
    metadata_dir: PathLike = "data/metadata",
    copy_images: bool = True,
) -> str:
    """
    Exporta o dataset para uma pasta de treino, aplicando filtros.

    Returns:
        Caminho para o manifest gerado (ex: '{output_folder}/train_manifest.csv').
    """
    ...
```

-----

## 8\. Módulo `calibration.py`

**Responsabilidade:** Importar e listar perfis de calibração.

### Importar Calibração

```python
def import_calibration(
    json_path: PathLike,
    name: str,
    model_type: str,
    calibration_dir: PathLike = "config/calibration",
) -> dict:
    """
    Importa um ficheiro de calibração JSON, adiciona metadados e guarda-o.

    Returns:
        Dicionário com o conteúdo da calibração carregada.
    """
    ...
```

### Listar Calibrações

```python
from typing import List, Dict

def list_calibrations(
    calibration_dir: PathLike = "config/calibration",
) -> List[Dict]:
    """
    Lista todos os perfis de calibração disponíveis.

    Returns:
        Lista de dicionários, cada um representando um perfil.
    """
    ...
```

-----

## 9\. Pipeline de Alto Nível (Exemplo Conceptual)

Abaixo um exemplo de uma função que orquestra todo o fluxo, útil para ser chamada pela UI.

```python
def process_session_from_folder(
    input_folder: PathLike,
    sku: str,
    operator: str,
    notes: str,
    normalization_profile: dict,
    tests_config: dict,
    model_config: dict,
    output_base: PathLike = "data",
) -> str:
    """
    Processa todas as folhas de uma pasta, criando uma nova sessão completa e registando em CSVs.

    Returns:
        session_id criada.
    """
    ...
```
