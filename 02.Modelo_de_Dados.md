
# Modelo de Dados e Estrutura de Ficheiros  
*(Dataset Builder / Calibrator)*

Este documento define a estrutura de pastas, convenções de nomes e formatos de ficheiros utilizados pela aplicação.  
O objetivo é garantir consistência, rastreabilidade e compatibilidade entre a Dataset Builder e a aplicação de inspeção final.

---

# 1. Estrutura de Pastas

A seguinte estrutura de pastas é a recomendada:

```text
project_root/
│
├─ data/
│   ├─ raw_sheets/               # folhas originais carregadas
│   ├─ normalized_sheets/        # folhas após normalize_image
│   ├─ cans/
│   │   ├─ session_0001/
│   │   │   ├─ sheet_0001/
│   │   │   │   ├─ can_01.png
│   │   │   │   ├─ can_02.png
│   │   │   │   └─ ...
│   │   │   ├─ sheet_0002/
│   │   │   └─ ...
│   │   ├─ session_0002/
│   │   └─ ...
│   ├─ metadata/
│   │   ├─ sessions.csv
│   │   └─ cans.csv
│   └─ exports/
│       └─ ...
│
├─ config/
│   ├─ camera_profile.json
│   ├─ normalization_profile.json
│   └─ calibration/
│       ├─ cvae_calib_v1.json
│       ├─ padim_calib_v1.json
│       └─ ...
│
├─ models/
│   ├─ can_seg.onnx
│   └─ ...
│
├─ core/
│   ├─ capture.py
│   ├─ preprocess.py
│   ├─ tests_quality.py
│   ├─ can_detector.py
│   ├─ cropper.py
│   ├─ dataset_manager.py
│   └─ calibration.py
│
├─ ui/
│   └─ ...
│
└─ logs/
    ├─ dataset_builder.log
    └─ tests_history.csv


# 2. Convenções de Nomes

### Sessões
session_0001
session_0002
...

### Folhas dentro de uma sessão
sheet_0001
sheet_0002
...

### Latas
can_01.png ... can_48.png

### Exports
export_train_2025-01-04/
export_train_2025-01-04_manifest.csv

Estas convenções garantem ordem lexical perfeita no disco e nomeação fácil em CSVs.

---

# 3. Ficheiros de Metadados

A base da aplicação são **dois CSVs**:

---

## 3.1. `sessions.csv`

Localização:
data/metadata/sessions.csv

Contém **uma linha por sessão criada**.

### Colunas recomendadas:

| Campo           | Tipo     | Descrição |
|-----------------|----------|-----------|
| session_id      | string   | ex: `session_0001` |
| created_at      | datetime | timestamp da sessão |
| sku             | string   | opcional |
| operator        | string   | opcional |
| notes           | string   | livre |
| num_sheets      | int      | total de folhas processadas |
| num_sheets_pass | int      | folhas aprovadas |
| num_sheets_fail | int      | folhas reprovadas |

### Exemplo:

session_id,created_at,sku,operator,notes,num_sheets,num_sheets_pass,num_sheets_fail
session_0001,2025-01-03 14:21:54,SKU_A,Vitor,,15,14,1

---

## 3.2. `cans.csv`

Localização:
data/metadata/cans.csv

Contém **uma linha por lata**.

Cada lata = uma imagem.

### Colunas recomendadas:

| Campo            | Tipo     | Descrição |
|------------------|----------|-----------|
| session_id       | string   | sessão origem |
| sheet_id         | string   | ex: `sheet_0001` |
| can_id           | int      | 1–48 |
| image_path       | string   | path relativo (ex: `data/cans/session_0001/sheet_0001/can_01.png`) |
| label            | string   | `good`, `defect`, `duvidosa`, `excluded` |
| quality_ok       | bool     | passou nos testes da folha |
| brightness_score | float    | delta vs referência |
| lab_delta_L      | float    | diferença canal L |
| lab_delta_ab     | float    | diferença cromática |
| rotation_score   | float    | alinhamento |
| centering_score  | float    | centragem média |
| created_at       | datetime | timestamp |
| notes            | string   | opcional |

### Exemplo:

session_id,sheet_id,can_id,image_path,label,quality_ok,brightness_score,lab_delta_L,lab_delta_ab,rotation_score,centering_score,created_at,notes
session_0001,sheet_0003,17,data/cans/session_0001/sheet_0003/can_17.png,good,True,0.12,1.44,0.98,0.01,0.97,2025-01-03 14:45:03,

---

# 4. Modelo de Dados (Entidades)

Mesmo que não cries classes complexas, estas entidades vão orientar os módulos do core.

---

## 4.1. `CaptureSession`

Representa uma sessão de captura.

{
"session_id": "session_0001",
"created_at": "...",
"sku": "SKU_A",
"operator": "Vitor",
"num_sheets": 15,
"num_sheets_pass": 14,
"num_sheets_fail": 1,
"notes": ""
}

---

## 4.2. `Sheet`

Representa uma folha processada.

{
"session_id": "session_0001",
"sheet_id": "sheet_0003",
"raw_path": "...",
"normalized_path": "...",
"tests_pass": true,
"tests_details": {
"brightness": 0.12,
"lab_delta_L": 1.44,
"lab_delta_ab": 0.98,
"rotation": 0.01,
"centering": 0.97
}
}

---

## 4.3. `CanSample`

Uma imagem de lata.

{
"session_id": "session_0001",
"sheet_id": "sheet_0003",
"can_id": 17,
"image_path": "data/cans/session_0001/sheet_0003/can_17.png",
"label": "good",
"quality_ok": true,
"brightness_score": 0.12,
"lab_delta_L": 1.44,
"lab_delta_ab": 0.98,
"rotation_score": 0.01,
"centering_score": 0.97,
"created_at": "...",
"notes": ""
}

---

## 4.4. `CalibrationProfile`

Usado para importar thresholds do Colab.

{
"name": "SKU_A_v1_illumination_ok",
"model_type": "CVAE",
"dataset_version": "export_train_2025-01-04",
"thresholds": {
"p99_5": 123.45,
"p99_7": 140.12
},
"created_at": "2025-01-04",
"notes": ""
}

---

# 5. Regras Importantes

1. **Nunca apagar imagens fisicamente**  
   Apenas marcar como `excluded` no CSV.

2. **Todos os paths guardados devem ser relativos**  
   Ex.:  
   `data/cans/session_0002/sheet_0003/can_11.png`

3. **CSV sempre ordenado por:**
session_id, sheet_id, can_id

4. **Nomes sempre zero-padded** (0001, 0002…)  
Para manter ordenação lexicográfica correta.

5. **Tudo o que a app fizer deve ser reprodutível**  
O dataset deve conseguir ser reconstruído.

---

# 6. Próximo Passo

Seguir para o **Passo 4 – Design da Arquitectura de Módulos (core e UI)**, onde vamos definir:

- módulos do core,
- responsabilidades de cada um,
- chamadas entre módulos,
- e como isso sustenta os fluxos principais.


