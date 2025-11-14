# Fluxos Principais da Aplicação Dataset Builder / Calibrator

Este documento descreve os fluxos fundamentais de utilização da aplicação, desde a criação de uma sessão de captura até à exportação do dataset e importação de calibração.

---

## Fluxo A – Criar Nova Sessão e Processar Folhas

**Objetivo:**  
Gerar automaticamente as 48 imagens de latas por folha, já normalizadas e registadas com metadados consistentes.

**Actor Principal:**  
Administrador.

### Passos (nível alto)

1. O utilizador abre a app e escolhe **"Nova sessão de captura"**.
2. Introduz um nome para a sessão (ex.: `session_0001`) e opcionalmente notas adicionais (SKU, máquina, operador, etc.).
3. Seleciona:
   - modo de captura (pasta com imagens),
   - pasta de destino onde a sessão será guardada.
4. Para cada imagem de folha presente na pasta selecionada (1 ou várias):
   1. A app aplica o pipeline de normalização (`normalize_image`).
   2. Corre os testes de qualidade:
      - ΔLAB vs template,
      - brilho homogéneo,
      - rotação,
      - centragem básica vs template.
   3. Se algum teste falhar:
      - marca a folha como **reprovada** e mostra mensagem com motivos,
      - permite **"ignorar e continuar"** ou **"descartar folha"**.
   4. Se os testes passarem:
      - atribui automaticamente um `sheet_id` incremental (`0001`, `0002`, …),
      - guarda a folha normalizada em `data/normalized_sheets`,
      - corre o modelo de deteção/segmentação de latas,
      - faz **crop das 48 latas**, aplica máscara e guarda em `data/cans/session_xxx/...`,
      - cria/atualiza registos em `cans.csv` com:
        - `session_id`
        - `sheet_id`
        - `can_id`
        - `image_path`
        - resultados dos testes
        - metadados da captura.
5. No fim, a app mostra um resumo da sessão:
   - número de folhas processadas,
   - número aprovadas/reprovadas,
   - número total de latas geradas.

---

## Fluxo B – Rever e Etiquetar Latas (Curadoria)

**Objetivo:**  
Permitir ao utilizador rever rapidamente as latas e ajustar as labels: `good`, `defect`, `duvidosa`.

### Passos

1. O utilizador abre a secção **"Rever dataset"**.
2. Escolhe filtros:
   - sessão,
   - folha,
   - label atual,
   - estado dos testes (aprovado / reprovado).
3. A app mostra uma grelha (ex.: 8×6) com thumbnails das latas.
4. Ao clicar numa lata, o utilizador pode:
   - ver a imagem ampliada,
   - ver metadados (session, sheet, can_id, scores dos testes),
   - alterar a label para:
     - `good`
     - `defect`
     - `duvidosa`
   - **excluir do dataset**  
     _(excluir = marcação no `cans.csv`, a imagem não é apagada fisicamente)_.
5. A app grava todas as alterações em `cans.csv`.

---

## Fluxo C – Exportar Dataset para Treino

**Objetivo:**  
Gerar uma estrutura de ficheiros limpa + manifest (CSV/JSON) totalmente preparada para treinar modelos no Colab.

### Passos

1. O utilizador abre **"Exportar dataset"**.
2. Seleciona os critérios de exportação:
   - labels incluídas (ex.: apenas `good`),
   - sessões a incluir,
   - aplicar ou não filtros adicionais (ex.: `quality_tests_ok == True`).
3. Escolhe a pasta de destino.
4. A app:
   - percorre as entradas válidas de `cans.csv`,
   - copia (ou cria links simbólicos) das imagens para subpastas, ex.:
     ```
     export/train/good
     ```
   - gera um `train_manifest.csv` ou `.json` contendo:
     - path da imagem
     - session_id, sheet_id, can_id
     - parâmetros de câmara
     - scores dos testes LAB e brilho
     - flags adicionais.
5. No fim, a app mostra:
   - número total de imagens exportadas,
   - caminho completo da exportação,
   - nota explicativa: “usar este path no Colab”.

---

## Fluxo D – Importar Calibração Vinda do Treino

**Objetivo:**  
Guardar os parâmetros de calibração (ex.: percentis e thresholds dos modelos CVAE/Padim) para uso direto na app de inspeção.

### Passos

1. O utilizador abre **"Calibração"**.
2. Clica em **"Importar calibração"**.
3. Seleciona o ficheiro JSON gerado no Colab (ex.: `cvae_calib_v1.json`).
4. A app lê:
   - thresholds,
   - percentis,
   - versão do dataset usada no treino,
   - notas sobre iluminação ou condições específicas.
5. O utilizador atribui um nome amigável (ex.: `SKU_A_v1_illumination_ok`).
6. A app guarda o perfil de calibração em:
   config/calibration/
   incluindo:
- data
- nome fornecido
- ficheiro original.
7. A app mostra a lista de calibrações disponíveis.

---


