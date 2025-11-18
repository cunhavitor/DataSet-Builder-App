# DataSet-Builder-App
Aplicação para criar dataset robusto e pré-processado para treinar CVAE - detectar defeitos em latas

## 1. Nome e Descrição curta

 - Nome: Dataset Builder / Calibrator

 - Descrição 1 linha:

 - Aplicação para captar, validar e organizar imagens de latas para treinar modelos de detecção de defeitos.

## 2. Problema que resolve

 - Scores dos modelos variam demasiado porque o dataset não é consistente.

 - Latas mal centradas, iluminação diferente, rotação diferente…

 - É trabalhoso e confuso organizar 3k–5k imagens manualmente.

## 3. Objetivo principal

 - Garantir que cada imagem de lata usada no treino passou por um pipeline controlado de normalização, validação e etiquetagem, produzindo um dataset limpo e reprodutível.

## 4. Resultados que a app tem de entregar (outcomes)

 - Gerar um conjunto de imagens de latas normalizadas (mesmo tamanho, mesma máscara, mesma rotação).

 - Garantir que toda imagem do dataset passou em testes de centragem, iluminação e LAB.

 - Guardar imagens e metadados de forma organizada (sessão, folha, lata, parâmetros de câmara).

 - Permitir etiquetar amostras como good / defect / duvidosa.

 - Exportar datasets + manifest (CSV/JSON) prontos para treinar CVAE / Padim / etc.

 - Importar parâmetros de calibração vindos do Colab (thresholds, percentis) e guardá-los para uso na app de inspeção.

## 5. Escopo IN (o que está incluído)

 - Carregamento de imagens de folha completa.

 - Aplicar pipeline normalize_image e testes de qualidade.

 - Detetar 48 latas e fazer crop + máscara.

 - Gestão de sessões de captura (session_0001, 0002, …).

 - Browser simples para rever latas e mudar label.

 - Exportar dataset para treino e importar calibrações.

## 6. Escopo OUT (explicitamente fora)

 - Não vai treinar modelos (CVAE, Padim, etc.) localmente – treino é no Colab.

 - Não vai fazer a inspeção em tempo real na linha (isso é outra app).

 - Não vai gerir múltiplos SKUs complexamente (numa primeira versão talvez só 1 ou poucos).

 - Não vai fazer análise estatística avançada (apenas o necessário para calibração básica).

## 7. Utilizador alvo e contexto

 - Utilizador: Vitor (ou operadores específicos) a correr a app num PC/laptop ligado ao Pi ou a uma pasta partilhada.

 - Contexto: captura e preparação de datasets em sessões offline, fora da linha de produção.aptar, validar e organizar imagens de latas para treinar modelos de       detecção de defeitos.
