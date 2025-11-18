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