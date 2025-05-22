from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uuid
import os
import subprocess
import glob
import pathlib

app = FastAPI()

# Diretórios
UPLOAD_DIR = "uploads"
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "/brplates/runs/train11/weights/best.pt")
YOLO_IMAGE_SIZE = 640
YOLO_OUTPUT_DIR = os.getenv("YOLO_OUTPUT_DIR", "/brplates/runs")

# Garante que os diretórios existam
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/detectar-placa")
async def detectar_placa_api(file: UploadFile = File(...)):

    # 1. Salvar a imagem temporária com nome seguro
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2. Comando YOLO
    cmd = (
        f"yolo detect predict "
        f"source={input_path} "
        f"model={YOLO_WEIGHTS} "
        f"save_crop=True "
        f"max_det=1 "
        f"imgsz={YOLO_IMAGE_SIZE} "
        f"project={YOLO_OUTPUT_DIR} "
        f"name={file_id}"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # 3. Verifica se YOLO executou corretamente
    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={"erro": "Falha na execução do YOLO", "detalhe": result.stderr},
        )

    # 4. Procura a imagem recortada da placa
    crop_glob = os.path.join(YOLO_OUTPUT_DIR, f"{file_id}/crops/*/*.jpg")
    crops = glob.glob(crop_glob)
    if not crops:
        return JSONResponse(
            status_code=404,
            content={"erro": "Nenhuma placa detectada.", "file_id": file_id},
        )

    # 5. Extrai o nome da classe (última pasta antes da imagem)
    crop_path = crops[0]
    classe_detectada = pathlib.Path(crop_path).parent.name

    # Remove o arquivo enviado após o processamento
    try:
        os.remove(input_path)
    except Exception:
        pass

    # 6. Retorna imagem da placa + nome da classe
    return {"arquivo": crop_path, "classe": classe_detectada}
