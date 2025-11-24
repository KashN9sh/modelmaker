from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .stl_parser import STLParser
from .segmenter import ModelSegmenter
from .sprue_generator import SprueGenerator
from .exporter import STLExporter

app = FastAPI(title="STL Sprue Generator")

# CORS для работы с frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Также обслуживаем index.html напрямую
    @app.get("/index.html")
    async def index_html():
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"message": "Index file not found"}

# Временная директория для загруженных файлов
UPLOAD_DIR = Path(tempfile.gettempdir()) / "stl_sprue_generator"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)


@app.get("/")
async def root():
    """Главная страница"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return {"message": "STL Sprue Generator API"}


@app.post("/api/upload")
async def upload_stl(file: UploadFile = File(...)):
    """Загрузка STL файла"""
    try:
        logger.info(f"Получен запрос на загрузку файла: {file.filename}, content_type: {file.content_type}")
        
        # Проверяем наличие имени файла
        if not file.filename:
            logger.error("Имя файла не указано")
            raise HTTPException(status_code=400, detail="Имя файла не указано")
        
        # Проверяем расширение файла (без учета регистра)
        if not file.filename.lower().endswith('.stl'):
            logger.error(f"Неправильное расширение файла: {file.filename}")
            raise HTTPException(status_code=400, detail="Файл должен быть в формате STL")
        
        # Сохраняем файл
        # Используем безопасное имя файла
        safe_filename = file.filename.replace('/', '_').replace('\\', '_')
        file_path = UPLOAD_DIR / safe_filename
        
        logger.info(f"Сохранение файла в: {file_path}")
        
        # Читаем содержимое файла
        contents = await file.read()
        logger.info(f"Прочитано байт: {len(contents)}")
        
        # Сохраняем файл
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Проверяем, что файл не пустой
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.error("Загруженный файл пуст")
            raise HTTPException(status_code=400, detail="Загруженный файл пуст")
        
        logger.info(f"Файл сохранен, размер: {file_size} байт")
        
        # Парсим STL
        parser = STLParser()
        try:
            logger.info("Начало парсинга STL файла")
            mesh = parser.load(str(file_path))
            logger.info(f"STL файл успешно загружен: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
            info = parser.get_mesh_info(mesh)
            return JSONResponse({
                "filename": safe_filename,
                "filepath": str(file_path),
                "info": info
            })
        except ValueError as e:
            logger.error(f"Ошибка при парсинге STL файла: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Ошибка при парсинге STL файла: {str(e)}")
        except Exception as e:
            logger.error(f"Ошибка при обработке STL: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Ошибка при обработке STL: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке файла: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Неожиданная ошибка при загрузке файла: {str(e)}")


@app.post("/api/segment")
async def segment_model(data: dict):
    """Сегментация модели на части"""
    filepath = data.get("filepath")
    num_parts = data.get("num_parts", 4)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    parser = STLParser()
    segmenter = ModelSegmenter()
    
    try:
        logger.info(f"Начало сегментации модели: {filepath}")
        mesh = parser.load(filepath)
        logger.info(f"Модель загружена: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
        
        segments = segmenter.segment(mesh)
        logger.info(f"Получено {len(segments)} сегментов после первичной сегментации")
        
        # Если нужно разбить на больше частей, используем разбиение по осям
        # Но только если деталь достаточно большая
        if len(segments) == 1 and num_parts > 1:
            # Проверяем размер детали перед разбиением
            if len(mesh.faces) >= 500:  # Минимум 500 граней для разбиения
                logger.info(f"Разбиваем модель на {num_parts} частей по осям...")
                segments = segmenter._segment_by_axes(mesh, num_parts)
                logger.info(f"Получено {len(segments)} сегментов после разбиения по осям")
            else:
                logger.info(f"Деталь слишком маленькая для разбиения: {len(mesh.faces)} граней")
        
        # Сериализуем сегменты (убираем меши, оставляем только метаданные)
        logger.info("Сериализация сегментов...")
        serialized_segments = []
        for seg in segments:
            try:
                serialized_segments.append({
                    "id": seg["id"],
                    "type": seg["type"],
                    "bounds": seg["bounds"],
                    "center": seg["center"],
                    "vertices": len(seg["mesh"].vertices),
                    "faces": len(seg["mesh"].faces)
                })
            except Exception as e:
                logger.error(f"Ошибка при сериализации сегмента {seg.get('id', 'unknown')}: {e}", exc_info=True)
                continue
        
        # Сохраняем меши сегментов во временные файлы для последующего использования
        segments_dir = UPLOAD_DIR / "segments"
        segments_dir.mkdir(exist_ok=True)
        
        logger.info(f"Сохранение {len(segments)} сегментов в файлы...")
        for seg in segments:
            try:
                seg_file = segments_dir / f"segment_{seg['id']}.stl"
                parser.save(seg["mesh"], str(seg_file))
            except Exception as e:
                logger.error(f"Ошибка при сохранении сегмента {seg['id']}: {e}", exc_info=True)
                continue
        
        logger.info(f"Сегментация завершена успешно: {len(serialized_segments)} сегментов")
        return JSONResponse({
            "segments": serialized_segments,
            "num_parts": len(serialized_segments)
        })
    except Exception as e:
        logger.error(f"Ошибка сегментации: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка сегментации: {str(e)}")


@app.post("/api/generate-sprue")
async def generate_sprue(data: dict):
    """Генерация спрусов для разбитой модели"""
    filepath = data.get("filepath")
    segments_meta = data.get("segments", [])
    sprue_params = data.get("sprue_params", {})
    num_parts = data.get("num_parts", 4)
    
    logger.info(f"Генерация спрусов для файла: {filepath}, сегментов: {len(segments_meta)}")
    
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    parser = STLParser()
    segmenter = ModelSegmenter()
    sprue_gen = SprueGenerator()
    exporter = STLExporter()
    
    try:
        logger.info("Загрузка исходной модели...")
        mesh = parser.load(filepath)
        
        # Применяем масштабирование если указано
        scale = sprue_params.get("scale", 1.0)
        if scale != 1.0:
            logger.info(f"Применение масштаба {scale}x к модели")
            mesh.apply_scale(scale)
        
        logger.info(f"Модель загружена: {len(mesh.vertices)} вершин, {len(mesh.faces)} граней")
        
        # Загружаем сегменты из файлов или создаем заново
        segments_dir = UPLOAD_DIR / "segments"
        segments = []
        
        if segments_dir.exists() and len(segments_meta) > 0:
            logger.info(f"Загрузка {len(segments_meta)} сегментов из файлов...")
            # Загружаем сохраненные сегменты
            for seg_meta in segments_meta:
                seg_file = segments_dir / f"segment_{seg_meta['id']}.stl"
                if seg_file.exists():
                    try:
                        seg_mesh = parser.load(str(seg_file))
                        segments.append({
                            "id": seg_meta["id"],
                            "mesh": seg_mesh,
                            "type": seg_meta.get("type", "unknown"),
                            "bounds": seg_mesh.bounds.tolist(),
                            "center": seg_mesh.centroid.tolist()
                        })
                        logger.info(f"Сегмент {seg_meta['id']} загружен: {len(seg_mesh.vertices)} вершин")
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке сегмента {seg_meta['id']}: {str(e)}")
                        continue
        
            # Если сегменты не загружены, создаем заново
            if not segments:
                logger.info("Сегменты не найдены, создаем заново...")
                segments = segmenter.segment(mesh)
                if len(segments) == 1 and num_parts > 1:
                    # Проверяем размер детали перед разбиением
                    if len(mesh.faces) >= 500:  # Минимум 500 граней для разбиения
                        logger.info(f"Разбиваем модель на {num_parts} частей...")
                        segments = segmenter._segment_by_axes(mesh, num_parts)
                    else:
                        logger.info(f"Деталь слишком маленькая для разбиения: {len(mesh.faces)} граней")
        
        logger.info(f"Всего сегментов для обработки: {len(segments)}")
        
        # Генерируем спрусы
        logger.info("Генерация спрусов...")
        try:
            # Добавляем параметры для спрусов
            sprue_params_full = {
                "wall_thickness": sprue_params.get("wall_thickness", 1.0),  # толщина стенок деталей
                "parts_per_sprue": 15  # деталей на спрус
            }
            parts_with_sprue = sprue_gen.generate(mesh, segments, sprue_params_full)
            logger.info(f"Сгенерировано {len(parts_with_sprue)} частей со спрусами")
        except Exception as e:
            logger.error(f"Ошибка при генерации спрусов: {str(e)}", exc_info=True)
            raise
        
        # Экспортируем в отдельные файлы
        output_dir = UPLOAD_DIR / "output"
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Экспорт частей в STL файлы...")
        exported_files = exporter.export_parts(parts_with_sprue, str(output_dir))
        logger.info(f"Экспортировано {len(exported_files)} файлов")
        
        return JSONResponse({
            "parts": exported_files,
            "num_parts": len(exported_files)
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка генерации спрусов: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации спрусов: {str(e)}")


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Скачивание сгенерированного файла"""
    file_path = UPLOAD_DIR / "output" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(str(file_path), filename=filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

