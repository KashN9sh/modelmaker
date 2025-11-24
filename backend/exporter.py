import trimesh
from pathlib import Path
from typing import List, Dict, Any
import os


class STLExporter:
    """Класс для экспорта разбитых деталей в STL файлы"""
    
    def export_parts(self, 
                    parts: List[Dict[str, Any]], 
                    output_dir: str,
                    base_filename: str = "part") -> List[str]:
        """
        Экспортирует части в отдельные STL файлы
        
        Args:
            parts: Список частей со спрусами (теперь это спрусы, а не отдельные детали)
            output_dir: Директория для сохранения
            base_filename: Базовое имя файла
            
        Returns:
            Список путей к экспортированным файлам
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = []
        
        for part in parts:
            # Проверяем тип - спрус или отдельная деталь
            if part.get("type") == "sprue":
                sprue_id = part["id"]
                mesh = part["mesh"]
                
                # Формируем имя файла для спруса
                filename = f"{base_filename}_sprue_{sprue_id:02d}.stl"
                filepath = output_path / filename
                
                try:
                    # Экспортируем весь спрус в STL
                    mesh.export(str(filepath))
                    exported_files.append(filename)
                except Exception as e:
                    print(f"Ошибка при экспорте спруса {sprue_id}: {str(e)}")
                    continue
            else:
                # Старый формат - отдельные детали
                part_id = part["id"]
                mesh = part["mesh"]
                
                filename = f"{base_filename}_{part_id:02d}.stl"
                filepath = output_path / filename
                
                try:
                    mesh.export(str(filepath))
                    exported_files.append(filename)
                except Exception as e:
                    print(f"Ошибка при экспорте части {part_id}: {str(e)}")
                    continue
        
        return exported_files
    
    def export_single(self, mesh: trimesh.Trimesh, filepath: str) -> bool:
        """
        Экспортирует один меш в STL файл
        
        Args:
            mesh: Меш для экспорта
            filepath: Путь к файлу
            
        Returns:
            True если успешно, False иначе
        """
        try:
            mesh.export(filepath)
            return True
        except Exception as e:
            print(f"Ошибка при экспорте: {str(e)}")
            return False

