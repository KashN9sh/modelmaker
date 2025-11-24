import trimesh
import numpy as np
import os
from typing import Dict, Any


class STLParser:
    """Класс для загрузки и парсинга STL файлов"""
    
    def load(self, filepath: str) -> trimesh.Trimesh:
        """
        Загружает STL файл
        
        Args:
            filepath: Путь к STL файлу
            
        Returns:
            Trimesh объект
        """
        try:
            # Проверяем существование файла
            if not os.path.exists(filepath):
                raise ValueError(f"Файл не найден: {filepath}")
            
            # Проверяем размер файла
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("Файл пуст")
            
            # Загружаем меш
            mesh = trimesh.load(filepath, file_type='stl')
            
            # Если trimesh вернул None
            if mesh is None:
                raise ValueError("Не удалось загрузить STL файл. Возможно, файл поврежден или имеет неподдерживаемый формат.")
            
            # Если это сцена, берем первый меш
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    raise ValueError("STL файл не содержит геометрии")
                mesh = list(mesh.geometry.values())[0]
            
            # Убеждаемся, что это триангулированная сетка
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Файл не содержит триангулированную сетку. Получен тип: {type(mesh)}")
            
            # Проверяем, что меш не пустой
            if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                raise ValueError("Загруженный меш пуст (нет вершин или граней)")
            
            # Очистка и исправление меша
            try:
                mesh.fix_normals()
            except:
                pass  # Игнорируем ошибки исправления нормалей
            
            try:
                mesh.remove_duplicate_faces()
            except:
                pass  # Игнорируем ошибки удаления дубликатов
            
            try:
                mesh.remove_unreferenced_vertices()
            except:
                pass  # Игнорируем ошибки удаления неиспользуемых вершин
            
            return mesh
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке STL: {str(e)}")
    
    def get_mesh_info(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Получает информацию о меше
        
        Args:
            mesh: Trimesh объект
            
        Returns:
            Словарь с информацией о меше
        """
        return {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "volume": float(mesh.volume) if mesh.is_volume else None,
            "bounds": mesh.bounds.tolist(),
            "center": mesh.centroid.tolist(),
            "is_watertight": mesh.is_watertight,
            "is_winding_consistent": mesh.is_winding_consistent
        }
    
    def save(self, mesh: trimesh.Trimesh, filepath: str) -> None:
        """
        Сохраняет меш в STL файл
        
        Args:
            mesh: Trimesh объект
            filepath: Путь для сохранения
        """
        mesh.export(filepath)

