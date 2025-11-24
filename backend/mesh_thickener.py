import trimesh
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MeshThickener:
    """Класс для добавления толщины к мешам (создание оболочки)"""
    
    def __init__(self, default_thickness: float = 1.0):
        """
        Args:
            default_thickness: Толщина стенок по умолчанию (мм)
        """
        self.default_thickness = default_thickness
    
    def add_thickness(self, mesh: trimesh.Trimesh, thickness: Optional[float] = None) -> trimesh.Trimesh:
        """
        Добавляет толщину к мешу, создавая оболочку
        
        Args:
            mesh: Исходный меш
            thickness: Толщина стенок (если None, используется default_thickness)
            
        Returns:
            Меш с толщиной
        """
        if thickness is None:
            thickness = self.default_thickness
        
        if thickness <= 0:
            return mesh
        
        try:
            # Используем метод создания оболочки через нормали
            # buffer() не существует в trimesh, используем смещение по нормалям
            return self._create_shell_by_normals(mesh, thickness)
            
        except Exception as e:
            logger.error(f"Ошибка при добавлении толщины: {e}")
            return mesh
    
    def _create_shell_by_normals(self, mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
        """
        Создает оболочку, смещая грани по нормалям
        
        Args:
            mesh: Исходный меш
            thickness: Толщина стенок
            
        Returns:
            Меш с оболочкой
        """
        try:
            # Копируем меш
            mesh_copy = mesh.copy()
            
            # Вычисляем нормали вершин
            mesh_copy.vertex_normals
            
            # Создаем две копии меша - внутреннюю и внешнюю
            inner = mesh_copy.copy()
            outer = mesh_copy.copy()
            
            # Смещаем вершины по нормалям
            if hasattr(mesh_copy, 'vertex_normals') and mesh_copy.vertex_normals is not None:
                normals = mesh_copy.vertex_normals
            else:
                # Вычисляем нормали вершин
                normals = np.zeros_like(mesh_copy.vertices)
                for i, vertex in enumerate(mesh_copy.vertices):
                    # Находим грани, содержащие эту вершину
                    face_indices = np.where(np.any(mesh_copy.faces == i, axis=1))[0]
                    if len(face_indices) > 0:
                        # Вычисляем среднюю нормаль граней
                        face_normals = mesh_copy.face_normals[face_indices]
                        normals[i] = face_normals.mean(axis=0)
                        normals[i] = normals[i] / np.linalg.norm(normals[i])
                    else:
                        normals[i] = np.array([0, 0, 1])
            
            # Нормализуем нормали
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normals = normals / norms
            
            # Смещаем вершины
            offset = thickness / 2
            inner.vertices = mesh_copy.vertices - normals * offset
            outer.vertices = mesh_copy.vertices + normals * offset
            
            # Создаем боковые грани, соединяющие внутреннюю и внешнюю поверхности
            shell_faces = []
            shell_vertices = []
            
            # Добавляем вершины внутренней поверхности
            inner_start = len(shell_vertices)
            shell_vertices.extend(inner.vertices)
            
            # Добавляем вершины внешней поверхности
            outer_start = len(shell_vertices)
            shell_vertices.extend(outer.vertices)
            
            # Создаем боковые грани для каждого ребра исходного меша
            for face in mesh_copy.faces:
                # Для каждой грани создаем боковые грани
                for i in range(3):
                    v1_inner = inner_start + face[i]
                    v2_inner = inner_start + face[(i + 1) % 3]
                    v1_outer = outer_start + face[i]
                    v2_outer = outer_start + face[(i + 1) % 3]
                    
                    # Создаем две треугольные грани для боковой поверхности
                    shell_faces.append([v1_inner, v2_inner, v2_outer])
                    shell_faces.append([v1_inner, v2_outer, v1_outer])
            
            # Добавляем грани внутренней и внешней поверхностей
            # Внутренняя поверхность (инвертированные нормали)
            for face in inner.faces:
                shell_faces.append([inner_start + face[0], inner_start + face[2], inner_start + face[1]])
            
            # Внешняя поверхность
            for face in outer.faces:
                shell_faces.append([outer_start + face[0], outer_start + face[1], outer_start + face[2]])
            
            # Создаем новый меш
            shell_vertices = np.array(shell_vertices)
            shell_faces = np.array(shell_faces)
            
            shell_mesh = trimesh.Trimesh(vertices=shell_vertices, faces=shell_faces)
            
            # Очищаем меш
            shell_mesh.remove_duplicate_faces()
            shell_mesh.remove_unreferenced_vertices()
            
            return shell_mesh
            
        except Exception as e:
            logger.error(f"Ошибка при создании оболочки через нормали: {e}")
            # Fallback: просто возвращаем исходный меш
            return mesh
    
    def _create_shell_simple(self, mesh: trimesh.Trimesh, thickness: float) -> trimesh.Trimesh:
        """
        Упрощенный метод создания оболочки через смещение по нормалям
        
        Args:
            mesh: Исходный меш
            thickness: Толщина стенок
            
        Returns:
            Меш с оболочкой
        """
        try:
            # Простой метод: создаем две копии меша и соединяем их боковыми гранями
            mesh_copy = mesh.copy()
            
            # Вычисляем нормали граней
            if not hasattr(mesh_copy, 'face_normals') or mesh_copy.face_normals is None:
                mesh_copy.fix_normals()
            
            # Создаем внутреннюю и внешнюю поверхности
            inner = mesh_copy.copy()
            outer = mesh_copy.copy()
            
            # Смещаем по нормалям граней (упрощенный подход)
            offset = thickness / 2
            
            # Для каждой грани смещаем её вершины
            for i, face in enumerate(mesh_copy.faces):
                normal = mesh_copy.face_normals[i]
                
                # Смещаем вершины грани
                for j in range(3):
                    vertex_idx = face[j]
                    inner.vertices[vertex_idx] = mesh_copy.vertices[vertex_idx] - normal * offset
                    outer.vertices[vertex_idx] = mesh_copy.vertices[vertex_idx] + normal * offset
            
            # Объединяем внутреннюю и внешнюю поверхности
            # Это упрощенный подход - в реальности нужно создавать боковые грани правильно
            try:
                combined = trimesh.util.concatenate([inner, outer])
                return combined
            except:
                return outer  # Возвращаем хотя бы внешнюю поверхность
                
        except Exception as e:
            logger.error(f"Ошибка в упрощенном методе: {e}")
            return mesh

