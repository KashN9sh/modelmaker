import trimesh
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NestingAlgorithm:
    """Алгоритм автоматической укладки деталей на плоскость (nesting)"""
    
    def __init__(self, padding: float = 2.0):
        """
        Args:
            padding: Отступ между деталями (мм)
        """
        self.padding = padding
    
    def layout_parts(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Раскладывает детали на плоскости оптимальным образом
        
        Args:
            parts: Список деталей с мешами
            
        Returns:
            Список деталей с обновленными позициями
        """
        if len(parts) == 0:
            return []
        
        # Вычисляем проекции деталей на плоскость XY
        part_profiles = []
        for i, part in enumerate(parts):
            profile = self._get_part_profile(part["mesh"])
            part_profiles.append({
                "id": part["id"],
                "profile": profile,
                "mesh": part["mesh"],
                "center": part["center"],
                "bounds": part["bounds"]
            })
        
        # Используем алгоритм bottom-left fill для укладки
        laid_out = self._bottom_left_fill(part_profiles)
        
        return laid_out
    
    def _get_part_profile(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Получает профиль детали (проекцию на плоскость XY)
        
        Args:
            mesh: Меш детали
            
        Returns:
            Словарь с информацией о профиле
        """
        bounds = mesh.bounds
        min_bounds = bounds[0]
        max_bounds = bounds[1]
        
        # Размеры в плоскости XY
        width = max_bounds[0] - min_bounds[0]
        height = max_bounds[1] - min_bounds[1]
        depth = max_bounds[2] - min_bounds[2]
        
        # Центр в плоскости XY
        center_x = (min_bounds[0] + max_bounds[0]) / 2
        center_y = (min_bounds[1] + max_bounds[1]) / 2
        
        return {
            "width": width,
            "height": height,
            "depth": depth,
            "center_x": center_x,
            "center_y": center_y,
            "min_x": min_bounds[0],
            "min_y": min_bounds[1],
            "max_x": max_bounds[0],
            "max_y": max_bounds[1],
            "area": width * height
        }
    
    def _bottom_left_fill(self, part_profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Алгоритм bottom-left fill для укладки деталей
        
        Args:
            part_profiles: Список профилей деталей
            
        Returns:
            Список деталей с новыми позициями
        """
        # Сортируем детали по площади (большие сначала) для лучшей укладки
        sorted_parts = sorted(part_profiles, key=lambda p: p["profile"]["area"], reverse=True)
        
        # Позиции размещенных деталей
        placed_parts = []
        occupied_rectangles = []  # Список занятых прямоугольников (x, y, width, height)
        
        for part in sorted_parts:
            profile = part["profile"]
            width = profile["width"] + self.padding
            height = profile["height"] + self.padding
            
            # Находим позицию для размещения (bottom-left)
            position = self._find_best_position(width, height, occupied_rectangles)
            
            # Вычисляем смещение от текущей позиции детали
            current_center_x = profile["center_x"]
            current_center_y = profile["center_y"]
            new_center_x = position[0] + width / 2 - self.padding / 2
            new_center_y = position[1] + height / 2 - self.padding / 2
            
            offset_x = new_center_x - current_center_x
            offset_y = new_center_y - current_center_y
            
            # Перемещаем деталь
            part_mesh = part["mesh"].copy()
            part_mesh.apply_translation([offset_x, offset_y, 0])
            
            # Обновляем информацию о детали
            new_center = np.array(part["center"])
            new_center[0] += offset_x
            new_center[1] += offset_y
            
            new_bounds = part_mesh.bounds
            
            # Добавляем в список размещенных
            placed_parts.append({
                "id": part["id"],
                "mesh": part_mesh,
                "center": new_center.tolist() if isinstance(new_center, np.ndarray) else new_center,
                "bounds": new_bounds.tolist() if isinstance(new_bounds, np.ndarray) else new_bounds
            })
            
            # Добавляем занятый прямоугольник
            occupied_rectangles.append({
                "x": position[0],
                "y": position[1],
                "width": width,
                "height": height
            })
        
        return placed_parts
    
    def _find_best_position(self, width: float, height: float, occupied: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        Находит лучшую позицию для размещения детали (bottom-left алгоритм)
        
        Args:
            width: Ширина детали
            height: Высота детали
            occupied: Список занятых прямоугольников
            
        Returns:
            Позиция (x, y) для размещения
        """
        if len(occupied) == 0:
            return (0.0, 0.0)
        
        # Находим все возможные позиции (кандидаты)
        candidates = []
        
        # Позиции рядом с уже размещенными деталями
        for rect in occupied:
            # Справа от прямоугольника
            candidates.append((rect["x"] + rect["width"], rect["y"]))
            # Сверху от прямоугольника
            candidates.append((rect["x"], rect["y"] + rect["height"]))
            # Внизу от прямоугольника (если помещается)
            if rect["y"] >= height:
                candidates.append((rect["x"], rect["y"] - height))
            # Слева от прямоугольника (если помещается)
            if rect["x"] >= width:
                candidates.append((rect["x"] - width, rect["y"]))
        
        # Добавляем позицию (0, 0)
        candidates.append((0.0, 0.0))
        
        # Фильтруем кандидатов - проверяем, не пересекаются ли они с занятыми областями
        valid_candidates = []
        for x, y in candidates:
            if self._can_place_at(x, y, width, height, occupied):
                valid_candidates.append((x, y))
        
        if len(valid_candidates) == 0:
            # Если нет валидных позиций, размещаем справа от всех
            max_x = max([r["x"] + r["width"] for r in occupied])
            return (max_x, 0.0)
        
        # Выбираем позицию с минимальным y, затем минимальным x (bottom-left)
        valid_candidates.sort(key=lambda pos: (pos[1], pos[0]))
        return valid_candidates[0]
    
    def _can_place_at(self, x: float, y: float, width: float, height: float, occupied: List[Dict[str, Any]]) -> bool:
        """
        Проверяет, можно ли разместить деталь в указанной позиции
        
        Args:
            x, y: Позиция
            width, height: Размеры детали
            occupied: Список занятых прямоугольников
            
        Returns:
            True если можно разместить, False иначе
        """
        # Проверяем, что позиция не отрицательная
        if x < 0 or y < 0:
            return False
        
        # Проверяем пересечения с занятыми областями
        new_rect = {"x": x, "y": y, "width": width, "height": height}
        
        for rect in occupied:
            if self._rectangles_intersect(new_rect, rect):
                return False
        
        return True
    
    def _rectangles_intersect(self, rect1: Dict[str, float], rect2: Dict[str, float]) -> bool:
        """
        Проверяет пересечение двух прямоугольников
        
        Args:
            rect1, rect2: Прямоугольники с полями x, y, width, height
            
        Returns:
            True если пересекаются, False иначе
        """
        return not (
            rect1["x"] + rect1["width"] <= rect2["x"] or
            rect2["x"] + rect2["width"] <= rect1["x"] or
            rect1["y"] + rect1["height"] <= rect2["y"] or
            rect2["y"] + rect2["height"] <= rect1["y"]
        )
    
    def _optimize_rotation(self, part: Dict[str, Any]) -> Dict[str, Any]:
        """
        Оптимизирует ориентацию детали для лучшей укладки
        
        Args:
            part: Деталь с мешем
            
        Returns:
            Деталь с оптимизированной ориентацией
        """
        mesh = part["mesh"]
        profile = self._get_part_profile(mesh)
        
        # Пробуем повернуть на 90, 180, 270 градусов
        best_rotation = 0
        best_area = profile["area"]
        
        for angle in [90, 180, 270]:
            test_mesh = mesh.copy()
            test_mesh.apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(angle), [0, 0, 1]
            ))
            test_profile = self._get_part_profile(test_mesh)
            
            # Выбираем ориентацию с минимальной площадью проекции
            if test_profile["area"] < best_area:
                best_area = test_profile["area"]
                best_rotation = angle
        
        # Применяем лучшую ориентацию
        if best_rotation > 0:
            part["mesh"].apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(best_rotation), [0, 0, 1]
            ))
        
        return part

