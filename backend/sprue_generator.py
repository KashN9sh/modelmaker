import trimesh
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from .mesh_thickener import MeshThickener
from .nesting import NestingAlgorithm

logger = logging.getLogger(__name__)


class SprueGenerator:
    """Класс для генерации спрусов (литниковых систем) для модели"""
    
    def __init__(self):
        self.parts_per_sprue = 15          # деталей на один спрус
        self.thickener = MeshThickener(default_thickness=1.0)  # для добавления толщины деталям
        self.frame_thickness = 2.0         # толщина рамки (высота)
        self.frame_width = 3.0             # ширина рамки
        self.frame_size = 200.0            # размер рамки 20x20 см (200x200 мм)
        self.gate_diameter = 2.0            # диаметр ворот (спрута) 2мм
        self.nesting = NestingAlgorithm(padding=2.0)  # алгоритм укладки деталей
    
    def generate(self, 
                 original_mesh: trimesh.Trimesh,
                 segments: List[Dict[str, Any]],
                 sprue_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Генерирует спрусы для разбитой модели
        
        Новая логика:
        1. Раскладывает все детали в одной плоскости (Z=0)
        2. Укладывает детали по очереди в рамки 20x20 см
        3. Если деталь не помещается - разрезает ее на части
        4. Создает гейты между рамкой и деталями, и между деталями
        
        Args:
            original_mesh: Исходная модель
            segments: Список сегментов модели
            sprue_params: Параметры спрусов
            
        Returns:
            Список спрусов, каждый содержит несколько деталей с воротами
        """
        if sprue_params is None:
            sprue_params = {}
        
        wall_thickness = sprue_params.get("wall_thickness", 1.0)
        
        logger.info(f"Генерация спрусов для {len(segments)} деталей")
        
        # 1. Добавляем толщину к деталям и раскладываем их в одной плоскости (Z=0)
        flattened_parts = self._flatten_parts_to_plane(segments, wall_thickness)
        logger.info(f"Разложено {len(flattened_parts)} деталей в плоскости")
        
        # 2. Укладываем детали в рамки 20x20 см, разрезая при необходимости
        sprue_assemblies = self._pack_parts_into_frames(flattened_parts)
        logger.info(f"Создано {len(sprue_assemblies)} спрусов")
        
        return sprue_assemblies
    
    def _flatten_parts_to_plane(self, segments: List[Dict[str, Any]], wall_thickness: float) -> List[Dict[str, Any]]:
        """
        Раскладывает все детали в одной плоскости (Z=0), как слайсеры для 3D печати
        Автоматически поворачивает детали для максимальной площади контакта с плоскостью
        
        Args:
            segments: Список сегментов модели
            wall_thickness: Толщина стенок деталей
            
        Returns:
            Список деталей, разложенных в плоскости Z=0
        """
        flattened_parts = []
        
        for seg in segments:
            part_mesh = seg["mesh"].copy()
            original_mesh = seg["mesh"].copy()  # Сохраняем исходный меш для определения внутренней стороны
            
            # Добавляем толщину к детали
            try:
                part_mesh = self.thickener.add_thickness(part_mesh, wall_thickness)
            except Exception as e:
                logger.warning(f"Не удалось добавить толщину к детали {seg['id']}: {e}")
            
            # Находим оптимальную ориентацию для максимальной площади контакта
            # Передаем исходный меш для правильного определения внутренней стороны
            part_mesh = self._orient_part_for_largest_contact_area(part_mesh, original_mesh)
            
            # Перемещаем деталь так, чтобы она лежала на плоскости Z=0
            # Находим минимальную Z координату
            min_z = part_mesh.bounds[0][2]
            
            # Смещаем деталь так, чтобы нижняя точка была на Z=0
            part_mesh.apply_translation([0, 0, -min_z])
            
            flattened_parts.append({
                "id": seg["id"],
                "mesh": part_mesh,
                "center": part_mesh.centroid.tolist(),
                "bounds": part_mesh.bounds.tolist()
            })
        
        return flattened_parts
    
    def _orient_part_for_largest_contact_area(self, mesh: trimesh.Trimesh, original_mesh: trimesh.Trimesh = None) -> trimesh.Trimesh:
        """
        Поворачивает деталь так, чтобы она лежала наибольшей площадью на плоскости
        и внутренняя оболочка была внизу
        
        Использует быстрый метод: пробует только основные ориентации (6 сторон куба)
        и выбирает ту, где площадь проекции максимальна и внутренняя сторона внизу.
        
        Args:
            mesh: Меш детали (уже с толщиной)
            original_mesh: Исходный меш (до добавления толщины) для определения внутренней стороны
            
        Returns:
            Повернутый меш
        """
        # Определяем направление внутренней стороны (используя scipy для эффективности)
        inner_direction = self._determine_inner_side(mesh, original_mesh)
        
        # Быстрый метод: пробуем только 6 основных ориентаций (стороны куба)
        test_orientations = [
            # Оригинальная ориентация (без поворота)
            None,
            # Повороты вокруг X на 90 и 270 градусов
            (np.pi/2, [1, 0, 0]),
            (3*np.pi/2, [1, 0, 0]),
            # Повороты вокруг Y на 90 и 270 градусов
            (np.pi/2, [0, 1, 0]),
            (3*np.pi/2, [0, 1, 0]),
            # Поворот вокруг X на 180 (переворот)
            (np.pi, [1, 0, 0]),
        ]
        
        best_mesh = mesh
        best_score = -1.0
        
        # Вычисляем оценку для каждой ориентации
        for rotation_params in test_orientations:
            try:
                if rotation_params is None:
                    # Без поворота
                    test_mesh = mesh
                    rotated_inner_direction = inner_direction
                else:
                    angle, axis = rotation_params
                    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                    test_mesh = mesh.copy()
                    test_mesh.apply_transform(rotation_matrix)
                    # Поворачиваем направление внутренней стороны (используем только вращательную часть 3x3)
                    rotation_3x3 = rotation_matrix[:3, :3]
                    rotated_inner_direction = rotation_3x3 @ inner_direction
                    rotated_inner_direction = rotated_inner_direction / np.linalg.norm(rotated_inner_direction)
                
                # Быстро вычисляем площадь проекции через bounding box
                bounds = test_mesh.bounds
                projection_area = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
                
                # Проверяем, направлена ли внутренняя сторона вниз (Z отрицательное)
                is_inner_down = rotated_inner_direction[2] < -0.3  # Порог для определения "вниз"
                
                # Оценка: площадь проекции + бонус если внутренняя сторона внизу
                score = projection_area
                if is_inner_down:
                    score *= 1.15  # Бонус 15% за правильную ориентацию внутренней стороны
                
                if score > best_score:
                    best_score = score
                    if rotation_params is None:
                        best_mesh = mesh
                    else:
                        best_mesh = test_mesh
                        
            except Exception as e:
                logger.debug(f"Ошибка при повороте детали: {e}")
                continue
        
        # Если нашли лучшую ориентацию, возвращаем повернутый меш
        if best_mesh is not mesh:
            bounds = best_mesh.bounds
            projection_area = (bounds[1][0] - bounds[0][0]) * (bounds[1][1] - bounds[0][1])
            logger.debug(f"Деталь повернута, площадь проекции: {projection_area:.2f} мм²")
            return best_mesh
        else:
            return mesh
    
    def _determine_inner_side(self, mesh: trimesh.Trimesh, original_mesh: trimesh.Trimesh = None) -> np.ndarray:
        """
        Определяет направление внутренней стороны детали (оболочки)
        
        Внутренняя сторона оболочки - это та, где была исходная модель.
        Использует scipy.spatial.cKDTree для быстрого поиска ближайших точек.
        
        Args:
            mesh: Меш детали (оболочка с толщиной)
            original_mesh: Исходный меш (до добавления толщины)
            
        Returns:
            Направление внутренней стороны (вектор нормали, указывающий вниз для внутренней стороны)
        """
        try:
            from scipy.spatial import cKDTree
            
            if original_mesh is None:
                # Если нет исходного меша, используем направление по умолчанию
                return np.array([0, 0, -1])
            
            # Используем scipy для быстрого поиска ближайших точек
            tree = cKDTree(original_mesh.vertices)
            
            # Получаем центры граней оболочки
            face_centers = mesh.triangles_center
            face_normals = mesh.face_normals
            
            # Находим ближайшие точки исходного меша для каждой грани
            distances, closest_indices = tree.query(face_centers)
            closest_points = original_mesh.vertices[closest_indices]
            
            # Вычисляем векторы от ближайших точек исходного меша к центрам граней оболочки
            vectors_from_surface = face_centers - closest_points
            
            # Нормализуем векторы
            norms = np.linalg.norm(vectors_from_surface, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors_from_surface_normalized = vectors_from_surface / norms
            
            # Внутренние грани: нормали направлены К исходному мешу (положительное скалярное произведение)
            # Но из-за того, как создается оболочка, внутренние грани могут иметь инвертированные нормали
            # Поэтому проверяем расстояние - внутренние грани должны быть ближе к исходному мешу
            dot_products = np.sum(face_normals * vectors_from_surface_normalized, axis=1)
            
            # Внутренние грани: близко к поверхности И нормали направлены к поверхности
            # Используем порог расстояния (нижние 50% по расстоянию)
            distance_threshold = np.percentile(distances, 50)
            inner_faces = (distances < distance_threshold) & (dot_products > -0.3)
            
            if np.sum(inner_faces) < len(face_normals) * 0.1:
                # Если слишком мало внутренних граней, используем более мягкий критерий
                inner_faces = distances < distance_threshold
            
            if np.sum(inner_faces) > 0:
                # Вычисляем среднее направление нормалей внутренних граней
                inner_normals = face_normals[inner_faces]
                
                # Внутренние грани должны иметь нормали, направленные к исходному мешу
                # Но для определения "вниз" нам нужно направление, противоположное нормалям
                inner_direction = -np.mean(inner_normals, axis=0)
                
                # Нормализуем
                norm = np.linalg.norm(inner_direction)
                if norm > 1e-6:
                    inner_direction = inner_direction / norm
                else:
                    inner_direction = np.array([0, 0, -1])
                
                # Внутренняя сторона должна указывать вниз (Z отрицательное)
                if inner_direction[2] > 0:
                    inner_direction = -inner_direction
                
                logger.debug(f"Определена внутренняя сторона: {inner_direction}, внутренних граней: {np.sum(inner_faces)}/{len(face_normals)}")
                return inner_direction
            else:
                logger.debug("Не удалось определить внутренние грани, используем направление по умолчанию")
                return np.array([0, 0, -1])
                
        except Exception as e:
            logger.debug(f"Ошибка при определении внутренней стороны: {e}")
            return np.array([0, 0, -1])
    
    
    def _pack_parts_into_frames(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Укладывает детали в рамки 20x20 см, разрезая при необходимости
        
        Args:
            parts: Список деталей, разложенных в плоскости
            
        Returns:
            Список спрусов с уложенными деталями
        """
        sprue_assemblies = []
        remaining_parts = parts.copy()
        frame_id = 0
        
        # Сортируем детали по размеру (большие сначала)
        remaining_parts.sort(key=lambda p: np.linalg.norm(np.array(p["bounds"][1]) - np.array(p["bounds"][0])), reverse=True)
        
        while len(remaining_parts) > 0:
            logger.info(f"Создание рамки {frame_id + 1}, осталось деталей: {len(remaining_parts)}")
            
            # Создаем новую рамку
            frame_parts = []
            frame_bounds = {
                "min_x": -self.frame_size / 2,
                "max_x": self.frame_size / 2,
                "min_y": -self.frame_size / 2,
                "max_y": self.frame_size / 2
            }
            
            # Укладываем детали в рамку
            parts_to_process = remaining_parts.copy()
            remaining_parts = []
            
            for part in parts_to_process:
                # Проверяем, помещается ли деталь в рамку
                bounds = np.array(part["bounds"])
                part_width = bounds[1][0] - bounds[0][0]
                part_height = bounds[1][1] - bounds[0][1]
                
                if part_width <= self.frame_size and part_height <= self.frame_size:
                    # Деталь помещается - пытаемся разместить
                    placed = self._try_place_part_in_frame(part, frame_parts, frame_bounds)
                    if not placed:
                        # Не поместилась - оставляем для следующей рамки
                        remaining_parts.append(part)
                else:
                    # Деталь слишком большая - разрезаем
                    logger.info(f"Деталь {part['id']} слишком большая ({part_width:.1f}x{part_height:.1f}мм), разрезаем...")
                    cut_parts = self._cut_part_to_fit(part, self.frame_size)
                    remaining_parts.extend(cut_parts)
            
            # Если в рамке есть детали, создаем спрус
            if len(frame_parts) > 0:
                sprue_assembly = self._create_sprue_with_gates(frame_parts, frame_id)
                if sprue_assembly is not None:
                    sprue_assemblies.append(sprue_assembly)
                frame_id += 1
            else:
                # Если рамка пустая, но есть детали - значит все детали слишком большие
                # Разрезаем первую деталь и продолжаем
                if len(remaining_parts) > 0:
                    part = remaining_parts.pop(0)
                    cut_parts = self._cut_part_to_fit(part, self.frame_size)
                    remaining_parts.extend(cut_parts)
        
        return sprue_assemblies
    
    def _try_place_part_in_frame(self, part: Dict[str, Any], frame_parts: List[Dict[str, Any]], frame_bounds: Dict[str, float]) -> bool:
        """
        Пытается разместить деталь в рамке
        
        Args:
            part: Деталь для размещения
            frame_parts: Уже размещенные детали в рамке
            frame_bounds: Границы рамки
            
        Returns:
            True если деталь размещена, False иначе
        """
        bounds = np.array(part["bounds"])
        part_width = bounds[1][0] - bounds[0][0]
        part_height = bounds[1][1] - bounds[0][1]
        
        # Пробуем разместить деталь в разных позициях (bottom-left алгоритм)
        padding = 2.0  # Отступ между деталями
        
        # Генерируем кандидатов для размещения
        candidates = [(frame_bounds["min_x"], frame_bounds["min_y"])]  # Начальная позиция
        
        # Добавляем позиции рядом с уже размещенными деталями
        for placed_part in frame_parts:
            placed_bounds = np.array(placed_part["bounds"])
            # Справа от размещенной детали
            candidates.append((placed_bounds[1][0] + padding, placed_bounds[0][1]))
            # Сверху от размещенной детали
            candidates.append((placed_bounds[0][0], placed_bounds[1][1] + padding))
        
        # Пробуем разместить в каждой позиции
        for x, y in candidates:
            # Проверяем, помещается ли деталь в этой позиции
            if (x + part_width <= frame_bounds["max_x"] and 
                y + part_height <= frame_bounds["max_y"]):
                
                # Проверяем пересечения с уже размещенными деталями
                overlaps = False
                new_bounds = np.array([[x, y, bounds[0][2]], [x + part_width, y + part_height, bounds[1][2]]])
                
                for placed_part in frame_parts:
                    placed_bounds = np.array(placed_part["bounds"])
                    if self._bounds_overlap(new_bounds, placed_bounds, padding):
                        overlaps = True
                        break
                
                if not overlaps:
                    # Размещаем деталь
                    offset_x = x - bounds[0][0]
                    offset_y = y - bounds[0][1]
                    
                    part_mesh = part["mesh"].copy()
                    part_mesh.apply_translation([offset_x, offset_y, 0])
                    
                    frame_parts.append({
                        "id": part["id"],
                        "mesh": part_mesh,
                        "center": (np.array(part["center"]) + [offset_x, offset_y, 0]).tolist(),
                        "bounds": part_mesh.bounds.tolist()
                    })
                    return True
        
        return False
    
    def _bounds_overlap(self, bounds1: np.ndarray, bounds2: np.ndarray, padding: float) -> bool:
        """Проверяет, пересекаются ли два bounding box с учетом отступа"""
        return not (bounds1[1][0] + padding <= bounds2[0][0] or
                   bounds1[0][0] >= bounds2[1][0] + padding or
                   bounds1[1][1] + padding <= bounds2[0][1] or
                   bounds1[0][1] >= bounds2[1][1] + padding)
    
    def _cut_part_to_fit(self, part: Dict[str, Any], max_size: float) -> List[Dict[str, Any]]:
        """
        Разрезает деталь на части, чтобы они поместились в рамку
        
        Args:
            part: Деталь для разрезания
            max_size: Максимальный размер части (размер рамки)
            
        Returns:
            Список частей детали
        """
        mesh = part["mesh"]
        bounds = mesh.bounds
        size_x = bounds[1][0] - bounds[0][0]
        size_y = bounds[1][1] - bounds[0][1]
        
        # Определяем, по какой оси резать
        cuts_x = max(1, int(np.ceil(size_x / max_size)))
        cuts_y = max(1, int(np.ceil(size_y / max_size)))
        
        cut_parts = []
        part_id_base = part["id"]
        
        for i in range(cuts_x):
            for j in range(cuts_y):
                # Вычисляем границы части
                x_min = bounds[0][0] + (size_x * i / cuts_x)
                x_max = bounds[0][0] + (size_x * (i + 1) / cuts_x)
                y_min = bounds[0][1] + (size_y * j / cuts_y)
                y_max = bounds[0][1] + (size_y * (j + 1) / cuts_y)
                
                # Извлекаем часть меша
                mask = ((mesh.vertices[:, 0] >= x_min) & (mesh.vertices[:, 0] < x_max) &
                       (mesh.vertices[:, 1] >= y_min) & (mesh.vertices[:, 1] < y_max))
                
                if np.sum(mask) > 0:
                    # Находим грани, все вершины которых в этой части
                    vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
                    vertex_mask[mask] = True
                    face_mask = vertex_mask[mesh.faces].all(axis=1)
                    
                    if np.sum(face_mask) > 0:
                        part_faces = mesh.faces[face_mask]
                        used_vertices = np.unique(part_faces.flatten())
                        vertex_map = {old: new for new, old in enumerate(used_vertices)}
                        new_faces = np.array([[vertex_map[v] for v in face] for face in part_faces])
                        new_vertices = mesh.vertices[used_vertices]
                        
                        part_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                        
                        cut_parts.append({
                            "id": f"{part_id_base}_{i}_{j}",
                            "mesh": part_mesh,
                            "center": part_mesh.centroid.tolist(),
                            "bounds": part_mesh.bounds.tolist()
                        })
        
        logger.info(f"Деталь {part['id']} разрезана на {len(cut_parts)} частей")
        return cut_parts if len(cut_parts) > 0 else [part]
    
    def _create_sprue_with_gates(self, frame_parts: List[Dict[str, Any]], frame_id: int) -> Dict[str, Any]:
        """
        Создает спрус с гейтами между рамкой и деталями, и между деталями
        
        Args:
            frame_parts: Детали в рамке
            frame_id: ID рамки
            
        Returns:
            Словарь с информацией о спрусе
        """
        # Создаем рамку
        frame_mesh = self._create_frame_around_parts(frame_parts)
        
        # Создаем гейты
        gates = []
        sprue_meshes = [frame_mesh]
        
        # Добавляем детали и создаем гейты от рамки к деталям
        for part in frame_parts:
            sprue_meshes.append(part["mesh"])
            
            # Гейт от рамки к детали
            gate = self._create_gate_to_part(part, frame_mesh)
            if gate is not None:
                gates.append(gate)
                sprue_meshes.append(gate)
        
        # Создаем гейты между деталями
        for i in range(len(frame_parts)):
            for j in range(i + 1, len(frame_parts)):
                gate = self._create_gate_between_parts(frame_parts[i], frame_parts[j])
                if gate is not None:
                    gates.append(gate)
                    sprue_meshes.append(gate)
        
        # Объединяем все части
        try:
            complete_sprue = trimesh.util.concatenate(sprue_meshes)
            return {
                "id": frame_id,
                "mesh": complete_sprue,
                "num_parts": len(frame_parts),
                "type": "sprue"
            }
        except Exception as e:
            logger.error(f"Ошибка при объединении спруса {frame_id}: {e}")
            return None
    
    def _create_gate_between_parts(self, part1: Dict[str, Any], part2: Dict[str, Any]) -> Optional[trimesh.Trimesh]:
        """
        Создает гейт между двумя деталями
        
        Args:
            part1: Первая деталь
            part2: Вторая деталь
            
        Returns:
            Меш гейта или None
        """
        try:
            # Находим ближайшие точки на деталях
            prox1 = trimesh.proximity.ProximityQuery(part1["mesh"])
            prox2 = trimesh.proximity.ProximityQuery(part2["mesh"])
            
            center1 = np.array(part1["center"])
            center2 = np.array(part2["center"])
            
            # Находим ближайшую точку на part2 к центру part1
            closest_on_part2, _, _ = prox2.on_surface(np.array([center1]))
            if len(closest_on_part2) == 0:
                return None
            
            point2 = np.array(closest_on_part2[0])
            
            # Находим ближайшую точку на part1 к point2
            closest_on_part1, _, _ = prox1.on_surface(np.array([point2]))
            if len(closest_on_part1) == 0:
                point1 = center1
            else:
                point1 = np.array(closest_on_part1[0])
            
            # Проверяем расстояние - создаем гейт только если детали близко
            distance = np.linalg.norm(point1 - point2)
            if distance > 20.0:  # Максимум 20мм между деталями для гейта
                return None
            
            # Создаем гейт
            return self._create_gate(point1, point2, self.gate_diameter)
            
        except Exception as e:
            logger.debug(f"Не удалось создать гейт между деталями: {e}")
            return None
    
    def _group_parts_into_sprues(self, segments: List[Dict[str, Any]], parts_per_sprue: int) -> List[List[Dict[str, Any]]]:
        """
        Группирует детали на спрусы
        
        Args:
            segments: Список всех деталей
            parts_per_sprue: Количество деталей на один спрус
            
        Returns:
            Список групп деталей для каждого спруса
        """
        groups = []
        
        # Сортируем детали по размеру (большие сначала)
        sorted_segments = sorted(segments, key=lambda s: len(s["mesh"].vertices), reverse=True)
        
        # Группируем детали
        for i in range(0, len(sorted_segments), parts_per_sprue):
            group = sorted_segments[i:i + parts_per_sprue]
            groups.append(group)
        
        return groups
    
    def _create_complete_sprue(self,
                              segments: List[Dict[str, Any]],
                              sprue_id: int,
                              sprue_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Создает полный спрус: центральный ствол + детали + ворота
        
        Returns:
            Словарь с информацией о спрусе и его мешем
        """
        try:
            # Добавляем толщину к деталям
            wall_thickness = sprue_params.get("wall_thickness", 1.0)
            thickened_parts = []
            
            for seg in segments:
                part_mesh = seg["mesh"].copy()
                try:
                    part_mesh = self.thickener.add_thickness(part_mesh, wall_thickness)
                except Exception as e:
                    logger.warning(f"Не удалось добавить толщину к детали {seg['id']}: {e}")
                
                thickened_parts.append({
                    "id": seg["id"],
                    "mesh": part_mesh,
                    "center": seg["center"],
                    "bounds": seg["bounds"]
                })
            
            # Раскладываем детали на плоскости используя алгоритм nesting
            laid_out_parts = self.nesting.layout_parts(thickened_parts)
            
            # Создаем рамку вокруг деталей
            frame_mesh = self._create_frame_around_parts(laid_out_parts)
            
            # Создаем ворота от рамки к деталям
            sprue_meshes = [frame_mesh]
            
            for part in laid_out_parts:
                # Добавляем деталь
                sprue_meshes.append(part["mesh"])
                
                # Создаем ворот от рамки к детали
                gate = self._create_gate_to_part(part, frame_mesh)
                if gate is not None:
                    sprue_meshes.append(gate)
            
            # Объединяем все части спруса
            try:
                complete_sprue = trimesh.util.concatenate(sprue_meshes)
                return {
                    "id": sprue_id,
                    "mesh": complete_sprue,
                    "num_parts": len(segments),
                    "type": "sprue"
                }
            except Exception as e:
                logger.error(f"Ошибка при объединении спруса {sprue_id}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при создании спруса {sprue_id}: {e}")
            return None
    
    
    def _create_frame_around_parts(self, parts: List[Dict[str, Any]]) -> trimesh.Trimesh:
        """
        Создает рамку (frame) фиксированного размера 20x20 см вокруг деталей
        
        Args:
            parts: Список деталей, разложенных на плоскости
            
        Returns:
            Меш рамки
        """
        # Создаем рамку фиксированного размера 20x20 см (200x200 мм)
        frame_size_xy = self.frame_size  # 200 мм
        
        # Центрируем рамку в начале координат
        frame_min = np.array([-frame_size_xy / 2, -frame_size_xy / 2, -self.frame_thickness / 2])
        frame_max = np.array([frame_size_xy / 2, frame_size_xy / 2, self.frame_thickness / 2])
        
        # Создаем внешний и внутренний боксы
        outer_box = trimesh.creation.box(extents=frame_size_xy + np.array([self.frame_width * 2, self.frame_width * 2, self.frame_thickness]))
        inner_box = trimesh.creation.box(extents=frame_size_xy + np.array([0, 0, self.frame_thickness * 1.1]))
        
        # Центрируем
        outer_center = (frame_min + frame_max) / 2
        outer_box.apply_translation(outer_center - outer_box.centroid)
        inner_box.apply_translation(outer_center - inner_box.centroid)
        
        # Вычитаем внутренний бокс из внешнего (создаем рамку)
        try:
            frame = outer_box.difference(inner_box)
            if frame is None or len(frame.vertices) == 0:
                # Fallback: создаем простую рамку из 4 сторон
                return self._create_simple_frame(frame_min, frame_max)
            return frame
        except:
            # Fallback: создаем простую рамку
            return self._create_simple_frame(frame_min, frame_max)
    
    def _create_simple_frame(self, min_bounds: np.ndarray, max_bounds: np.ndarray) -> trimesh.Trimesh:
        """Создает простую рамку из 4 прямоугольных сторон фиксированного размера 20x20 см"""
        # Используем фиксированный размер рамки
        frame_size_xy = self.frame_size  # 300 мм
        center = (min_bounds + max_bounds) / 2
        
        # Создаем 4 стороны рамки
        sides = []
        
        # Верхняя и нижняя стороны
        top_bottom_size = [frame_size_xy + self.frame_width * 2, self.frame_width, self.frame_thickness]
        for y_pos in [min_bounds[1] - self.frame_width/2, max_bounds[1] + self.frame_width/2]:
            side = trimesh.creation.box(extents=top_bottom_size)
            side.apply_translation([center[0], y_pos, center[2]] - side.centroid)
            sides.append(side)
        
        # Левая и правая стороны
        left_right_size = [self.frame_width, frame_size_xy + self.frame_width * 2, self.frame_thickness]
        for x_pos in [min_bounds[0] - self.frame_width/2, max_bounds[0] + self.frame_width/2]:
            side = trimesh.creation.box(extents=left_right_size)
            side.apply_translation([x_pos, center[1], center[2]] - side.centroid)
            sides.append(side)
        
        # Объединяем все стороны
        try:
            frame = trimesh.util.concatenate(sides)
            return frame
        except:
            return sides[0] if sides else trimesh.creation.box(extents=[300, 300, 2])
    
    def _create_gate_to_part(self, part: Dict[str, Any], frame: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
        """
        Создает ворот от рамки к детали
        
        Args:
            part: Деталь
            frame: Рамка
            
        Returns:
            Меш ворот или None
        """
        try:
            # Убеждаемся, что center и bounds - numpy массивы
            part_center = np.array(part["center"])
            part_mesh = part["mesh"]
            
            # Используем фиксированный диаметр ворот 2мм
            gate_diameter = self.gate_diameter
            
            # Находим ближайшую точку на рамке к центру детали
            prox = trimesh.proximity.ProximityQuery(frame)
            closest_on_frame, distance, _ = prox.on_surface(np.array([part_center]))
            
            if len(closest_on_frame) == 0:
                return None
            
            frame_point = np.array(closest_on_frame[0])
            
            # Находим ближайшую точку на детали к точке на рамке
            part_prox = trimesh.proximity.ProximityQuery(part_mesh)
            closest_on_part, _, _ = part_prox.on_surface(np.array([frame_point]))
            
            if len(closest_on_part) == 0:
                part_point = part_center
            else:
                part_point = np.array(closest_on_part[0])
            
            # Создаем ворот
            return self._create_gate(frame_point, part_point, gate_diameter)
            
        except Exception as e:
            logger.warning(f"Ошибка при создании ворот к детали {part['id']}: {e}")
            return None
    
    def _position_part_on_sprue(self,
                               part: Dict[str, Any],
                               runner: trimesh.Trimesh,
                               gate_diameter: float,
                               gate_length: float) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """
        Располагает деталь на спрусе и создает ворот
        
        Returns:
            (позиционированная деталь, ворот)
        """
        try:
            part_mesh = part["mesh"].copy()
            part_center = np.array(part["center"])
            
            # Находим ближайшую точку на стволе к центру детали
            prox = trimesh.proximity.ProximityQuery(runner)
            closest_on_runner, distance, _ = prox.on_surface(np.array([part_center]))
            
            if len(closest_on_runner) == 0:
                return part_mesh, None
            
            sprue_point = np.array(closest_on_runner[0])
            
            # Находим точку на детали для ворот
            part_prox = trimesh.proximity.ProximityQuery(part_mesh)
            closest_on_part, _, _ = part_prox.on_surface(np.array([sprue_point]))
            
            if len(closest_on_part) == 0:
                part_gate_point = part_center
            else:
                part_gate_point = np.array(closest_on_part[0])
            
            # Создаем ворот
            gate = self._create_gate(sprue_point, part_gate_point, gate_diameter)
            
            # Позиционируем деталь (она уже в правильной позиции, просто возвращаем)
            return part_mesh, gate
            
        except Exception as e:
            logger.warning(f"Ошибка при позиционировании детали {part['id']}: {e}")
            return part["mesh"], None
    
    def _create_gate(self, sprue_point: np.ndarray, part_point: np.ndarray, gate_diameter: float) -> Optional[trimesh.Trimesh]:
        """Создает ворот между точкой на спрусе и точкой на детали"""
        try:
            # Убеждаемся, что точки - numpy массивы
            sprue_point = np.array(sprue_point)
            part_point = np.array(part_point)
            
            gate_direction = part_point - sprue_point
            gate_distance = np.linalg.norm(gate_direction)
            
            if gate_distance < 0.5:
                return None
            
            gate_direction_normalized = gate_direction / gate_distance
            
            # Создаем цилиндр для ворот
            gate = trimesh.creation.cylinder(
                radius=gate_diameter / 2,
                height=gate_distance,
                sections=8
            )
            
            # Поворачиваем и перемещаем ворот
            z_axis = np.array([0, 0, 1])
            if np.abs(np.dot(z_axis, gate_direction_normalized) - 1.0) > 1e-6:
                rotation_axis = np.cross(z_axis, gate_direction_normalized)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    angle = np.arccos(np.clip(np.dot(z_axis, gate_direction_normalized), -1, 1))
                    rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
                else:
                    rotation_matrix = np.eye(4)
            else:
                rotation_matrix = np.eye(4)
            
            gate_center = (sprue_point + part_point) / 2
            translation = trimesh.transformations.translation_matrix(gate_center)
            transform = translation @ rotation_matrix
            
            gate.apply_transform(transform)
            
            return gate
            
        except Exception as e:
            logger.warning(f"Ошибка при создании ворот: {e}")
            return None
    
