import trimesh
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.cluster.hierarchy import linkage, fcluster
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelSegmenter:
    """Класс для сегментации 3D модели на логические части"""
    
    def __init__(self, min_component_size: int = 100, max_segments_for_small_parts: int = 1):
        """
        Args:
            min_component_size: Минимальный размер компонента для включения
            max_segments_for_small_parts: Максимальное количество сегментов для маленьких деталей (1 = не разбивать)
        """
        self.min_component_size = min_component_size
        self.max_segments_for_small_parts = max_segments_for_small_parts
    
    def segment(self, mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
        """
        Разбивает модель на логические части
        
        Args:
            mesh: Исходная триангулированная сетка
            
        Returns:
            Список сегментов с информацией о каждой части
        """
        # Сначала разбиваем на связанные компоненты
        components = self._get_connected_components(mesh)
        
        # Если модель состоит из нескольких несвязанных частей
        if len(components) > 1:
            segments = []
            
            # Вычисляем размеры всех компонентов для определения "маленьких"
            # Используем и количество граней, и физический размер
            component_info = []
            for c in components:
                face_count = len(c.faces)
                physical_size = np.linalg.norm(c.bounds[1] - c.bounds[0])
                component_info.append({
                    "component": c,
                    "face_count": face_count,
                    "physical_size": physical_size
                })
            
            if len(component_info) > 0:
                max_face_count = max([info["face_count"] for info in component_info])
                max_physical_size = max([info["physical_size"] for info in component_info])
                # Пороги для трех категорий (как в _get_connected_components)
                small_threshold_faces = max(max_face_count * 0.1, self.min_component_size)
                small_threshold_size = max(max_physical_size * 0.1, 8.0)  # Минимум 8мм
                medium_threshold_faces = max(max_face_count * 0.3, self.min_component_size * 5)
                medium_threshold_size = max(max_physical_size * 0.3, 25.0)  # Минимум 25мм
            else:
                small_threshold_faces = self.min_component_size
                small_threshold_size = 8.0
                medium_threshold_faces = self.min_component_size * 5
                medium_threshold_size = 25.0
            
            for i, info in enumerate(component_info):
                component = info["component"]
                face_count = info["face_count"]
                physical_size = info["physical_size"]
                
                # Фильтруем только достаточно большие компоненты
                if face_count >= max(10, self.min_component_size // 10):
                    # Определяем категорию детали
                    is_small = (face_count < small_threshold_faces) and (physical_size < small_threshold_size)
                    is_medium = (face_count >= small_threshold_faces and face_count < medium_threshold_faces) and \
                               (physical_size >= small_threshold_size and physical_size < medium_threshold_size)
                    
                    if is_small:
                        # Маленькая деталь - оставляем как есть, не разбиваем
                        segments.append({
                            "id": i,
                            "mesh": component,
                            "type": "small_component",
                            "bounds": component.bounds.tolist(),
                            "center": component.centroid.tolist()
                        })
                    elif is_medium:
                        # Средняя деталь - оставляем как есть, не разбиваем
                        segments.append({
                            "id": i,
                            "mesh": component,
                            "type": "medium_component",
                            "bounds": component.bounds.tolist(),
                            "center": component.centroid.tolist()
                        })
                    else:
                        # Большая деталь - НЕ группируем с другими, оставляем отдельно
                        segments.append({
                            "id": i,
                            "mesh": component,
                            "type": "large_component",
                            "bounds": component.bounds.tolist(),
                            "center": component.centroid.tolist()
                        })
            
            logger.info(f"Разбито на {len(segments)} компонентов (было {len(components)} связанных частей)")
            return segments
        
        # Если модель монолитная, пытаемся найти естественные границы раздела
        return self._segment_monolithic(mesh)
    
    def _get_connected_components(self, mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
        """Получает связанные компоненты модели с группировкой близких мелких деталей"""
        # Используем встроенный метод trimesh
        components = mesh.split(only_watertight=False)
        
        if len(components) == 0:
            return []
        
        if len(components) == 1:
            return components
        
        # Вычисляем размеры компонентов (используем физический размер, а не только количество граней)
        component_info = []
        for c in components:
            face_count = len(c.faces)
            # Физический размер компонента (диагональ bounding box)
            physical_size = np.linalg.norm(c.bounds[1] - c.bounds[0])
            component_info.append({
                "component": c,
                "face_count": face_count,
                "physical_size": physical_size,
                "size_score": face_count * physical_size  # Комбинированный показатель размера
            })
        
        # Сортируем по комбинированному размеру
        component_info.sort(reverse=True, key=lambda x: x["size_score"])
        
        # Определяем пороги для трех категорий: маленькие, средние, большие
        # Используем и количество граней, и физический размер
        if len(component_info) > 0:
            max_face_count = component_info[0]["face_count"]
            max_physical_size = component_info[0]["physical_size"]
            
            # Пороги для маленьких деталей (меньше 10% от самого большого)
            small_threshold_faces = max(max_face_count * 0.1, self.min_component_size * 2)
            small_threshold_size = max(max_physical_size * 0.1, 8.0)  # Минимум 8мм
            
            # Пороги для средних деталей (10-30% от самого большого)
            medium_threshold_faces = max(max_face_count * 0.3, self.min_component_size * 5)
            medium_threshold_size = max(max_physical_size * 0.3, 25.0)  # Минимум 25мм
            
            logger.info(f"Пороги категорий:")
            logger.info(f"  Маленькие: грани < {small_threshold_faces:.0f}, размер < {small_threshold_size:.2f}мм")
            logger.info(f"  Средние: грани {small_threshold_faces:.0f}-{medium_threshold_faces:.0f}, размер {small_threshold_size:.2f}-{medium_threshold_size:.2f}мм")
            logger.info(f"  Большие: грани > {medium_threshold_faces:.0f}, размер > {medium_threshold_size:.2f}мм")
        else:
            small_threshold_faces = self.min_component_size * 2
            small_threshold_size = 8.0
            medium_threshold_faces = self.min_component_size * 5
            medium_threshold_size = 25.0
        
        large_components = []
        medium_components = []
        small_components = []
        
        # Разделяем на три категории
        for info in component_info:
            component = info["component"]
            face_count = info["face_count"]
            physical_size = info["physical_size"]
            
            # Определяем категорию детали
            is_small = (face_count < small_threshold_faces) and (physical_size < small_threshold_size)
            is_medium = (face_count >= small_threshold_faces and face_count < medium_threshold_faces) and \
                       (physical_size >= small_threshold_size and physical_size < medium_threshold_size)
            
            if is_small and face_count >= 10:  # Минимальный размер для учета
                small_components.append(component)
            elif is_medium and face_count >= self.min_component_size:
                medium_components.append(component)
            elif face_count >= self.min_component_size:
                large_components.append(component)
        
        # Группируем маленькие компоненты по близости (более гибко)
        if len(small_components) > 0:
            logger.info(f"Найдено {len(small_components)} маленьких компонентов для группировки")
            grouped_small = self._group_nearby_components(small_components, max_distance_ratio=0.01, category="small")
            logger.info(f"После группировки: {len(grouped_small)} групп маленьких компонентов")
            large_components.extend(grouped_small)
        
        # Группируем средние компоненты по близости (более строго)
        if len(medium_components) > 0:
            logger.info(f"Найдено {len(medium_components)} средних компонентов для группировки")
            grouped_medium = self._group_nearby_components(medium_components, max_distance_ratio=0.01, category="medium")
            logger.info(f"После группировки: {len(grouped_medium)} групп средних компонентов")
            large_components.extend(grouped_medium)
        
        return large_components
    
    def _group_nearby_components(self, components: List[trimesh.Trimesh], max_distance_ratio: float = 0.01, category: str = "small") -> List[trimesh.Trimesh]:
        """
        Группирует близкие компоненты в одну деталь используя scipy для оптимизации
        
        Использует scipy.spatial.cKDTree для быстрого поиска ближайших соседей
        и scipy.cluster.hierarchy для автоматической кластеризации по расстоянию.
        
        Args:
            components: Список компонентов
            max_distance_ratio: Максимальное расстояние между компонентами (относительно размера модели)
            category: Категория компонентов ("small", "medium")
            
        Returns:
            Список сгруппированных компонентов
        """
        if len(components) == 0:
            return []
        
        if len(components) == 1:
            return components
        
        # Вычисляем центроиды и размеры каждого компонента
        centroids = np.array([comp.centroid for comp in components])
        component_sizes = np.array([np.linalg.norm(comp.bounds[1] - comp.bounds[0]) for comp in components])
        
        # Определяем максимальное расстояние в зависимости от категории
        # Для маленьких деталей используем более строгий критерий, чтобы избежать дырок
        min_size = np.min(component_sizes)
        if category == "small":
            # Уменьшаем предел для маленьких деталей, чтобы они объединялись только если очень близко
            # Это предотвращает создание дырок при объединении
            max_allowed_distance = min(min_size * 0.15, 5.0)  # Максимум 5мм (было 8мм)
        elif category == "medium":
            max_allowed_distance = min(min_size * 0.1, 3.0)  # Максимум 3мм
        else:
            max_allowed_distance = min(min_size * 0.1, 3.0)
        
        logger.info(f"Группировка {len(components)} {category} компонентов с максимальным расстоянием {max_allowed_distance:.2f}мм")
        
        # Используем cKDTree для быстрого поиска ближайших соседей
        tree = cKDTree(centroids)
        
        # Находим все пары компонентов, которые находятся в пределах max_allowed_distance
        pairs = tree.query_pairs(max_allowed_distance)
        
        # Создаем граф связности для группировки
        # Используем Union-Find структуру для объединения связанных компонентов
        parent = list(range(len(components)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Объединяем только если компоненты действительно близки
                # Проверяем расстояние между bounding boxes для более точной проверки
                bbox_dist = self._bbox_min_distance(components[px].bounds, components[py].bounds)
                
                # Дополнительная проверка: расстояние между центроидами
                centroid_dist = np.linalg.norm(centroids[px] - centroids[py])
                combined_size = component_sizes[px] + component_sizes[py]
                
                # Критерии объединения в зависимости от категории
                if category == "small":
                    # Для маленьких деталей используем более строгий критерий
                    # Компоненты должны быть очень близко, чтобы избежать дырок
                    centroid_threshold = combined_size * 0.5  # Было 0.7, теперь строже
                    # Дополнительно: bbox расстояние должно быть меньше половины размера меньшего компонента
                    min_component_size = min(component_sizes[px], component_sizes[py])
                    bbox_threshold = min_component_size * 0.3  # Компоненты должны почти касаться
                else:
                    centroid_threshold = combined_size * 0.4
                    bbox_threshold = max_allowed_distance
                
                # Объединяем только если компоненты действительно очень близко
                # Это предотвращает создание дырок в геометрии
                if (bbox_dist <= min(bbox_threshold, max_allowed_distance) and 
                    centroid_dist <= centroid_threshold):
                    parent[px] = py
                    return True
            return False
        
        # Объединяем компоненты
        num_merged = 0
        for i, j in pairs:
            if union(i, j):
                num_merged += 1
        
        logger.info(f"Найдено {len(pairs)} пар близких компонентов, объединено {num_merged}")
        
        # Группируем компоненты по их корневым родителям
        groups_dict = {}
        for i in range(len(components)):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(components[i])
        
        # Объединяем компоненты в каждой группе
        groups = []
        for root, group_components in groups_dict.items():
            if len(group_components) > 1:
                try:
                    combined = trimesh.util.concatenate(group_components)
                    groups.append(combined)
                    logger.debug(f"Объединено {len(group_components)} компонентов в группу {root}")
                except Exception as e:
                    logger.warning(f"Не удалось объединить {len(group_components)} компонентов: {e}. Добавляем как отдельные.")
                    groups.extend(group_components)
            else:
                groups.append(group_components[0])
        
        logger.info(f"Группировка завершена: {len(components)} компонентов -> {len(groups)} групп")
        return groups
    
    def _bbox_min_distance(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Вычисляет минимальное расстояние между двумя bounding boxes"""
        # Минимальное расстояние между AABB
        min_dist_sq = 0.0
        for dim in range(3):
            if bbox1[1, dim] < bbox2[0, dim]:
                # bbox1 слева от bbox2
                min_dist_sq += (bbox2[0, dim] - bbox1[1, dim]) ** 2
            elif bbox2[1, dim] < bbox1[0, dim]:
                # bbox2 слева от bbox1
                min_dist_sq += (bbox1[0, dim] - bbox2[1, dim]) ** 2
            # Иначе они перекрываются по этой оси
        
        return np.sqrt(min_dist_sq)
    
    def _segment_monolithic(self, mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
        """
        Сегментирует монолитную модель на части
        
        Использует комбинацию методов:
        1. Анализ выпуклости
        2. Поиск узких перемычек
        3. Кластеризация по геометрическим признакам
        
        НЕ разбивает маленькие детали дальше.
        """
        # Проверяем размер детали - если она маленькая, не разбиваем
        num_faces = len(mesh.faces)
        mesh_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        
        # Если деталь маленькая (меньше 500 граней или меньше 10% от среднего размера модели),
        # не разбиваем её дальше
        if num_faces < 500 or mesh_size < 10.0:
            logger.info(f"Деталь слишком маленькая для дальнейшего разбиения: {num_faces} граней, размер {mesh_size:.2f}мм")
            return [{
                "id": 0,
                "mesh": mesh,
                "type": "small_monolithic",
                "bounds": mesh.bounds.tolist(),
                "center": mesh.centroid.tolist()
            }]
        
        segments = []
        
        # Метод 1: Поиск узких перемычек (thin bridges)
        thin_segments = self._segment_by_thin_bridges(mesh)
        if len(thin_segments) > 1:
            return thin_segments
        
        # Метод 2: Кластеризация по выпуклости
        convex_segments = self._segment_by_convexity(mesh)
        if len(convex_segments) > 1:
            return convex_segments
        
        # Метод 3: Простое разбиение по осям (fallback)
        return self._segment_by_axes(mesh)
    
    def _segment_by_thin_bridges(self, mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
        """Разбиение по узким перемычкам"""
        # Анализируем толщину модели в разных местах
        # Ищем области с минимальной толщиной
        
        # Вычисляем расстояния от вершин до поверхности
        # Это упрощенный подход - в реальности нужен более сложный анализ
        
        # Пока возвращаем исходную модель как один сегмент
        # TODO: Реализовать более сложный алгоритм поиска перемычек
        return [{
            "id": 0,
            "mesh": mesh,
            "type": "monolithic",
            "bounds": mesh.bounds.tolist(),
            "center": mesh.centroid.tolist()
        }]
    
    def _segment_by_convexity(self, mesh: trimesh.Trimesh) -> List[Dict[str, Any]]:
        """Разбиение на основе анализа выпуклости"""
        # Вычисляем выпуклую оболочку
        try:
            hull = ConvexHull(mesh.vertices)
            
            # Находим вершины, которые далеко от выпуклой оболочки
            # Это могут быть вогнутые области, которые можно разделить
            
            # Упрощенный подход: разбиваем по главным осям
            return self._segment_by_axes(mesh)
        except:
            return [{
                "id": 0,
                "mesh": mesh,
                "type": "monolithic",
                "bounds": mesh.bounds.tolist(),
                "center": mesh.centroid.tolist()
            }]
    
    def _segment_by_axes(self, mesh: trimesh.Trimesh, num_parts: int = 4) -> List[Dict[str, Any]]:
        """
        Разбиение модели по осям на заданное количество частей
        
        Args:
            mesh: Исходная сетка
            num_parts: Количество частей (по умолчанию 4)
        """
        # Проверяем размер - если деталь маленькая, не разбиваем
        num_faces = len(mesh.faces)
        if num_faces < 500:
            logger.info(f"Деталь слишком маленькая для разбиения по осям: {num_faces} граней")
            return [{
                "id": 0,
                "mesh": mesh,
                "type": "small_monolithic",
                "bounds": mesh.bounds.tolist(),
                "center": mesh.centroid.tolist()
            }]
        
        bounds = mesh.bounds
        center = mesh.centroid
        
        # Определяем главную ось (самая длинная)
        sizes = bounds[1] - bounds[0]
        main_axis = np.argmax(sizes)
        
        segments = []
        
        # Разбиваем по главной оси
        axis_min = bounds[0][main_axis]
        axis_max = bounds[1][main_axis]
        axis_range = axis_max - axis_min
        
        for i in range(num_parts):
            part_min = axis_min + (axis_range * i / num_parts)
            part_max = axis_min + (axis_range * (i + 1) / num_parts)
            
            # Создаем ограничивающий бокс для этой части
            bbox = bounds.copy()
            bbox[0][main_axis] = part_min
            bbox[1][main_axis] = part_max
            
            # Извлекаем часть меша внутри этого бокса
            part_mesh = mesh.slice_plane(
                plane_origin=[center[0], center[1], part_min] if main_axis == 2 
                else [part_min, center[1], center[2]] if main_axis == 0
                else [center[0], part_min, center[2]],
                plane_normal=[0, 0, 1] if main_axis == 2 
                else [1, 0, 0] if main_axis == 0
                else [0, 1, 0]
            )
            
            # Альтернативный подход: используем булевы операции
            # Для упрощения используем фильтрацию вершин
            mask = (mesh.vertices[:, main_axis] >= part_min) & (mesh.vertices[:, main_axis] < part_max)
            
            if np.sum(mask) > 0:
                # Находим грани, которые принадлежат этой части
                vertex_mask = np.zeros(len(mesh.vertices), dtype=bool)
                vertex_mask[mask] = True
                
                # Находим грани, все вершины которых в этой части
                face_mask = vertex_mask[mesh.faces].all(axis=1)
                
                # Увеличиваем минимальный размер для частей при разбиении
                min_part_size = max(self.min_component_size, num_faces // (num_parts * 2))
                if np.sum(face_mask) > min_part_size:
                    part_faces = mesh.faces[face_mask]
                    # Перенумеровываем вершины
                    used_vertices = np.unique(part_faces.flatten())
                    vertex_map = {old: new for new, old in enumerate(used_vertices)}
                    new_faces = np.array([[vertex_map[v] for v in face] for face in part_faces])
                    new_vertices = mesh.vertices[used_vertices]
                    
                    part_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                    
                    segments.append({
                        "id": i,
                        "mesh": part_mesh,
                        "type": "axis_split",
                        "bounds": part_mesh.bounds.tolist(),
                        "center": part_mesh.centroid.tolist(),
                        "axis": main_axis,
                        "position": (part_min + part_max) / 2
                    })
        
        # Если не удалось разбить, возвращаем исходную модель
        if len(segments) == 0:
            return [{
                "id": 0,
                "mesh": mesh,
                "type": "monolithic",
                "bounds": mesh.bounds.tolist(),
                "center": mesh.centroid.tolist()
            }]
        
        return segments
    
    def get_segment_info(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """Получает информацию о сегменте"""
        mesh = segment["mesh"]
        return {
            "id": segment["id"],
            "type": segment["type"],
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "bounds": segment["bounds"],
            "center": segment["center"]
        }

