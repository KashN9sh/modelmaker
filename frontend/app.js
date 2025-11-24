// Глобальные переменные
let currentMesh = null;
let currentFilepath = null;
let currentSegments = null;
let scene, camera, renderer, controls;
let meshObject = null;
let originalMeshGeometry = null;  // Оригинальная геометрия для масштабирования
let currentScale = 1.0;
let modelBounds = null;  // Габариты модели

// Инициализация Three.js
function initThreeJS() {
    const canvas = document.getElementById('canvas');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Сцена
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a1a);

    // Камера
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.set(0, 0, 5);

    // Рендерер
    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Освещение
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 1, 1);
    scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-1, -1, -1);
    scene.add(directionalLight2);

    // Контролы
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Обработка изменения размера
    window.addEventListener('resize', onWindowResize);

    // Анимационный цикл
    animate();
}

function onWindowResize() {
    const canvas = document.getElementById('canvas');
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Загрузка файла
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].name.endsWith('.stl')) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

async function handleFile(file) {
    showStatus('Загрузка файла...', 'info');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка загрузки');
        }

        const data = await response.json();
        currentFilepath = data.filepath;

        // Отображаем информацию о файле
        document.getElementById('fileName').textContent = data.filename;
        document.getElementById('vertexCount').textContent = data.info.vertices.toLocaleString();
        document.getElementById('faceCount').textContent = data.info.faces.toLocaleString();
        document.getElementById('volume').textContent = data.info.volume 
            ? data.info.volume.toFixed(2) + ' мм³' 
            : 'N/A';
        
        // Отображаем габариты модели
        if (data.info.bounds && data.info.bounds.length === 2) {
            const bounds = data.info.bounds;
            const sizeX = (bounds[1][0] - bounds[0][0]).toFixed(2);
            const sizeY = (bounds[1][1] - bounds[0][1]).toFixed(2);
            const sizeZ = (bounds[1][2] - bounds[0][2]).toFixed(2);
            
            document.getElementById('sizeX').textContent = sizeX + ' мм';
            document.getElementById('sizeY').textContent = sizeY + ' мм';
            document.getElementById('sizeZ').textContent = sizeZ + ' мм';
            
            // Сохраняем габариты для масштабирования
            modelBounds = {
                sizeX: parseFloat(sizeX),
                sizeY: parseFloat(sizeY),
                sizeZ: parseFloat(sizeZ),
                bounds: bounds
            };
        } else {
            document.getElementById('sizeX').textContent = 'N/A';
            document.getElementById('sizeY').textContent = 'N/A';
            document.getElementById('sizeZ').textContent = 'N/A';
        }
        
        document.getElementById('fileInfo').style.display = 'block';
        document.getElementById('controls').style.display = 'block';
        
        // Инициализируем слайдер масштабирования
        initScaleSlider();

        // Загружаем модель для визуализации
        await loadModelForPreview(file);

        showStatus('Файл успешно загружен!', 'success');
    } catch (error) {
        showStatus('Ошибка: ' + error.message, 'error');
    }
}

async function loadModelForPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const loader = new THREE.STLLoader();
        const geometry = loader.parse(e.target.result);
        
        // Сохраняем оригинальную геометрию для масштабирования
        originalMeshGeometry = geometry.clone();
        
        // Удаляем предыдущую модель
        if (meshObject) {
            scene.remove(meshObject);
            meshObject.geometry.dispose();
            meshObject.material.dispose();
        }

        // Создаем материал
        const material = new THREE.MeshPhongMaterial({
            color: 0x667eea,
            specular: 0x111111,
            shininess: 200,
            flatShading: true
        });

        // Создаем меш
        meshObject = new THREE.Mesh(geometry, material);
        
        // Центрируем и масштабируем для визуализации
        geometry.computeBoundingBox();
        const center = new THREE.Vector3();
        geometry.boundingBox.getCenter(center);
        geometry.translate(-center.x, -center.y, -center.z);

        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const displayScale = 3 / maxDim;
        meshObject.scale.set(displayScale, displayScale, displayScale);

        scene.add(meshObject);

        // Обновляем камеру
        camera.position.set(0, 0, 5);
        controls.reset();
        
        // Сбрасываем масштаб для обработки
        currentScale = 1.0;
    };

    reader.readAsArrayBuffer(file);
}

// Сегментация модели
async function segmentModel() {
    if (!currentFilepath) {
        showStatus('Сначала загрузите файл', 'error');
        return;
    }

    showStatus('Разбиение модели на части...', 'info');
    document.getElementById('segmentBtn').disabled = true;

    try {
        const numParts = parseInt(document.getElementById('numParts').value);
        
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentFilepath,
                num_parts: numParts
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка сегментации');
        }

        const data = await response.json();
        currentSegments = data.segments;

        showStatus(`Модель разбита на ${data.num_parts} частей!`, 'success');
        document.getElementById('generateBtn').disabled = false;
    } catch (error) {
        showStatus('Ошибка: ' + error.message, 'error');
    } finally {
        document.getElementById('segmentBtn').disabled = false;
    }
}

// Генерация спрусов
async function generateSprue() {
    if (!currentFilepath || !currentSegments) {
        showStatus('Сначала разбейте модель на части', 'error');
        return;
    }

    showStatus('Генерация спрусов...', 'info');
    document.getElementById('generateBtn').disabled = true;

    try {
        const sprueParams = {
            wall_thickness: parseFloat(document.getElementById('wallThickness').value)
        };

        const response = await fetch('/api/generate-sprue', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filepath: currentFilepath,
                segments: currentSegments,
                sprue_params: {
                    ...sprueParams,
                    scale: currentScale  // Добавляем масштаб
                },
                num_parts: parseInt(document.getElementById('numParts').value)
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Ошибка генерации спрусов');
        }

        const data = await response.json();
        displayDownloads(data.parts);

        showStatus(`Сгенерировано ${data.num_parts} деталей со спрусами!`, 'success');
    } catch (error) {
        showStatus('Ошибка: ' + error.message, 'error');
    } finally {
        document.getElementById('generateBtn').disabled = false;
    }
}

// Отображение списка файлов для скачивания
function displayDownloads(files) {
    const downloadList = document.getElementById('downloadList');
    downloadList.innerHTML = '';

    files.forEach(filename => {
        const item = document.createElement('div');
        item.className = 'download-item';
        item.innerHTML = `
            <span>${filename}</span>
            <a href="/api/download/${filename}" download class="btn">
                ⬇️ Скачать
            </a>
        `;
        downloadList.appendChild(item);
    });

    document.getElementById('downloads').style.display = 'block';
}

// Сброс вида
function resetView() {
    if (controls) {
        camera.position.set(0, 0, 5);
        controls.reset();
    }
}

// Показ статуса
function showStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status ${type} show`;
    
    setTimeout(() => {
        statusEl.classList.remove('show');
    }, 5000);
}

// Инициализация поля масштабирования
function initScaleSlider() {
    const scaleInput = document.getElementById('scaleInput');
    
    if (!scaleInput) return;
    
    // Устанавливаем начальное значение
    scaleInput.value = 1.0;
    currentScale = 1.0;
    
    scaleInput.addEventListener('input', (e) => {
        const scale = parseFloat(e.target.value);
        
        // Проверяем валидность значения
        if (isNaN(scale) || scale <= 0) {
            return;
        }
        
        // Ограничиваем максимальное значение
        if (scale > 1000) {
            scaleInput.value = 1000;
            return;
        }
        
        currentScale = scale;
        
        // Обновляем визуализацию (применяем масштаб к уже отмасштабированной для отображения модели)
        if (meshObject && originalMeshGeometry) {
            // Вычисляем базовый масштаб для отображения
            const baseGeometry = originalMeshGeometry.clone();
            baseGeometry.computeBoundingBox();
            const size = baseGeometry.boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const displayScale = 3 / maxDim;
            
            // Применяем пользовательский масштаб
            meshObject.scale.set(displayScale * scale, displayScale * scale, displayScale * scale);
        }
        
        // Обновляем отображаемые габариты
        if (modelBounds) {
            document.getElementById('sizeX').textContent = (modelBounds.sizeX * scale).toFixed(2) + ' мм';
            document.getElementById('sizeY').textContent = (modelBounds.sizeY * scale).toFixed(2) + ' мм';
            document.getElementById('sizeZ').textContent = (modelBounds.sizeZ * scale).toFixed(2) + ' мм';
        }
    });
    
    // Также обрабатываем изменение при потере фокуса (для случаев, когда пользователь вводит значение вручную)
    scaleInput.addEventListener('change', (e) => {
        const scale = parseFloat(e.target.value);
        
        if (isNaN(scale) || scale <= 0) {
            scaleInput.value = 1.0;
            currentScale = 1.0;
            return;
        }
        
        if (scale > 1000) {
            scaleInput.value = 1000;
            currentScale = 1000;
        } else {
            currentScale = scale;
        }
        
        // Обновляем визуализацию
        if (meshObject && originalMeshGeometry) {
            const baseGeometry = originalMeshGeometry.clone();
            baseGeometry.computeBoundingBox();
            const size = baseGeometry.boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const displayScale = 3 / maxDim;
            meshObject.scale.set(displayScale * currentScale, displayScale * currentScale, displayScale * currentScale);
        }
        
        // Обновляем отображаемые габариты
        if (modelBounds) {
            document.getElementById('sizeX').textContent = (modelBounds.sizeX * currentScale).toFixed(2) + ' мм';
            document.getElementById('sizeY').textContent = (modelBounds.sizeY * currentScale).toFixed(2) + ' мм';
            document.getElementById('sizeZ').textContent = (modelBounds.sizeZ * currentScale).toFixed(2) + ' мм';
        }
    });
}

// Инициализация при загрузке страницы
window.addEventListener('load', () => {
    initThreeJS();
});

