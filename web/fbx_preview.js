/**
 * FBX 3D Preview - Interactive viewer with Three.js
 * Based on UniRig's implementation pattern
 */

import { app } from "../../scripts/app.js";

console.log("[FBXPreview] Loading FBX Preview extension");

// Inline HTML viewer with Three.js
const VIEWER_HTML = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { margin: 0; overflow: hidden; background: #1a1a1a; font-family: Arial, sans-serif; }
        #canvas-container { width: 100%; height: 100%; }
        #controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            z-index: 100;
        }
        #controls button {
            display: block;
            width: 100%;
            margin: 5px 0;
            padding: 8px;
            background: #444;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        #controls button:hover { background: #666; }
        #controls label {
            display: block;
            margin: 10px 0 5px 0;
        }
        #loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 18px;
            z-index: 50;
        }
    </style>
</head>
<body>
    <div id="loading">Loading FBX...</div>
    <div id="canvas-container"></div>
    <div id="controls">
        <div style="font-weight: bold; margin-bottom: 10px;">FBX Viewer</div>
        <label><input type="checkbox" id="showSkeleton" checked> Show Skeleton</label>
        <label><input type="checkbox" id="showMesh" checked> Show Mesh</label>
        <label><input type="checkbox" id="wireframe"> Wireframe</label>
        <label><input type="checkbox" id="xraySkeleton"> X-Ray Skeleton</label>
        <button id="resetCamera">Reset Camera</button>
    </div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

        let scene, camera, renderer, controls;
        let currentModel = null;
        let skeletonHelper = null;
        let modelBoundingBox = null;

        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            // Camera
            camera = new THREE.PerspectiveCamera(
                50,
                window.innerWidth / window.innerHeight,
                0.1,
                10000
            );
            camera.position.set(0, 1.5, 3);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.shadowMap.enabled = true;
            document.getElementById('canvas-container').appendChild(renderer.domElement);

            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 7.5);
            directionalLight.castShadow = true;
            scene.add(directionalLight);

            // Grid
            const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
            scene.add(gridHelper);

            // Axes
            const axesHelper = new THREE.AxesHelper(2);
            scene.add(axesHelper);

            // Event listeners
            window.addEventListener('resize', onWindowResize);
            setupControls();

            // Listen for messages from parent
            window.addEventListener('message', handleMessage);

            // Notify parent that viewer is ready
            window.parent.postMessage({ type: 'VIEWER_READY' }, '*');

            animate();
        }

        function handleMessage(event) {
            const { type, filepath } = event.data;
            if (type === 'LOAD_FBX') {
                loadFBX(filepath);
            }
        }

        function loadFBX(filepath) {
            document.getElementById('loading').style.display = 'block';

            // Clear previous model
            if (currentModel) {
                scene.remove(currentModel);
            }
            if (skeletonHelper) {
                scene.remove(skeletonHelper);
            }

            const loader = new FBXLoader();

            // Convert absolute path to relative path for /view endpoint
            // ComfyUI's /view expects filenames relative to input/output dirs
            // and needs type=input for input folder files
            let relativePath = filepath;
            let fileType = 'output';  // default

            if (filepath.includes('/output/')) {
                relativePath = filepath.split('/output/')[1];
                fileType = 'output';
            } else if (filepath.includes('/input/')) {
                relativePath = filepath.split('/input/')[1];
                fileType = 'input';
            } else if (!filepath.startsWith('http')) {
                // Just use the basename if no standard directory found
                relativePath = filepath.split('/').pop();
            }

            // FBX files need to be served - use the file path directly
            // ComfyUI serves files from input/output directories
            // Use absolute URL since iframe runs from blob URL context
            const url = filepath.startsWith('http')
                ? filepath
                : \`\${window.parent.location.origin}/view?filename=\${encodeURIComponent(relativePath)}&type=\${fileType}\`;

            console.log('[FBXPreview] Loading URL:', url, '(type:', fileType, ')');

            loader.load(
                url,
                (fbx) => {
                    currentModel = fbx;

                    // Center model
                    const box = new THREE.Box3().setFromObject(currentModel);
                    modelBoundingBox = box;
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());

                    currentModel.position.sub(center);

                    // Find skeleton
                    currentModel.traverse((child) => {
                        if (child.isSkinnedMesh) {
                            if (child.skeleton) {
                                skeletonHelper = new THREE.SkeletonHelper(child);
                                skeletonHelper.material.linewidth = 2;
                                scene.add(skeletonHelper);
                            }
                        }
                        // Enable shadows
                        if (child.isMesh) {
                            child.castShadow = true;
                            child.receiveShadow = true;
                        }
                    });

                    scene.add(currentModel);

                    // Adjust camera to fit model
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                    cameraZ *= 2; // Zoom out a bit
                    camera.position.set(cameraZ, cameraZ * 0.5, cameraZ);
                    camera.lookAt(0, 0, 0);
                    controls.target.set(0, 0, 0);
                    controls.update();

                    document.getElementById('loading').style.display = 'none';
                    console.log('[FBXPreview] FBX loaded successfully');
                },
                (progress) => {
                    const percent = (progress.loaded / progress.total * 100).toFixed(0);
                    document.getElementById('loading').textContent = \`Loading FBX... \${percent}%\`;
                },
                (error) => {
                    console.error('[FBXPreview] Error loading FBX:', error);
                    document.getElementById('loading').textContent = 'Error loading FBX';
                    document.getElementById('loading').style.color = '#ff4444';
                }
            );
        }

        function setupControls() {
            document.getElementById('showSkeleton').addEventListener('change', (e) => {
                if (skeletonHelper) {
                    skeletonHelper.visible = e.target.checked;
                }
            });

            document.getElementById('showMesh').addEventListener('change', (e) => {
                if (currentModel) {
                    currentModel.traverse((child) => {
                        if (child.isMesh) {
                            child.visible = e.target.checked;
                        }
                    });
                }
            });

            document.getElementById('wireframe').addEventListener('change', (e) => {
                if (currentModel) {
                    currentModel.traverse((child) => {
                        if (child.isMesh && child.material) {
                            child.material.wireframe = e.target.checked;
                        }
                    });
                }
            });

            document.getElementById('xraySkeleton').addEventListener('change', (e) => {
                if (skeletonHelper) {
                    skeletonHelper.material.depthTest = !e.target.checked;
                }
            });

            document.getElementById('resetCamera').addEventListener('click', () => {
                if (modelBoundingBox) {
                    const size = modelBoundingBox.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const fov = camera.fov * (Math.PI / 180);
                    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                    cameraZ *= 2;
                    camera.position.set(cameraZ, cameraZ * 0.5, cameraZ);
                    camera.lookAt(0, 0, 0);
                    controls.target.set(0, 0, 0);
                    controls.update();
                }
            });
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }

        init();
    </script>
</body>
</html>
`;

app.registerExtension({
    name: "Comfy.FBXPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FBXPreview") {
            console.log("[FBXPreview] Registering FBXPreview node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Create iframe widget
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; height: 500px; display: flex; flex-direction: column; border: 2px solid #444; border-radius: 8px; overflow: hidden; background: #1a1a1a; box-sizing: border-box;";

                const iframe = document.createElement("iframe");
                iframe.style.cssText = "display: block; max-width: 100%; max-height: 100%; width: 100%; height: 100%; border: none; flex: 1 1 0; object-fit: contain;";

                // Create blob URL for viewer
                const blob = new Blob([VIEWER_HTML], { type: 'text/html' });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;

                container.appendChild(iframe);

                // Add to node
                const widget = this.addDOMWidget("fbx_preview", "preview", container, {
                    serialize: false,
                    hideOnZoom: false,
                });

                // Make widget dynamically sized - override computeSize
                widget.computeSize = (width) => {
                    const nodeHeight = this.size ? this.size[1] : 500;
                    const widgetHeight = Math.max(300, nodeHeight - 80); // 80px for title bar + padding
                    return [width, widgetHeight];
                };

                // Store iframe reference
                this.fbxIframe = iframe;
                this.fbxViewerReady = false;

                // Listen for viewer ready message
                window.addEventListener('message', (event) => {
                    if (event.data.type === 'VIEWER_READY' && event.source === iframe.contentWindow) {
                        console.log("[FBXPreview] Viewer ready");
                        this.fbxViewerReady = true;
                        // Load FBX if we already have a path
                        if (this.fbxPathToLoad) {
                            this.loadFBXInViewer(this.fbxPathToLoad);
                        }
                    }
                });

                // Override onResize to update container height when node is resized
                const originalOnResize = this.onResize;
                this.onResize = function(size) {
                    if (originalOnResize) {
                        originalOnResize.apply(this, arguments);
                    }
                    const containerHeight = Math.max(300, size[1] - 80);
                    container.style.height = containerHeight + "px";
                    console.log(`[FBXPreview] Node resized to: ${size[0]}x${size[1]}, container height: ${containerHeight}px`);
                };

                // Override onDrawForeground to ensure container height syncs on every frame
                const originalOnDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    if (originalOnDrawForeground) {
                        originalOnDrawForeground.apply(this, arguments);
                    }
                    // Update container height based on current node size
                    const containerHeight = Math.max(300, this.size[1] - 80);
                    if (container.style.height !== containerHeight + "px") {
                        container.style.height = containerHeight + "px";
                    }
                };

                // Set initial node size
                const nodeWidth = Math.max(400, this.size[0] || 400);
                const nodeHeight = 500; // Initial height: viewer (420) + overhead (80)
                this.setSize([nodeWidth, nodeHeight]);

                // Set initial container height
                container.style.height = "420px";

                console.log("[FBXPreview] Node setup complete");
                return result;
            };

            // Add method to load FBX
            nodeType.prototype.loadFBXInViewer = function(fbxPath) {
                if (!this.fbxIframe || !this.fbxIframe.contentWindow) {
                    console.warn("[FBXPreview] Iframe not ready");
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                if (!this.fbxViewerReady) {
                    console.log("[FBXPreview] Viewer not ready yet, will load when ready");
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                console.log("[FBXPreview] Loading FBX:", fbxPath);
                this.fbxIframe.contentWindow.postMessage({
                    type: 'LOAD_FBX',
                    filepath: fbxPath
                }, '*');
                this.fbxPathToLoad = null;
            };

            // Override onExecuted to load FBX when node executes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                console.log("[FBXPreview] onExecuted called");
                console.log("[FBXPreview] Message:", message);

                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Get fbx_path from output (message is object with named keys from RETURN_NAMES)
                if (message?.fbx_path?.[0]) {
                    const fbxPath = message.fbx_path[0];
                    console.log("[FBXPreview] Node executed with FBX path:", fbxPath);
                    this.loadFBXInViewer(fbxPath);
                } else {
                    console.warn("[FBXPreview] No fbx_path in message");
                    console.warn("[FBXPreview] Available keys:", Object.keys(message || {}));
                    console.warn("[FBXPreview] Full message:", message);
                }
            };
        }
    }
});

console.log("[FBXPreview] Extension registered");
