/**
 * FBX Animation Viewer - Interactive animation playback with Three.js AnimationMixer
 * Controls are in a unified widget
 */

import { app } from "../../scripts/app.js";

console.log("[FBXAnimationViewer] Loading FBX Animation Viewer extension");

// Inline HTML viewer - NO CONTROLS, just the 3D view
const ANIMATION_VIEWER_HTML = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { margin: 0; overflow: hidden; background: #1a1a1a; font-family: Arial, sans-serif; }
        #canvas-container { width: 100%; height: 100%; }
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
    <div id="loading">Loading animated FBX...</div>
    <div id="canvas-container"></div>

    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    <\/script>

    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';

        let scene, camera, renderer, controls;
        let currentModel = null;
        let skeletonHelper = null;
        let modelBoundingBox = null;
        let mixer = null;
        let currentAction = null;
        let animations = [];
        let clock = new THREE.Clock();
        let isPlaying = false;
        let showSkeleton = true;
        let showMesh = true;

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
            const axesHelper = new THREE.AxesHelper(1);
            scene.add(axesHelper);

            animate();
            window.addEventListener('resize', onWindowResize);
        }

        function loadFBX(path) {
            const loader = new FBXLoader();

            // Ensure absolute URL
            const url = (path.startsWith('http') || path.startsWith('blob:'))
                ? path
                : window.parent.location.origin + path;

            loader.load(
                url,
                (fbx) => {
                    console.log('[FBXAnimationViewer] FBX loaded successfully');
                    console.log('[FBXAnimationViewer] === FBX STRUCTURE ===');
                    console.log('[FBXAnimationViewer] Root name: ' + fbx.name);
                    console.log('[FBXAnimationViewer] Root type: ' + fbx.type);
                    console.log('[FBXAnimationViewer] Direct children: ' + fbx.children.length);

                    fbx.children.forEach((child, i) => {
                        console.log('[FBXAnimationViewer] Child[' + i + ']: ' + child.name + ' (type: ' + child.type + ')');
                    });

                    // Clear previous model
                    if (currentModel) scene.remove(currentModel);
                    if (skeletonHelper) scene.remove(skeletonHelper);
                    if (mixer) mixer.stopAllAction();

                    currentModel = fbx;
                    scene.add(fbx);

                    // Enable shadows and check mesh visibility
                    let meshCount = 0;
                    let skinnedMeshCount = 0;
                    fbx.traverse((child) => {
                        if (child.isMesh) {
                            meshCount++;
                            child.castShadow = true;
                            child.receiveShadow = true;
                            child.visible = showMesh;

                            const vertCount = child.geometry.attributes.position?.count || 0;
                            const isSkinned = child.isSkinnedMesh;
                            if (isSkinned) skinnedMeshCount++;

                            console.log('[FBXAnimationViewer] === MESH DEBUG ===');
                            console.log('[FBXAnimationViewer] Name: ' + child.name);
                            console.log('[FBXAnimationViewer] Vertices: ' + vertCount);
                            console.log('[FBXAnimationViewer] Is SkinnedMesh: ' + isSkinned);
                            console.log('[FBXAnimationViewer] Visible: ' + child.visible);

                            // Material info
                            const materials = Array.isArray(child.material) ? child.material : [child.material];
                            console.log('[FBXAnimationViewer] Material count: ' + materials.length);

                            materials.forEach((mat, i) => {
                                if (mat) {
                                    console.log('[FBXAnimationViewer] Material[' + i + ']: ' + mat.name + ' (type: ' + mat.type + ')');
                                    console.log('[FBXAnimationViewer]   Color: ' + (mat.color ? mat.color.getHexString() : 'none'));
                                    console.log('[FBXAnimationViewer]   Map (diffuse): ' + (mat.map ? mat.map.name || 'yes' : 'none'));
                                    console.log('[FBXAnimationViewer]   NormalMap: ' + (mat.normalMap ? 'yes' : 'none'));
                                    console.log('[FBXAnimationViewer]   Transparent: ' + mat.transparent);
                                    console.log('[FBXAnimationViewer]   Opacity: ' + mat.opacity);
                                    console.log('[FBXAnimationViewer]   Side: ' + mat.side);
                                } else {
                                    console.log('[FBXAnimationViewer] Material[' + i + ']: NULL');
                                }
                            });
                        }
                    });
                    console.log('[FBXAnimationViewer] Total meshes: ' + meshCount + ', SkinnedMeshes: ' + skinnedMeshCount);

                    // Create skeleton helper - search all descendants, not just direct children
                    let skinnedMesh = null;
                    fbx.traverse((child) => {
                        if (child.isSkinnedMesh && !skinnedMesh) {
                            skinnedMesh = child;
                        }
                    });

                    if (skinnedMesh && skinnedMesh.skeleton) {
                        console.log('[FBXAnimationViewer] Found skeleton on: ' + skinnedMesh.name);
                        console.log('[FBXAnimationViewer] Skeleton bones: ' + skinnedMesh.skeleton.bones.length);
                        skeletonHelper = new THREE.SkeletonHelper(fbx);
                        skeletonHelper.material.linewidth = 2;
                        skeletonHelper.visible = showSkeleton;
                        scene.add(skeletonHelper);
                    } else {
                        console.warn('[FBXAnimationViewer] No skinned mesh with skeleton found');
                    }

                    // Setup animations
                    animations = fbx.animations || [];
                    console.log('[FBXAnimationViewer] Found ' + animations.length + ' animation(s)');

                    if (animations.length > 0) {
                        setupAnimations();
                    } else {
                        console.warn('[FBXAnimationViewer] No animations found in FBX');
                        notifyParent({ type: 'NO_ANIMATIONS' });
                    }

                    // Center and frame model
                    centerModel(fbx);
                    document.getElementById('loading').style.display = 'none';
                },
                (xhr) => {
                    const percent = (xhr.loaded / xhr.total * 100).toFixed(0);
                    document.getElementById('loading').textContent = 'Loading... ' + percent + '%';
                },
                (error) => {
                    console.error('[FBXAnimationViewer] Error loading FBX:', error);
                    document.getElementById('loading').textContent = 'Error loading FBX';
                }
            );
        }

        function setupAnimations() {
            mixer = new THREE.AnimationMixer(currentModel);

            // Notify parent of available animations
            const animationNames = animations.map((clip, i) => ({
                index: i,
                name: clip.name || ('Animation ' + (i + 1)),
                duration: clip.duration
            }));
            notifyParent({
                type: 'ANIMATIONS_LOADED',
                animations: animationNames
            });

            // Play first animation
            playAnimation(0);
        }

        function playAnimation(index) {
            if (mixer && animations[index]) {
                // Stop current animation
                if (currentAction) {
                    currentAction.stop();
                }

                // Play new animation
                currentAction = mixer.clipAction(animations[index]);
                currentAction.setLoop(THREE.LoopRepeat);
                currentAction.timeScale = 1.0;
                currentAction.play();

                isPlaying = true;

                // Update parent
                notifyParent({
                    type: 'ANIMATION_CHANGED',
                    index: index,
                    duration: currentAction.getClip().duration
                });
            }
        }

        function togglePlayPause() {
            if (!currentAction) return;

            if (isPlaying) {
                currentAction.paused = true;
                isPlaying = false;
            } else {
                currentAction.paused = false;
                isPlaying = true;
            }
            notifyParent({ type: 'PLAY_STATE_CHANGED', isPlaying });
        }

        function resetAnimation() {
            if (currentAction) {
                currentAction.reset();
                currentAction.play();
                isPlaying = true;
                notifyParent({ type: 'PLAY_STATE_CHANGED', isPlaying: true });
            }
        }

        function setTimeline(progress) {
            if (currentAction) {
                const duration = currentAction.getClip().duration;
                currentAction.time = progress * duration;
            }
        }

        function setSpeed(speed) {
            if (currentAction) {
                currentAction.timeScale = speed;
            }
        }

        function setLoop(loop) {
            if (currentAction) {
                currentAction.setLoop(loop ? THREE.LoopRepeat : THREE.LoopOnce);
            }
        }

        function toggleSkeleton(visible) {
            showSkeleton = visible;
            if (skeletonHelper) skeletonHelper.visible = visible;
        }

        function toggleMesh(visible) {
            showMesh = visible;
            if (currentModel) {
                currentModel.traverse((child) => {
                    if (child.isMesh) child.visible = visible;
                });
            }
        }

        function toggleXRay(xray) {
            if (skeletonHelper) {
                skeletonHelper.material.depthTest = !xray;
                skeletonHelper.material.depthWrite = !xray;
            }
        }

        function centerModel(model) {
            const box = new THREE.Box3().setFromObject(model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            model.position.sub(center);

            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.5;

            camera.position.set(0, size.y / 2, cameraZ);
            camera.lookAt(0, size.y / 2, 0);

            controls.target.set(0, size.y / 2, 0);
            controls.update();

            modelBoundingBox = box;
        }

        function resetCamera() {
            if (modelBoundingBox) {
                const size = modelBoundingBox.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.5;

                camera.position.set(0, size.y / 2, cameraZ);
                camera.lookAt(0, size.y / 2, 0);
                controls.target.set(0, size.y / 2, 0);
                controls.update();
            }
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function notifyParent(data) {
            window.parent.postMessage(data, '*');
        }

        // Send time updates to parent
        let lastTimeUpdate = 0;
        function sendTimeUpdate() {
            if (!currentAction) return;

            const currentTime = currentAction.time;
            const duration = currentAction.getClip().duration;
            const progress = (currentTime / duration) * 100;
            const fps = 30;
            const currentFrame = Math.floor(currentTime * fps);
            const totalFrames = Math.floor(duration * fps);

            // Only update if enough time has passed (reduce spam)
            const now = Date.now();
            if (now - lastTimeUpdate > 50) { // 20 updates per second max
                notifyParent({
                    type: 'TIME_UPDATE',
                    time: currentTime,
                    duration: duration,
                    progress: progress,
                    frame: currentFrame,
                    totalFrames: totalFrames
                });
                lastTimeUpdate = now;
            }
        }

        function animate() {
            requestAnimationFrame(animate);

            const delta = clock.getDelta();

            // Update animation mixer
            if (mixer && isPlaying && !currentAction?.paused) {
                mixer.update(delta);
                sendTimeUpdate();
            }

            controls.update();
            renderer.render(scene, camera);
        }

        // Listen for commands from parent
        window.addEventListener('message', (event) => {
            const { type, ...data } = event.data;

            switch(type) {
                case 'LOAD_FBX':
                    loadFBX(data.path);
                    break;
                case 'PLAY_PAUSE':
                    togglePlayPause();
                    break;
                case 'RESET':
                    resetAnimation();
                    break;
                case 'SET_TIMELINE':
                    setTimeline(data.progress);
                    break;
                case 'SET_SPEED':
                    setSpeed(data.speed);
                    break;
                case 'SET_LOOP':
                    setLoop(data.loop);
                    break;
                case 'CHANGE_ANIMATION':
                    playAnimation(data.index);
                    break;
                case 'TOGGLE_SKELETON':
                    toggleSkeleton(data.visible);
                    break;
                case 'TOGGLE_MESH':
                    toggleMesh(data.visible);
                    break;
                case 'TOGGLE_XRAY':
                    toggleXRay(data.xray);
                    break;
                case 'RESET_CAMERA':
                    resetCamera();
                    break;
            }
        });

        // Initialize
        init();

        // Signal ready
        window.parent.postMessage({ type: 'VIEWER_READY' }, '*');
    </script>
</body>
</html>
`;

// Register extension
app.registerExtension({
    name: "Comfy.MotionCapture.FBXAnimationViewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FBXAnimationViewer") {
            console.log("[FBXAnimationViewer] Registering FBXAnimationViewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                console.log("[FBXAnimationViewer] Node created, adding unified widget");

                // 1. Main Container (Holds Viewer + Controls)
                const mainContainer = document.createElement("div");
                mainContainer.style.cssText = "width: 100%; height: 100%; display: flex; flex-direction: column; overflow: hidden; background: #1a1a1a; border: 1px solid #444; box-sizing: border-box;";

                // 2. Viewer Area (Iframe) - Flex Grow to fill space
                const viewerArea = document.createElement("div");
                viewerArea.style.cssText = "position: relative; flex-grow: 1; min-height: 200px; overflow: hidden;";
                
                const iframe = document.createElement("iframe");
                iframe.style.cssText = "display: block; width: 100%; height: 100%; border: none;";
                
                const blob = new Blob([ANIMATION_VIEWER_HTML], { type: 'text/html' });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;
                
                viewerArea.appendChild(iframe);
                mainContainer.appendChild(viewerArea);

                // 3. Controls Area - Fixed/Shrinkable
                const controlsContainer = document.createElement("div");
                controlsContainer.style.cssText = "flex-shrink: 0; background: #2a2a2a; border-top: 1px solid #444; padding: 10px; box-sizing: border-box; color: white; font-size: 12px; font-family: Arial, sans-serif;";

                controlsContainer.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                        <button id="playPauseBtn" style="padding: 6px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;">▶ Play</button>
                        <button id="resetBtn" style="padding: 6px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;">⟲ Reset</button>
                    </div>

                    <div style="margin-bottom: 8px;">
                        <input type="range" id="timeline" min="0" max="100" value="0" style="width: 100%; display: block; margin-bottom: 2px;" disabled>
                        <div style="display: flex; justify-content: space-between; font-size: 10px; color: #aaa;">
                            <span><span id="currentFrame">0</span> / <span id="totalFrames">0</span></span>
                            <span id="currentTime">0.00s</span>
                        </div>
                    </div>

                    <div style="display: flex; gap: 8px; margin-bottom: 8px;">
                        <select id="animationSelect" style="flex-grow: 1; padding: 4px; background: #333; color: white; border: 1px solid #555; border-radius: 4px;" disabled></select>
                        <select id="speedControl" style="width: 60px; padding: 4px; background: #333; color: white; border: 1px solid #555; border-radius: 4px;" disabled>
                            <option value="0.25">0.25x</option>
                            <option value="0.5">0.5x</option>
                            <option value="1" selected>1x</option>
                            <option value="2">2x</option>
                        </select>
                    </div>

                    <div style="display: flex; flex-wrap: wrap; gap: 8px; font-size: 11px;">
                        <label><input type="checkbox" id="loop" checked disabled> Loop</label>
                        <label><input type="checkbox" id="showSkeleton" checked> Bones</label>
                        <label><input type="checkbox" id="showMesh" checked> Mesh</label>
                        <label><input type="checkbox" id="xraySkeleton"> X-Ray</label>
                    </div>

                    <button id="resetCamera" style="width: 100%; padding: 8px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Reset Camera</button>
                `;
                
                mainContainer.appendChild(controlsContainer);

                // 4. Add Single Widget
                const widget = this.addDOMWidget("fbx_viewer_unified", "main", mainContainer, {
                    serialize: false,
                    hideOnZoom: false
                });

                // 5. Compute Size
                widget.computeSize = (width) => {
                    // Return a fixed height that includes viewer + controls. 
                    // The flex layout will distribute it.
                    return [width, 500];
                };
                
                // Get control elements
                const playPauseBtn = controlsContainer.querySelector('#playPauseBtn');
                const resetBtn = controlsContainer.querySelector('#resetBtn');
                const timeline = controlsContainer.querySelector('#timeline');
                const speedControl = controlsContainer.querySelector('#speedControl');
                const loopCheckbox = controlsContainer.querySelector('#loop');
                const animationSelect = controlsContainer.querySelector('#animationSelect');
                const showSkeleton = controlsContainer.querySelector('#showSkeleton');
                const showMesh = controlsContainer.querySelector('#showMesh');
                const xraySkeleton = controlsContainer.querySelector('#xraySkeleton');
                const resetCamera = controlsContainer.querySelector('#resetCamera');
                const currentTimeEl = controlsContainer.querySelector('#currentTime');
                const currentFrameEl = controlsContainer.querySelector('#currentFrame');
                const totalFramesEl = controlsContainer.querySelector('#totalFrames');

                // Store references
                this.animationViewerIframe = iframe;
                this.animationViewerReady = false;
                this.animationControls = {
                    playPauseBtn, resetBtn, timeline, speedControl, loopCheckbox, animationSelect,
                    showSkeleton, showMesh, xraySkeleton, resetCamera,
                    currentTimeEl, currentFrameEl, totalFramesEl
                };

                // Wire up controls to send commands to iframe
                playPauseBtn.addEventListener('click', () => {
                    iframe.contentWindow.postMessage({ type: 'PLAY_PAUSE' }, '*');
                });

                resetBtn.addEventListener('click', () => {
                    iframe.contentWindow.postMessage({ type: 'RESET' }, '*');
                });

                timeline.addEventListener('input', (e) => {
                    const progress = parseFloat(e.target.value) / 100;
                    iframe.contentWindow.postMessage({ type: 'SET_TIMELINE', progress }, '*');
                });

                speedControl.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'SET_SPEED', speed: parseFloat(e.target.value) }, '*');
                });

                loopCheckbox.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'SET_LOOP', loop: e.target.checked }, '*');
                });

                animationSelect.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'CHANGE_ANIMATION', index: parseInt(e.target.value) }, '*');
                });

                showSkeleton.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'TOGGLE_SKELETON', visible: e.target.checked }, '*');
                });

                showMesh.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'TOGGLE_MESH', visible: e.target.checked }, '*');
                });

                xraySkeleton.addEventListener('change', (e) => {
                    iframe.contentWindow.postMessage({ type: 'TOGGLE_XRAY', xray: e.target.checked }, '*');
                });

                // Listen for messages from iframe
                window.addEventListener('message', (event) => {
                    if (event.source !== iframe.contentWindow) return;

                    const { type, ...data } = event.data;

                    switch(type) {
                        case 'VIEWER_READY':
                            console.log("[FBXAnimationViewer] Viewer ready");
                            this.animationViewerReady = true;
                            if (this.fbxPathToLoad) {
                                this.loadAnimationInViewer(this.fbxPathToLoad);
                            }
                            break;

                        case 'ANIMATIONS_LOADED':
                            console.log("[FBXAnimationViewer] ✓ Animations loaded successfully!");
                            animationSelect.innerHTML = '';
                            data.animations.forEach(anim => {
                                const option = document.createElement('option');
                                option.value = anim.index;
                                option.textContent = anim.name;
                                animationSelect.appendChild(option);
                            });

                            // Enable controls
                            playPauseBtn.disabled = false;
                            resetBtn.disabled = false;
                            timeline.disabled = false;
                            speedControl.disabled = false;
                            loopCheckbox.disabled = false;
                            animationSelect.disabled = data.animations.length <= 1;
                            break;

                        case 'TIME_UPDATE':
                            if (currentTimeEl) currentTimeEl.textContent = data.time.toFixed(2) + 's';
                            if (currentFrameEl) currentFrameEl.textContent = data.frame;
                            if (totalFramesEl) totalFramesEl.textContent = data.totalFrames;
                            timeline.value = data.progress;
                            break;

                        case 'PLAY_STATE_CHANGED':
                            playPauseBtn.textContent = data.isPlaying ? '⏸ Pause' : '▶ Play';
                            break;

                        case 'ANIMATION_CHANGED':
                            const fps = 30;
                            if (totalFramesEl) totalFramesEl.textContent = Math.floor(data.duration * fps);
                            playPauseBtn.textContent = '⏸ Pause';
                            break;

                        case 'NO_ANIMATIONS':
                            console.warn("[FBXAnimationViewer] ⚠ No animations found in FBX file");
                            playPauseBtn.textContent = 'No Animation';
                            playPauseBtn.disabled = true;
                            break;
                    }
                });

                // Set initial node size
                const nodeWidth = Math.max(512, this.size[0] || 512);
                this.setSize([nodeWidth, 500]); // Slightly larger than widget height to account for header

                console.log("[FBXAnimationViewer] Node setup complete");
                return result;
            };

            // Add method to load FBX
            nodeType.prototype.loadAnimationInViewer = function(fbxPath) {
                console.log("[FBXAnimationViewer] loadAnimationInViewer called with:", fbxPath);

                if (!this.animationViewerIframe || !this.animationViewerIframe.contentWindow) {
                    console.warn("[FBXAnimationViewer] Iframe not ready, deferring load");
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                if (!this.animationViewerReady) {
                    console.log("[FBXAnimationViewer] Viewer not ready yet, deferring load");
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                // Convert absolute path to relative path for /view endpoint
                // ComfyUI's /view expects filenames relative to input/output dirs
                let relativePath = fbxPath;
                if (fbxPath.includes('/output/')) {
                    relativePath = fbxPath.split('/output/')[1];
                } else if (fbxPath.includes('/input/')) {
                    relativePath = fbxPath.split('/input/')[1];
                } else {
                    // Just use the basename if no standard directory found
                    relativePath = fbxPath.split('/').pop();
                }

                // Construct absolute URL (iframe runs from blob URL, needs absolute path)
                const viewPath = window.location.origin + "/view?filename=" + encodeURIComponent(relativePath);
                console.log("[FBXAnimationViewer] Sending LOAD_FBX message to iframe");
                console.log("[FBXAnimationViewer] Relative path:", relativePath);
                console.log("[FBXAnimationViewer] View URL:", viewPath);

                this.animationViewerIframe.contentWindow.postMessage({
                    type: 'LOAD_FBX',
                    path: viewPath
                }, '*');
                this.fbxPathToLoad = null;
            };

            // Override onExecuted to load FBX when node executes
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                console.log("[FBXAnimationViewer] onExecuted called");
                console.log("[FBXAnimationViewer] Message:", message);

                if (onExecuted) {
                    onExecuted.apply(this, arguments);
                }

                // Get fbx_path from output (message is object with named keys from RETURN_NAMES)
                if (message?.fbx_path?.[0]) {
                    const fbxPath = message.fbx_path[0];
                    console.log("[FBXAnimationViewer] ✓ Node executed with FBX path:", fbxPath);
                    this.loadAnimationInViewer(fbxPath);
                } else {
                    console.warn("[FBXAnimationViewer] ⚠ No fbx_path in message");
                    console.warn("[FBXAnimationViewer] Available keys:", Object.keys(message || {}));
                    console.warn("[FBXAnimationViewer] Full message:", message);
                }
            };
        }
    }
});

console.log("[FBXAnimationViewer] Extension registered successfully");