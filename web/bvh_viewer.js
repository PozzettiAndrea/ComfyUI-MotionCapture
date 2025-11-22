/**
 * BVH Viewer - Interactive BVH skeleton animation viewer with Three.js
 * Displays skeletal animations in BVH format
 */

import { app } from "../../scripts/app.js";

console.log("[BVHViewer] Loading BVH Viewer extension");

// Inline HTML viewer with Three.js BVHLoader
const BVH_VIEWER_HTML = `
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
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            font-family: monospace;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="loading">Loading BVH animation...</div>
    <div id="info" style="display: none;"></div>
    <div id="canvas-container"></div>

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
        import { BVHLoader } from 'three/addons/loaders/BVHLoader.js';

        let scene, camera, renderer, controls;
        let skeletonHelper = null;
        let mixer = null;
        let currentAction = null;
        let clip = null;
        let clock = new THREE.Clock();
        let isPlaying = false;
        let currentFrame = 0;
        let totalFrames = 0;
        let frameTime = 0;
        let skeleton = null;

        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a1a);

            // Camera
            camera = new THREE.PerspectiveCamera(
                50,
                window.innerWidth / window.innerHeight,
                0.01,
                1000
            );
            camera.position.set(2, 2, 3);

            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('canvas-container').appendChild(renderer.domElement);

            // Controls
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.target.set(0, 1, 0);
            controls.update();

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
            directionalLight.position.set(5, 10, 7.5);
            scene.add(directionalLight);

            // Grid
            const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
            scene.add(gridHelper);

            // Axes helper
            const axesHelper = new THREE.AxesHelper(1);
            scene.add(axesHelper);

            // Handle window resize
            window.addEventListener('resize', onWindowResize);

            // Message handler for BVH data
            window.addEventListener('message', handleMessage);

            // Animation loop
            animate();

            console.log('[BVHViewer] Initialized');
        }

        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }

        function handleMessage(event) {
            console.log('[BVHViewer iframe DEBUG] Received message:', event);
            console.log('[BVHViewer iframe DEBUG] Message type:', event.data?.type);
            console.log('[BVHViewer iframe DEBUG] Message data keys:', Object.keys(event.data || {}));

            const data = event.data;

            if (data.type === 'loadBVH') {
                console.log('[BVHViewer iframe DEBUG] loadBVH message received');
                console.log('[BVHViewer iframe DEBUG] bvhContent length:', data.bvhContent?.length);
                console.log('[BVHViewer iframe DEBUG] bvhInfo:', data.bvhInfo);
                loadBVHFromString(data.bvhContent, data.bvhInfo);
            } else if (data.type === 'play') {
                console.log('[BVHViewer iframe DEBUG] play message');
                playAnimation();
            } else if (data.type === 'pause') {
                console.log('[BVHViewer iframe DEBUG] pause message');
                pauseAnimation();
            } else if (data.type === 'setFrame') {
                console.log('[BVHViewer iframe DEBUG] setFrame message');
                setFrame(data.frame);
            } else if (data.type === 'setSpeed') {
                console.log('[BVHViewer iframe DEBUG] setSpeed message');
                if (mixer) {
                    mixer.timeScale = data.speed;
                }
            }
        }

        function loadBVHFromString(bvhContent, bvhInfo) {
            console.log('[BVHViewer] Loading BVH data');

            // Clear existing animation
            if (skeletonHelper) {
                scene.remove(skeletonHelper);
                skeletonHelper = null;
            }
            if (skeleton && skeleton.bones[0]) {
                scene.remove(skeleton.bones[0]);
                skeleton = null;
            }

            // Parse BVH
            const loader = new BVHLoader();
            let result;

            try {
                result = loader.parse(bvhContent);
            } catch (error) {
                console.error('[BVHViewer] Failed to parse BVH:', error);
                document.getElementById('loading').textContent = 'Error loading BVH: ' + error.message;
                return;
            }

            skeleton = result.skeleton;
            clip = result.clip;

            // Add skeleton to scene
            scene.add(skeleton.bones[0]);

            // Create skeleton helper for visualization
            skeletonHelper = new THREE.SkeletonHelper(skeleton.bones[0]);
            skeletonHelper.skeleton = skeleton;

            // Make skeleton lines bright and visible
            skeletonHelper.material = new THREE.LineBasicMaterial({
                color: 0x00ff00,  // Bright green
                linewidth: 3,      // Will be ignored by WebGL but doesn't hurt
                depthTest: false   // Always visible, even behind grid
            });
            scene.add(skeletonHelper);

            // Add sphere markers at each joint for better visibility
            const jointGeometry = new THREE.SphereGeometry(0.02, 8, 8);
            const jointMaterial = new THREE.MeshBasicMaterial({
                color: 0xff0000,  // Red joints
                depthTest: false
            });

            skeleton.bones.forEach(bone => {
                const jointMarker = new THREE.Mesh(jointGeometry, jointMaterial);
                bone.add(jointMarker);
            });

            // Setup animation
            mixer = new THREE.AnimationMixer(skeleton.bones[0]);
            currentAction = mixer.clipAction(clip);
            currentAction.setEffectiveWeight(1.0);

            // Get animation info
            totalFrames = Math.floor(clip.duration / (1/30)); // Approximate frame count
            frameTime = clip.duration / totalFrames;

            console.log('[BVHViewer] BVH loaded:', {
                bones: skeleton.bones.length,
                duration: clip.duration,
                frames: totalFrames
            });

            // Center camera on skeleton
            const box = new THREE.Box3().setFromObject(skeleton.bones[0]);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            cameraZ *= 1.5; // Zoom out a bit

            camera.position.set(center.x + cameraZ, center.y + cameraZ * 0.5, center.z + cameraZ);
            camera.lookAt(center);
            controls.target.copy(center);
            controls.update();

            // Update info display
            const infoDiv = document.getElementById('info');
            infoDiv.innerHTML = \`
                <strong>BVH Animation</strong><br>
                Bones: \${skeleton.bones.length}<br>
                Frames: \${bvhInfo.num_frames || totalFrames}<br>
                FPS: \${bvhInfo.fps || 30}<br>
                Duration: \${clip.duration.toFixed(2)}s
            \`;
            infoDiv.style.display = 'block';

            document.getElementById('loading').style.display = 'none';

            // Auto-play
            playAnimation();
        }

        function playAnimation() {
            if (!currentAction) return;

            isPlaying = true;
            currentAction.play();
            clock.start();

            // Notify parent
            window.parent.postMessage({ type: 'playing' }, '*');
        }

        function pauseAnimation() {
            if (!currentAction) return;

            isPlaying = false;
            currentAction.paused = true;

            // Notify parent
            window.parent.postMessage({ type: 'paused' }, '*');
        }

        function setFrame(frame) {
            if (!mixer || !currentAction || !clip) return;

            currentFrame = Math.max(0, Math.min(frame, totalFrames - 1));
            const time = (currentFrame / totalFrames) * clip.duration;

            mixer.setTime(time);
            currentAction.time = time;

            // Update once without playing
            mixer.update(0);

            // Notify parent
            window.parent.postMessage({
                type: 'frameChanged',
                frame: currentFrame,
                totalFrames: totalFrames
            }, '*');
        }

        function animate() {
            requestAnimationFrame(animate);

            // Update controls
            controls.update();

            // Update animation
            if (mixer && isPlaying) {
                const delta = clock.getDelta();
                mixer.update(delta);

                // Calculate current frame
                if (currentAction && clip) {
                    const progress = currentAction.time / clip.duration;
                    currentFrame = Math.floor(progress * totalFrames);

                    // Loop notification
                    if (currentAction.time >= clip.duration - 0.01) {
                        window.parent.postMessage({ type: 'looped' }, '*');
                    }
                }
            }

            // Render
            renderer.render(scene, camera);
        }

        // Initialize when page loads
        init();
    </script>
</body>
</html>
`;

app.registerExtension({
    name: "Comfy.BVH.Viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BVHViewer") {
            console.log("[BVHViewer] Registering BVHViewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                console.log("[BVHViewer] onNodeCreated called");
                const result = onNodeCreated?.apply(this, arguments);

                // Create container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222;";

                // Create iframe for Three.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.cssText = "width: 100%; height: 600px; border: none; display: block; background: #1a1a1a;";
                iframe.srcdoc = BVH_VIEWER_HTML;
                container.appendChild(iframe);

                // Controls bar
                const controlsBar = document.createElement("div");
                controlsBar.style.cssText = "display: flex; gap: 10px; padding: 10px; background: #252525; align-items: center;";

                // Play/Pause button
                const playButton = document.createElement("button");
                playButton.textContent = "▶";
                playButton.style.cssText = "width: 40px; height: 40px; border: none; border-radius: 6px; background: #4a9eff; color: white; font-size: 16px; cursor: pointer;";
                playButton.disabled = true;
                controlsBar.appendChild(playButton);

                // Frame slider
                const frameSlider = document.createElement("input");
                frameSlider.type = "range";
                frameSlider.min = 0;
                frameSlider.max = 100;
                frameSlider.value = 0;
                frameSlider.disabled = true;
                frameSlider.style.cssText = "flex-grow: 1; height: 6px;";
                controlsBar.appendChild(frameSlider);

                // Frame counter
                const frameCounter = document.createElement("div");
                frameCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace; min-width: 100px; text-align: center;";
                frameCounter.textContent = "0 / 0";
                controlsBar.appendChild(frameCounter);

                // Speed control
                const speedLabel = document.createElement("span");
                speedLabel.textContent = "Speed:";
                speedLabel.style.cssText = "color: #fff; font-size: 12px; margin-left: 10px;";
                controlsBar.appendChild(speedLabel);

                const speedSlider = document.createElement("input");
                speedSlider.type = "range";
                speedSlider.min = 0.1;
                speedSlider.max = 2.0;
                speedSlider.step = 0.1;
                speedSlider.value = 1.0;
                speedSlider.style.cssText = "width: 100px;";
                controlsBar.appendChild(speedSlider);

                const speedValue = document.createElement("span");
                speedValue.textContent = "1.0x";
                speedValue.style.cssText = "color: #fff; font-size: 12px; min-width: 40px;";
                controlsBar.appendChild(speedValue);

                container.appendChild(controlsBar);

                // State
                this.bvhViewerState = {
                    iframe: iframe,
                    container: container,
                    playButton: playButton,
                    frameSlider: frameSlider,
                    frameCounter: frameCounter,
                    speedSlider: speedSlider,
                    speedValue: speedValue,
                    isPlaying: false,
                    currentFrame: 0,
                    totalFrames: 0,
                    bvhData: null
                };

                // Add DOM widget
                this.addDOMWidget("bvh_viewer", "customIframe", container);

                // Play button handler
                playButton.onclick = () => {
                    const state = this.bvhViewerState;
                    state.isPlaying = !state.isPlaying;
                    playButton.textContent = state.isPlaying ? "⏸" : "▶";

                    iframe.contentWindow.postMessage({
                        type: state.isPlaying ? 'play' : 'pause'
                    }, '*');
                };

                // Frame slider handler
                frameSlider.oninput = (e) => {
                    const frame = parseInt(e.target.value);
                    this.bvhViewerState.currentFrame = frame;

                    iframe.contentWindow.postMessage({
                        type: 'setFrame',
                        frame: frame
                    }, '*');
                };

                // Speed slider handler
                speedSlider.oninput = (e) => {
                    const speed = parseFloat(e.target.value);
                    speedValue.textContent = speed.toFixed(1) + 'x';

                    iframe.contentWindow.postMessage({
                        type: 'setSpeed',
                        speed: speed
                    }, '*');
                };

                // Listen for messages from iframe
                window.addEventListener('message', (event) => {
                    if (event.source !== iframe.contentWindow) return;

                    const data = event.data;
                    const state = this.bvhViewerState;

                    if (data.type === 'playing') {
                        state.isPlaying = true;
                        playButton.textContent = "⏸";
                    } else if (data.type === 'paused') {
                        state.isPlaying = false;
                        playButton.textContent = "▶";
                    } else if (data.type === 'frameChanged') {
                        state.currentFrame = data.frame;
                        state.totalFrames = data.totalFrames;
                        frameSlider.value = data.frame;
                        frameCounter.textContent = `${data.frame} / ${data.totalFrames}`;
                    } else if (data.type === 'looped') {
                        state.currentFrame = 0;
                        frameSlider.value = 0;
                    }
                });

                // Handle data from backend
                this.onExecuted = (message) => {
                    console.log("[BVHViewer DEBUG] onExecuted called");
                    console.log("[BVHViewer DEBUG] Full message:", message);
                    console.log("[BVHViewer DEBUG] message keys:", Object.keys(message || {}));
                    console.log("[BVHViewer DEBUG] bvh_content exists?", !!message?.bvh_content);
                    console.log("[BVHViewer DEBUG] bvh_info exists?", !!message?.bvh_info);

                    if (message?.bvh_content) {
                        const bvhContent = message.bvh_content[0];
                        const bvhInfo = message.bvh_info ? message.bvh_info[0] : {};

                        console.log("[BVHViewer DEBUG] bvhContent type:", typeof bvhContent);
                        console.log("[BVHViewer DEBUG] bvhContent length:", bvhContent?.length);
                        console.log("[BVHViewer DEBUG] bvhContent first 200 chars:", bvhContent?.substring(0, 200));
                        console.log("[BVHViewer DEBUG] bvhInfo:", bvhInfo);

                        this.bvhViewerState.bvhData = { bvhContent, bvhInfo };
                        this.bvhViewerState.totalFrames = bvhInfo.num_frames || 0;

                        // Enable controls
                        playButton.disabled = false;
                        frameSlider.disabled = false;
                        frameSlider.max = this.bvhViewerState.totalFrames - 1;
                        frameCounter.textContent = `0 / ${this.bvhViewerState.totalFrames}`;

                        console.log("[BVHViewer DEBUG] iframe.contentDocument:", iframe.contentDocument);
                        console.log("[BVHViewer DEBUG] iframe readyState:", iframe.contentDocument?.readyState);

                        // Wait for iframe to load, then send BVH data
                        iframe.onload = () => {
                            console.log("[BVHViewer DEBUG] iframe.onload triggered");
                            setTimeout(() => {
                                console.log("[BVHViewer DEBUG] Sending loadBVH message to iframe (delayed)");
                                iframe.contentWindow.postMessage({
                                    type: 'loadBVH',
                                    bvhContent: bvhContent,
                                    bvhInfo: bvhInfo
                                }, '*');
                            }, 100);
                        };

                        // If already loaded, send immediately
                        if (iframe.contentDocument && iframe.contentDocument.readyState === 'complete') {
                            console.log("[BVHViewer DEBUG] iframe already loaded, sending immediately");
                            iframe.contentWindow.postMessage({
                                type: 'loadBVH',
                                bvhContent: bvhContent,
                                bvhInfo: bvhInfo
                            }, '*');
                        } else {
                            console.log("[BVHViewer DEBUG] iframe not ready, will wait for onload");
                        }
                    } else {
                        console.log("[BVHViewer DEBUG] No bvh_content in message!");
                    }
                };

                console.log("[BVHViewer] Node setup complete");
                this.setSize([Math.max(400, this.size[0] || 400), 720]);

                return result;
            };
        }
    }
});
