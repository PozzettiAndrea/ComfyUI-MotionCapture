/**
 * SMPL Camera Viewer Widget - Renders SMPL mesh from estimated camera trajectory
 * with optional side-by-side reference video.
 * Uses Three.js via iframe with trajectory/exterior camera toggle.
 */

import { app } from "../../../../scripts/app.js";

console.log("[SMPLCameraViewer] Loading SMPL Camera Viewer extension");

const SMPL_CAMERA_NODES = [
    {
        nodeName: "SMPLCameraViewer",
        extensionName: "comfy3d.smplcameraviewer",
        logPrefix: "[SMPLCameraViewer]"
    }
];

function detectExtensionFolder() {
    try {
        if (typeof import.meta !== 'undefined' && import.meta.url) {
            const match = import.meta.url.match(/\/extensions\/([^\/]+)\//);
            if (match) return match[1];
        }
        const scripts = document.getElementsByTagName('script');
        for (let i = scripts.length - 1; i >= 0; i--) {
            const src = scripts[i].src;
            if (src) {
                const match = src.match(/\/extensions\/([^\/]+)\//);
                if (match) return match[1];
            }
        }
    } catch (e) {
        console.warn('[SMPLCameraViewer] Could not detect extension folder:', e);
    }
    return null;
}

function getViewerUrl(extensionFolder) {
    return `/extensions/${extensionFolder}/viewer_smpl_camera.html?v=` + Date.now();
}

function createSMPLCameraViewerExtension(config) {
    const { extensionName, nodeName, logPrefix } = config;

    console.log(`${logPrefix} Loading extension...`);

    const extensionFolder = detectExtensionFolder();
    if (!extensionFolder) {
        console.error(`${logPrefix} Could not detect extension folder`);
        return;
    }

    app.registerExtension({
        name: extensionName,

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name !== nodeName) return;

            console.log(`${logPrefix} Registering ${nodeName} node`);

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);

                // Create outer container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; height: 100%; display: flex; flex-direction: column; background: #222;";

                // Split view container (flexbox row)
                const splitView = document.createElement("div");
                splitView.style.cssText = "display: flex; flex-direction: row; width: 100%; flex: 1; min-height: 0;";
                container.appendChild(splitView);

                // Left panel: reference video (hidden by default)
                const videoPanel = document.createElement("div");
                videoPanel.style.cssText = "display: none; flex: 1; position: relative; background: #111; overflow: hidden;";
                splitView.appendChild(videoPanel);

                const videoElement = document.createElement("video");
                videoElement.muted = true;
                videoElement.playsInline = true;
                videoElement.preload = "auto";
                videoElement.style.cssText = "position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; background: #111;";
                videoPanel.appendChild(videoElement);

                // "Reference" label
                const refLabel = document.createElement("div");
                refLabel.textContent = "Reference";
                refLabel.style.cssText = "position: absolute; top: 8px; left: 8px; color: rgba(255,255,255,0.5); font-size: 11px; font-family: monospace; background: rgba(0,0,0,0.4); padding: 2px 6px; border-radius: 3px; pointer-events: none;";
                videoPanel.appendChild(refLabel);

                // Right panel: Three.js iframe
                const iframePanel = document.createElement("div");
                iframePanel.style.cssText = "flex: 1; position: relative; overflow: hidden;";
                splitView.appendChild(iframePanel);

                const iframe = document.createElement("iframe");
                iframe.style.cssText = "width: 100%; height: 100%; border: none; display: block; background: #1a1a1a;";
                iframe.src = getViewerUrl(extensionFolder);
                iframePanel.appendChild(iframe);

                // Controls bar
                const controlsBar = document.createElement("div");
                controlsBar.style.cssText = "display: flex; flex-wrap: wrap; gap: 10px; padding: 10px; background: #252525; align-items: center; border-top: 1px solid #333;";

                // Play/Pause button
                const playButton = document.createElement("button");
                playButton.textContent = "\u25B6";
                playButton.style.cssText = "width: 30px; height: 30px; border: none; border-radius: 4px; background: #4a9eff; color: white; font-size: 14px; cursor: pointer; flex-shrink: 0;";
                playButton.disabled = true;
                controlsBar.appendChild(playButton);

                // Frame slider
                const frameSlider = document.createElement("input");
                frameSlider.type = "range";
                frameSlider.min = 0;
                frameSlider.max = 100;
                frameSlider.value = 0;
                frameSlider.disabled = true;
                frameSlider.style.cssText = "flex-grow: 1; height: 6px; min-width: 100px;";
                controlsBar.appendChild(frameSlider);

                // Frame counter
                const frameCounter = document.createElement("div");
                frameCounter.style.cssText = "padding: 4px 8px; background: rgba(0,0,0,0.3); color: #aaa; border-radius: 3px; font-size: 11px; font-family: monospace; min-width: 80px; text-align: center;";
                frameCounter.textContent = "0 / 0";
                controlsBar.appendChild(frameCounter);

                // Separator
                const sep = document.createElement("div");
                sep.style.cssText = "width: 1px; height: 20px; background: #444; margin: 0 5px;";
                controlsBar.appendChild(sep);

                // Camera mode toggle button (cycles: Through Camera -> Exterior)
                const cameraModes = ['through_camera', 'exterior'];
                const cameraModeLabels = { through_camera: 'Through Camera', exterior: 'Exterior' };
                const cameraModeColors = { through_camera: '#e67e22', exterior: '#8e44ad' };
                let currentCamModeIndex = 0;
                const cameraToggle = document.createElement("button");
                cameraToggle.textContent = "Through Camera";
                cameraToggle.style.cssText = "width: auto; padding: 0 10px; height: 30px; border: none; border-radius: 4px; background: #e67e22; color: white; font-size: 11px; cursor: pointer; flex-shrink: 0;";
                cameraToggle.disabled = true;
                controlsBar.appendChild(cameraToggle);

                // Speed control
                const speedContainer = document.createElement("div");
                speedContainer.style.cssText = "display: flex; align-items: center; gap: 5px;";

                const speedLabel = document.createElement("span");
                speedLabel.textContent = "Spd:";
                speedLabel.style.cssText = "color: #aaa; font-size: 11px;";
                speedContainer.appendChild(speedLabel);

                const speedSlider = document.createElement("input");
                speedSlider.type = "range";
                speedSlider.min = 0.1;
                speedSlider.max = 2.0;
                speedSlider.step = 0.1;
                speedSlider.value = 1.0;
                speedSlider.style.cssText = "width: 60px; height: 4px;";
                speedContainer.appendChild(speedSlider);

                const speedValue = document.createElement("span");
                speedValue.textContent = "1.0x";
                speedValue.style.cssText = "color: #fff; font-size: 11px; min-width: 30px;";
                speedContainer.appendChild(speedValue);

                controlsBar.appendChild(speedContainer);
                container.appendChild(controlsBar);

                // State
                this.smplCameraViewerState = {
                    iframe: iframe,
                    container: container,
                    splitView: splitView,
                    videoPanel: videoPanel,
                    videoElement: videoElement,
                    playButton: playButton,
                    frameSlider: frameSlider,
                    frameCounter: frameCounter,
                    cameraToggle: cameraToggle,
                    isPlaying: false,
                    currentFrame: 0,
                    totalFrames: 0,
                    hasCamera: false,
                    viewerReady: false,
                    pendingBuffer: null,
                    videoUrl: null,
                    videoFps: 30,
                };

                // Add DOM widget
                this.addDOMWidget("smpl_camera_viewer", "customIframe", container);

                // --- Video seek helper (1:1 frame mapping) ---
                const seekVideo = (meshFrame) => {
                    const state = this.smplCameraViewerState;
                    if (!state.videoUrl || !videoElement.readyState) return;
                    videoElement.currentTime = meshFrame / state.videoFps;
                };

                // Play button handler
                playButton.onclick = () => {
                    const state = this.smplCameraViewerState;
                    state.isPlaying = !state.isPlaying;
                    playButton.textContent = state.isPlaying ? "\u23F8" : "\u25B6";
                    iframe.contentWindow.postMessage({
                        type: state.isPlaying ? 'play' : 'pause'
                    }, '*');
                };

                // Frame slider handler
                frameSlider.oninput = (e) => {
                    const frame = parseInt(e.target.value);
                    this.smplCameraViewerState.currentFrame = frame;
                    iframe.contentWindow.postMessage({ type: 'setFrame', frame: frame }, '*');
                    seekVideo(frame);
                };

                // Camera toggle handler -- cycles through modes
                cameraToggle.onclick = () => {
                    currentCamModeIndex = (currentCamModeIndex + 1) % cameraModes.length;
                    const mode = cameraModes[currentCamModeIndex];
                    cameraToggle.textContent = cameraModeLabels[mode];
                    cameraToggle.style.background = cameraModeColors[mode];
                    iframe.contentWindow.postMessage({
                        type: 'setCameraMode',
                        mode: mode
                    }, '*');
                };

                // Speed slider handler
                speedSlider.oninput = (e) => {
                    const speed = parseFloat(e.target.value);
                    speedValue.textContent = speed.toFixed(1) + 'x';
                    iframe.contentWindow.postMessage({ type: 'setSpeed', speed: speed }, '*');
                };

                // Listen for messages from iframe
                const messageHandler = (event) => {
                    if (event.source !== iframe.contentWindow) return;

                    const data = event.data;
                    const state = this.smplCameraViewerState;

                    if (data.type === 'VIEWER_READY') {
                        state.viewerReady = true;
                        if (state.pendingBuffer) {
                            this.sendMeshData(state.pendingBuffer);
                            state.pendingBuffer = null;
                        }
                    } else if (data.type === 'meshLoaded') {
                        state.totalFrames = data.totalFrames || 0;
                        state.hasCamera = data.hasCamera || false;
                        playButton.disabled = false;
                        frameSlider.disabled = false;
                        frameSlider.max = state.totalFrames - 1;
                        frameCounter.textContent = `0 / ${state.totalFrames}`;
                        // Enable/disable camera toggle based on whether camera data exists
                        cameraToggle.disabled = !state.hasCamera;
                        if (!state.hasCamera) {
                            cameraToggle.textContent = "Exterior";
                            cameraToggle.style.background = "#555";
                            currentCamModeIndex = 1; // exterior
                        } else {
                            currentCamModeIndex = 0; // through_camera
                            cameraToggle.textContent = cameraModeLabels.through_camera;
                            cameraToggle.style.background = cameraModeColors.through_camera;
                        }
                        // Seek video to first frame
                        seekVideo(0);
                    } else if (data.type === 'frameChanged') {
                        state.currentFrame = data.frame;
                        frameSlider.value = data.frame;
                        frameCounter.textContent = `${data.frame} / ${state.totalFrames}`;
                        seekVideo(data.frame);
                    }
                };
                window.addEventListener('message', messageHandler);

                // Cleanup on removal
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    window.removeEventListener('message', messageHandler);
                    const state = this.smplCameraViewerState;
                    if (state.videoElement) state.videoElement.src = "";
                    if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                };

                // Send mesh data to iframe (structured clone, NOT Transferable -- parent keeps access)
                this.sendMeshData = function(buffer) {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: 'loadMesh',
                            buffer: buffer
                        }, '*');
                    }
                };

                // Handle data from backend
                this.onExecuted = async (message) => {
                    const state = this.smplCameraViewerState;

                    // Handle video info
                    if (message?.video_info) {
                        const vi = message.video_info[0];
                        state.videoFps = vi.fps || 30;
                        const params = new URLSearchParams({
                            filename: vi.filename,
                            type: vi.type,
                            subfolder: vi.subfolder || ""
                        });
                        const url = `/view?${params}`;
                        state.videoUrl = url;
                        videoElement.src = url;
                        videoPanel.style.display = "block";
                        console.log(`${logPrefix} Video: ${vi.filename} (fps=${state.videoFps})`);
                    } else {
                        state.videoUrl = null;
                        videoElement.src = "";
                        videoPanel.style.display = "none";
                    }

                    // Handle mesh file
                    if (message?.smpl_camera_mesh_file) {
                        const filename = message.smpl_camera_mesh_file[0];
                        console.log(`${logPrefix} Loading mesh file: ${filename}`);

                        try {
                            const resp = await fetch(`/view?filename=${encodeURIComponent(filename)}&type=output`);
                            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                            const buffer = await resp.arrayBuffer();
                            console.log(`${logPrefix} Fetched ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

                            // Send buffer to iframe for mesh rendering
                            if (state.viewerReady) {
                                this.sendMeshData(buffer);
                            } else {
                                state.pendingBuffer = buffer;
                            }
                        } catch (e) {
                            console.error(`${logPrefix} Error loading mesh:`, e);
                        }
                    }
                };

                this.setSize([Math.max(400, this.size[0] || 400), 720]);
                return result;
            };
        }
    });

    console.log(`${logPrefix} Extension registered: ${extensionName}`);
}

// Auto-register all SMPL camera viewer node types
SMPL_CAMERA_NODES.forEach(config => createSMPLCameraViewerExtension(config));
