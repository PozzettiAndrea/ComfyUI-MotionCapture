/**
 * FBX Animation Viewer Widget - Interactive animation playback with Three.js
 * Controls animation playback, speed, looping, and skeleton visibility
 */

import { app } from "../../../../scripts/app.js";

console.log("[FBXAnimationViewer] Loading FBX Animation Viewer extension");

// Node types that use this viewer
const FBX_ANIMATION_NODES = [
    {
        nodeName: "FBXAnimationViewer",
        extensionName: "comfy3d.fbxanimationviewer",
        logPrefix: "[FBXAnimationViewer]"
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
        console.warn('[FBXAnimationViewer] Could not detect extension folder:', e);
    }
    return null;
}

function getViewerUrl(extensionFolder) {
    return `/extensions/${extensionFolder}/viewer_fbx_animation.html?v=` + Date.now();
}

function createFBXAnimationViewerExtension(config) {
    const { extensionName, nodeName, logPrefix } = config;

    console.log(`${logPrefix} Loading FBX Animation Viewer extension...`);

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

                // Main Container
                const mainContainer = document.createElement("div");
                mainContainer.style.cssText = "width: 100%; height: 100%; display: flex; flex-direction: column; overflow: hidden; background: #1a1a1a; border: 1px solid #444; box-sizing: border-box;";

                // Viewer Area (Iframe)
                const viewerArea = document.createElement("div");
                viewerArea.style.cssText = "position: relative; flex-grow: 1; min-height: 200px; overflow: hidden;";

                const iframe = document.createElement("iframe");
                iframe.style.cssText = "display: block; width: 100%; height: 100%; border: none;";
                iframe.src = getViewerUrl(extensionFolder);

                viewerArea.appendChild(iframe);
                mainContainer.appendChild(viewerArea);

                // Controls Area
                const controlsContainer = document.createElement("div");
                controlsContainer.style.cssText = "flex-shrink: 0; background: #2a2a2a; border-top: 1px solid #444; padding: 10px; box-sizing: border-box; color: white; font-size: 12px; font-family: Arial, sans-serif;";

                controlsContainer.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                        <button id="playPauseBtn" style="padding: 6px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;">\u25B6 Play</button>
                        <button id="resetBtn" style="padding: 6px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer;">\u27F2 Reset</button>
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

                    <button id="resetCamera" style="width: 100%; padding: 8px; background: #444; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; margin-top: 8px;">Reset Camera</button>
                `;

                mainContainer.appendChild(controlsContainer);

                // Add Single Widget
                const widget = this.addDOMWidget("fbx_viewer_unified", "main", mainContainer, {
                    serialize: false,
                    hideOnZoom: false
                });

                widget.computeSize = (width) => [width, 500];

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

                // Wire up controls
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

                resetCamera.addEventListener('click', () => {
                    iframe.contentWindow.postMessage({ type: 'RESET_CAMERA' }, '*');
                });

                // Listen for messages from iframe
                const messageHandler = (event) => {
                    if (event.source !== iframe.contentWindow) return;

                    const { type, ...data } = event.data;

                    switch(type) {
                        case 'VIEWER_READY':
                            this.animationViewerReady = true;
                            if (this.fbxPathToLoad) {
                                this.loadAnimationInViewer(this.fbxPathToLoad);
                            }
                            break;

                        case 'ANIMATIONS_LOADED':
                            animationSelect.innerHTML = '';
                            data.animations.forEach(anim => {
                                const option = document.createElement('option');
                                option.value = anim.index;
                                option.textContent = anim.name;
                                animationSelect.appendChild(option);
                            });

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
                            playPauseBtn.textContent = data.isPlaying ? '\u23F8 Pause' : '\u25B6 Play';
                            break;

                        case 'ANIMATION_CHANGED':
                            const fps = 30;
                            if (totalFramesEl) totalFramesEl.textContent = Math.floor(data.duration * fps);
                            playPauseBtn.textContent = '\u23F8 Pause';
                            break;

                        case 'NO_ANIMATIONS':
                            playPauseBtn.textContent = 'No Animation';
                            playPauseBtn.disabled = true;
                            break;
                    }
                };
                window.addEventListener('message', messageHandler);

                // Cleanup on removal
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    window.removeEventListener('message', messageHandler);
                    if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                };

                // Set initial node size
                const nodeWidth = Math.max(512, this.size[0] || 512);
                this.setSize([nodeWidth, 500]);

                return result;
            };

            // Add method to load FBX
            nodeType.prototype.loadAnimationInViewer = function(fbxPath) {
                if (!this.animationViewerIframe || !this.animationViewerIframe.contentWindow) {
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                if (!this.animationViewerReady) {
                    this.fbxPathToLoad = fbxPath;
                    return;
                }

                let relativePath = fbxPath;
                if (fbxPath.includes('/output/')) {
                    relativePath = fbxPath.split('/output/')[1];
                } else if (fbxPath.includes('/input/')) {
                    relativePath = fbxPath.split('/input/')[1];
                } else {
                    relativePath = fbxPath.split('/').pop();
                }

                const viewPath = window.location.origin + "/view?filename=" + encodeURIComponent(relativePath);

                this.animationViewerIframe.contentWindow.postMessage({
                    type: 'LOAD_FBX',
                    path: viewPath
                }, '*');
                this.fbxPathToLoad = null;
            };

            // Override onExecuted to load FBX
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);

                if (message?.fbx_path?.[0]) {
                    const fbxPath = message.fbx_path[0];
                    this.loadAnimationInViewer(fbxPath);
                }
            };
        }
    });

    console.log(`${logPrefix} Extension registered: ${extensionName}`);
}

// Auto-register all known FBX animation viewer node types
FBX_ANIMATION_NODES.forEach(config => createFBXAnimationViewerExtension(config));

export { createFBXAnimationViewerExtension, detectExtensionFolder };
