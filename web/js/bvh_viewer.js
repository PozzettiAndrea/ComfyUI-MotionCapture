/**
 * BVH Viewer Widget - Interactive BVH skeleton animation viewer
 * Displays skeletal animations in BVH format with Three.js
 */

import { app } from "../../../../scripts/app.js";

console.log("[BVHViewer] Loading BVH Viewer extension");

// Node types that use this viewer
const BVH_VIEWER_NODES = [
    {
        nodeName: "BVHViewer",
        extensionName: "comfy3d.bvhviewer",
        logPrefix: "[BVHViewer]"
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
        console.warn('[BVHViewer] Could not detect extension folder:', e);
    }
    return null;
}

function getViewerUrl(extensionFolder) {
    return `/extensions/${extensionFolder}/viewer_bvh.html?v=` + Date.now();
}

function createBVHViewerExtension(config) {
    const { extensionName, nodeName, logPrefix } = config;

    console.log(`${logPrefix} Loading BVH viewer extension...`);

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

                // Create container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222;";

                // Create iframe for Three.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.cssText = "width: 100%; height: 600px; border: none; display: block; background: #1a1a1a;";
                iframe.src = getViewerUrl(extensionFolder);
                container.appendChild(iframe);

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

                // Speed control container
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
                    bvhData: null,
                    viewerReady: false
                };

                // Add DOM widget
                this.addDOMWidget("bvh_viewer", "customIframe", container);

                // Play button handler
                playButton.onclick = () => {
                    const state = this.bvhViewerState;
                    state.isPlaying = !state.isPlaying;
                    playButton.textContent = state.isPlaying ? "\u23F8" : "\u25B6";
                    iframe.contentWindow.postMessage({
                        type: state.isPlaying ? 'play' : 'pause'
                    }, '*');
                };

                // Frame slider handler
                frameSlider.oninput = (e) => {
                    const frame = parseInt(e.target.value);
                    this.bvhViewerState.currentFrame = frame;
                    iframe.contentWindow.postMessage({ type: 'setFrame', frame: frame }, '*');
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
                    const state = this.bvhViewerState;

                    if (data.type === 'VIEWER_READY') {
                        state.viewerReady = true;
                        if (state.bvhData) {
                            this.loadBVHData(state.bvhData.bvhContent, state.bvhData.bvhInfo);
                        }
                    } else if (data.type === 'playing') {
                        state.isPlaying = true;
                        playButton.textContent = "\u23F8";
                    } else if (data.type === 'paused') {
                        state.isPlaying = false;
                        playButton.textContent = "\u25B6";
                    } else if (data.type === 'frameChanged') {
                        state.currentFrame = data.frame;
                        state.totalFrames = data.totalFrames;
                        frameSlider.value = data.frame;
                        frameCounter.textContent = `${data.frame} / ${data.totalFrames}`;
                    } else if (data.type === 'looped') {
                        state.currentFrame = 0;
                        frameSlider.value = 0;
                    }
                };
                window.addEventListener('message', messageHandler);

                // Cleanup on removal
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    window.removeEventListener('message', messageHandler);
                    if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                };

                // Handle data from backend
                this.onExecuted = (message) => {
                    if (message?.bvh_content) {
                        const bvhContent = message.bvh_content[0];
                        const bvhInfo = message.bvh_info ? message.bvh_info[0] : {};

                        this.bvhViewerState.bvhData = { bvhContent, bvhInfo };
                        this.bvhViewerState.totalFrames = bvhInfo.num_frames || 0;

                        playButton.disabled = false;
                        frameSlider.disabled = false;
                        frameSlider.max = this.bvhViewerState.totalFrames - 1;
                        frameCounter.textContent = `0 / ${this.bvhViewerState.totalFrames}`;

                        if (this.bvhViewerState.viewerReady) {
                            this.loadBVHData(bvhContent, bvhInfo);
                        }
                    }
                };

                // Method to load BVH data into viewer
                this.loadBVHData = function(bvhContent, bvhInfo) {
                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: 'loadBVH',
                            bvhContent: bvhContent,
                            bvhInfo: bvhInfo
                        }, '*');
                    }
                };

                this.setSize([Math.max(400, this.size[0] || 400), 720]);
                return result;
            };
        }
    });

    console.log(`${logPrefix} Extension registered: ${extensionName}`);
}

// Auto-register all known BVH viewer node types
BVH_VIEWER_NODES.forEach(config => createBVHViewerExtension(config));

export { createBVHViewerExtension, detectExtensionFolder };
