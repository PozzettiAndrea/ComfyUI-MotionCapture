/**
 * SMPL 3D Viewer - Plain Canvas 2D Implementation
 * Following SAM3 pattern - no external dependencies
 */

import { app } from "../../scripts/app.js";

console.log("[SMPL] ===== SMPL VIEWER LOADED =====");

// Simple 3D to 2D projection
function project3D(point, camera) {
    const dx = point[0] - camera.x;
    const dy = point[1] - camera.y;
    const dz = point[2] - camera.z;

    // Rotate around Y axis
    const cosY = Math.cos(camera.rotY);
    const sinY = Math.sin(camera.rotY);
    const x1 = dx * cosY - dz * sinY;
    const z1 = dx * sinY + dz * cosY;

    // Rotate around X axis
    const cosX = Math.cos(camera.rotX);
    const sinX = Math.sin(camera.rotX);
    const y2 = dy * cosX - z1 * sinX;
    const z2 = dy * sinX + z1 * cosX;

    // Perspective projection
    const scale = camera.distance / (camera.distance + z2);
    return {
        x: x1 * scale * 300 + camera.centerX,
        y: -y2 * scale * 300 + camera.centerY,
        z: z2
    };
}

app.registerExtension({
    name: "Comfy.SMPL.Viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SMPLViewer") {
            console.log("[SMPL] Registering SMPLViewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                console.log("[SMPL] onNodeCreated called");
                const result = onNodeCreated?.apply(this, arguments);

                // Create container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #222; overflow: hidden;";

                // Info bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
                container.appendChild(infoBar);

                // Frame counter
                const frameCounter = document.createElement("div");
                frameCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
                frameCounter.textContent = "No data loaded";
                infoBar.appendChild(frameCounter);

                // Canvas
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: grab; margin: 0 auto;";
                container.appendChild(canvas);

                const ctx = canvas.getContext("2d");

                // Controls bar
                const controlsBar = document.createElement("div");
                controlsBar.style.cssText = "display: flex; gap: 10px; padding: 10px; background: #252525; align-items: center;";

                // Play button
                const playButton = document.createElement("button");
                playButton.textContent = "▶";
                playButton.style.cssText = "width: 40px; height: 40px; border: none; border-radius: 6px; background: #4a9eff; color: white; font-size: 16px; cursor: pointer;";
                playButton.disabled = true;
                controlsBar.appendChild(playButton);

                // Frame slider
                const frameSlider = document.createElement("input");
                frameSlider.type = "range";
                frameSlider.min = 0;
                frameSlider.max = 0;
                frameSlider.value = 0;
                frameSlider.disabled = true;
                frameSlider.style.cssText = "flex-grow: 1; height: 6px;";
                controlsBar.appendChild(frameSlider);

                container.appendChild(controlsBar);

                // State
                this.smplViewerState = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    frameCounter: frameCounter,
                    playButton: playButton,
                    frameSlider: frameSlider,
                    controlsBar: controlsBar,

                    // Data
                    meshData: null,
                    currentFrame: 0,
                    isPlaying: false,

                    // Camera
                    camera: {
                        x: 0,
                        y: 1,
                        z: 0,
                        rotX: 0,
                        rotY: 0,
                        distance: 3,
                        centerX: canvas.width / 2,
                        centerY: canvas.height / 2
                    },

                    // Mouse
                    mouseDown: false,
                    lastMouseX: 0,
                    lastMouseY: 0
                };

                // Add DOM widget
                this.addDOMWidget("smpl_viewer", "customCanvas", container);

                // Mouse controls
                canvas.addEventListener("mousedown", (e) => {
                    this.smplViewerState.mouseDown = true;
                    this.smplViewerState.lastMouseX = e.clientX;
                    this.smplViewerState.lastMouseY = e.clientY;
                    canvas.style.cursor = "grabbing";
                });

                window.addEventListener("mousemove", (e) => {
                    if (this.smplViewerState.mouseDown) {
                        const dx = e.clientX - this.smplViewerState.lastMouseX;
                        const dy = e.clientY - this.smplViewerState.lastMouseY;
                        this.smplViewerState.camera.rotY += dx * 0.01;
                        this.smplViewerState.camera.rotX += dy * 0.01;
                        this.smplViewerState.lastMouseX = e.clientX;
                        this.smplViewerState.lastMouseY = e.clientY;
                        this.redrawSMPLCanvas();
                    }
                });

                window.addEventListener("mouseup", () => {
                    this.smplViewerState.mouseDown = false;
                    canvas.style.cursor = "grab";
                });

                canvas.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    this.smplViewerState.camera.distance += e.deltaY * 0.01;
                    this.smplViewerState.camera.distance = Math.max(1, Math.min(10, this.smplViewerState.camera.distance));
                    this.redrawSMPLCanvas();
                });

                // Play button
                playButton.onclick = () => {
                    this.smplViewerState.isPlaying = !this.smplViewerState.isPlaying;
                    playButton.textContent = this.smplViewerState.isPlaying ? "⏸" : "▶";
                    if (this.smplViewerState.isPlaying) {
                        this.smplViewerState.lastFrameTime = performance.now();
                        this.animateSMPL();
                    }
                };

                // Frame slider
                frameSlider.oninput = (e) => {
                    this.smplViewerState.currentFrame = parseInt(e.target.value);
                    this.updateSMPLFrame();
                };

                // Handle data from backend
                this.onExecuted = (message) => {
                    console.log("[SMPL] onExecuted:", message);
                    if (message?.smpl_mesh) {
                        const data = message.smpl_mesh[0];
                        console.log("[SMPL] Received mesh data:", data.frames, "frames");
                        this.smplViewerState.meshData = data;
                        this.smplViewerState.currentFrame = 0;
                        frameSlider.max = data.frames - 1;
                        frameSlider.disabled = false;
                        playButton.disabled = false;
                        this.updateSMPLFrame();
                    }
                };

                // Draw placeholder
                this.redrawSMPLCanvas();

                container.style.height = "600px";
                this.setSize([Math.max(400, this.size[0] || 400), 680]);

                console.log("[SMPL] Node setup complete");
                return result;
            };

            // Update frame display
            nodeType.prototype.updateSMPLFrame = function() {
                const state = this.smplViewerState;
                if (!state.meshData) return;

                state.frameCounter.textContent = `Frame ${state.currentFrame + 1} / ${state.meshData.frames}`;
                state.frameSlider.value = state.currentFrame;
                this.redrawSMPLCanvas();
            };

            // Animation loop
            nodeType.prototype.animateSMPL = function() {
                const state = this.smplViewerState;
                if (!state.isPlaying || !state.meshData) return;

                const now = performance.now();
                const elapsed = now - state.lastFrameTime;
                const frameDuration = 1000 / state.meshData.fps;

                if (elapsed >= frameDuration) {
                    state.currentFrame = (state.currentFrame + 1) % state.meshData.frames;
                    this.updateSMPLFrame();
                    state.lastFrameTime = now - (elapsed % frameDuration);
                }

                requestAnimationFrame(() => this.animateSMPL());
            };

            // Render mesh
            nodeType.prototype.redrawSMPLCanvas = function() {
                const state = this.smplViewerState;
                const {canvas, ctx, meshData, currentFrame, camera} = state;

                // Clear
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                if (!meshData) {
                    // Placeholder
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Waiting for SMPL data...", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Connect SMPL params from GVHMR Inference", canvas.width / 2, canvas.height / 2 + 25);
                    return;
                }

                // Draw grid
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 1;
                for (let i = -2; i <= 2; i++) {
                    const p1 = project3D([i, 0, -2], camera);
                    const p2 = project3D([i, 0, 2], camera);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();

                    const p3 = project3D([-2, 0, i], camera);
                    const p4 = project3D([2, 0, i], camera);
                    ctx.beginPath();
                    ctx.moveTo(p3.x, p3.y);
                    ctx.lineTo(p4.x, p4.y);
                    ctx.stroke();
                }

                // Get current frame vertices and faces
                const vertices = meshData.vertices[currentFrame];
                const faces = meshData.faces;

                // Project vertices
                const projected = vertices.map(v => project3D(v, camera));

                // Sort faces by depth (painter's algorithm)
                const facesWithDepth = faces.map(face => {
                    const avgZ = (projected[face[0]].z + projected[face[1]].z + projected[face[2]].z) / 3;
                    return {face, depth: avgZ};
                });
                facesWithDepth.sort((a, b) => a.depth - b.depth);

                // Draw faces (wireframe)
                ctx.strokeStyle = meshData.mesh_color || "#4a9eff";
                ctx.lineWidth = 1;
                ctx.globalAlpha = 0.5;

                for (const {face} of facesWithDepth) {
                    const v0 = projected[face[0]];
                    const v1 = projected[face[1]];
                    const v2 = projected[face[2]];

                    ctx.beginPath();
                    ctx.moveTo(v0.x, v0.y);
                    ctx.lineTo(v1.x, v1.y);
                    ctx.lineTo(v2.x, v2.y);
                    ctx.closePath();
                    ctx.stroke();
                }

                ctx.globalAlpha = 1.0;
            };
        }
    }
});
