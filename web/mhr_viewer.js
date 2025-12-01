/**
 * MHR Skeleton 3D Viewer - Canvas 2D Implementation
 * Renders MHR 70-keypoint skeleton with colored bones
 *
 * Controls:
 * - Left-drag: Orbit/rotate around skeleton
 * - Right-drag or Shift+left-drag: Pan view
 * - Scroll wheel: Zoom in/out
 * - Double-click or 'R' key: Reset view
 */

import { app } from "../../scripts/app.js";

console.log("[MHR] ===== MHR VIEWER LOADED =====");

// Calculate bounding box center and size of skeleton
function getSkeletonBounds(keypoints) {
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const kp of keypoints) {
        minX = Math.min(minX, kp[0]); maxX = Math.max(maxX, kp[0]);
        minY = Math.min(minY, kp[1]); maxY = Math.max(maxY, kp[1]);
        minZ = Math.min(minZ, kp[2]); maxZ = Math.max(maxZ, kp[2]);
    }

    return {
        center: {
            x: (minX + maxX) / 2,
            y: (minY + maxY) / 2,
            z: (minZ + maxZ) / 2
        },
        size: Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1
    };
}

// 3D to 2D projection with orbit camera
function project3D(point, camera, canvasWidth, canvasHeight) {
    // Translate point relative to target
    let x = point[0] - camera.target.x;
    let y = point[1] - camera.target.y;
    let z = point[2] - camera.target.z;

    // Rotate around Y axis (yaw)
    const cosY = Math.cos(camera.rotY);
    const sinY = Math.sin(camera.rotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Rotate around X axis (pitch)
    const cosX = Math.cos(camera.rotX);
    const sinX = Math.sin(camera.rotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Apply camera distance (move back from target)
    const z3 = z2 + camera.distance;

    // Perspective projection
    const fov = 500;  // Field of view factor
    const scale = fov / (fov + z3);

    // Apply pan offset and center on canvas
    const screenX = x1 * scale * fov / camera.distance + canvasWidth / 2 + camera.panX;
    const screenY = -y2 * scale * fov / camera.distance + canvasHeight / 2 + camera.panY;

    return {
        x: screenX,
        y: screenY,
        z: z3,
        scale: scale
    };
}

// Convert RGB array to CSS color string
function rgbToHex(rgb) {
    return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

app.registerExtension({
    name: "Comfy.MHR.Viewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MHRViewer") {
            console.log("[MHR] Registering MHRViewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                console.log("[MHR] onNodeCreated called");
                const result = onNodeCreated?.apply(this, arguments);

                // Create container
                const container = document.createElement("div");
                container.style.cssText = "position: relative; width: 100%; background: #1a1a1a; overflow: hidden;";

                // Info bar
                const infoBar = document.createElement("div");
                infoBar.style.cssText = "position: absolute; top: 5px; left: 5px; right: 5px; z-index: 10; display: flex; justify-content: space-between; align-items: center;";
                container.appendChild(infoBar);

                // Frame counter
                const frameCounter = document.createElement("div");
                frameCounter.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #fff; border-radius: 3px; font-size: 12px; font-family: monospace;";
                frameCounter.textContent = "No data loaded";
                infoBar.appendChild(frameCounter);

                // Skeleton info
                const skeletonInfo = document.createElement("div");
                skeletonInfo.style.cssText = "padding: 5px 10px; background: rgba(0,0,0,0.7); color: #4a9eff; border-radius: 3px; font-size: 11px; font-family: monospace;";
                skeletonInfo.textContent = "MHR 70 Keypoints";
                infoBar.appendChild(skeletonInfo);

                // Canvas
                const canvas = document.createElement("canvas");
                canvas.width = 512;
                canvas.height = 512;
                canvas.style.cssText = "display: block; max-width: 100%; max-height: 100%; object-fit: contain; cursor: grab; margin: 0 auto;";
                canvas.tabIndex = 0;  // Make canvas focusable for keyboard events
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

                // Reset button
                const resetButton = document.createElement("button");
                resetButton.textContent = "⟲";
                resetButton.title = "Reset View (R)";
                resetButton.style.cssText = "width: 40px; height: 40px; border: none; border-radius: 6px; background: #444; color: white; font-size: 18px; cursor: pointer;";
                controlsBar.appendChild(resetButton);

                container.appendChild(controlsBar);

                // Help text
                const helpText = document.createElement("div");
                helpText.style.cssText = "position: absolute; bottom: 55px; left: 10px; background: rgba(0,0,0,0.6); color: #888; padding: 4px 8px; border-radius: 3px; font-size: 10px; font-family: monospace;";
                helpText.textContent = "Drag: rotate | Shift+drag: pan | Scroll: zoom | R: reset";
                container.appendChild(helpText);

                // State
                this.mhrViewerState = {
                    canvas: canvas,
                    ctx: ctx,
                    container: container,
                    frameCounter: frameCounter,
                    skeletonInfo: skeletonInfo,
                    playButton: playButton,
                    frameSlider: frameSlider,
                    resetButton: resetButton,
                    controlsBar: controlsBar,

                    // Data
                    skeletonData: null,
                    currentFrame: 0,
                    isPlaying: false,
                    bounds: null,

                    // Camera - orbit style
                    camera: {
                        target: { x: 0, y: 0, z: 0 },  // Point camera orbits around
                        distance: 2,                    // Distance from target
                        rotX: 0.3,                      // Pitch (slight downward angle)
                        rotY: 0,                        // Yaw
                        panX: 0,                        // Screen-space pan X
                        panY: 0                         // Screen-space pan Y
                    },
                    defaultCamera: null,

                    // Mouse state
                    mouseDown: false,
                    mouseButton: 0,
                    shiftKey: false,
                    lastMouseX: 0,
                    lastMouseY: 0
                };

                // Add DOM widget
                this.addDOMWidget("mhr_viewer", "customCanvas", container);

                const self = this;

                // Reset view function
                const resetView = () => {
                    const state = self.mhrViewerState;
                    if (state.bounds) {
                        state.camera.target = { ...state.bounds.center };
                        state.camera.distance = state.bounds.size * 2.5;
                        state.camera.rotX = 0.3;
                        state.camera.rotY = 0;
                        state.camera.panX = 0;
                        state.camera.panY = 0;
                    } else {
                        state.camera.target = { x: 0, y: 0, z: 0 };
                        state.camera.distance = 2;
                        state.camera.rotX = 0.3;
                        state.camera.rotY = 0;
                        state.camera.panX = 0;
                        state.camera.panY = 0;
                    }
                    self.redrawMHRCanvas();
                };

                // Prevent context menu on canvas
                canvas.addEventListener("contextmenu", (e) => e.preventDefault());

                // Mouse down
                canvas.addEventListener("mousedown", (e) => {
                    const state = self.mhrViewerState;
                    state.mouseDown = true;
                    state.mouseButton = e.button;
                    state.shiftKey = e.shiftKey;
                    state.lastMouseX = e.clientX;
                    state.lastMouseY = e.clientY;
                    canvas.style.cursor = "grabbing";
                    canvas.focus();
                    e.preventDefault();
                });

                // Mouse move (on window to catch moves outside canvas)
                window.addEventListener("mousemove", (e) => {
                    const state = self.mhrViewerState;
                    if (!state.mouseDown) return;

                    const dx = e.clientX - state.lastMouseX;
                    const dy = e.clientY - state.lastMouseY;
                    state.lastMouseX = e.clientX;
                    state.lastMouseY = e.clientY;

                    // Right-drag or Shift+left-drag = pan
                    if (state.mouseButton === 2 || state.shiftKey) {
                        state.camera.panX += dx;
                        state.camera.panY += dy;
                    } else {
                        // Left-drag = orbit
                        state.camera.rotY += dx * 0.01;
                        state.camera.rotX += dy * 0.01;
                        // Clamp pitch to avoid gimbal lock
                        state.camera.rotX = Math.max(-Math.PI / 2 + 0.1, Math.min(Math.PI / 2 - 0.1, state.camera.rotX));
                    }

                    self.redrawMHRCanvas();
                });

                // Mouse up
                window.addEventListener("mouseup", () => {
                    const state = self.mhrViewerState;
                    state.mouseDown = false;
                    canvas.style.cursor = "grab";
                });

                // Scroll wheel for zoom
                canvas.addEventListener("wheel", (e) => {
                    e.preventDefault();
                    const state = self.mhrViewerState;
                    const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
                    state.camera.distance *= zoomFactor;
                    state.camera.distance = Math.max(0.1, Math.min(50, state.camera.distance));
                    self.redrawMHRCanvas();
                }, { passive: false });

                // Keyboard events
                canvas.addEventListener("keydown", (e) => {
                    if (e.key === "r" || e.key === "R") {
                        resetView();
                    }
                });

                // Double-click to reset
                canvas.addEventListener("dblclick", () => {
                    resetView();
                });

                // Reset button
                resetButton.onclick = resetView;

                // Play button
                playButton.onclick = () => {
                    const state = self.mhrViewerState;
                    state.isPlaying = !state.isPlaying;
                    playButton.textContent = state.isPlaying ? "⏸" : "▶";
                    if (state.isPlaying) {
                        state.lastFrameTime = performance.now();
                        self.animateMHR();
                    }
                };

                // Frame slider
                frameSlider.oninput = (e) => {
                    self.mhrViewerState.currentFrame = parseInt(e.target.value);
                    self.updateMHRFrame();
                };

                // Handle data from backend
                this.onExecuted = (message) => {
                    console.log("[MHR] onExecuted:", message);
                    if (message?.mhr_skeleton) {
                        const data = message.mhr_skeleton[0];
                        console.log("[MHR] Received skeleton data:", data.frames, "frames,", data.num_keypoints, "keypoints");

                        const state = self.mhrViewerState;
                        state.skeletonData = data;
                        state.currentFrame = 0;
                        frameSlider.max = data.frames - 1;
                        frameSlider.disabled = false;
                        playButton.disabled = false;
                        skeletonInfo.textContent = `MHR ${data.num_keypoints} Keypoints | ${data.skeleton.length} Bones`;

                        // Calculate bounds and auto-center
                        if (data.keypoints && data.keypoints.length > 0) {
                            state.bounds = getSkeletonBounds(data.keypoints[0]);
                            resetView();
                        }

                        self.updateMHRFrame();
                    }
                };

                // Draw placeholder
                this.redrawMHRCanvas();

                container.style.height = "600px";
                this.setSize([Math.max(400, this.size[0] || 400), 680]);

                console.log("[MHR] Node setup complete");
                return result;
            };

            // Update frame display
            nodeType.prototype.updateMHRFrame = function() {
                const state = this.mhrViewerState;
                if (!state.skeletonData) return;

                state.frameCounter.textContent = `Frame ${state.currentFrame + 1} / ${state.skeletonData.frames}`;
                state.frameSlider.value = state.currentFrame;
                this.redrawMHRCanvas();
            };

            // Animation loop
            nodeType.prototype.animateMHR = function() {
                const state = this.mhrViewerState;
                if (!state.isPlaying || !state.skeletonData) return;

                const now = performance.now();
                const elapsed = now - state.lastFrameTime;
                const frameDuration = 1000 / state.skeletonData.fps;

                if (elapsed >= frameDuration) {
                    state.currentFrame = (state.currentFrame + 1) % state.skeletonData.frames;
                    this.updateMHRFrame();
                    state.lastFrameTime = now - (elapsed % frameDuration);
                }

                requestAnimationFrame(() => this.animateMHR());
            };

            // Render skeleton
            nodeType.prototype.redrawMHRCanvas = function() {
                const state = this.mhrViewerState;
                const { canvas, ctx, skeletonData, currentFrame, camera } = state;

                // Clear
                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                if (!skeletonData) {
                    // Placeholder
                    ctx.fillStyle = "#666";
                    ctx.font = "16px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Waiting for MHR skeleton data...", canvas.width / 2, canvas.height / 2);
                    ctx.fillText("Connect MHR params from SAM3D Inference", canvas.width / 2, canvas.height / 2 + 25);
                    return;
                }

                // Draw grid centered on target
                ctx.strokeStyle = "#333";
                ctx.lineWidth = 1;
                const gridSize = state.bounds ? state.bounds.size : 2;
                const gridStep = gridSize / 4;
                const gridY = camera.target.y - (state.bounds ? state.bounds.size / 2 : 0);

                for (let i = -2; i <= 2; i++) {
                    const offset = i * gridStep;
                    const p1 = project3D([camera.target.x + offset, gridY, camera.target.z - gridSize/2], camera, canvas.width, canvas.height);
                    const p2 = project3D([camera.target.x + offset, gridY, camera.target.z + gridSize/2], camera, canvas.width, canvas.height);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();

                    const p3 = project3D([camera.target.x - gridSize/2, gridY, camera.target.z + offset], camera, canvas.width, canvas.height);
                    const p4 = project3D([camera.target.x + gridSize/2, gridY, camera.target.z + offset], camera, canvas.width, canvas.height);
                    ctx.beginPath();
                    ctx.moveTo(p3.x, p3.y);
                    ctx.lineTo(p4.x, p4.y);
                    ctx.stroke();
                }

                // Get current frame keypoints
                const keypoints = skeletonData.keypoints[currentFrame];
                const skeleton = skeletonData.skeleton;
                const jointSize = skeletonData.joint_size || 5;
                const boneWidth = skeletonData.bone_width || 2;

                // Project all keypoints
                const projected = keypoints.map(kp => project3D(kp, camera, canvas.width, canvas.height));

                // Sort bones by depth for painter's algorithm (far to near)
                const bonesWithDepth = skeleton.map(bone => {
                    const p1 = projected[bone.from];
                    const p2 = projected[bone.to];
                    const avgZ = (p1.z + p2.z) / 2;
                    return { bone, p1, p2, depth: avgZ };
                });
                bonesWithDepth.sort((a, b) => b.depth - a.depth);  // Far first

                // Draw bones
                ctx.lineWidth = boneWidth;
                ctx.lineCap = "round";
                for (const { bone, p1, p2 } of bonesWithDepth) {
                    ctx.strokeStyle = rgbToHex(bone.color);
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }

                // Sort joints by depth (far to near)
                const jointsWithDepth = projected.map((p, i) => ({
                    point: p,
                    index: i,
                    depth: p.z
                }));
                jointsWithDepth.sort((a, b) => b.depth - a.depth);  // Far first

                // Draw joints
                for (const { point, index } of jointsWithDepth) {
                    // Color based on body part
                    let color;
                    if (index < 5) {
                        color = "#3399ff";  // Face (blue)
                    } else if (index < 21) {
                        color = "#ffffff";  // Body (white)
                    } else if (index < 42) {
                        color = "#ff8800";  // Right hand (orange)
                    } else if (index < 63) {
                        color = "#00ff00";  // Left hand (green)
                    } else {
                        color = "#00ffff";  // Extra points (cyan)
                    }

                    // Scale joint size by depth for perspective effect
                    const scaledSize = jointSize * point.scale;

                    // Draw joint circle
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, Math.max(2, scaledSize), 0, Math.PI * 2);
                    ctx.fill();

                    // Draw outline
                    ctx.strokeStyle = "#000";
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            };
        }
    }
});
