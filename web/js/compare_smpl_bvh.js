/**
 * Compare SMPL to BVH Viewer Widget
 * Side-by-side comparison of SMPL mesh and BVH skeleton using split view
 */

import { app } from "../../../../scripts/app.js";

console.log("[CompareSMPLBVH] Loading Compare SMPL BVH extension");

// Node types that use this viewer
const COMPARE_VIEWER_NODES = [
    {
        nodeName: "CompareSMPLtoBVH",
        extensionName: "comfy3d.comparesmplbvh",
        logPrefix: "[CompareSMPLBVH]",
        smplMeshApiPath: "/motioncapture/smpl_mesh"
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
        console.warn('[CompareSMPLBVH] Could not detect extension folder:', e);
    }
    return null;
}

function getViewerUrl(extensionFolder) {
    return `/extensions/${extensionFolder}/viewer_compare_smpl_bvh.html?v=` + Date.now();
}

function createCompareViewerExtension(config) {
    const { extensionName, nodeName, logPrefix, smplMeshApiPath } = config;

    console.log(`${logPrefix} Loading Compare SMPL BVH extension...`);

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

                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.background = "#222";
                container.style.display = "flex";
                container.style.flexDirection = "column";

                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.flexGrow = "1";
                iframe.src = getViewerUrl(extensionFolder);
                container.appendChild(iframe);

                // Store references
                this.compareViewerIframe = iframe;
                this.compareViewerReady = false;
                this.compareDataToLoad = null;

                // Add widget with computeSize for automatic resizing
                const widget = this.addDOMWidget("compare_viewer", "iframe", container);
                widget.computeSize = (w) => [w, 500];

                // Set an initial size for the node
                this.setSize([this.size[0], 500 + 30]);

                // Listen for viewer ready message
                const messageHandler = (event) => {
                    if (event.source !== iframe.contentWindow) return;
                    if (event.data.type === 'VIEWER_READY') {
                        this.compareViewerReady = true;
                        if (this.compareDataToLoad) {
                            this.loadCompareData(this.compareDataToLoad);
                        }
                    }
                };
                window.addEventListener('message', messageHandler);

                // Cleanup on removal
                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    window.removeEventListener('message', messageHandler);
                    if (originalOnRemoved) originalOnRemoved.apply(this, arguments);
                };

                // Method to load compare data
                this.loadCompareData = function(data) {
                    if (!iframe.contentWindow) {
                        this.compareDataToLoad = data;
                        return;
                    }

                    if (!this.compareViewerReady) {
                        this.compareDataToLoad = data;
                        return;
                    }

                    iframe.contentWindow.postMessage({
                        type: 'loadData',
                        smplFilename: data.smplFilename,
                        bvhContent: data.bvhContent,
                        smplMeshApiPath: smplMeshApiPath
                    }, '*');
                    this.compareDataToLoad = null;
                };

                this.onExecuted = (msg) => {
                    if (msg?.smpl_mesh_filename && msg?.bvh_content) {
                        const data = {
                            smplFilename: msg.smpl_mesh_filename[0],
                            bvhContent: msg.bvh_content[0]
                        };

                        if (this.compareViewerReady) {
                            this.loadCompareData(data);
                        } else {
                            this.compareDataToLoad = data;
                        }
                    }
                };

                return result;
            };
        }
    });

    console.log(`${logPrefix} Extension registered: ${extensionName}`);
}

// Auto-register all known compare viewer node types
COMPARE_VIEWER_NODES.forEach(config => createCompareViewerExtension(config));

export { createCompareViewerExtension, detectExtensionFolder };
