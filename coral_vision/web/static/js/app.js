        // Use structured logging (logger.js must be loaded first)
        const appLogger = typeof logger !== 'undefined' ? logger : {
            debug: () => {},
            info: () => {},
            warn: console.warn.bind(console),
            error: console.error.bind(console)
        };

        // Use error handler (error-handler.js must be loaded first)
        const handleError = typeof errorHandler !== 'undefined'
            ? (error, elementId) => {
                const { userMessage } = errorHandler.handleError(error);
                if (elementId) {
                    errorHandler.showError(userMessage, elementId);
                } else {
                    errorHandler.showError(userMessage);
                }
            }
            : (error, elementId) => {
                const message = error?.message || error?.error || String(error);
                const container = elementId ? document.getElementById(elementId) : null;
                if (container) {
                    container.innerHTML = `<div class="error-message">❌ ${message}</div>`;
                } else {
                    alert(message);
                }
            };

        // API key storage key for localStorage
        const API_KEY_STORAGE_KEY = 'coral_vision_api_key';

        // Get API key from localStorage
        function getApiKey() {
            return localStorage.getItem(API_KEY_STORAGE_KEY) || '';
        }

        // Save API key to localStorage
        function saveApiKey() {
            const input = document.getElementById('api-key-input');
            const apiKey = input.value.trim();

            if (!apiKey) {
                showApiKeyStatus('missing', '⚠️ API key cannot be empty');
                return;
            }

            localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
            showApiKeyStatus('valid', '✅ API key saved successfully');
            input.value = ''; // Clear input for security
        }

        // Show API key status
        function showApiKeyStatus(status, message) {
            const statusDiv = document.getElementById('api-key-status');
            statusDiv.className = `api-key-status ${status}`;
            statusDiv.textContent = message;
            statusDiv.style.display = 'block';

            // Auto-hide success message after 3 seconds
            if (status === 'valid') {
                setTimeout(() => {
                    statusDiv.style.display = 'none';
                }, 3000);
            }
        }

        // Check API key status on page load
        function checkApiKeyStatus() {
            const apiKey = getApiKey();
            const statusDiv = document.getElementById('api-key-status');

            if (!apiKey) {
                statusDiv.className = 'api-key-status missing';
                statusDiv.textContent = '⚠️ API key is required to use the API';
                statusDiv.style.display = 'block';
            } else {
                statusDiv.className = 'api-key-status valid';
                statusDiv.textContent = '✅ API key is configured';
                statusDiv.style.display = 'block';
            }
        }

        // Helper function to add API key headers to fetch options
        function addApiKeyHeaders(options = {}) {
            const apiKey = getApiKey();

            if (!apiKey) {
                throw new Error('API key is required. Please configure it in the API Key section above.');
            }

            const headers = options.headers || {};
            headers['X-API-Key'] = apiKey;
            return { ...options, headers };
        }

        // Initialize API key status on page load
        window.addEventListener('DOMContentLoaded', () => {
            checkApiKeyStatus();

            // Add Enter key handler for API key input
            const apiKeyInput = document.getElementById('api-key-input');
            if (apiKeyInput) {
                apiKeyInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        saveApiKey();
                    }
                });
            }
        });

        // State management
        let recognizeFiles = [];
        let enrollFiles = [];
        let cameraStreamActive = false;
        let cameraFeedInterval = null;
        let capturedFrames = [];
        let videoStream = null;
        let recognitionInterval = null;
        let overlayRedrawInterval = null;
        let lastRecognitionResults = null;

        // WebSocket connection
        let socket = null;
        let videoImage = null; // For displaying WebSocket frames
        let isConnecting = false; // Prevent multiple simultaneous connection attempts

        // Helper function to detect localhost/private IP addresses
        function isLocalhost(hostname) {
            return hostname === 'localhost' ||
                   hostname === '127.0.0.1' ||
                   hostname === '[::1]' ||
                   hostname.startsWith('192.168.') ||
                   hostname.startsWith('10.') ||
                   hostname.startsWith('172.16.') ||
                   hostname.startsWith('172.17.') ||
                   hostname.startsWith('172.18.') ||
                   hostname.startsWith('172.19.') ||
                   hostname.startsWith('172.20.') ||
                   hostname.startsWith('172.21.') ||
                   hostname.startsWith('172.22.') ||
                   hostname.startsWith('172.23.') ||
                   hostname.startsWith('172.24.') ||
                   hostname.startsWith('172.25.') ||
                   hostname.startsWith('172.26.') ||
                   hostname.startsWith('172.27.') ||
                   hostname.startsWith('172.28.') ||
                   hostname.startsWith('172.29.') ||
                   hostname.startsWith('172.30.') ||
                   hostname.startsWith('172.31.') ||
                   hostname === '0.0.0.0';
        }

        // Tab switching
        function switchTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            event.target.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');

            // Stop camera if switching away from camera tab
            if (tabName !== 'camera' && cameraStreamActive) {
                stopCameraFeed();
            }
        }

        // Drag and drop handlers
        function setupDragDrop(uploadAreaId, inputId) {
            const uploadArea = document.getElementById(uploadAreaId);

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'));
            });

            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'));
            });

            uploadArea.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                document.getElementById(inputId).files = files;
                if (inputId === 'recognize-files') {
                    handleRecognizeFiles(files);
                } else {
                    handleEnrollFiles(files);
                }
            });
        }

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        // File preview
        function createPreview(file, previewContainer) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'preview-image';
                div.innerHTML = `
                    <img src="${e.target.result}" alt="${file.name}">
                    <div style="margin-top: 5px; font-size: 0.85em; color: #666; text-align: center;">${file.name}</div>
                `;
                previewContainer.appendChild(div);
            };
            reader.readAsDataURL(file);
        }

        // Recognize handlers
        function handleRecognizeFiles(files) {
            recognizeFiles = Array.from(files);
            const previewContainer = document.getElementById('recognize-previews');
            previewContainer.innerHTML = '';

            recognizeFiles.forEach(file => createPreview(file, previewContainer));
            document.getElementById('recognize-btn').disabled = recognizeFiles.length === 0;
        }

        async function recognizeFaces() {
            const resultsDiv = document.getElementById('recognize-results');
            const threshold = parseFloat(document.getElementById('threshold').value);
            const btn = document.getElementById('recognize-btn');

            btn.disabled = true;
            resultsDiv.innerHTML = '<div class="loading">🔄 Processing images...</div>';

            try {
                const results = [];
                for (const file of recognizeFiles) {
                    const formData = new FormData();
                    formData.append('image', file);
                    formData.append('threshold', threshold);

                    const response = await fetch('/api/recognize', addApiKeyHeaders({
                        method: 'POST',
                        body: formData
                    }));

                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ error: response.statusText }));
                        if (response.status === 401) {
                            showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                            throw new Error('Authentication failed. Please check your API key.');
                        }
                        throw new Error(`Failed to process ${file.name}: ${errorData.error || response.statusText}`);
                    }

                    const result = await response.json();
                    results.push({ filename: file.name, ...result });
                }

                displayRecognitionResults(results);
            } catch (error) {
                if (error.message.includes('API key is required')) {
                    showApiKeyStatus('missing', '⚠️ ' + error.message);
                    resultsDiv.innerHTML = `<div class="error-message">❌ ${error.message}</div>`;
                } else {
                    resultsDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
                }
            } finally {
                btn.disabled = false;
            }
        }

        function displayRecognitionResults(results) {
            const resultsDiv = document.getElementById('recognize-results');

            if (results.length === 0) {
                resultsDiv.innerHTML = '<div class="error-message">No results to display</div>';
                return;
            }

            let html = '<div class="success-message">✅ Recognition complete!</div>';

            results.forEach(result => {
                html += `<div class="result-item">
                    <h4>📄 ${result.filename}</h4>
                    <p><strong>Detected Faces:</strong> ${result.faces.length}</p>`;

                result.faces.forEach((face, idx) => {
                    const bbox = face.bbox;
                    html += `
                        <div class="face-box">
                            <h5>Face ${idx + 1}</h5>
                            <p><strong>Detection Score:</strong> ${(face.score * 100).toFixed(1)}%</p>
                            <p><strong>Bounding Box:</strong> (${bbox.x.toFixed(0)}, ${bbox.y.toFixed(0)}, ${bbox.width.toFixed(0)}, ${bbox.height.toFixed(0)})</p>

                            ${face.matches && face.matches.length > 0 ? `
                                <p><strong>Matches Found:</strong> ${face.matches.length}</p>
                                <div class="matches-list">
                                    ${face.matches.map(match => {
                                        const distClass = match.distance < 0.4 ? 'good' : (match.distance < 0.6 ? 'fair' : 'poor');
                                        return `
                                            <div class="match-item">
                                                <div>
                                                    <strong>${match.name}</strong> (ID: ${match.person_id})
                                                </div>
                                                <span class="distance-badge ${distClass}">
                                                    Distance: ${match.distance.toFixed(3)}
                                                </span>
                                            </div>
                                        `;
                                    }).join('')}
                                </div>
                            ` : '<p style="color: #666;">No matches found above threshold</p>'}
                        </div>
                    `;
                });

                html += '</div>';
            });

            resultsDiv.innerHTML = html;
        }

        // Enroll handlers
        function handleEnrollFiles(files) {
            enrollFiles = Array.from(files);
            const previewContainer = document.getElementById('enroll-previews');
            previewContainer.innerHTML = '';

            enrollFiles.forEach(file => createPreview(file, previewContainer));
            updateEnrollButton();
        }

        function updateEnrollButton() {
            const personId = document.getElementById('person-id').value.trim();
            const personName = document.getElementById('person-name').value.trim();
            document.getElementById('enroll-btn').disabled =
                !personId || !personName || enrollFiles.length === 0;
        }

        // Listen to input changes
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('person-id').addEventListener('input', updateEnrollButton);
            document.getElementById('person-name').addEventListener('input', updateEnrollButton);
            setupDragDrop('recognize-upload', 'recognize-files');
            setupDragDrop('enroll-upload', 'enroll-files');
        });

        async function enrollPerson() {
            const resultsDiv = document.getElementById('enroll-results');
            const personId = document.getElementById('person-id').value.trim();
            const personName = document.getElementById('person-name').value.trim();
            const btn = document.getElementById('enroll-btn');

            btn.disabled = true;
            resultsDiv.innerHTML = '<div class="loading">🔄 Enrolling person and processing images...</div>';

            try {
                // Step 1: Create person
                const createResponse = await fetch('/api/persons', addApiKeyHeaders({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ person_id: personId, name: personName })
                }));

                if (!createResponse.ok) {
                    const error = await createResponse.json();
                    if (createResponse.status === 401) {
                        showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                        throw new Error('Authentication failed. Please check your API key.');
                    }
                    throw new Error(error.error || 'Failed to create person');
                }

                // Step 2: Upload images
                const uploadResults = [];
                for (const file of enrollFiles) {
                    const formData = new FormData();
                    formData.append('image', file);

                    const uploadResponse = await fetch(`/api/persons/${personId}/train`, addApiKeyHeaders({
                        method: 'POST',
                        body: formData
                    }));

                    if (!uploadResponse.ok) {
                        if (uploadResponse.status === 401) {
                            showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                            throw new Error('Authentication failed. Please check your API key.');
                        }
                        throw new Error(`Failed to upload ${file.name}`);
                    }

                    const result = await uploadResponse.json();
                    uploadResults.push({ filename: file.name, ...result });
                }

                displayEnrollmentResults(personId, personName, uploadResults);

                // Clear form
                document.getElementById('person-id').value = '';
                document.getElementById('person-name').value = '';
                document.getElementById('enroll-files').value = '';
                document.getElementById('enroll-previews').innerHTML = '';
                enrollFiles = [];
            } catch (error) {
                if (error.message && error.message.includes('API key is required')) {
                    showApiKeyStatus('missing', '⚠️ ' + error.message);
                    resultsDiv.innerHTML = `<div class="error-message">❌ ${error.message}</div>`;
                } else {
                    resultsDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
                }
            } finally {
                btn.disabled = false;
                updateEnrollButton();
            }
        }

        function displayEnrollmentResults(personId, personName, results) {
            const resultsDiv = document.getElementById('enroll-results');

            const totalFaces = results.reduce((sum, r) => sum + r.faces_detected, 0);
            const totalEmbeddings = results.reduce((sum, r) => sum + r.embeddings_added, 0);

            let html = `
                <div class="success-message">
                    ✅ Successfully enrolled <strong>${personName}</strong> (ID: ${personId})
                </div>
                <div class="result-item">
                    <h4>📊 Summary</h4>
                    <p><strong>Images Processed:</strong> ${results.length}</p>
                    <p><strong>Total Faces Detected:</strong> ${totalFaces}</p>
                    <p><strong>Embeddings Generated:</strong> ${totalEmbeddings}</p>
                </div>
            `;

            results.forEach(result => {
                html += `
                    <div class="result-item">
                        <h4>📄 ${result.filename}</h4>
                        <p><strong>Faces Detected:</strong> ${result.faces_detected}</p>
                        <p><strong>Embeddings Added:</strong> ${result.embeddings_added}</p>
                        ${result.faces_detected === 0 ?
                            '<p style="color: #f39c12;">⚠️ No faces detected in this image</p>' : ''}
                    </div>
                `;
            });

            resultsDiv.innerHTML = html;
        }

        // Camera feed handlers - WebSocket version with browser camera
        async function startCameraFeed() {
            const container = document.getElementById('camera-container');
            const statusDiv = document.getElementById('camera-status');
            const enrollmentSection = document.getElementById('camera-enrollment-section');
            const startBtn = document.getElementById('start-camera-btn');
            const stopBtn = document.getElementById('stop-camera-btn');
            const video = document.getElementById('camera-video');
            const overlay = document.getElementById('camera-overlay');

            // Prevent multiple simultaneous connection attempts
            if (isConnecting || cameraStreamActive) {
                appLogger.debug('Already connecting or stream active');
                return;
            }

            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<div class="loading">🔄 Requesting camera access...</div>';
            isConnecting = true;
            startBtn.disabled = true;

            try {
                // First, get browser camera access
                let constraints = {
                    video: {
                        facingMode: 'user',
                        width: { ideal: 640 },
                        height: { ideal: 480 }
                    }
                };

                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
                } else if (navigator.getUserMedia) {
                    videoStream = await new Promise((resolve, reject) => {
                        navigator.getUserMedia({ video: true }, resolve, reject);
                    });
                } else if (navigator.webkitGetUserMedia) {
                    videoStream = await new Promise((resolve, reject) => {
                        navigator.webkitGetUserMedia({ video: true }, resolve, reject);
                    });
                } else if (navigator.mozGetUserMedia) {
                    videoStream = await new Promise((resolve, reject) => {
                        navigator.mozGetUserMedia({ video: true }, resolve, reject);
                    });
                } else {
                    throw new Error('Camera API not supported in this browser.');
                }

                // Set video source
                video.srcObject = videoStream;
                container.style.display = 'block';
                enrollmentSection.style.display = 'block';

                // Initialize overlay canvas
                if (overlay) {
                    overlay.style.position = 'absolute';
                    overlay.style.top = '0';
                    overlay.style.left = '0';
                    overlay.style.pointerEvents = 'none';
                    overlay.style.zIndex = '10';
                }

                // Wait for video to be ready, then connect WebSocket
                video.onloadedmetadata = async () => {
                    // Update overlay size when video metadata loads
                    setTimeout(() => {
                        updateOverlaySize();
                    }, 100);
                    try {
                        const apiKey = getApiKey();
                        if (!apiKey) {
                            throw new Error('API key is required. Please configure it in the API Key section above.');
                        }

                        // Determine WebSocket protocol (ws:// or wss://)
                        // Use the same protocol as the current page
                        const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${wsProtocol}//${location.host}`;

                        appLogger.debug('Connecting to WebSocket', { url: wsUrl, protocol: location.protocol });

                        // Connect to WebSocket server
                        // Try websocket first, fallback to polling if needed
                        socket = io(wsUrl, {
                            auth: {
                                api_key: apiKey
                            },
                            transports: ['websocket', 'polling'],
                            reconnection: false,
                            // Add timeout for connection
                            timeout: 10000,
                            // Force new connection
                            forceNew: true
                        });

                        // Handle connection
                        socket.on('connect', () => {
                            appLogger.info('WebSocket connected');
                            isConnecting = false;

                            const threshold = parseFloat(document.getElementById('camera-threshold').value);

                            // Start video stream processing
                            socket.emit('start_video_stream', {
                                threshold: threshold
                            });
                        });

                        // Handle connection error
                        socket.on('connect_error', (error) => {
                            appLogger.error('WebSocket connection error', error);
                            isConnecting = false;
                            let errorMsg = 'Failed to connect to server. ';

                            // Provide more helpful error messages
                            if (error.message && error.message.includes('API key')) {
                                errorMsg = 'Invalid API key. Please check your API key configuration.';
                            } else if (error.message && (error.message.includes('ECONNREFUSED') || error.message.includes('Failed to fetch'))) {
                                errorMsg = 'Cannot connect to server. Please check if the server is running and accessible.';
                            } else if (error.message && error.message.includes('websocket')) {
                                errorMsg = 'WebSocket connection failed. The server may not support WebSockets or may require HTTPS.';
                            } else if (error.type === 'TransportError') {
                                errorMsg = 'Connection transport error. Trying polling fallback...';
                                // Socket.IO will automatically try polling if websocket fails
                            } else {
                                errorMsg += error.message || 'Unknown error occurred.';
                            }

                            statusDiv.innerHTML = `<div class="error-message">❌ ${errorMsg}</div>`;
                            startBtn.disabled = false;
                            stopBtn.disabled = true;
                            cameraStreamActive = false;
                            if (socket) {
                                socket.disconnect();
                                socket = null;
                            }
                        });

                        // Handle stream started
                        socket.on('stream_started', () => {
                            appLogger.info('Video stream started');

                            // Wait a bit for video to render, then update overlay
                            setTimeout(() => {
                                updateOverlaySize();
                            }, 200);

                            statusDiv.innerHTML = '<div class="success-message">✅ Camera feed active (WebSocket)</div>';
                            startBtn.disabled = true;
                            stopBtn.disabled = false;
                            cameraStreamActive = true;

                            // Start sending frames to server
                            startFrameCapture();
                        });

                        // Handle recognition results
                        socket.on('recognition_result', (data) => {
                            if (data.error) {
                                appLogger.error('Recognition error', new Error(data.error));
                                return;
                            }

                            // Update overlay size before drawing (handles mobile resize)
                            updateOverlaySize();

                            // Draw recognition results on overlay
                            if (data.faces && data.faces.length > 0) {
                                lastRecognitionResults = { faces: data.faces };
                                if (video && video.videoWidth > 0 && video.videoHeight > 0) {
                                    drawRecognitionResults(overlay, lastRecognitionResults, video.videoWidth, video.videoHeight);
                                }
                            } else {
                                // Clear overlay if no faces
                                const ctx = overlay.getContext('2d');
                                if (ctx && overlay.width > 0 && overlay.height > 0) {
                                    ctx.clearRect(0, 0, overlay.width, overlay.height);
                                }
                                lastRecognitionResults = null;
                            }
                        });

                        // Handle stream errors
                        socket.on('stream_error', (data) => {
                            appLogger.error('Stream error', new Error(data.error));
                            handleError(new Error(data.error), 'camera-status');
                        });

                        // Handle disconnection
                        socket.on('disconnect', () => {
                            appLogger.info('WebSocket disconnected');
                            if (cameraStreamActive) {
                                statusDiv.innerHTML = '<div class="error-message">❌ Connection lost. Please restart the camera feed.</div>';
                                cameraStreamActive = false;
                                startBtn.disabled = false;
                                stopBtn.disabled = true;
                            }
                        });

                    } catch (error) {
                        appLogger.error('WebSocket setup error', error);
                        isConnecting = false;
                        statusDiv.innerHTML = `<div class="error-message">❌ ${error.message || 'Failed to connect to server'}</div>`;
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                };

                // Update overlay size when window resizes or orientation changes
                let resizeTimeout;
                const handleResize = () => {
                    clearTimeout(resizeTimeout);
                    resizeTimeout = setTimeout(() => {
                        if (cameraStreamActive) {
                            updateOverlaySize();
                            // Redraw recognition results after resize
                            if (lastRecognitionResults && video && video.videoWidth > 0) {
                                drawRecognitionResults(overlay, lastRecognitionResults, video.videoWidth, video.videoHeight);
                            }
                        }
                    }, 100);
                };
                window.addEventListener('resize', handleResize);
                window.addEventListener('orientationchange', handleResize);

                video.onerror = (error) => {
                    appLogger.error('Video error', error);
                    handleError(new Error('Failed to start camera feed'), 'camera-status');
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    cameraStreamActive = false;
                    isConnecting = false;
                };

            } catch (error) {
                appLogger.error('Camera access error', error);
                isConnecting = false;
                let errorMsg = 'Failed to access camera. ';

                if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    errorMsg += 'Please allow camera access in your browser settings and reload the page.';
                } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                    errorMsg += 'No camera found on this device.';
                } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
                    errorMsg += 'Camera is already in use by another application.';
                } else if (error.message) {
                    errorMsg += error.message;
                } else {
                    errorMsg += 'Unknown error occurred.';
                }

                statusDiv.innerHTML = `<div class="error-message">❌ ${errorMsg}</div>`;
                startBtn.disabled = false;
                stopBtn.disabled = true;
                cameraStreamActive = false;
            }
        }

        // Function to capture and send frames via WebSocket
        function startFrameCapture() {
            const video = document.getElementById('camera-video');
            const canvas = document.getElementById('camera-canvas');
            const threshold = parseFloat(document.getElementById('camera-threshold').value);

            if (!video || !socket || !cameraStreamActive) return;

            // Capture frame at ~10 FPS (every 100ms) to balance performance and responsiveness
            const captureInterval = setInterval(() => {
                if (!cameraStreamActive || !socket || !socket.connected) {
                    clearInterval(captureInterval);
                    return;
                }

                try {
                    if (video.readyState !== video.HAVE_ENOUGH_DATA) return;

                    // Set canvas to video dimensions
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                    const ctx = canvas.getContext('2d');

                    // Draw current frame to canvas
                    ctx.drawImage(video, 0, 0);

                    // Convert to blob and send via WebSocket
                    canvas.toBlob((blob) => {
                        if (!blob || !socket || !socket.connected) return;

                        // Convert blob to base64
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            const base64data = reader.result.split(',')[1]; // Remove data:image/jpeg;base64, prefix

                            // Send frame to server for processing
                            socket.emit('process_frame', {
                                frame: base64data,
                                threshold: threshold,
                                timestamp: Date.now()
                            });
                        };
                        reader.readAsDataURL(blob);
                    }, 'image/jpeg', 0.85); // 85% quality

                } catch (error) {
                    appLogger.error('Frame capture error', error);
                }
            }, 100); // 10 FPS

            // Store interval ID for cleanup
            if (recognitionInterval) {
                clearInterval(recognitionInterval);
            }
            recognitionInterval = captureInterval;
        }

        function startRecognitionLoop() {
            // This function is kept for backward compatibility but is not used
            // with WebSocket streaming since recognition happens server-side
            // Recognition results are already drawn on frames by the server
        }

        async function processFrameForRecognition() {
            const video = document.getElementById('camera-video');
            const canvas = document.getElementById('camera-canvas');
            const overlay = document.getElementById('camera-overlay');
            const threshold = parseFloat(document.getElementById('camera-threshold').value);

            if (!video || video.readyState !== video.HAVE_ENOUGH_DATA) {
                return;
            }

            try {
                // Update overlay size to match current video display (handles redraw if needed)
                updateOverlaySize();

                // Set canvas to video dimensions
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');

                // Draw current frame to canvas
                ctx.drawImage(video, 0, 0);

                // Convert to blob and send to server
                canvas.toBlob(async (blob) => {
                    if (!blob) return;

                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    formData.append('threshold', threshold);

                    try {
                        const response = await fetch('/api/process_frame', addApiKeyHeaders({
                            method: 'POST',
                            body: formData
                        }));

                        if (response.ok) {
                            const result = await response.json();
                            lastRecognitionResults = result;
                            appLogger.debug('Recognition result', result);
                            // Always draw results, even if empty (will clear overlay if no faces)
                            drawRecognitionResults(overlay, result, video.videoWidth, video.videoHeight);
                        } else {
                            const error = await response.json().catch(() => ({ error: response.statusText }));
                            if (response.status === 401) {
                                showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                            }
                            appLogger.error('Recognition API error', error);
                            // On error, keep last results visible (don't clear overlay)
                        }
                    } catch (error) {
                        if (error.message && error.message.includes('API key is required')) {
                            showApiKeyStatus('missing', '⚠️ ' + error.message);
                        }
                        appLogger.error('Recognition error', error);
                    }
                }, 'image/jpeg', 0.9);
            } catch (error) {
                appLogger.error('Frame processing error', error);
            }
        }

        function updateOverlaySize() {
            const video = document.getElementById('camera-video');
            const overlay = document.getElementById('camera-overlay');

            if (video && overlay) {
                // Get the actual displayed size of the video
                const videoRect = video.getBoundingClientRect();
                const newWidth = videoRect.width;
                const newHeight = videoRect.height;

                // Only update if size actually changed to avoid clearing canvas unnecessarily
                if (overlay.width !== newWidth || overlay.height !== newHeight) {
                    const needsRedraw = overlay.width > 0 && overlay.height > 0; // Had previous size

                    // Set canvas internal dimensions to match display size
                    overlay.width = newWidth;
                    overlay.height = newHeight;

                    // Set CSS size to match
                    overlay.style.width = newWidth + 'px';
                    overlay.style.height = newHeight + 'px';

                    appLogger.debug('Overlay size updated', {
                        overlay: `${overlay.width}x${overlay.height}`,
                        video: `${video.videoWidth}x${video.videoHeight}`
                    });

                    // Redraw last results if we had them (canvas was cleared by size change)
                    if (needsRedraw && lastRecognitionResults && video.videoWidth > 0) {
                        drawRecognitionResults(overlay, lastRecognitionResults, video.videoWidth, video.videoHeight);
                    }
                }
            }
        }

        function drawRecognitionResults(canvas, result, width, height) {
            if (!canvas) {
                appLogger.error('Canvas element not found');
                return;
            }

            const ctx = canvas.getContext('2d');
            if (!ctx) {
                appLogger.error('Could not get canvas context');
                return;
            }

            // Always clear canvas first to remove old drawings
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // If no faces detected, clear overlay and return
            if (!result || !result.faces || result.faces.length === 0) {
                appLogger.debug('No faces detected in result');
                return;
            }

            appLogger.debug(`Drawing ${result.faces.length} face(s)`);

            // Canvas matches video display size
            // Video is mirrored with CSS scaleX(-1), so we need to flip X coordinates
            const video = document.getElementById('camera-video');
            if (!video) {
                console.error('Video element not found');
                return;
            }

            // Get actual displayed dimensions
            const videoRect = video.getBoundingClientRect();
            const videoDisplayWidth = videoRect.width || video.offsetWidth || canvas.width;
            const videoDisplayHeight = videoRect.height || video.offsetHeight || canvas.height;
            const scaleX = videoDisplayWidth / width;
            const scaleY = videoDisplayHeight / height;

            result.faces.forEach((face, index) => {
                if (!face.bbox) {
                    appLogger.warn(`Face ${index} missing bbox`, face);
                    return;
                }

                const bbox = face.bbox;
                // Video is mirrored with CSS, so flip X coordinates to match
                const xmin = (width - bbox.xmax) * scaleX;
                const ymin = bbox.ymin * scaleY;
                const xmax = (width - bbox.xmin) * scaleX;
                const ymax = bbox.ymax * scaleY;

                // Determine color and label
                let color = '#00ff00'; // Green for recognized
                let label = 'Unknown';
                let distance = null;

                if (face.predicted && face.accepted) {
                    label = face.predicted.name;
                    distance = face.predicted.distance;
                    color = '#00ff00'; // Green for recognized
                } else if (face.predicted) {
                    // Show best match even if not accepted (distance too high)
                    label = face.predicted.name;
                    distance = face.predicted.distance;
                    color = '#ffaa00'; // Orange for match but above threshold
                } else if (face.matches && face.matches.length > 0) {
                    // Show best match if available
                    label = face.matches[0].name;
                    distance = face.matches[0].distance;
                    color = '#ffaa00'; // Orange for match but above threshold
                } else {
                    color = '#ff0000'; // Red for unknown
                }

                // Ensure coordinates are valid
                const boxWidth = Math.abs(xmax - xmin);
                const boxHeight = Math.abs(ymax - ymin);

                if (boxWidth > 0 && boxHeight > 0 && xmin >= 0 && ymin >= 0 &&
                    xmax <= canvas.width && ymax <= canvas.height) {
                    // Draw bounding box with responsive line width
                    const lineWidth = Math.max(2, Math.min(4, boxWidth / 100));
                    ctx.strokeStyle = color;
                    ctx.lineWidth = lineWidth;
                    ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);

                    // Draw label background with responsive font
                    const fontSize = Math.max(12, Math.min(18, boxWidth / 15));
                    ctx.fillStyle = color;
                    ctx.font = `bold ${fontSize}px Arial`;
                    const labelText = distance !== null
                        ? `${label} (${distance.toFixed(3)})`
                        : `${label} (No match)`;
                    const textMetrics = ctx.measureText(labelText);
                    const textHeight = fontSize + 4;
                    const textY = Math.max(textHeight + 4, ymin);
                    const textPadding = 8;
                    ctx.fillRect(xmin, textY - textHeight - 2, textMetrics.width + textPadding, textHeight + 2);

                    // Draw label text
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(labelText, xmin + textPadding / 2, textY - 4);

                    appLogger.debug(`Drew face ${index}: ${label}`, { x: xmin.toFixed(0), y: ymin.toFixed(0) });
                } else {
                    appLogger.warn(`Invalid coordinates for face ${index}`, {
                        xmin, ymin, xmax, ymax,
                        canvas: `${canvas.width}x${canvas.height}`
                    });
                }
            });

            // Force canvas redraw
            canvas.style.display = 'block';
        }

        function updateCameraThreshold() {
            // Threshold will be used in next recognition cycle
            // No need to restart the camera
        }

        function stopCameraFeed() {
            // Prevent multiple calls
            if (!cameraStreamActive) {
                return;
            }

            const container = document.getElementById('camera-container');
            const statusDiv = document.getElementById('camera-status');
            const enrollmentSection = document.getElementById('camera-enrollment-section');
            const startBtn = document.getElementById('start-camera-btn');
            const stopBtn = document.getElementById('stop-camera-btn');
            const video = document.getElementById('camera-video');
            const overlay = document.getElementById('camera-overlay');

            // Stop recognition loop (if still running from old implementation)
            if (recognitionInterval) {
                clearInterval(recognitionInterval);
                recognitionInterval = null;
            }

            // Stop overlay redraw loop
            if (overlayRedrawInterval) {
                clearInterval(overlayRedrawInterval);
                overlayRedrawInterval = null;
            }

            // Stop WebSocket stream
            if (socket) {
                socket.emit('stop_video_stream');
                socket.disconnect();
                socket = null;
            }

            // Stop video stream (for old client-side camera implementation)
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
            }

            if (video) {
                video.srcObject = null;
                video.src = '';
            }

            // Clear video canvas
            const videoCanvas = document.getElementById('camera-video-canvas');
            if (videoCanvas) {
                const ctx = videoCanvas.getContext('2d');
                ctx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
            }

            // Clear overlay
            if (overlay) {
                const ctx = overlay.getContext('2d');
                ctx.clearRect(0, 0, overlay.width, overlay.height);
            }

            container.style.display = 'none';
            enrollmentSection.style.display = 'none';
            statusDiv.style.display = 'none';
            statusDiv.innerHTML = '';

            startBtn.disabled = false;
            stopBtn.disabled = true;
            cameraStreamActive = false;
            isConnecting = false;
            lastRecognitionResults = null;
            videoImage = null;

            // Clear captured frames
            clearCapturedFrames();
        }

        // Frame capture functions
        function captureFrame() {
            const video = document.getElementById('camera-video');
            const canvas = document.getElementById('camera-canvas');

            if (!video || !cameraStreamActive || video.readyState !== video.HAVE_ENOUGH_DATA) {
                alert('Please start the camera feed first and wait for it to load');
                return;
            }

            try {
                const ctx = canvas.getContext('2d');

                // Set canvas dimensions to match video
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;

                // Draw current frame to canvas (mirror it back for natural view)
                ctx.save();
                ctx.scale(-1, 1);
                ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
                ctx.restore();

                // Convert canvas to blob
                canvas.toBlob(function(blob) {
                    if (blob) {
                        // Create a File object from blob
                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                        const file = new File([blob], `capture_${timestamp}.jpg`, { type: 'image/jpeg' });

                        capturedFrames.push(file);
                        updateCapturedPreviews();
                        updateEnrollButton();

                        // Show brief feedback
                        const statusDiv = document.getElementById('camera-status');
                        statusDiv.innerHTML = '<div class="success-message">📷 Frame captured! (' + capturedFrames.length + ' total)</div>';
                        setTimeout(() => {
                            if (cameraStreamActive) {
                                statusDiv.innerHTML = '<div class="success-message">✅ Camera feed active (WebSocket)</div>';
                            }
                        }, 2000);
                    }
                }, 'image/jpeg', 0.95);
            } catch (error) {
                appLogger.error('Error capturing frame', error);
                handleError(new Error('Error capturing frame. Please try again.'));
            }
        }

        function clearCapturedFrames() {
            capturedFrames = [];
            updateCapturedPreviews();
            updateEnrollButton();
        }

        function updateCapturedPreviews() {
            const previewContainer = document.getElementById('captured-previews');
            const countDiv = document.getElementById('captured-count');

            previewContainer.innerHTML = '';
            countDiv.textContent = `Captured: ${capturedFrames.length} images`;

            capturedFrames.forEach((file, index) => {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const div = document.createElement('div');
                    div.className = 'preview-image';
                    div.style.position = 'relative';
                    div.style.display = 'inline-block';
                    div.innerHTML = `
                        <img src="${e.target.result}" alt="${file.name}" style="width: 100%; height: auto; max-width: 100px; max-height: 100px; object-fit: cover; border-radius: 4px; border: 2px solid #e0e0e0; display: block;">
                        <button onclick="removeCapturedFrame(${index})" style="position: absolute; top: -5px; right: -5px; background: #dc3545; color: white; border: none; border-radius: 50%; width: 24px; height: 24px; cursor: pointer; font-size: 14px; line-height: 1; min-width: 24px; touch-action: manipulation; -webkit-tap-highlight-color: rgba(220, 53, 69, 0.3);">×</button>
                    `;
                    previewContainer.appendChild(div);
                };
                reader.readAsDataURL(file);
            });
        }

        function removeCapturedFrame(index) {
            capturedFrames.splice(index, 1);
            updateCapturedPreviews();
            updateEnrollButton();
        }

        function updateEnrollButton() {
            const personId = document.getElementById('camera-person-id').value.trim();
            const personName = document.getElementById('camera-person-name').value.trim();
            const enrollBtn = document.getElementById('enroll-captured-btn');
            enrollBtn.disabled = !personId || !personName || capturedFrames.length === 0;
        }

        // Listen to input changes for enrollment form
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('camera-person-id').addEventListener('input', updateEnrollButton);
            document.getElementById('camera-person-name').addEventListener('input', updateEnrollButton);
        });

        async function enrollCapturedFrames() {
            const resultsDiv = document.getElementById('camera-enroll-results');
            const personId = document.getElementById('camera-person-id').value.trim();
            const personName = document.getElementById('camera-person-name').value.trim();
            const btn = document.getElementById('enroll-captured-btn');

            if (!personId || !personName || capturedFrames.length === 0) {
                resultsDiv.innerHTML = '<div class="error-message">❌ Please fill in all fields and capture at least one image</div>';
                resultsDiv.style.display = 'block';
                return;
            }

            btn.disabled = true;
            resultsDiv.style.display = 'block';
            resultsDiv.innerHTML = '<div class="loading">🔄 Enrolling person and processing images...</div>';

            try {
                // Step 1: Create person
                const createResponse = await fetch('/api/persons', addApiKeyHeaders({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ person_id: personId, name: personName })
                }));

                if (!createResponse.ok) {
                    const error = await createResponse.json();
                    if (createResponse.status === 401) {
                        showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                        throw new Error('Authentication failed. Please check your API key.');
                    }
                    throw new Error(error.error || 'Failed to create person');
                }

                // Step 2: Upload all captured images at once
                const formData = new FormData();
                capturedFrames.forEach(file => {
                    formData.append('images', file);
                });

                const uploadResponse = await fetch(`/api/persons/${personId}/train`, addApiKeyHeaders({
                    method: 'POST',
                    body: formData
                }));

                if (!uploadResponse.ok) {
                    const error = await uploadResponse.json();
                    if (uploadResponse.status === 401) {
                        showApiKeyStatus('invalid', '❌ Invalid API key. Please check your API key.');
                        throw new Error('Authentication failed. Please check your API key.');
                    }
                    throw new Error(error.error || 'Failed to upload images');
                }

                const result = await uploadResponse.json();
                displayCameraEnrollmentResults(personId, personName, result, capturedFrames.length);

                // Clear form and captured frames
                document.getElementById('camera-person-id').value = '';
                document.getElementById('camera-person-name').value = '';
                clearCapturedFrames();
            } catch (error) {
                if (error.message && error.message.includes('API key is required')) {
                    showApiKeyStatus('missing', '⚠️ ' + error.message);
                    resultsDiv.innerHTML = `<div class="error-message">❌ ${error.message}</div>`;
                } else {
                    resultsDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
                }
            } finally {
                btn.disabled = false;
                updateEnrollButton();
            }
        }

        function displayCameraEnrollmentResults(personId, personName, result, imageCount) {
            const resultsDiv = document.getElementById('camera-enroll-results');

            let html = `
                <div class="success-message">
                    ✅ Successfully enrolled <strong>${personName}</strong> (ID: ${personId})
                </div>
                <div class="result-item">
                    <h4>📊 Summary</h4>
                    <p><strong>Images Processed:</strong> ${result.images_processed || imageCount}</p>
                    <p><strong>Total Embeddings Generated:</strong> ${result.total_embeddings || 0}</p>
                    <p style="color: #28a745; margin-top: 10px;">
                        <strong>✅ Person is now enrolled and can be recognized in the live feed!</strong>
                    </p>
                </div>
            `;

            resultsDiv.innerHTML = html;
        }
