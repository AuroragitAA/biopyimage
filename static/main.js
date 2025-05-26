        // Global variables
        let socket;
        let currentAnalysisId = null;
        let currentBatchId = null;
        let selectedFiles = [];

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            setupEventListeners();
            checkSystemStatus();
        });

        // WebSocket initialization
        function initializeWebSocket() {
            socket = io({
                transports: ['websocket', 'polling']
            });

            socket.on('connect', function() {
                updateConnectionStatus(true);
                showNotification('Connected to live analysis system', 'success');
            });

            socket.on('disconnect', function() {
                updateConnectionStatus(false);
                showNotification('Disconnected from server', 'error');
            });

            socket.on('status', function(data) {
                updateSystemStatus(data.analyzer_ready);
            });

            socket.on('analysis_progress', function(data) {
                updateAnalysisProgress(data.progress, data.stage);
            });

            socket.on('analysis_complete', function(data) {
                handleAnalysisComplete(data);
            });

            socket.on('analysis_error', function(data) {
                handleAnalysisError(data.error);
            });

            socket.on('batch_progress', function(data) {
                updateBatchProgress(data.progress, data.stage, data.completed, data.total);
            });

            socket.on('batch_complete', function(data) {
                handleBatchComplete(data);
            });

            socket.on('batch_error', function(data) {
                handleBatchError(data.error);
            });
        }

        // Event listeners setup
        function setupEventListeners() {
            // File input
            const imageInput = document.getElementById('imageInput');
            imageInput.addEventListener('change', handleFileSelect);

            // Batch file input
            const batchInput = document.getElementById('batchInput');
            batchInput.addEventListener('change', handleBatchFileSelect);

            // Upload area drag and drop
            setupDragAndDrop('uploadArea', 'imageInput');
            setupDragAndDrop('batchUploadArea', 'batchInput');

            // Analysis button
            document.getElementById('analyzeBtn').addEventListener('click', startAnalysis);

            // Batch analysis button
            document.getElementById('batchAnalyzeBtn').addEventListener('click', startBatchAnalysis);

            // Tab switching
            document.querySelectorAll('.tab-button').forEach(button => {
                button.addEventListener('click', function() {
                    switchTab(this.dataset.tab);
                });
            });

            // Cancel button
            document.getElementById('cancelBtn').addEventListener('click', cancelAnalysis);
        }

        // Drag and drop setup
        function setupDragAndDrop(areaId, inputId) {
            const area = document.getElementById(areaId);
            const input = document.getElementById(inputId);

            area.addEventListener('dragover', function(e) {
                e.preventDefault();
                area.classList.add('dragover');
            });

            area.addEventListener('dragleave', function(e) {
                e.preventDefault();
                area.classList.remove('dragover');
            });

            area.addEventListener('drop', function(e) {
                e.preventDefault();
                area.classList.remove('dragover');
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    if (inputId === 'imageInput') {
                        input.files = files;
                        handleFileSelect({ target: input });
                    } else {
                        input.files = files;
                        handleBatchFileSelect({ target: input });
                    }
                }
            });
        }

        // File selection handling
        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                document.getElementById('analyzeBtn').disabled = false;
                showNotification(`Image selected: ${file.name}`, 'info');
            }
        }

        function handleBatchFileSelect(event) {
            const files = Array.from(event.target.files);
            selectedFiles = files;
            
            if (files.length > 0) {
                updateBatchFileList(files);
                document.getElementById('batchAnalyzeBtn').disabled = false;
                showNotification(`${files.length} images selected for batch analysis`, 'info');
            }
        }

        function updateBatchFileList(files) {
            const fileList = document.getElementById('batchFileList');
            fileList.innerHTML = '';
            
            files.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <span>${file.name} (${formatFileSize(file.size)})</span>
                    <button class="btn btn-secondary" style="padding: 5px 10px;" onclick="removeFile(${index})">❌</button>
                `;
                fileList.appendChild(fileItem);
            });
            
            fileList.style.display = files.length > 0 ? 'block' : 'none';
        }

        function removeFile(index) {
            selectedFiles.splice(index, 1);
            updateBatchFileList(selectedFiles);
            
            if (selectedFiles.length === 0) {
                document.getElementById('batchAnalyzeBtn').disabled = true;
            }
        }

        function clearBatchFiles() {
            selectedFiles = [];
            document.getElementById('batchFileList').style.display = 'none';
            document.getElementById('batchInput').value = '';
            document.getElementById('batchAnalyzeBtn').disabled = true;
        }

        // Analysis functions
        async function startAnalysis() {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showNotification('Please select an image file', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('image', file);
            formData.append('pixel_ratio', document.getElementById('pixelRatio').value);
            formData.append('debug_mode', document.getElementById('debugMode').checked);
            formData.append('auto_export', document.getElementById('autoExport').checked);

            try {
                setAnalysisInProgress(true);
                
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.analysis_id) {
                    currentAnalysisId = result.analysis_id;
                    socket.emit('join_analysis', { analysis_id: result.analysis_id });
                    showNotification('Analysis started - receiving live updates', 'success');
                } else {
                    throw new Error(result.error || 'Failed to start analysis');
                }

            } catch (error) {
                setAnalysisInProgress(false);
                showNotification(`Analysis failed: ${error.message}`, 'error');
            }
        }

        async function startBatchAnalysis() {
            if (selectedFiles.length === 0) {
                showNotification('Please select images for batch analysis', 'error');
                return;
            }

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('images', file);
            });

            try {
                setBatchAnalysisInProgress(true);
                
                const response = await fetch('/api/batch/analyze', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                
                if (result.batch_id) {
                    currentBatchId = result.batch_id;
                    socket.emit('join_analysis', { analysis_id: result.batch_id });
                    showNotification(`Batch analysis started: ${result.total_images} images`, 'success');
                } else {
                    throw new Error(result.error || 'Failed to start batch analysis');
                }

            } catch (error) {
                setBatchAnalysisInProgress(false);
                showNotification(`Batch analysis failed: ${error.message}`, 'error');
            }
        }

        function cancelAnalysis() {
            if (currentAnalysisId) {
                socket.emit('leave_analysis', { analysis_id: currentAnalysisId });
                currentAnalysisId = null;
            }
            setAnalysisInProgress(false);
            showNotification('Analysis cancelled', 'info');
        }

        // Progress update functions
        function updateAnalysisProgress(progress, stage) {
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = stage;
            
            if (progress > 0) {
                document.getElementById('liveUpdateIndicator').style.display = 'inline-flex';
            }
        }

        function updateBatchProgress(progress, stage, completed, total) {
            document.getElementById('batchProgressFill').style.width = progress + '%';
            document.getElementById('batchProgressText').textContent = `${stage} (${completed}/${total})`;
        }

        // Analysis completion handlers
        async function handleAnalysisComplete(data) {
            setAnalysisInProgress(false);
            document.getElementById('liveUpdateIndicator').style.display = 'none';
            
            showNotification(`Analysis complete: ${data.total_cells} cells detected`, 'success');
            
            // Fetch full results
            try {
                const response = await fetch(`/api/analysis/${data.analysis_id}`);
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                showNotification('Failed to load analysis results', 'error');
            }
        }

        function handleAnalysisError(error) {
            setAnalysisInProgress(false);
            document.getElementById('liveUpdateIndicator').style.display = 'none';
            showNotification(`Analysis error: ${error}`, 'error');
        }

        async function handleBatchComplete(data) {
            setBatchAnalysisInProgress(false);
            
            showNotification(`Batch complete: ${data.success_rate.toFixed(1)}% success rate`, 'success');
            
            // Fetch batch results
            try {
                const response = await fetch(`/api/analysis/${data.batch_id}`);
                const result = await response.json();
                displayBatchResults(result);
            } catch (error) {
                showNotification('Failed to load batch results', 'error');
            }
        }

        function handleBatchError(error) {
            setBatchAnalysisInProgress(false);
            showNotification(`Batch error: ${error}`, 'error');
        }

        // Display functions
        function displayResults(result) {
            if (!result.success) {
                showNotification(`Analysis failed: ${result.error}`, 'error');
                return;
            }

            // Show results panel
            document.getElementById('resultsEmpty').style.display = 'none';
            document.getElementById('resultsContent').style.display = 'block';

            // Update summary
            updateSummary(result.summary);

            // Load visualizations
            loadVisualizations(result.analysis_id);

            // Update cell data table
            updateCellDataTable(result.cell_data);
        }

        function updateSummary(summary) {
            const summaryGrid = document.getElementById('summaryGrid');
            summaryGrid.innerHTML = `
                <div class="summary-item">
                    <span class="summary-value">${summary.total_cells}</span>
                    <span class="summary-label">Total Cells</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${summary.avg_area.toFixed(1)}</span>
                    <span class="summary-label">Avg Area (px)</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${summary.chlorophyll_ratio.toFixed(1)}%</span>
                    <span class="summary-label">Chlorophyll</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${summary.coverage_percent ? summary.coverage_percent.toFixed(1) : 'N/A'}%</span>
                    <span class="summary-label">Coverage</span>
                </div>
                <div class="summary-item">
                    <span class="summary-value">${summary.cell_density ? summary.cell_density.toFixed(2) : 'N/A'}</span>
                    <span class="summary-label">Density</span>
                </div>
            `;
        }

        async function loadVisualizations(analysisId) {
            try {
                const response = await fetch(`/api/visualizations/${analysisId}`);
                const visualizations = await response.json();

                // Load each visualization
                Object.keys(visualizations).forEach(key => {
                    const imgElement = document.getElementById(key + 'Image');
                    if (imgElement && visualizations[key]) {
                        imgElement.src = 'data:image/png;base64,' + visualizations[key];
                        imgElement.style.display = 'block';
                        imgElement.nextElementSibling.style.display = 'none';
                    }
                });

            } catch (error) {
                showNotification('Failed to load visualizations', 'error');
            }
        }

        function updateCellDataTable(cellData) {
            const tableContainer = document.getElementById('cellDataTable');
            
            if (!cellData || cellData.length === 0) {
                tableContainer.innerHTML = '<p style="text-align: center; padding: 20px; color: #7f8c8d;">No cell data available</p>';
                return;
            }

            // Create table
            const table = document.createElement('table');
            
            // Headers
            const headers = ['Cell ID', 'Area (px)', 'Perimeter', 'Circularity', 'Chlorophyll', 'Health Score'];
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            
            headers.forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            // Body
            const tbody = document.createElement('tbody');
            cellData.forEach(cell => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${cell.cell_id || 'N/A'}</td>
                    <td>${(cell.area_pixels || cell.area || 0).toFixed(1)}</td>
                    <td>${(cell.perimeter || 0).toFixed(1)}</td>
                    <td>${(cell.circularity || 0).toFixed(3)}</td>
                    <td>${(cell.chlorophyll_content || 0).toFixed(3)}</td>
                    <td>${(cell.health_score || 0).toFixed(3)}</td>
                `;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);

            tableContainer.innerHTML = '';
            tableContainer.appendChild(table);
        }

        // UI state functions
        function setAnalysisInProgress(inProgress) {
            document.getElementById('analyzeBtn').disabled = inProgress;
            document.getElementById('cancelBtn').style.display = inProgress ? 'inline-flex' : 'none';
            document.getElementById('progressContainer').style.display = inProgress ? 'block' : 'none';
            
            if (!inProgress) {
                document.getElementById('progressFill').style.width = '0%';
                document.getElementById('progressText').textContent = 'Ready to analyze...';
            }
        }

        function setBatchAnalysisInProgress(inProgress) {
            document.getElementById('batchAnalyzeBtn').disabled = inProgress;
            document.getElementById('batchProgressContainer').style.display = inProgress ? 'block' : 'none';
            
            if (!inProgress) {
                document.getElementById('batchProgressFill').style.width = '0%';
                document.getElementById('batchProgressText').textContent = 'Ready for batch analysis...';
            }
        }

        function updateConnectionStatus(connected) {
            const status = document.getElementById('connectionStatus');
            if (connected) {
                status.className = 'connection-status connected';
                status.innerHTML = '🔗 Connected';
            } else {
                status.className = 'connection-status disconnected';
                status.innerHTML = '🔌 Disconnected';
            }
        }

        function updateSystemStatus(analyzerReady) {
            const status = document.getElementById('systemStatus');
            if (analyzerReady) {
                status.className = 'status-indicator status-ready';
                status.innerHTML = '<span>✅</span><span>System Ready</span>';
            } else {
                status.className = 'status-indicator status-error';
                status.innerHTML = '<span>❌</span><span>System Error</span>';
            }
        }

        // Tab switching
        function switchTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`tab-${tabName}`).classList.add('active');
        }

        // Export functions
        async function exportResults(format) {
            if (!currentAnalysisId) {
                showNotification('No analysis results to export', 'error');
                return;
            }

            try {
                const response = await fetch(`/api/export/${currentAnalysisId}/${format}`);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `wolffia_analysis_${currentAnalysisId}.${format}`;
                    a.click();
                    window.URL.revokeObjectURL(url);
                    
                    showNotification(`Results exported as ${format.toUpperCase()}`, 'success');
                } else {
                    throw new Error('Export failed');
                }
            } catch (error) {
                showNotification(`Export failed: ${error.message}`, 'error');
            }
        }

        // Utility functions
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function showNotification(message, type = 'info') {
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.textContent = message;
            
            document.getElementById('notificationContainer').appendChild(notification);
            
            setTimeout(() => {
                notification.classList.add('show');
            }, 100);
            
            setTimeout(() => {
                notification.classList.remove('show');
                setTimeout(() => {
                    notification.remove();
                }, 300);
            }, 4000);
        }

        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                updateSystemStatus(data.status === 'healthy');
            } catch (error) {
                updateSystemStatus(false);
            }
        }

        function displayBatchResults(result) {
            if (result.batch_summary) {
                const summary = result.batch_summary;
                showNotification(
                    `Batch complete: ${summary.successful}/${summary.total_images} successful (${summary.success_rate.toFixed(1)}%)`,
                    'success'
                );
            }
        }


        // WebSocket initialization
        function initializeWebSocket() {
            // Check if Socket.IO is available
            if (typeof io === 'undefined') {
                console.warn('Socket.IO not available - using polling fallback');
                updateConnectionStatus(false);
                // Set up polling fallback for progress updates
                setupPollingFallback();
                return;
            }

            socket = io({
                transports: ['websocket', 'polling']
            });

            socket.on('connect', function() {
                updateConnectionStatus(true);
                showNotification('Connected to live analysis system', 'success');
            });

            socket.on('disconnect', function() {
                updateConnectionStatus(false);
                showNotification('Disconnected from server', 'error');
            });

            socket.on('status', function(data) {
                updateSystemStatus(data.analyzer_ready);
            });

            socket.on('analysis_progress', function(data) {
                updateAnalysisProgress(data.progress, data.stage);
            });

            socket.on('analysis_complete', function(data) {
                handleAnalysisComplete(data);
            });

            socket.on('analysis_error', function(data) {
                handleAnalysisError(data.error);
            });

            socket.on('batch_progress', function(data) {
                updateBatchProgress(data.progress, data.stage, data.completed, data.total);
            });

            socket.on('batch_complete', function(data) {
                handleBatchComplete(data);
            });

            socket.on('batch_error', function(data) {
                handleBatchError(data.error);
            });
        }

        // Polling fallback for when WebSocket is not available
        function setupPollingFallback() {
            console.log('Setting up polling fallback for progress updates');
            
            // This will be used to poll for progress when WebSocket is not available
            window.pollingInterval = null;
            
            window.startProgressPolling = function(analysisId) {
                if (window.pollingInterval) {
                    clearInterval(window.pollingInterval);
                }
                
                window.pollingInterval = setInterval(async function() {
                    try {
                        const response = await fetch(`/api/analysis/${analysisId}`);
                        const data = await response.json();
                        
                        if (data.status === 'running') {
                            updateAnalysisProgress(data.progress, data.stage);
                        } else if (data.success !== undefined) {
                            clearInterval(window.pollingInterval);
                            if (data.success) {
                                handleAnalysisComplete({
                                    analysis_id: analysisId,
                                    total_cells: data.total_cells,
                                    quality_score: data.quality_score,
                                    processing_time: data.processing_time
                                });
                            } else {
                                handleAnalysisError(data.error || 'Analysis failed');
                            }
                        }
                    } catch (error) {
                        console.error('Polling error:', error);
                    }
                }, 1000); // Poll every second
            };
            
            window.stopProgressPolling = function() {
                if (window.pollingInterval) {
                    clearInterval(window.pollingInterval);
                    window.pollingInterval = null;
                }
            };
        }