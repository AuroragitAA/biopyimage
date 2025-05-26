        // Global variables
        let socket;
        let currentAnalysisId = null;
        let currentBatchId = null;
        let selectedFiles = [];
        let analysisResults = {}; // Store analysis results
        let pollingInterval = null;


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


// Load comprehensive visualizations
async function loadComprehensiveVisualizations(analysisId) {
    try {
        const response = await fetch(`/api/visualizations/comprehensive/${analysisId}`);
        const visualizations = await response.json();
        
        // Display each visualization type
        if (visualizations.biomass_charts) {
            document.getElementById('biomassCharts').innerHTML = 
                `<img src="data:image/png;base64,${visualizations.biomass_charts}" style="width: 100%;">`;
        }
        
        if (visualizations.spectral_charts) {
            document.getElementById('spectralCharts').innerHTML = 
                `<img src="data:image/png;base64,${visualizations.spectral_charts}" style="width: 100%;">`;
        }
        
        if (visualizations.similarity_charts) {
            document.getElementById('similarityCharts').innerHTML = 
                `<img src="data:image/png;base64,${visualizations.similarity_charts}" style="width: 100%;">`;
        }
        
        if (visualizations.temporal_charts) {
            document.getElementById('temporalCharts').innerHTML = 
                `<img src="data:image/png;base64,${visualizations.temporal_charts}" style="width: 100%;">`;
        }
        
    } catch (error) {
        showNotification('Failed to load comprehensive visualizations', 'error');
    }
}

// Export comprehensive results
async function exportComprehensiveResults() {
    if (!currentAnalysisId) {
        showNotification('No analysis results to export', 'error');
        return;
    }
    
    try {
        const response = await fetch(`/api/export/comprehensive/${currentAnalysisId}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `wolffia_comprehensive_analysis_${currentAnalysisId}.xlsx`;
            a.click();
            window.URL.revokeObjectURL(url);
            
            showNotification('Comprehensive results exported', 'success');
        } else {
            throw new Error('Export failed');
        }
    } catch (error) {
        showNotification(`Export failed: ${error.message}`, 'error');
    }
}


// Add these functions to handle comprehensive analysis results
function displayComprehensiveResults(result) {
    // Show comprehensive sections
    document.getElementById('comprehensiveCharts').style.display = 'block';
    document.getElementById('histogramSection').style.display = 'block';
    
    // Display biomass chart
    if (result.biomass_analysis) {
        displayBiomassChart(result.biomass_analysis);
    }
    
    // Display spectral analysis
    if (result.spectral_analysis) {
        displaySpectralChart(result.spectral_analysis);
    }
    
    // Display temporal tracking if available
    if (result.temporal_analysis) {
        displayTemporalChart(result.temporal_analysis);
    }
    
    // Display cell similarity
    if (result.similarity_analysis) {
        displaySimilarityChart(result.similarity_analysis);
    }
    
    // Display histograms
    if (result.cell_data && result.cell_data.length > 0) {
        displayHistograms(result.cell_data);
    }
}

function displayBiomassChart(biomassData) {
    const container = document.getElementById('biomassChart');
    container.innerHTML = `
        <div class="biomass-results">
            <div class="biomass-estimate">
                <h5>Combined Estimate</h5>
                <p class="biomass-value">${biomassData.combined_estimate.fresh_biomass_g.toFixed(4)} g</p>
                <p class="biomass-detail">Dry: ${biomassData.combined_estimate.dry_biomass_g.toFixed(4)} g</p>
                <p class="biomass-confidence">CI: ${biomassData.combined_estimate.confidence_interval[0].toFixed(4)} - ${biomassData.combined_estimate.confidence_interval[1].toFixed(4)} g</p>
            </div>
            <div class="biomass-methods">
                <p>Area-based: ${biomassData.area_based.fresh_biomass_g.toFixed(4)} g</p>
                <p>Chlorophyll-based: ${biomassData.chlorophyll_based.estimated_biomass_g.toFixed(4)} g</p>
                <p>Allometric: ${biomassData.allometric.estimated_biomass_g.toFixed(4)} g</p>
            </div>
        </div>
    `;
}

function displaySpectralChart(spectralData) {
    const container = document.getElementById('spectralChart');
    const stats = spectralData.population_statistics;
    
    container.innerHTML = `
        <div class="spectral-results">
            <p>Mean Wavelength: ${stats.mean_wavelength.toFixed(1)} nm</p>
            <p>Green Intensity (550nm): ${stats.mean_green_intensity.toFixed(1)}</p>
            <p>Vegetation Index: ${stats.mean_vegetation_index.toFixed(3)}</p>
            <p>Photosynthetic Efficiency: ${(stats.photosynthetic_efficiency_index * 100).toFixed(1)}%</p>
            <p>Mean Chlorophyll: ${stats.mean_chlorophyll_content.toFixed(1)}%</p>
        </div>
    `;
}










// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing BIOIMAGIN System...');
    initializeWebSocket();
    setupEventListeners();
    checkSystemStatus();
});

// WebSocket initialization with proper error handling
function initializeWebSocket() {
    console.log('[WS] Initializing WebSocket connection...');
    
    // Check if Socket.IO is available
    if (typeof io === 'undefined') {
        console.warn('[WS] Socket.IO not available - using polling fallback');
        updateConnectionStatus(false);
        setupPollingFallback();
        return;
    }

    try {
        socket = io({
            transports: ['websocket', 'polling'],
            timeout: 20000,
            forceNew: true
        });

        // Connection events
        socket.on('connect', function() {
            console.log('[WS] ✅ Connected to server');
            updateConnectionStatus(true);
            showNotification('Connected to BIOIMAGIN Live Analysis', 'success');
        });

        socket.on('disconnect', function() {
            console.log('[WS] ❌ Disconnected from server');
            updateConnectionStatus(false);
            showNotification('Disconnected from server', 'error');
        });

        socket.on('connect_error', function(error) {
            console.error('[WS] Connection error:', error);
            updateConnectionStatus(false);
        });

        // System status
        socket.on('status', function(data) {
            console.log('[WS] Status update:', data);
            updateSystemStatus(data.analyzer_ready);
        });

        // Analysis events - FIXED
        socket.on('analysis_progress', function(data) {
            console.log('[WS] Progress:', data);
            updateAnalysisProgress(data.progress, data.stage);
        });

        socket.on('analysis_complete', function(data) {
            console.log('[WS] ✅ Analysis complete:', data);
            handleAnalysisComplete(data);
        });

        socket.on('analysis_error', function(data) {
            console.error('[WS] ❌ Analysis error:', data);
            handleAnalysisError(data.error);
        });

        // Batch events
        socket.on('batch_progress', function(data) {
            console.log('[WS] Batch progress:', data);
            updateBatchProgress(data.progress, data.stage, data.completed, data.total);
        });

        socket.on('batch_complete', function(data) {
            console.log('[WS] ✅ Batch complete:', data);
            handleBatchComplete(data);
        });

        socket.on('batch_error', function(data) {
            console.error('[WS] ❌ Batch error:', data);
            handleBatchError(data.error);
        });

    } catch (error) {
        console.error('[WS] WebSocket initialization failed:', error);
        updateConnectionStatus(false);
        setupPollingFallback();
    }
}

// Polling fallback when WebSocket fails
function setupPollingFallback() {
    console.log('[POLLING] Setting up polling fallback...');
    // This will be used if WebSocket fails
}

function startPollingBackup(analysisId) {
    if (!analysisId) {
        console.warn('[POLLING] No analysis ID provided!');
        return;
    }
    console.log(`[POLLING] Starting backup polling for ${analysisId}`);
    
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/analysis/${analysisId}`);
            if (response.ok) {
                const result = await response.json();
                if (result.success) {
                    clearInterval(pollingInterval);
                    analysisResults = result;
                    displayResults(result);
                    setAnalysisInProgress(false);
                    showNotification(`Analysis complete: ${result.total_cells} cells detected`, 'success');
                }
            }
        } catch (error) {
            console.error('[POLLING] Error:', error);
        }
    }, 2000); // Poll every 2 seconds
}

// Event listeners setup
function setupEventListeners() {
    console.log('[SETUP] Setting up event listeners...');
    
    // File input
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', handleFileSelect);
    }

    // Batch file input
    const batchInput = document.getElementById('batchInput');
    if (batchInput) {
        batchInput.addEventListener('change', handleBatchFileSelect);
    }

    // Upload areas drag and drop
    setupDragAndDrop('uploadArea', 'imageInput');
    setupDragAndDrop('batchUploadArea', 'batchInput');

    // Analysis button
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', startAnalysis);
    }

    // Batch analysis button
    const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
    if (batchAnalyzeBtn) {
        batchAnalyzeBtn.addEventListener('click', startBatchAnalysis);
    }

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            switchTab(this.dataset.tab);
        });
    });

    // Cancel button
    const cancelBtn = document.getElementById('cancelBtn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', cancelAnalysis);
    }

    // Analysis type change
    const analysisType = document.getElementById('analysisType');
    if (analysisType) {
        analysisType.addEventListener('change', function(e) {
            const timestampGroup = document.getElementById('timestampGroup');
            if (timestampGroup) {
                timestampGroup.style.display = e.target.value === 'temporal' ? 'block' : 'none';
            }
        });
    }

    console.log('[SETUP] ✅ Event listeners set up successfully');
}

// Drag and drop setup
function setupDragAndDrop(areaId, inputId) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);

    if (!area || !input) return;

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
            input.files = files;
            if (inputId === 'imageInput') {
                handleFileSelect({ target: input });
            } else {
                handleBatchFileSelect({ target: input });
            }
        }
    });
}

// File selection handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('[FILE] Image selected:', file.name);
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
        showNotification(`Image selected: ${file.name}`, 'info');
    }
}

function handleBatchFileSelect(event) {
    const files = Array.from(event.target.files);
    selectedFiles = files;
    
    if (files.length > 0) {
        updateBatchFileList(files);
        const batchAnalyzeBtn = document.getElementById('batchAnalyzeBtn');
        if (batchAnalyzeBtn) {
            batchAnalyzeBtn.disabled = false;
        }
        showNotification(`${files.length} images selected for batch analysis`, 'info');
    }
}

function updateBatchFileList(files) {
    const batchFileList = document.getElementById('batchFileList');
    if (!batchFileList) return;
    
    batchFileList.style.display = 'block';
    batchFileList.innerHTML = '';
    
    files.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${file.name}</span>
            <span>${(file.size / 1024 / 1024).toFixed(2)} MB</span>
        `;
        batchFileList.appendChild(fileItem);
    });
}

// Analysis functions - FIXED
async function startAnalysis() {
    console.log('[ANALYSIS] Starting analysis...');
    
    const fileInput = document.getElementById('imageInput');
    const file = fileInput?.files[0];
    const analysisType = document.getElementById('analysisType')?.value || 'standard';
    
    if (!file) {
        showNotification('Please select an image file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('pixel_ratio', document.getElementById('pixelRatio')?.value || '1.0');
    formData.append('debug_mode', document.getElementById('debugMode')?.checked || false);
    formData.append('auto_export', document.getElementById('autoExport')?.checked || false);
    
    // Add timestamp if temporal analysis
    if (analysisType === 'temporal') {
        const timestamp = document.getElementById('imageTimestamp')?.value;
        if (timestamp) {
            formData.append('timestamp', timestamp);
        }
    }

    try {
        setAnalysisInProgress(true);
        
        // Choose endpoint based on analysis type
        const endpoint = analysisType === 'comprehensive' ? '/api/analyze/comprehensive' : '/api/analyze';
        
        console.log(`[ANALYSIS] Sending request to ${endpoint}`);
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('[ANALYSIS] Server response:', result);
        
        if (result.analysis_id) {
            currentAnalysisId = result.analysis_id;
            
            // Join WebSocket room for real-time updates
            if (socket && socket.connected) {
                socket.emit('join_analysis', { analysis_id: result.analysis_id });
                console.log(`[WS] Joined analysis room: ${result.analysis_id}`);
            }
            
            showNotification('Analysis started - receiving live updates', 'success');
            
            // Start polling as backup if WebSocket fails
            setTimeout(() => {
                startPollingBackup(result.analysis_id);
            }, 10000); // Start polling after 10 seconds as backup
            
        } else if (result.success) {
            // Direct result (not using WebSocket)
            console.log('[ANALYSIS] Direct result received');
            handleDirectResult(result);
        } else {
            throw new Error(result.error || 'Failed to start analysis');
        }

    } catch (error) {
        console.error('[ANALYSIS] Error:', error);
        setAnalysisInProgress(false);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    }
}

// FIXED: Analysis completion handler
async function handleAnalysisComplete(data) {
    console.log('[RESULT] Handling analysis completion:', data);
    
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    setAnalysisInProgress(false);
    document.getElementById('liveUpdateIndicator').style.display = 'none';
    
    const totalCells = data.total_cells || 0;
    showNotification(`Analysis complete: ${totalCells} cells detected`, 'success');
    
    // Fetch full results
    try {
        console.log(`[RESULT] Fetching full results for analysis ${data.analysis_id}`);
        const response = await fetch(`/api/analysis/${data.analysis_id}`);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch results: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('[RESULT] Full analysis result:', result);
        
        // Store results globally
        analysisResults = result;
        
        // Display results
        displayResults(result);
        
    } catch (error) {
        console.error('[RESULT] Failed to load analysis results:', error);
        showNotification('Failed to load analysis results', 'error');
    }
}

// NEW: Handle direct results (when WebSocket is not used)
function handleDirectResult(result) {
    console.log('[RESULT] Handling direct result:', result);
    
    setAnalysisInProgress(false);
    analysisResults = result;
    displayResults(result);
    
    const totalCells = result.total_cells || 0;
    showNotification(`Analysis complete: ${totalCells} cells detected`, 'success');
}

function handleAnalysisError(error) {
    console.error('[ERROR] Analysis error:', error);
    setAnalysisInProgress(false);
    showNotification(`Analysis failed: ${error}`, 'error');
}

// FIXED: Results display function
function displayResults(result) {
    console.log('[DISPLAY] Displaying results:', result);
    
    if (!result || !result.success) {
        showNotification(`Analysis failed: ${result?.error || 'Unknown error'}`, 'error');
        return;
    }

    // Show results panel and hide empty state
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsContent = document.getElementById('resultsContent');
    
    if (resultsEmpty) resultsEmpty.style.display = 'none';
    if (resultsContent) resultsContent.style.display = 'block';

    // Update summary
    updateSummary(result.summary || result);

    // Load visualizations
    if (result.analysis_id) {
        loadVisualizations(result.analysis_id);
    } else {
        // Create basic visualizations from result data
        createBasicVisualizations(result);
    }

    // Update cell data table
    updateCellDataTable(result.cell_data || []);

    // Display comprehensive results if available
    if (result.biomass_analysis || result.spectral_analysis || result.similarity_analysis || result.temporal_analysis) {
        displayComprehensiveResults(result);
    }

    // Show live dashboard
    const liveDashboard = document.getElementById('liveDashboard');
    if (liveDashboard) {
        liveDashboard.style.display = 'block';
        updateLiveDashboard(result);
    }

    console.log('[DISPLAY] ✅ Results displayed successfully');
}

// FIXED: Summary update function
function updateSummary(data) {
    console.log('[SUMMARY] Updating summary with data:', data);
    
    const summaryGrid = document.getElementById('summaryGrid');
    if (!summaryGrid) return;

    // Extract summary data with fallbacks
    const summary = data.summary || data;
    const totalCells = data.total_cells || summary.total_cells || 0;
    const avgArea = summary.avg_area || summary.mean_area_pixels || summary.morphological_statistics?.mean_area || 0;
    const chlorophyllRatio = summary.chlorophyll_ratio || summary.mean_chlorophyll || summary.biological_statistics?.mean_chlorophyll || 0;
    const coverage = summary.coverage_percent || summary.spatial_analysis?.image_coverage || 0;
    const density = summary.cell_density || summary.spatial_analysis?.cell_density || 0;

    summaryGrid.innerHTML = `
        <div class="summary-item">
            <span class="summary-value">${totalCells}</span>
            <span class="summary-label">Total Cells</span>
        </div>
        <div class="summary-item">
            <span class="summary-value">${avgArea.toFixed(1)}</span>
            <span class="summary-label">Avg Area (px)</span>
        </div>
        <div class="summary-item">
            <span class="summary-value">${(chlorophyllRatio * 100).toFixed(1)}%</span>
            <span class="summary-label">Chlorophyll</span>
        </div>
        <div class="summary-item">
            <span class="summary-value">${coverage.toFixed(1)}%</span>
            <span class="summary-label">Coverage</span>
        </div>
        <div class="summary-item">
            <span class="summary-value">${density.toFixed(2)}</span>
            <span class="summary-label">Density</span>
        </div>
    `;
    
    console.log('[SUMMARY] ✅ Summary updated');
}

// FIXED: Load visualizations
async function loadVisualizations(analysisId) {
    console.log(`[VIZ] Loading visualizations for analysis ${analysisId}`);
    
    try {
        const response = await fetch(`/api/visualizations/${analysisId}`);
        
        if (!response.ok) {
            console.warn(`[VIZ] Visualizations not available: ${response.status}`);
            return;
        }
        
        const visualizations = await response.json();
        console.log('[VIZ] Loaded visualizations:', Object.keys(visualizations));

        // Load each visualization
        Object.keys(visualizations).forEach(key => {
            const imgElement = document.getElementById(key + 'Image');
            if (imgElement && visualizations[key]) {
                imgElement.src = 'data:image/png;base64,' + visualizations[key];
                imgElement.style.display = 'block';
                
                // Hide placeholder text
                const nextElement = imgElement.nextElementSibling;
                if (nextElement && nextElement.tagName === 'P') {
                    nextElement.style.display = 'none';
                }
                
                console.log(`[VIZ] ✅ Loaded ${key} visualization`);
            }
        });

    } catch (error) {
        console.error('[VIZ] Failed to load visualizations:', error);
        showNotification('Some visualizations may not be available', 'warning');
    }
}

function createBasicVisualizations(result) {
    console.log('[VIZ] Creating basic visualizations from result data');
    // Placeholder for creating visualizations from result data when API isn't available
}

// FIXED: Cell data table update
function updateCellDataTable(cellData) {
    console.log('[TABLE] Updating cell data table with', cellData?.length || 0, 'cells');
    
    const tableContainer = document.getElementById('cellDataTable');
    if (!tableContainer) return;
    
    if (!cellData || cellData.length === 0) {
        tableContainer.innerHTML = '<p style="text-align: center; padding: 20px; color: #7f8c8d;">No cell data available</p>';
        return;
    }

    // Create table with enhanced data
    const table = document.createElement('table');
    table.className = 'table table-striped table-hover';
    
    // Headers - include all available columns
    const sampleCell = cellData[0];
    const headers = ['Cell ID', 'Area (px)', 'Area (μm²)', 'Perimeter', 'Circularity', 'Chlorophyll', 'Health Score'];
    
    const thead = document.createElement('thead');
    thead.className = 'thead-dark';
    const headerRow = document.createElement('tr');
    
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.position = 'sticky';
        th.style.top = '0';
        th.style.backgroundColor = '#667eea';
        th.style.color = 'white';
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Body
    const tbody = document.createElement('tbody');
    cellData.forEach((cell, index) => {
        const row = document.createElement('tr');
        
        // Add hover effect
        row.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f8f9fa';
        });
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
        
        row.innerHTML = `
            <td><strong>${cell.cell_id || (index + 1)}</strong></td>
            <td>${(cell.area_pixels || cell.area || 0).toFixed(1)}</td>
            <td>${(cell.area_microns_sq || (cell.area_pixels || cell.area || 0) * 0.01).toFixed(2)}</td>
            <td>${(cell.perimeter || 0).toFixed(1)}</td>
            <td>${(cell.circularity || 0).toFixed(3)}</td>
            <td><span style="color: ${getChlorophyllColor(cell.chlorophyll_content || 0)}">${(cell.chlorophyll_content || 0).toFixed(3)}</span></td>
            <td><span style="color: ${getHealthColor(cell.health_score || 0)}">${(cell.health_score || 0).toFixed(3)}</span></td>
        `;
        tbody.appendChild(row);
    });
    table.appendChild(tbody);

    tableContainer.innerHTML = '';
    tableContainer.appendChild(table);
    
    console.log('[TABLE] ✅ Cell data table updated');
}

// Helper functions for cell data styling
function getChlorophyllColor(value) {
    if (value > 0.7) return '#27ae60'; // Green
    if (value > 0.4) return '#f39c12'; // Orange
    return '#e74c3c'; // Red
}

function getHealthColor(value) {
    if (value > 0.7) return '#27ae60'; // Green
    if (value > 0.4) return '#f39c12'; // Orange
    return '#e74c3c'; // Red
}

// FIXED: Tab switching
function switchTab(tabName) {
    console.log(`[TAB] Switching to tab: ${tabName}`);
    
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const activeContent = document.getElementById(`tab-${tabName}`);
    if (activeContent) {
        activeContent.classList.add('active');
    }

    // Load specific visualizations for the tab
    loadTabSpecificContent(tabName);
}

// Load content specific to each tab
function loadTabSpecificContent(tabName) {
    if (!analysisResults) return;
    
    switch (tabName) {
        case 'biomass':
            loadBiomassContent();
            break;
        case 'spectral':
            loadSpectralContent();
            break;
        case 'similarity':
            loadSimilarityContent();
            break;
        case 'temporal':
            loadTemporalContent();
            break;
        case 'histograms':
            loadHistogramContent();
            break;
    }
}

// Load biomass content
function loadBiomassContent() {
    console.log('[BIOMASS] Loading biomass content');
    const biomassChart = document.getElementById('biomassChart');
    const biomassValues = document.getElementById('biomassValues');
    
    if (analysisResults.biomass_analysis && biomassChart && biomassValues) {
        const biomass = analysisResults.biomass_analysis;
        
        biomassValues.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="metric-card">
                        <h6>Fresh Biomass</h6>
                        <div class="metric-value">${(biomass.combined_estimate?.fresh_biomass_g || 0).toFixed(4)} g</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <h6>Dry Biomass</h6>
                        <div class="metric-value">${(biomass.combined_estimate?.dry_biomass_g || 0).toFixed(4)} g</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="metric-card">
                        <h6>Total Area</h6>
                        <div class="metric-value">${(biomass.area_based?.total_area_mm2 || 0).toFixed(2)} mm²</div>
                    </div>
                </div>
            </div>
        `;
        
        // Create biomass chart
        createBiomassChart(biomass);
    } else {
        console.log('[BIOMASS] No biomass data available');
    }
}

// Load spectral content
function loadSpectralContent() {
    console.log('[SPECTRAL] Loading spectral content');
    const spectralChart = document.getElementById('spectralChart');
    
    if (analysisResults.spectral_analysis && spectralChart) {
        const spectral = analysisResults.spectral_analysis;
        const stats = spectral.population_statistics || {};
        
        spectralChart.innerHTML = `
            <div class="spectral-results">
                <p><strong>Mean Wavelength:</strong> ${(stats.mean_wavelength || 0).toFixed(1)} nm</p>
                <p><strong>Green Intensity:</strong> ${(stats.mean_green_intensity || 0).toFixed(1)}</p>
                <p><strong>Vegetation Index:</strong> ${(stats.mean_vegetation_index || 0).toFixed(3)}</p>
                <p><strong>Chlorophyll Content:</strong> ${(stats.mean_chlorophyll_content || 0).toFixed(1)}%</p>
            </div>
        `;
        
        // Create spectral visualization
        createSpectralChart(spectral);
    } else {
        console.log('[SPECTRAL] No spectral data available');
    }
}

function loadSimilarityContent() {
    console.log('[SIMILARITY] Loading similarity content');
    const similarityMatrix = document.getElementById('similarityMatrix');
    
    if (analysisResults.similarity_analysis && similarityMatrix) {
        const similarity = analysisResults.similarity_analysis;
        
        similarityMatrix.innerHTML = `
            <div class="similarity-results">
                <p><strong>Similar Cell Groups:</strong> ${similarity.similar_cell_groups?.length || 0}</p>
                <p><strong>Similarity Threshold:</strong> ${(similarity.similarity_threshold || 0).toFixed(3)}</p>
            </div>
        `;
        
        // Display similar cell groups
        if (similarity.similar_cell_groups && similarity.similar_cell_groups.length > 0) {
            const groups = similarity.similar_cell_groups.slice(0, 10); // Show top 10
            const groupsList = document.getElementById('similarCellGroups');
            if (groupsList) {
                groupsList.innerHTML = '<h5>Top Similar Cell Pairs:</h5>' + 
                    groups.map(group => 
                        `<p>Cells ${group.cell_1} & ${group.cell_2}: ${(group.similarity_score * 100).toFixed(1)}% similar</p>`
                    ).join('');
            }
        }
    }
}

function loadTemporalContent() {
    console.log('[TEMPORAL] Loading temporal content');
    const growthChart = document.getElementById('growthChart');
    
    if (analysisResults.temporal_analysis && growthChart) {
        const temporal = analysisResults.temporal_analysis;
        
        if (temporal.population_dynamics) {
            const popDyn = temporal.population_dynamics;
            growthChart.innerHTML = `
                <div class="temporal-results">
                    <p><strong>Total Timepoints:</strong> ${popDyn.total_timepoints || 0}</p>
                    <p><strong>Average Growth Rate:</strong> ${((popDyn.average_growth_rate || 0) * 100).toFixed(2)}%</p>
                </div>
            `;
            
            // Create growth visualization
            createGrowthChart(temporal);
        }
    }
}

function loadHistogramContent() {
    console.log('[HISTOGRAM] Loading histogram content');
    
    if (analysisResults.cell_data && analysisResults.cell_data.length > 0) {
        createHistograms(analysisResults.cell_data);
    }
}

// Chart creation functions
function createBiomassChart(biomassData) {
    console.log('[CHART] Creating biomass chart');
    const biomassChart = document.getElementById('biomassChart');
    if (!biomassChart) return;
    
    // Create simple biomass visualization
    const estimates = [
        { method: 'Area-based', value: biomassData.area_based?.fresh_biomass_g || 0 },
        { method: 'Chlorophyll-based', value: biomassData.chlorophyll_based?.estimated_biomass_g || 0 },
        { method: 'Combined', value: biomassData.combined_estimate?.fresh_biomass_g || 0 }
    ];
    
    const chartHtml = estimates.map(est => `
        <div class="biomass-bar">
            <label>${est.method}</label>
            <div class="bar-container">
                <div class="bar" style="width: ${(est.value * 10000)}px; background: #27ae60;"></div>
                <span>${est.value.toFixed(4)} g</span>
            </div>
        </div>
    `).join('');
    
    biomassChart.innerHTML = `<div class="biomass-chart">${chartHtml}</div>`;
}

function createSpectralChart(spectralData) {
    console.log('[CHART] Creating spectral chart');
    // Implementation for spectral chart
}

function createGrowthChart(temporalData) {
    console.log('[CHART] Creating growth chart');
    // Implementation for growth chart
}

function createHistograms(cellData) {
    console.log('[CHART] Creating histograms');

    const histogramContainer = document.getElementById('histogramContainer');
    if (!histogramContainer || !cellData || cellData.length === 0) return;

    // Extract relevant data
    const areas = cellData.map(cell => cell.area || cell.area_pixels || 0);
    const chlorophylls = cellData.map(cell => cell.chlorophyll_content || 0);
    const healthScores = cellData.map(cell => cell.health_score || 0);
    const wavelengths = cellData.map(cell => cell.mean_wavelength || 550); // fallback to 550nm if missing

    // Helper function to create and render a histogram
    function renderHistogram(containerId, data, title, xLabel) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const trace = {
            x: data,
            type: 'histogram',
            marker: {
                color: '#764ba2',
                opacity: 0.7
            }
        };

        const layout = {
            title: title,
            xaxis: { title: xLabel },
            yaxis: { title: 'Frequency' },
            margin: { t: 40, b: 40, l: 40, r: 20 }
        };

        Plotly.newPlot(container, [trace], layout, { responsive: true });
    }

    // Render each histogram
    renderHistogram('sizeHistogram', areas, 'Cell Size Distribution', 'Area (pixels)');
    renderHistogram('colorHistogram', chlorophylls, 'Chlorophyll Content Distribution', 'Chlorophyll Index');
    renderHistogram('healthHistogram', healthScores, 'Cell Health Score Distribution', 'Health Score');
    renderHistogram('wavelengthHistogram', wavelengths, 'Wavelength Distribution', 'Dominant Wavelength (nm)');
}












// Add to main.js - Live parameter adjustment
function updateAnalysisParameter(paramName, value) {
    if (currentAnalysisId && active_analyses[currentAnalysisId]) {
        socket.emit('update_analysis_parameter', {
            analysis_id: currentAnalysisId,
            parameter: paramName,
            value: value
        });
        showNotification(`Parameter ${paramName} updated to ${value}`, 'info');
    }
}

// Add event listeners for live parameter changes
document.getElementById('pixelRatio').addEventListener('change', function() {
    if (currentAnalysisId) {
        updateAnalysisParameter('pixel_ratio', parseFloat(this.value));
    }
});

// Enhanced result display with live updates
function displayResults(result) {
    if (!result.success) {
        showNotification(`Analysis failed: ${result.error}`, 'error');
        return;
    }

    document.getElementById('resultsEmpty').style.display = 'none';
    document.getElementById('resultsContent').style.display = 'block';

    updateSummary(result.summary_statistics || result.summary);  // supports both keys
    loadVisualizations(result.analysis_id);
    updateCellDataTable(result.cell_data);

    if (result.biomass_analysis || result.temporal_analysis || result.similarity_analysis) {
        displayComprehensiveResults(result);
    }

    enableLiveEditing(result);
}


// Enable live editing of results
function enableLiveEditing(result) {
    // Add edit buttons to cell table
    const table = document.getElementById('cellDataTable');
    const rows = table.querySelectorAll('tr');
    
    rows.forEach((row, index) => {
        if (index === 0) return; // Skip header
        
        const editBtn = document.createElement('button');
        editBtn.className = 'btn btn-sm btn-secondary';
        editBtn.innerHTML = '✏️';
        editBtn.onclick = () => editCellData(index - 1);
        
        const actionCell = document.createElement('td');
        actionCell.appendChild(editBtn);
        row.appendChild(actionCell);
    });
}











function displayComprehensiveResults(result) {
    document.getElementById('comprehensiveResults').style.display = 'block';
    
    // Display Biomass
    if (result.biomass_analysis) {
        const biomassData = result.biomass_analysis.combined_estimate;
        document.getElementById('biomassValues').innerHTML = `
            <div class="metric-card">
                <h5>Fresh Biomass</h5>
                <p class="metric-value">${biomassData.fresh_biomass_g.toFixed(4)} g</p>
                <p class="metric-subtitle">Dry: ${biomassData.dry_biomass_g.toFixed(4)} g</p>
                <p class="confidence">CI: ${biomassData.confidence_interval[0].toFixed(4)} - ${biomassData.confidence_interval[1].toFixed(4)} g</p>
            </div>
        `;
    }
    
    // Display Spectral Analysis
    if (result.spectral_analysis) {
        const spectralData = result.spectral_analysis.population_statistics;
        // Create wavelength distribution chart
        createWavelengthChart(result.spectral_analysis.wavelength_distribution);
    }
    
    // Display Temporal Tracking
    if (result.temporal_analysis) {
        createGrowthChart(result.temporal_analysis.growth_curves);
    }
    
    // Display Cell Similarity
    if (result.similarity_analysis) {
        displaySimilarCells(result.similarity_analysis.similar_cell_groups);
    }
    // Display Histograms
    if (result.cell_data && result.cell_data.length > 0) {
        createHistograms(result.cell_data);
    }

    // Update Additional Metric Sections (Optional)
    if (result.biomass_analysis?.combined_estimate) {
        const combined = result.biomass_analysis.combined_estimate;
        document.getElementById('freshBiomass').textContent = `${combined.fresh_biomass_g.toFixed(4)} g`;
        document.getElementById('dryBiomass').textContent = `${combined.dry_biomass_g.toFixed(4)} g`;
        document.getElementById('biomassConfidence').textContent = `${combined.confidence_interval[0].toFixed(4)} - ${combined.confidence_interval[1].toFixed(4)} g`;
    }

    if (result.spectral_analysis?.population_statistics) {
        const stats = result.spectral_analysis.population_statistics;
        document.getElementById('dominantWavelength').textContent = `${stats.mean_wavelength.toFixed(1)} nm`;
        document.getElementById('greenIntensity').textContent = `${stats.mean_green_intensity.toFixed(1)}`;
        document.getElementById('chlorophyllIndex').textContent = `${(stats.mean_chlorophyll_content).toFixed(1)}%`;
    }

    if (result.temporal_analysis?.population_dynamics) {
        const dyn = result.temporal_analysis.population_dynamics;
        document.getElementById('trackedCellCount').textContent = dyn.tracked_cells || '--';
        document.getElementById('averageGrowthRate').textContent = dyn.average_growth_rate ? `${(dyn.average_growth_rate * 100).toFixed(2)}%` : '--';
        document.getElementById('cellDivisions').textContent = dyn.estimated_cell_divisions || '--';
    }
}
function displaySimilarCells(similarGroups) {
    const similarCellList = document.getElementById('similarCellGroups');
    if (!similarCellList) return;

    similarCellList.innerHTML = '';
    if (similarGroups.length === 0) {
        similarCellList.innerHTML = '<p>No similar cell groups found</p>';
        return;
    }

    similarGroups.forEach(group => {
        const item = document.createElement('div');
        item.className = 'similar-cell-group';
        item.innerHTML = `
            <strong>Group ${group.group_id}:</strong> Cells ${group.cell_1} & ${group.cell_2} - Similarity: ${(group.similarity_score * 100).toFixed(1)}%
        `;
        similarCellList.appendChild(item);
    });
}