// BIOIMAGIN Professional JavaScript - RESTORED WORKING VERSION
// Handles all web interface interactions and result display

// Global variables
let uploadedFiles = [];
let currentAnalysisId = null;
let currentAnnotationMode = 'correct';
let trainingSession = null;
let currentTrainingImageIndex = 0;

// Keep old annotations for backward compatibility - MUST BE DECLARED FIRST
let annotations = {};

// ✅ UNIFIED ANNOTATION SYSTEM - Per Image
let imageAnnotations = new Map(); // Map of imageId -> annotations
let currentImageId = null;

let unifiedAnnotations = {
    correct: [],
    false_positive: [],
    missed: []
};

// Global drawing state for both canvases
let drawingState = {
    isDrawing: false,
    isPanning: false,
    currentStroke: [],
    lastX: 0,
    lastY: 0,
    autoBorder: false
};

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    initializeInterface();
    checkTophatStatus();
});

// Interface initialization
function initializeInterface() {
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');

    // File upload handlers
    fileInput.addEventListener('change', handleFileUpload);
    
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });
    
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });
}

// File handling
function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const validFiles = files.filter(file => 
        file.type.startsWith('image/') && file.size <= 100 * 1024 * 1024
    );
    
    if (validFiles.length === 0) {
        alert('Please select valid image files (max 100MB each)');
        return;
    }
    
    uploadFiles(validFiles);
}

function uploadFiles(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    // Add analysis options
    formData.append('use_celldetection', document.getElementById('useCellDetection').checked);
    formData.append('enable_temporal', document.getElementById('enableTemporal').checked);
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            uploadedFiles = data.files;
            displayUploadedFiles();
            enableAnalysisButtons();
        } else {
            alert('Upload failed: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        alert('Upload failed: ' + error.message);
    });
}

function displayUploadedFiles() {
    const filesList = document.getElementById('filesList');
    filesList.innerHTML = '';
    
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-name">${file.original_name}</div>
                <div class="file-status">Ready for analysis (${(file.size / 1024).toFixed(1)} KB)</div>
            </div>
            <button class="btn btn-sm" onclick="removeFile(${index})">🗑️ Remove</button>
        `;
        filesList.appendChild(fileItem);
    });
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    displayUploadedFiles();
    
    if (uploadedFiles.length === 0) {
        disableAnalysisButtons();
    }
}

function enableAnalysisButtons() {
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('batchAnalyzeBtn').disabled = false;
}

function disableAnalysisButtons() {
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('batchAnalyzeBtn').disabled = true;
}

// Analysis functions
function startAnalysis() {
    if (uploadedFiles.length === 0) return;
    
    document.getElementById('resultsSection').style.display = 'block';
    
    uploadedFiles.forEach((file, index) => {
        analyzeImage(file.id, file.original_name, index);
    });
}

function startBatchAnalysis() {
    if (uploadedFiles.length === 0) return;
    
    document.getElementById('batchProgressSection').style.display = 'block';
    
    const analysisOptions = {
        files: uploadedFiles.map(f => ({ id: f.id, filename: f.original_name })),
        options: {
            use_watershed: document.getElementById('useWatershed').checked,
            use_tophat: document.getElementById('useTophat').checked,
            use_cnn: document.getElementById('useCNN').checked,
            use_celldetection: document.getElementById('useCellDetection').checked,
            enable_temporal: document.getElementById('enableTemporal').checked
        }
    };
    
    fetch('/api/batch_analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisOptions)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentAnalysisId = data.analysis_id;
            monitorBatchAnalysis(data.analysis_id);
        } else {
            alert('Batch analysis failed: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Batch analysis error:', error);
        alert('Batch analysis failed: ' + error.message);
    });
}

function analyzeImage(fileId, filename, index) {
    const analysisOptions = {
        use_watershed: document.getElementById('useWatershed').checked,
        use_tophat: document.getElementById('useTophat').checked,
        use_cnn: document.getElementById('useCNN').checked,
        use_celldetection: document.getElementById('useCellDetection').checked
    };
    
    fetch(`/api/analyze/${fileId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(analysisOptions)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            monitorAnalysis(data.analysis_id, filename, () => {
                if (index === uploadedFiles.length - 1) {
                    console.log('All individual analyses completed');
                }
            });
        } else {
            alert(`Analysis failed for ${filename}: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Analysis error:', error);
        alert(`Analysis failed for ${filename}: ${error.message}`);
    });
}

function monitorAnalysis(analysisId, filename, onComplete) {
    const poll = () => {
        fetch(`/api/status/${analysisId}`)
            .then(response => response.json())
            .then(data => {
                const analysis = data.analysis;
                const status = analysis ? analysis.status : 'unknown';
                
                if (status === 'completed') {
                    displayResults(analysisId, filename, analysis.result);
                    if (onComplete) onComplete();
                } else if (status === 'error') {
                    console.error('Analysis failed:', analysis.error);
                    if (onComplete) onComplete();
                } else {
                    setTimeout(poll, 1000);
                }
            })
            .catch(error => {
                console.error('Status check error:', error);
                if (onComplete) onComplete();
            });
    };
    poll();
}

function monitorBatchAnalysis(analysisId) {
    const progressFill = document.getElementById('batchProgressFill');
    const progressText = document.getElementById('batchProgressText');
    
    const poll = () => {
        fetch(`/api/status/${analysisId}`)
            .then(response => response.json())
            .then(data => {
                const analysis = data.analysis;
                const status = analysis ? analysis.status : 'unknown';
                
                if (analysis && analysis.progress) {
                    const progress = analysis.progress;
                    progressFill.style.width = `${progress.percentage}%`;
                    progressText.textContent = `Processing ${progress.current_file} (${progress.completed}/${progress.total})`;
                }
                
                if (status === 'completed') {
                    document.getElementById('batchProgressSection').style.display = 'none';
                    document.getElementById('resultsSection').style.display = 'block';
                    displayBatchResults(analysisId, analysis.result);
                } else if (status === 'error') {
                    progressText.textContent = 'Batch analysis failed: ' + (analysis.error || 'Unknown error');
                    progressText.style.color = 'red';
                } else {
                    setTimeout(poll, 1000);
                }
            })
            .catch(error => {
                console.error('Batch status check error:', error);
                progressText.textContent = 'Status check failed';
                progressText.style.color = 'red';
            });
    };
    poll();
}

// ✅ FIXED: Enhanced displayResults with better data detection for pipeline and methods
function displayResults(analysisId, filename, result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultItem = document.createElement('div');
    resultItem.className = 'result-item analysis-success';
    
    currentAnalysisId = analysisId;

    // ✅ FIXED: Enhanced pipeline data detection
    console.log('🔍 Searching for pipeline data in result:', result);
    
    let pipelineSteps = null;
    
    // Check multiple possible locations for pipeline data
    if (result?.visualizations?.pipeline_steps) {
        pipelineSteps = result.visualizations.pipeline_steps;
        console.log('✅ Found pipeline in visualizations.pipeline_steps');
    } else if (result?.pipeline_visualization) {
        pipelineSteps = { pipeline_overview: result.pipeline_visualization };
        console.log('✅ Found pipeline in pipeline_visualization');
    } else if (result?.detection_results?.pipeline_visualization) {
        pipelineSteps = { pipeline_overview: result.detection_results.pipeline_visualization };
        console.log('✅ Found pipeline in detection_results.pipeline_visualization');
    } else if (result?.visualizations?.pipeline_overview) {
        pipelineSteps = { pipeline_overview: result.visualizations.pipeline_overview };
        console.log('✅ Found pipeline in visualizations.pipeline_overview');
    } else if (result?.visualizations?.watershed_pipeline) {
        pipelineSteps = { pipeline_overview: result.visualizations.watershed_pipeline };
        console.log('✅ Found pipeline in visualizations.watershed_pipeline');
    }
    
    console.log('📊 Pipeline steps found:', !!pipelineSteps);

    // ✅ FIXED: Enhanced method results detection
    let methodResults = null;
    let hasMethodResults = false;
    
    // Check multiple possible locations for method results
    if (result?.method_results && Object.keys(result.method_results).length > 0) {
        methodResults = result.method_results;
        console.log('✅ Found method results in root.method_results');
    } else if (result?.detection_results?.method_results && Object.keys(result.detection_results.method_results).length > 0) {
        methodResults = result.detection_results.method_results;
        console.log('✅ Found method results in detection_results.method_results');
    } else if (result?.detection_results?.methods && Object.keys(result.detection_results.methods).length > 0) {
        methodResults = result.detection_results.methods;
        console.log('✅ Found method results in detection_results.methods');
    } else if (result?.methods && Object.keys(result.methods).length > 0) {
        methodResults = result.methods;
        console.log('✅ Found method results in root.methods');
    }
    
    hasMethodResults = methodResults && Object.keys(methodResults).length > 1;
    console.log('🎯 Method results:', hasMethodResults ? Object.keys(methodResults) : 'none found');

    const detectionMethod = result?.detection_results?.detection_method || 
                           result?.detection_method || 
                           (hasMethodResults ? `Multi-Method Analysis (${Object.keys(methodResults).length} methods)` : 'Single Method Analysis');

    // Always show header + pipeline ON TOP
    const headerHTML = `
        <div class="result-header">
            <div class="result-title">📄 ${filename}</div>
            <div class="detection-method">${detectionMethod}</div>
        </div>
    `;
    const pipelineHTML = createPipelineVisualization(pipelineSteps, detectionMethod);

    // Start with pipeline shown first
    resultItem.innerHTML = headerHTML + pipelineHTML;

    // ⚠️ Append method results or fallback legacy metrics
    if (hasMethodResults) {
        console.log('🎯 Rendering method tabs for multiple methods');
        resultItem.innerHTML += renderMethodTabs(analysisId, methodResults, result.best_method);
    } else {
        console.log('📊 Rendering legacy metrics for single method');
        resultItem.innerHTML += renderLegacyMetrics(analysisId, result);
    }

    // Add shared visualizations container
    const vizContainer = document.createElement('div');
    vizContainer.className = 'visualization-grid';
    vizContainer.id = `viz-${analysisId}`;
    resultItem.appendChild(vizContainer);

    resultsContainer.appendChild(resultItem);

    loadVisualizations(analysisId, result.visualizations || {});
    setupPipelineInteractions(analysisId);
    resultItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Also display enhanced results
    displayEnhancedResults(result, analysisId);
}

// ✅ ENHANCED: More flexible pipeline visualization that handles different data structures
function createPipelineVisualization(pipelineSteps, detectionMethod) {
    console.log('🔍 Creating pipeline visualization with data:', pipelineSteps);
    
    // Check if we have any pipeline visualization data
    let hasPipelineImage = false;
    let pipelineImageData = null;
    
    if (pipelineSteps) {
        if (pipelineSteps.pipeline_overview) {
            pipelineImageData = pipelineSteps.pipeline_overview;
            hasPipelineImage = true;
            console.log('✅ Found pipeline_overview image');
        } else if (typeof pipelineSteps === 'string') {
            // Direct base64 string
            pipelineImageData = pipelineSteps;
            hasPipelineImage = true;
            console.log('✅ Found direct pipeline image string');
        }
    }
    
    if (!hasPipelineImage) {
        console.log('⚠️ No pipeline visualization data found');
        return `
        <div class="pipeline-container">
            <div class="pipeline-header">
                <h4>🔬 Processing Pipeline</h4>
                <span style="color: #ffc107;">⚠️ Pipeline visualization not available</span>
            </div>
        </div>
        `;
    }
    
    // If we have pipeline image but no detailed step data, show simplified version
    if (!pipelineSteps.step_descriptions && !pipelineSteps.individual_steps) {
        console.log('📊 Showing simplified pipeline visualization');
        return `
        <div class="pipeline-container">
            <div class="pipeline-header" onclick="togglePipeline(this)">
                <h4>🔬 ${detectionMethod} Processing Pipeline</h4>
                <span class="pipeline-toggle">▼</span>
            </div>
            <div class="pipeline-content">
                <div class="pipeline-overview">
                    <h5>📊 Processing Pipeline</h5>
                    <p>Click the image below to see the complete processing pipeline:</p>
                    <img src="data:image/png;base64,${pipelineImageData}" 
                        alt="Processing Pipeline"
                        onclick="openStepModal('${detectionMethod} Processing Pipeline', 'data:image/png;base64,${pipelineImageData}', 'Complete processing pipeline showing each stage of the analysis')"
                        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); cursor: pointer;">
                </div>
            </div>
        </div>
        
        <!-- Modal for detailed view -->
        <div id="stepModal" class="step-modal">
            <div class="step-modal-content">
                <span class="step-modal-close" onclick="closeStepModal()">&times;</span>
                <h3 id="stepModalTitle"></h3>
                <p id="stepModalDescription"></p>
                <img id="stepModalImage" src="" alt="">
            </div>
        </div>
        `;
    }
    
    // Full sophisticated version with all step data
    const stepDescriptions = pipelineSteps.step_descriptions || {};
    const individualSteps = pipelineSteps.individual_steps || {};
    
    // Backward compatibility mapping for legacy step keys
    const stepKeyMapping = {
        'Denoised': 'Green_enhanced',        // Old "denoised" was actually green enhanced
        'Shape_index': 'Distance_transform', // Old "shape index" was actually distance transform
        'Preprocessed': 'Green_enhanced',    // Legacy preprocessing step
        'Segmented': 'Watershed_raw',        // Legacy segmentation step
        'Final': 'Watershed_final'           // Legacy final step
    };
    
    // Apply backward compatibility mapping
    Object.keys(stepKeyMapping).forEach(oldKey => {
        if (individualSteps[oldKey] && !individualSteps[stepKeyMapping[oldKey]]) {
            individualSteps[stepKeyMapping[oldKey]] = individualSteps[oldKey];
            console.log(`🔄 Mapped legacy step '${oldKey}' to '${stepKeyMapping[oldKey]}'`);
        }
    });
    
    const stepCount = pipelineSteps.step_count || 0;
    
    // Define pipeline steps with status
    // Unified Watershed Pipeline - Watershed Result is Final
    const pipelineDefinition = [
        { key: 'Original', name: 'Original Image', status: 'success', icon: '📸' },
        { key: 'Green_enhanced', name: 'Green Enhanced', status: 'success', icon: '🟢' },
        { key: 'Green_mask', name: 'Color Filtered', status: 'success', icon: '🎯' },
        { key: 'Distance_transform', name: 'Distance Transform', status: 'success', icon: '📏' },
        { key: 'Watershed_final', name: 'Watershed Result (Final)', status: 'success', icon: '🧬' },
        { key: 'Watershed_raw', name: 'Processing Reference', status: 'success', icon: '💧' }
    ];

    
    const uniqueId = `pipeline-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    console.log('📊 Showing full sophisticated pipeline visualization');
    return `
    <div class="pipeline-container" id="${uniqueId}">
        <div class="pipeline-header" onclick="togglePipeline(this)">
            <h4>
                <span class="step-indicator success"></span>
                🔬 Enhanced Detection Pipeline
                <span style="font-size: 0.8em; font-weight: normal; margin-left: 10px;">
                    (${stepCount || 'Multiple'} processing steps completed)
                </span>
            </h4>
            <span class="pipeline-toggle">▼</span>
        </div>
        <div class="pipeline-content">
            <!-- Steps Summary -->
            <div class="pipeline-steps-summary">
                ${pipelineDefinition.map(step => `
                    <div class="processing-step ${individualSteps[step.key] ? 'active' : 'inactive'}" 
                        onclick="showStepDetail('${step.key}', '${step.name}', '${stepDescriptions[step.key] || 'Processing step'}')">
                        <span class="step-indicator ${step.status}"></span>
                        ${step.icon} ${step.name}
                    </div>
                `).join('')}
            </div>
            
            <!-- Pipeline Overview -->
            <div class="pipeline-overview">
                <h5>📊 Complete Processing Pipeline</h5>
                <p>Click the image below to see the full step-by-step processing pipeline:</p>
                <img src="data:image/png;base64,${pipelineImageData}" 
                    alt="Complete Processing Pipeline"
                    onclick="openStepModal('Complete Processing Pipeline', 'data:image/png;base64,${pipelineImageData}', 'Full step-by-step processing pipeline showing each stage of the analysis')"
                    style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); cursor: pointer;">
            </div>
            
            <!-- Individual Steps -->
            ${Object.keys(individualSteps).length > 0 ? `
            <div class="individual-steps">
                <h5>🔍 Individual Processing Steps</h5>
                <p>Click any step below to see detailed results and statistics:</p>
                <div class="steps-grid">
                    ${pipelineDefinition.filter(step => individualSteps[step.key]).map(step => `
                        <div class="step-item" onclick="openStepModal('${step.name}', 'data:image/png;base64,${individualSteps[step.key]}', '${stepDescriptions[step.key] || 'Processing step details'}')">
                            <img src="data:image/png;base64,${individualSteps[step.key]}" alt="${step.name}">
                            <div class="step-item-info">
                                <div class="step-item-title">${step.icon} ${step.name}</div>
                                <div class="step-item-description">${stepDescriptions[step.key]?.substring(0, 80) || 'Click to view details'}...</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            ` : ''}
        </div>
    </div>
    
    <!-- Modal for detailed step view -->
    <div id="stepModal" class="step-modal">
        <div class="step-modal-content">
            <span class="step-modal-close" onclick="closeStepModal()">&times;</span>
            <h3 id="stepModalTitle"></h3>
            <p id="stepModalDescription"></p>
            <img id="stepModalImage" src="" alt="">
        </div>
    </div>
    `;
}

// ✅ RESTORED: Exact renderMethodTabs function as requested
function renderMethodTabs(analysisId, methods, bestKey) {
    let tabs = '', panels = '';
    const methodKeys = Object.keys(methods);

    methodKeys.forEach((key, i) => {
        const m = methods[key];
        const isActive = i === 0 ? 'active' : '';
        const isBest = key === bestKey ? '⭐' : '';

        tabs += `
            <button class="method-tab ${isActive}" onclick="switchMethodTab('${analysisId}', '${key}', this)">
                ${m.method_name} ${isBest}<br><small>${m.cells_detected} cells</small>
            </button>
        `;

        panels += `
            <div class="method-panel ${isActive}" id="${analysisId}-panel-${key}">
                <div class="method-result-header">
                    <div class="method-result-title">${m.method_name}</div>
                    <div style="color:#666">${m.cells_detected} cells detected</div>
                </div>
                <div class="method-result-stats">
                    <div class="method-stat"><div class="method-stat-value">${m.cells_detected}</div><div class="method-stat-label">Cells</div></div>
                    <div class="method-stat"><div class="method-stat-value">${m.total_area.toFixed(1)}</div><div class="method-stat-label">Total Area</div></div>
                    <div class="method-stat"><div class="method-stat-value">${m.average_area.toFixed(1)}</div><div class="method-stat-label">Avg Area</div></div>
                </div>
                ${m.visualization_b64 ? `<img src="data:image/png;base64,${m.visualization_b64}" class="method-visualization"/>` : `<div style='text-align:center;'>No visualization</div>`}
                <div style="text-align:center;margin-top:15px">
                    <button class="btn" onclick="exportMethodResults('${analysisId}', '${key}', 'csv')">📊 Export CSV</button>
                </div>
            </div>
        `;
    });

    return `
        <div class="method-panels-container">
            <div class="method-tabs">${tabs}</div>
            ${panels}
            <div style="background:#f8f9fa;padding:15px;margin-top:15px;border-radius:10px">
                <strong>Best Method:</strong> ${methods[bestKey]?.method_name || 'N/A'}
            </div>
        </div>
    `;
}


// ✅ RESTORED: Legacy metrics rendering
function renderLegacyMetrics(analysisId, result) {
    const d = result.detection_results || {};
    const q = result.quantitative_analysis || {};
    const b = q.biomass_analysis || {};
    const c = q.color_analysis || {};
    const h = q.health_assessment || {};
    const visual = result.visualizations || {};

    return `
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">${d.cells_detected || 0}</div>
                <div class="metric-label">Cells</div>
            </div>
            <div class="metric-card biomass-card">
                <div class="metric-value">${(b.total_biomass_mg || 0).toFixed(3)}</div>
                <div class="metric-label">Biomass</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(c.green_cell_percentage || 0).toFixed(1)}%</div>
                <div class="metric-label">Green Cells</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">${(q.average_cell_area || 0).toFixed(1)}</div>
                <div class="metric-label">Avg Area</div>
            </div>
        </div>
        ${h.overall_health ? `
            <div style="text-align:center;margin:20px 0;">
                <span class="health-indicator health-${h.overall_health}">Health: ${h.overall_health.toUpperCase()}</span> 
                <span style="margin-left:10px">Score: ${(h.health_score || 0).toFixed(2)}/1.0</span>
            </div>
        ` : ''}
        ${visual.detection_overview ? `
            <div class="method-visualization">
                <img src="data:image/png;base64,${visual.detection_overview}" 
                     alt="Detection Results"
                     onclick="showFullSizeImage(this.src, 'Detection Results')"
                     style="max-width: 100%; height: auto; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); cursor: pointer;">
            </div>
        ` : ''}
    `;
}

// ✅ RESTORED: Supporting functions
function loadVisualizations(analysisId, visualizations) {
    // Load additional visualizations into the viz container
    const vizContainer = document.getElementById(`viz-${analysisId}`);
    if (!vizContainer) return;
    
    // Clear existing content
    vizContainer.innerHTML = '';
    
    // Add any additional visualizations here if needed
}





        function togglePipeline(header) {
            const container = header.parentElement;
            const content = container.querySelector('.pipeline-content');
            const toggle = header.querySelector('.pipeline-toggle');
            
            if (content.classList.contains('expanded')) {
                content.classList.remove('expanded');
                toggle.classList.remove('expanded');
            } else {
                content.classList.add('expanded');
                toggle.classList.add('expanded');
            }
        }

        function showStepDetail(stepKey, stepName, description) {
            console.log(`Showing details for ${stepName}: ${description}`);
            
            // Highlight the step with animation
            const stepElement = event.target;
            const originalTransform = stepElement.style.transform;
            stepElement.style.transform = 'scale(1.05)';
            stepElement.style.boxShadow = '0 4px 15px rgba(0,123,255,0.3)';
            
            setTimeout(() => {
                stepElement.style.transform = originalTransform;
                stepElement.style.boxShadow = '';
            }, 300);
        }

        function openStepModal(title, imageSrc, description) {
            const modal = document.getElementById('stepModal');
            const modalTitle = document.getElementById('stepModalTitle');
            const modalDescription = document.getElementById('stepModalDescription');
            const modalImage = document.getElementById('stepModalImage');
            
            modalTitle.textContent = title;
            modalDescription.textContent = description;
            modalImage.src = imageSrc;
            modalImage.alt = title;
            
            modal.style.display = 'block';
            
            // Close modal when clicking outside
            modal.onclick = function(event) {
                if (event.target === modal) {
                    closeStepModal();
                }
            }
        }

        function closeStepModal() {
            const modal = document.getElementById('stepModal');
            if (modal) {
                modal.style.display = 'none';
            }
        }

        function setupPipelineInteractions(analysisId) {
            // Add keyboard support for modal
            document.addEventListener('keydown', function(event) {
                if (event.key === 'Escape') {
                    closeStepModal();
                }
            });
        }

        // Auto-expand first pipeline when page loads
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(() => {
                const firstPipeline = document.querySelector('.pipeline-header');
                if (firstPipeline) {
                    const content = firstPipeline.parentElement.querySelector('.pipeline-content');
                    if (content && !content.classList.contains('expanded')) {
                        togglePipeline(firstPipeline);
                    }
                }
            }, 1000);
        });


function setupPipelineInteractions(analysisId) {
    // Add keyboard support for modal
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape') {
            closeStepModal();
        }
    });
    
    // Auto-expand first pipeline when page loads
    setTimeout(() => {
        const firstPipeline = document.querySelector('.pipeline-header');
        if (firstPipeline) {
            const content = firstPipeline.parentElement.querySelector('.pipeline-content');
            if (content && !content.classList.contains('expanded')) {
                togglePipeline(firstPipeline);
            }
        }
    }, 1000);
}

function downloadPipelineImage(analysisId) {
    const pipelineImg = document.querySelector(`#pipeline-${analysisId} img`);
    if (pipelineImg && pipelineImg.src) {
        const link = document.createElement('a');
        link.download = `pipeline_${analysisId}_${new Date().getTime()}.png`;
        link.href = pipelineImg.src;
        link.click();
    }
}

function switchMethodTab(analysisId, methodKey, tabElement) {
    // Remove active class from all tabs in this container
    const container = tabElement.closest('.method-panels-container');
    const tabs = container.querySelectorAll('.method-tab');
    const panels = container.querySelectorAll('.method-panel');
    
    tabs.forEach(tab => tab.classList.remove('active'));
    panels.forEach(panel => panel.classList.remove('active'));
    
    // Add active class to clicked tab
    tabElement.classList.add('active');
    
    // Show corresponding panel
    const panel = document.getElementById(`${analysisId}-panel-${methodKey}`);
    if (panel) {
        panel.classList.add('active');
    }
}

function displayBatchResults(analysisId, result) {
    const resultsContainer = document.getElementById('resultsContainer');
    const resultItem = document.createElement('div');
    resultItem.className = 'result-item batch-analysis-success';
    
    currentAnalysisId = analysisId;
    
    const batchSummary = result.batch_summary || {};
    const aggregateStats = batchSummary.aggregate_statistics || {};
    
    resultItem.innerHTML = `
        <div class="result-header">
            <div class="result-title">📊 Batch Analysis Results</div>
            <div style="color: #28a745; font-weight: bold;">
                ${batchSummary.successful_analyses || 0}/${batchSummary.total_files || 0} files processed 
                (${(batchSummary.success_rate || 0).toFixed(1)}% success rate)
            </div>
        </div>
        
        <div class="method-stats">
            <div class="stat-card">
                <div class="stat-value">${aggregateStats.total_cells_detected || 0}</div>
                <div class="stat-label">Total Cells Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(aggregateStats.total_biomass_mg || 0).toFixed(3)}</div>
                <div class="stat-label">Total Biomass (mg)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(aggregateStats.average_green_cell_percentage || 0).toFixed(1)}%</div>
                <div class="stat-label">Avg Green Cell %</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(aggregateStats.average_cell_area || 0).toFixed(1)}</div>
                <div class="stat-label">Avg Cell Area (μm²)</div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <button class="btn btn-primary" onclick="exportEnhancedResults('json')">
                📄 Export Complete Report
            </button>
            <button class="btn btn-success" onclick="exportEnhancedResults('csv')">
                📊 Export Summary CSV
            </button>
            <button class="btn btn-info" onclick="exportEnhancedResults('zip')">
                📦 Export Complete Package
            </button>
        </div>
    `;
    
    resultsContainer.appendChild(resultItem);
    
    // Display enhanced results
    displayEnhancedResults(result, analysisId);
    
    // Display time series if available
    if (result.time_series_analysis) {
        displayTimeSeriesResults(result.time_series_analysis, analysisId);
    }
}

function displayEnhancedResults(result, analysisId) {
    console.log('Displaying enhanced results:', result);
    
    // Show enhanced results section
    const enhancedSection = document.getElementById('enhanced-results');
    if (enhancedSection) {
        enhancedSection.style.display = 'block';
    }
    
    // Update summary cards
    const totalCellsElement = document.getElementById('total-cells-enhanced');
    if (totalCellsElement) {
        totalCellsElement.textContent = result.total_cells || 0;
    }
    
    const quantitative = result.quantitative_analysis || {};
    const biomass = quantitative.biomass_analysis || {};
    const colorAnalysis = quantitative.color_analysis || {};
    const health = quantitative.health_assessment || {};
    
    // Update biomass information
    const totalBiomassElement = document.getElementById('total-biomass');
    if (totalBiomassElement) {
        totalBiomassElement.textContent = (biomass.total_biomass_mg || 0).toFixed(3) + ' mg';
    }
    
    // Update green percentage
    const greenPercentageElement = document.getElementById('green-percentage');
    if (greenPercentageElement) {
        greenPercentageElement.textContent = (colorAnalysis.green_cell_percentage || 0).toFixed(1) + '%';
    }
    
    // Update health score
    const healthScoreElement = document.getElementById('health-score');
    if (healthScoreElement) {
        healthScoreElement.textContent = (health.overall_health_score || 0).toFixed(1);
    }
    
    // Update detailed cell analysis table
    const cellTableElement = document.getElementById('cell-analysis-table');
    if (cellTableElement && quantitative.detailed_cell_analysis) {
        updateCellAnalysisTable(quantitative.detailed_cell_analysis);
    }
    
    // Handle time series data
    if (result.time_series_analysis) {
        displayTimeSeriesResults(result.time_series_analysis);
    }
    
    // Store current analysis ID for exports
    window.currentAnalysisId = analysisId;
}

function updateCellAnalysisTable(cellData) {
    const tableBody = document.getElementById('cell-analysis-table');
    if (!tableBody || !cellData || cellData.length === 0) {
        if (tableBody) {
            tableBody.innerHTML = '<tr><td colspan="8" class="text-center text-muted">No cell data available</td></tr>';
        }
        return;
    }
    
    tableBody.innerHTML = '';
    cellData.forEach((cell, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${(cell.area || 0).toFixed(2)}</td>
            <td>${(cell.estimated_biomass_mg || 0).toFixed(6)}</td>
            <td>${(cell.green_intensity || 0).toFixed(1)}</td>
            <td>${(cell.circularity || 0).toFixed(3)}</td>
            <td>${(cell.color_ratio_green || 0).toFixed(2)}</td>
            <td>${(cell.hue_mean || 0).toFixed(1)}</td>
            <td>${(cell.saturation_mean || 0).toFixed(1)}</td>
        `;
    });
}

function displayTimeSeriesResults(timeSeriesData, analysisId) {
    const timeSeriesSection = document.getElementById('time-series-section');
    if (timeSeriesSection) {
        timeSeriesSection.style.display = 'block';
        
        if (timeSeriesData.visualization_data && timeSeriesData.visualization_data.visualization_b64) {
            const chartElement = document.getElementById('time-series-chart');
            if (chartElement) {
                chartElement.src = 'data:image/png;base64,' + timeSeriesData.visualization_data.visualization_b64;
            }
        }
        
        const growthRateElement = document.getElementById('growth-rate');
        if (growthRateElement) {
            growthRateElement.textContent = (timeSeriesData.trends?.growth_rate_percent || 0).toFixed(1) + '%';
        }
        
        const cellTrendElement = document.getElementById('cell-trend');
        if (cellTrendElement) {
            const trend = timeSeriesData.trends?.cell_count_trend || 'Stable';
            cellTrendElement.textContent = trend.charAt(0).toUpperCase() + trend.slice(1);
        }
        
        const biomassTrendElement = document.getElementById('biomass-trend');
        if (biomassTrendElement) {
            const trend = timeSeriesData.trends?.biomass_trend || 'Stable';
            biomassTrendElement.textContent = trend.charAt(0).toUpperCase() + trend.slice(1);
        }
        
        const timePointsElement = document.getElementById('time-points');
        if (timePointsElement) {
            timePointsElement.textContent = timeSeriesData.time_points || 1;
        }
    }
}

// Image modal functions
function showFullSizeImage(src, title) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    
    if (modal && modalImg) {
        modal.style.display = 'block';
        modalImg.src = src;
        modalImg.alt = title || 'Full Size Image';
    }
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Export functions
function exportEnhancedResults(format) {
    if (!window.currentAnalysisId) {
        alert('No analysis results available for export');
        return;
    }
    
    window.location.href = `/api/export_enhanced/${window.currentAnalysisId}/${format}`;
}

function exportMethodResults(analysisId, methodKey, format) {
    window.location.href = `/api/export_enhanced/${window.currentAnalysisId}/${format}/${methodKey}`;
}

function downloadImage(imageId) {
    const img = document.getElementById(imageId);
    if (img && img.src) {
        const link = document.createElement('a');
        link.download = `${imageId}_${new Date().getTime()}.png`;
        link.href = img.src;
        link.click();
    }
}

// ✅ RESTORED: Original working tophat training functions
function checkTophatStatus() {
    fetch('/api/tophat/status')
        .then(response => response.json())
        .then(data => {
            const statusElement = document.getElementById('tophat-model-status');
            const sessionsElement = document.getElementById('tophat-sessions-count');
            
            console.log('🔍 Tophat status response:', data);
            
            if (statusElement) {
                // Enhanced status logic with detailed information
                if (data.success === false) {
                    statusElement.innerHTML = '❌ Status Check Failed';
                    statusElement.style.color = 'red';
                } else if (data.model_trained) {
                    // Model is actually trained and valid
                    const sessionsInfo = data.training_sessions_count > 0 ? ` (${data.training_sessions_count} sessions)` : '';
                    const sizeInfo = data.model_file_size > 0 ? ` ${Math.round(data.model_file_size / 1024)}KB` : '';
                    
                    statusElement.innerHTML = `✅ Model Trained & Ready${sizeInfo}`;
                    statusElement.style.color = 'green';
                    statusElement.title = `Last trained: ${data.last_trained || 'Unknown'}${sessionsInfo}`;
                    console.log('✅ Tophat model is trained and ready');
                } else if (data.model_available) {
                    // Model file exists but not valid/trained
                    statusElement.innerHTML = '⚠️ Model File Invalid';
                    statusElement.style.color = 'orange';
                    statusElement.title = 'Model file exists but appears to be corrupted or incomplete';
                    console.log('⚠️ Tophat model file exists but is invalid');
                } else if (data.has_training_data) {
                    // Has annotation data but no model
                    statusElement.innerHTML = `🔄 Ready to Train (${data.annotation_files_count} annotations)`;
                    statusElement.style.color = 'blue';
                    statusElement.title = 'Annotation data available - ready to train model';
                    console.log('🔄 Has training data but no model');
                } else {
                    // No model, no training data
                    statusElement.innerHTML = '❌ No Training Data';
                    statusElement.style.color = 'red';
                    statusElement.title = 'Start a training session to create annotation data';
                    console.log('❌ No Tophat model or training data available');
                }
            }
            
            if (sessionsElement) {
                const activeSessions = data.training_sessions_active || 0;
                sessionsElement.textContent = `${activeSessions} active`;
                
                if (activeSessions > 0) {
                    sessionsElement.style.color = 'orange';
                    sessionsElement.style.fontWeight = 'bold';
                } else {
                    sessionsElement.style.color = 'inherit';
                    sessionsElement.style.fontWeight = 'normal';
                }
            }
        })
        .catch(error => {
            console.error('Tophat status check error:', error);
            const statusElement = document.getElementById('tophat-model-status');
            if (statusElement) {
                statusElement.innerHTML = '❌ Connection Failed';
                statusElement.style.color = 'red';
            }
        });
}

function startTophatTraining() {
    if (uploadedFiles.length === 0) {
        alert('Please upload images first');
        return;
    }

    fetch('/api/tophat/status')
        .then(response => response.json())
        .then(statusData => {
            const useTophat = statusData.model_available && statusData.model_trained;
            const fallback = !useTophat;

            const files = uploadedFiles.map(f => ({
                id: f.id,
                filename: f.original_name,
                path: f.path
            }));

            fetch('/api/tophat/start_training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    files: files,
                    use_tophat: useTophat,
                    fallback_to_watershed: fallback
                })
            })
            .then(response => {
                console.log('Training start response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.text(); // Get as text first to debug
            })
            .then(responseText => {
                console.log('Training start response text:', responseText);
                try {
                    const data = JSON.parse(responseText);
                    if (data.success) {
                        trainingSession = data.session;
                        currentTrainingImageIndex = 0;
                        document.getElementById('tophat-training-interface').style.display = 'block';
                        loadTrainingImage();
                    } else {
                        alert('Training session failed to start: ' + data.error);
                    }
                } catch (jsonError) {
                    console.error('JSON parsing error:', jsonError);
                    console.error('Response text:', responseText);
                    alert('Server response error: Invalid JSON response');
                }
            })
            .catch(error => {
                console.error('Training start error:', error);
                alert('Training session failed to start: ' + error.message);
            });
        });
}


// ✅ FIXED: Look for simple detection image, not comprehensive chart
function loadTrainingImage() {
    if (!trainingSession || currentTrainingImageIndex >= trainingSession.images.length) {
        return;
    }

    const canvas = document.getElementById('training-canvas');
    const ctx = canvas.getContext('2d');
    const loadingIndicator = document.getElementById('canvas-loading');
    const imageData = trainingSession.images[currentTrainingImageIndex];

    // Set current image in annotation manager
    const imageId = `image_${currentTrainingImageIndex}_${imageData.filename || 'unknown'}`;
    try {
        if (annotationManager && typeof annotationManager.setCurrentImage === 'function') {
            annotationManager.setCurrentImage(imageId);
        }
    } catch (error) {
        console.error('Error setting current image in annotation manager:', error);
    }

    if (loadingIndicator) loadingIndicator.style.display = 'block';

    document.getElementById('current-training-image').textContent = currentTrainingImageIndex + 1;
    document.getElementById('total-training-images').textContent = trainingSession.images.length;

    const img = new Image();
    let detectionImageFound = false;

    img.onload = function() {
        if (loadingIndicator) loadingIndicator.style.display = 'none';

        const maxWidth = 800, maxHeight = 600;
        let canvasWidth = img.width, canvasHeight = img.height;

        // Store original dimensions for coordinate scaling
        window.originalImageWidth = img.width;
        window.originalImageHeight = img.height;
        
        // Calculate scale factors
        let scaleX = 1, scaleY = 1;
        if (canvasWidth > maxWidth) {
            scaleX = maxWidth / canvasWidth;
            canvasWidth = maxWidth;
            canvasHeight *= scaleX;
        }
        if (canvasHeight > maxHeight) {
            scaleY = maxHeight / canvasHeight;
            canvasHeight = maxHeight;
            canvasWidth *= scaleY;
        }
        
        // Store final scale for coordinate conversion
        window.canvasScaleX = canvasWidth / window.originalImageWidth;
        window.canvasScaleY = canvasHeight / window.originalImageHeight;

        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);

        // Redraw annotations for this image
        setTimeout(() => {
            unifiedDrawing.drawOnCanvas(canvas, window.canvasScaleX || 1, window.canvasScaleY || 1);
            console.log(`🎨 Redrawn annotations for image ${currentTrainingImageIndex + 1}`);
        }, 100);

        console.log(`✅ Loaded training image ${currentTrainingImageIndex + 1} (${canvasWidth}x${canvasHeight}, scale: ${window.canvasScaleX.toFixed(3)}x${window.canvasScaleY.toFixed(3)})`);
    };

    img.onerror = function() {
        console.error(`❌ Failed to load training image ${currentTrainingImageIndex + 1}`);
        if (loadingIndicator) {
            loadingIndicator.innerHTML = `
                <div style="color: #dc3545; font-size: 16px;">❌ Image loading failed</div>
                <div style="font-size: 14px; color: #666; margin-top: 10px;">You can still draw annotations</div>
            `;
            setTimeout(() => loadingIndicator.style.display = 'none', 3000);
        }

        canvas.width = 600;
        canvas.height = 400;
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#333';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Original image unavailable', canvas.width / 2, canvas.height / 2 - 10);
        ctx.fillText('Draw annotations here', canvas.width / 2, canvas.height / 2 + 10);
    };

    // ✅ PRIORITY LOADING (simple only)
    const imageBase64 =
        imageData.image_data?.training_overlay_b64 ||
        imageData.image_data?.method_visualization ||
        imageData.image_data?.labeled_image_b64 ||
        imageData.image_data?.simple_detection ||
        imageData.image_data?.raw_input_b64 ||
        imageData.detection_visualization ||
        imageData.image_data?.visualizations?.detection_overview;

    if (imageBase64) {
        console.log(`🎯 Using prioritized detection overlay for image ${currentTrainingImageIndex + 1}`);
        img.src = `data:image/png;base64,${imageBase64}`;
        detectionImageFound = true;
    }

    if (!detectionImageFound) {
        console.warn(`⚠️ No valid training image found for index ${currentTrainingImageIndex + 1}`);
        console.log('Available imageData:', Object.keys(imageData));
        if (imageData.image_data) {
            console.log('image_data keys:', Object.keys(imageData.image_data));
        }
        img.onerror(); // force fallback
    }

    annotations = {};
}


function setAnnotationMode(mode) {
    currentAnnotationMode = mode;
    
    // Update button styles
    document.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    console.log(`Annotation mode set to: ${mode}`);
}

function clearAnnotations() {
    if (confirm('Are you sure you want to clear all annotations for this image?')) {
        // Clear unified annotations
        unifiedDrawing.clearAnnotations();
        
        // Clear local annotations
        annotations = {};
        
        // Clear fullscreen annotations
        fullscreenState.annotations = {};
        
        // Clear from annotation manager for current image
        const currentImageId = annotationManager.getCurrentImageId();
        if (currentImageId) {
            annotationManager.imageAnnotations.delete(currentImageId);
            console.log(`🗑️ Cleared all annotations for image: ${currentImageId}`);
        }
        
        // Reload to clear canvas
        loadTrainingImage();
        
        console.log('🗑️ All annotations cleared from unified system');
    }
}

function saveAnnotations() {
    const currentImage = trainingSession.images[currentTrainingImageIndex];
    
    // Force save current annotations to annotation manager
    let currentImageId = null;
    let allImageAnnotations = null;
    
    try {
        if (annotationManager && typeof annotationManager.forceSave === 'function') {
            annotationManager.forceSave();
            currentImageId = annotationManager.getCurrentImageId();
            allImageAnnotations = annotationManager.getImageAnnotations(currentImageId);
        }
    } catch (error) {
        console.error('Error with annotation manager in save:', error);
    }
    
    // Convert unified annotations to legacy format for backend
    const unifiedAnns = allImageAnnotations ? allImageAnnotations.unified : unifiedDrawing.getAllAnnotations();
    const convertedAnnotations = convertUnifiedToLegacyFormat(unifiedAnns);
    
    // Merge with any existing legacy annotations
    const finalAnnotations = {
        correct: [...(annotations.correct || []), ...(convertedAnnotations.correct || [])],
        false_positive: [...(annotations.false_positive || []), ...(convertedAnnotations.false_positive || [])],
        missed: [...(annotations.missed || []), ...(convertedAnnotations.missed || [])]
    };
    
    console.log('📊 Final annotations to save:', {
        correct: finalAnnotations.correct.length,
        false_positive: finalAnnotations.false_positive.length,
        missed: finalAnnotations.missed.length
    });
    
    // Prepare comprehensive annotation data
    const annotationData = {
        session_id: trainingSession.id,
        image_filename: currentImage.filename,
        image_index: currentTrainingImageIndex,
        image_id: currentImageId,
        annotations: finalAnnotations, // Properly converted format
        unified_annotations: unifiedAnns,
        fullscreen_annotations: allImageAnnotations ? allImageAnnotations.fullscreen : {},
        annotation_timestamp: Date.now(),
        annotated_image: '' // Could add canvas data here if needed
    };
    
    fetch('/api/tophat/save_annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(annotationData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Annotations saved successfully');
            nextTrainingImage();
        } else {
            alert('Failed to save annotations: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Save annotations error:', error);
        alert('Failed to save annotations: ' + error.message);
    });
}

function nextTrainingImage() {
    // Save current annotations before moving
    try {
        if (annotationManager && typeof annotationManager.forceSave === 'function') {
            annotationManager.forceSave();
        }
    } catch (error) {
        console.error('Error saving annotations:', error);
    }
    
    currentTrainingImageIndex++;
    
    if (currentTrainingImageIndex >= trainingSession.images.length) {
        alert('Training session completed! You can now run the ML trainer.');
        document.getElementById('tophat-training-interface').style.display = 'none';
        
        // Refresh status after completing session
        setTimeout(() => {
            checkTophatStatus();
            console.log('🔄 Refreshed Tophat status after training session completion');
        }, 1000);
    } else {
        loadTrainingImage();
    }
}

function previousTrainingImage() {
    // Save current annotations before moving
    try {
        if (annotationManager && typeof annotationManager.forceSave === 'function') {
            annotationManager.forceSave();
        }
    } catch (error) {
        console.error('Error saving annotations:', error);
    }
    
    if (currentTrainingImageIndex > 0) {
        currentTrainingImageIndex--;
        loadTrainingImage();
    } else {
        alert('This is the first image in the training session.');
    }
}

function finishTraining() {
    if (confirm('Are you sure you want to finish training? This will end the current session.')) {
        document.getElementById('tophat-training-interface').style.display = 'none';
        trainingSession = null;
        currentTrainingImageIndex = 0;
        annotations = {};
        
        // Clear unified annotations
        unifiedDrawing.clearAnnotations();
        
        // Refresh status after finishing training
        setTimeout(() => {
            checkTophatStatus();
            console.log('🔄 Refreshed Tophat status after finishing training');
        }, 1000);
    }
}

// Canvas drawing for annotations (simplified)
document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('training-canvas');
    if (canvas) {
        // Mouse event handlers for drawing strokes
        let isDrawing = false;
        let currentStroke = null;
        
        canvas.addEventListener('mousedown', function(e) {
            if (!currentAnnotationMode) return;
            
            const rect = canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            // Convert canvas coordinates to original image coordinates
            const originalX = Math.round(canvasX / (window.canvasScaleX || 1));
            const originalY = Math.round(canvasY / (window.canvasScaleY || 1));
            
            isDrawing = true;
            currentStroke = unifiedDrawing.startStroke(originalX, originalY, currentAnnotationMode);
            
            // Visual feedback
            canvas.style.cursor = 'crosshair';
            console.log(`🖊️ Started drawing ${currentAnnotationMode} stroke`);
        });
        
        canvas.addEventListener('mousemove', function(e) {
            if (!isDrawing || !currentStroke || !currentAnnotationMode) return;
            
            const rect = canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            // Convert canvas coordinates to original image coordinates
            const originalX = Math.round(canvasX / (window.canvasScaleX || 1));
            const originalY = Math.round(canvasY / (window.canvasScaleY || 1));
            
            unifiedDrawing.addPointToStroke(originalX, originalY, currentAnnotationMode);
            
            // Redraw with current stroke
            loadTrainingImage();
            setTimeout(() => {
                unifiedDrawing.drawOnCanvas(canvas, window.canvasScaleX || 1, window.canvasScaleY || 1);
            }, 10);
        });
        
        canvas.addEventListener('mouseup', async function(e) {
            if (!isDrawing || !currentStroke || !currentAnnotationMode) return;
            
            isDrawing = false;
            canvas.style.cursor = 'default';
            
            // Complete the stroke with smart boundary if enabled
            const completedStroke = await unifiedDrawing.completeStroke(currentAnnotationMode, canvas);
            
            // Keep backward compatibility
            if (!annotations[currentAnnotationMode]) {
                annotations[currentAnnotationMode] = [];
            }
            
            if (completedStroke && completedStroke.points) {
                completedStroke.points.forEach(point => {
                    annotations[currentAnnotationMode].push({ 
                        x: point.x, 
                        y: point.y, 
                        canvas_x: point.x * (window.canvasScaleX || 1),
                        canvas_y: point.y * (window.canvasScaleY || 1),
                        timestamp: Date.now() 
                    });
                });
            }
            
            // Final redraw
            loadTrainingImage();
            setTimeout(() => {
                unifiedDrawing.drawOnCanvas(canvas, window.canvasScaleX || 1, window.canvasScaleY || 1);
            }, 100);
            
            console.log(`✅ Completed ${currentAnnotationMode} stroke:`, completedStroke);
            currentStroke = null;
        });
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    if (document.getElementById('tophat-training-interface').style.display !== 'none') {
        switch(e.key) {
            case '1':
                setAnnotationMode('correct');
                break;
            case '2':
                setAnnotationMode('false_positive');
                break;
            case '3':
                setAnnotationMode('missed');
                break;
            case 's':
                if (e.ctrlKey) {
                    e.preventDefault();
                    saveAnnotations();
                }
                break;
            case 'n':
                if (e.ctrlKey) {
                    e.preventDefault();
                    nextTrainingImage();
                }
                break;
        }
    }
});

// ✅ COMPLETELY REWRITTEN: Simple & Robust Fullscreen System
let fullscreenState = {
    isActive: false,
    canvas: null,
    ctx: null,
    image: null,
    
    // View
    zoom: 1,
    panX: 0,
    panY: 0,
    
    // Drawing
    mode: 'correct',
    isDrawing: false,
    isPanning: false,
    autoBorder: false,
    annotations: {},
    currentStroke: [],
    
    // Mouse
    lastX: 0,
    lastY: 0
};

function openFullscreenAnnotator() {
    console.log('🚀 Opening fullscreen annotator...');
    
    const modal = document.getElementById('fullscreenAnnotationModal');
    if (!modal) {
        alert('Fullscreen modal not found');
        return;
    }
    
    if (!trainingSession || currentTrainingImageIndex >= trainingSession.images.length) {
        alert('No training image available');
        return;
    }
    
    // Save main canvas annotations before opening fullscreen
    try {
        if (annotationManager && typeof annotationManager.forceSave === 'function') {
            annotationManager.forceSave();
        }
    } catch (error) {
        console.error('Error saving annotations before fullscreen:', error);
    }
    
    // Show modal
    modal.style.display = 'block';
    fullscreenState.isActive = true;
    
    // Reset state
    fullscreenState.zoom = 1;
    fullscreenState.panX = 0;
    fullscreenState.panY = 0;
    fullscreenState.isDrawing = false;
    fullscreenState.isPanning = false;
    
    // Load current image annotations into fullscreen
    let currentImageId = null;
    let imageAnnotations = null;
    
    try {
        if (annotationManager && typeof annotationManager.getCurrentImageId === 'function') {
            currentImageId = annotationManager.getCurrentImageId();
            imageAnnotations = annotationManager.getImageAnnotations(currentImageId);
        }
    } catch (error) {
        console.error('Error getting current image annotations:', error);
    }
    
    if (imageAnnotations && imageAnnotations.fullscreen) {
        fullscreenState.annotations = JSON.parse(JSON.stringify(imageAnnotations.fullscreen));
        console.log('📂 Loaded fullscreen annotations for current image');
    } else {
        fullscreenState.annotations = {};
        console.log('🆕 No existing fullscreen annotations for current image');
    }
    
    // Initialize canvas after DOM is ready
    requestAnimationFrame(() => {
        initializeFullscreenCanvas();
        loadFullscreenImage();
        setupFullscreenEvents();
        updateFullscreenUI();
        
        setTimeout(() => {
            showInitialShortcutsHelp();
        }, 1000);
    });
}

function closeFullscreenAnnotator() {
    console.log('🚪 Closing fullscreen annotator...');
    
    const modal = document.getElementById('fullscreenAnnotationModal');
    if (modal) {
        modal.style.display = 'none';
    }
    
    fullscreenState.isActive = false;
    
    // Save fullscreen annotations to current image
    const currentImageId = annotationManager.getCurrentImageId();
    if (currentImageId) {
        // Get current saved annotations
        const savedAnnotations = annotationManager.getImageAnnotations(currentImageId) || {
            unified: { correct: [], false_positive: [], missed: [] },
            legacy: {},
            fullscreen: {},
            timestamp: Date.now()
        };
        
        // Update fullscreen annotations
        savedAnnotations.fullscreen = JSON.parse(JSON.stringify(fullscreenState.annotations));
        savedAnnotations.timestamp = Date.now();
        
        // Save back to annotation manager
        annotationManager.imageAnnotations.set(currentImageId, savedAnnotations);
        
        console.log('💾 Saved fullscreen annotations to image:', currentImageId);
    }
    
    // Transfer annotations to unified system
    transferFullscreenAnnotations();
    
    // Redraw main canvas with updated annotations
    const mainCanvas = document.getElementById('training-canvas');
    if (mainCanvas) {
        setTimeout(() => {
            loadTrainingImage(); // This will reload the image and redraw all annotations
        }, 100);
    }
    
    // Clean up events
    cleanupFullscreenEvents();
}

function initializeFullscreenCanvas() {
    fullscreenState.canvas = document.getElementById('fullscreenCanvas');
    if (!fullscreenState.canvas) {
        console.error('❌ Canvas not found');
        return;
    }
    
    fullscreenState.ctx = fullscreenState.canvas.getContext('2d');
    
    // Get container dimensions for proper sizing
    const container = document.getElementById('fullscreenCanvasContainer');
    if (!container) {
        console.error('❌ Canvas container not found');
        return;
    }
    
    // Use container dimensions (flexbox will handle sizing)
    const containerRect = container.getBoundingClientRect();
    const canvasWidth = containerRect.width;
    const canvasHeight = containerRect.height;
    
    fullscreenState.canvas.width = canvasWidth;
    fullscreenState.canvas.height = canvasHeight;
    
    console.log(`✅ Canvas sized to container: ${canvasWidth}x${canvasHeight}`);
    
    // Hide loading
    hideFullscreenLoading();
    
    // Draw initial state
    drawFullscreenCanvas();
}

function loadFullscreenImage() {
    if (!trainingSession) {
        console.warn('⚠️ No training session');
        drawFullscreenError();
        return;
    }
    
    const imageData = trainingSession.images[currentTrainingImageIndex];
    if (!imageData) {
        console.warn('⚠️ No image data');
        drawFullscreenError();
        return;
    }
    
    // Try to find image data
    const imageBase64 = 
        imageData.image_data?.training_overlay_b64 ||
        imageData.image_data?.method_visualization ||
        imageData.image_data?.labeled_image_b64 ||
        imageData.image_data?.simple_detection ||
        imageData.image_data?.raw_input_b64 ||
        imageData.detection_visualization;
    
    if (imageBase64) {
        fullscreenState.image = new Image();
        fullscreenState.image.onload = function() {
            console.log(`✅ Image loaded: ${this.width}x${this.height}`);
            drawFullscreenCanvas();
        };
        fullscreenState.image.onerror = function() {
            console.error('❌ Image load failed');
            drawFullscreenError();
        };
        fullscreenState.image.src = `data:image/png;base64,${imageBase64}`;
    } else {
        console.warn('⚠️ No image data found');
        drawFullscreenError();
    }
}

function drawFullscreenCanvas() {
    if (!fullscreenState.ctx || !fullscreenState.canvas) return;
    
    const ctx = fullscreenState.ctx;
    const canvas = fullscreenState.canvas;
    
    // Clear background
    ctx.fillStyle = '#1a252f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw image if available
    if (fullscreenState.image && fullscreenState.image.complete) {
        drawFullscreenImage();
    }
    
    // Draw annotations
    drawFullscreenAnnotations();
    
    // Draw current stroke
    drawFullscreenStroke();
}

function drawFullscreenImage() {
    const ctx = fullscreenState.ctx;
    const canvas = fullscreenState.canvas;
    const img = fullscreenState.image;
    
    // Calculate scale to fit
    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;
    const scale = Math.min(scaleX, scaleY) * fullscreenState.zoom;
    
    // Calculate position
    const imgW = img.width * scale;
    const imgH = img.height * scale;
    const x = (canvas.width - imgW) / 2 + fullscreenState.panX;
    const y = (canvas.height - imgH) / 2 + fullscreenState.panY;
    
    // Draw image
    ctx.drawImage(img, x, y, imgW, imgH);
    
    // Store transform for coordinate conversion
    fullscreenState.transform = { x, y, scale, imgW, imgH };
}

function drawFullscreenError() {
    if (!fullscreenState.ctx || !fullscreenState.canvas) return;
    
    const ctx = fullscreenState.ctx;
    const canvas = fullscreenState.canvas;
    
    // Clear background
    ctx.fillStyle = '#1a252f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw error message
    ctx.fillStyle = '#ffffff';
    ctx.font = '24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    ctx.fillText('🖼️ Training Image Not Available', centerX, centerY - 30);
    
    ctx.font = '16px Arial';
    ctx.fillStyle = '#bdc3c7';
    ctx.fillText('You can still draw annotations here', centerX, centerY + 10);
    ctx.fillText('or exit and try a different image', centerX, centerY + 35);
    
    // Draw border
    ctx.strokeStyle = '#34495e';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);
    ctx.strokeRect(50, 50, canvas.width - 100, canvas.height - 100);
    ctx.setLineDash([]);
}

function drawFullscreenAnnotations() {
    if (!fullscreenState.ctx || !fullscreenState.transform) return;
    
    const { x: imgX, y: imgY, scale } = fullscreenState.transform;
    
    // Draw unified annotations
    unifiedDrawing.drawOnCanvas(
        fullscreenState.canvas, 
        scale, scale, 
        imgX, imgY
    );
    
    // Also draw any local fullscreen annotations for immediate feedback
    const ctx = fullscreenState.ctx;
    Object.keys(fullscreenState.annotations).forEach(mode => {
        const color = getFullscreenModeColor(mode);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = Math.max(2, 3 / fullscreenState.zoom);
        ctx.lineCap = 'round';
        
        fullscreenState.annotations[mode].forEach(annotation => {
            if (annotation.type === 'smart_boundary' && annotation.boundaryPoints) {
                // Draw smart boundary
                ctx.beginPath();
                annotation.boundaryPoints.forEach((point, i) => {
                    const canvasX = point.x * scale + imgX;
                    const canvasY = point.y * scale + imgY;
                    if (i === 0) ctx.moveTo(canvasX, canvasY);
                    else ctx.lineTo(canvasX, canvasY);
                });
                ctx.closePath();
                ctx.stroke();
            } else if (annotation.type === 'point') {
                const canvasX = annotation.x * scale + imgX;
                const canvasY = annotation.y * scale + imgY;
                
                ctx.beginPath();
                ctx.arc(canvasX, canvasY, Math.max(4, 6 / fullscreenState.zoom), 0, 2 * Math.PI);
                ctx.fill();
            }
        });
    });
}

function drawFullscreenStroke() {
    if (!fullscreenState.ctx || !fullscreenState.isDrawing) return;
    if (fullscreenState.currentStroke.length < 1 || !fullscreenState.transform) return;
    
    const ctx = fullscreenState.ctx;
    const color = getFullscreenModeColor(fullscreenState.mode);
    const { x: imgX, y: imgY, scale } = fullscreenState.transform;
    
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = Math.max(2, 3 / fullscreenState.zoom);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Make current stroke slightly more transparent to distinguish from completed strokes
    ctx.globalAlpha = 0.7;
    
    if (fullscreenState.currentStroke.length === 1) {
        const point = fullscreenState.currentStroke[0];
        // Convert image coordinates to canvas coordinates
        const canvasX = point.x * scale + imgX;
        const canvasY = point.y * scale + imgY;
        
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, Math.max(3, 4 / fullscreenState.zoom), 0, 2 * Math.PI);
        ctx.fill();
    } else {
        ctx.beginPath();
        fullscreenState.currentStroke.forEach((point, i) => {
            // Convert image coordinates to canvas coordinates
            const canvasX = point.x * scale + imgX;
            const canvasY = point.y * scale + imgY;
            
            if (i === 0) ctx.moveTo(canvasX, canvasY);
            else ctx.lineTo(canvasX, canvasY);
        });
        ctx.stroke();
    }
    
    // Reset transparency
    ctx.globalAlpha = 1.0;
}

function getFullscreenModeColor(mode) {
    switch (mode) {
        case 'correct': return '#28a745';
        case 'false_positive': return '#dc3545';
        case 'missed': return '#007bff';
        default: return '#6c757d';
    }
}

function hideFullscreenLoading() {
    const loading = document.getElementById('fullscreenLoading');
    if (loading) {
        loading.style.display = 'none';
    }
}

function setupFullscreenEvents() {
    const canvas = fullscreenState.canvas;
    if (!canvas) return;
    
    // Remove existing listeners first
    cleanupFullscreenEvents();
    
    // Mouse events
    canvas.addEventListener('mousedown', handleFullscreenMouseDown);
    canvas.addEventListener('mousemove', handleFullscreenMouseMove);
    canvas.addEventListener('mouseup', handleFullscreenMouseUp);
    canvas.addEventListener('wheel', handleFullscreenWheel);
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    
    // Touch events
    canvas.addEventListener('touchstart', handleFullscreenTouchStart);
    canvas.addEventListener('touchmove', handleFullscreenTouchMove);
    canvas.addEventListener('touchend', handleFullscreenTouchEnd);
}

function cleanupFullscreenEvents() {
    const canvas = fullscreenState.canvas;
    if (!canvas) return;
    
    canvas.removeEventListener('mousedown', handleFullscreenMouseDown);
    canvas.removeEventListener('mousemove', handleFullscreenMouseMove);
    canvas.removeEventListener('mouseup', handleFullscreenMouseUp);
    canvas.removeEventListener('wheel', handleFullscreenWheel);
    canvas.removeEventListener('touchstart', handleFullscreenTouchStart);
    canvas.removeEventListener('touchmove', handleFullscreenTouchMove);
    canvas.removeEventListener('touchend', handleFullscreenTouchEnd);
}

function getFullscreenMousePos(e) {
    const rect = fullscreenState.canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;
    
    // Convert canvas coordinates to image coordinates
    if (fullscreenState.transform && fullscreenState.image) {
        const { x: imgX, y: imgY, scale } = fullscreenState.transform;
        
        // Calculate position relative to image
        const imageX = (canvasX - imgX) / scale;
        const imageY = (canvasY - imgY) / scale;
        
        return {
            x: imageX,
            y: imageY,
            canvasX: canvasX,
            canvasY: canvasY
        };
    }
    
    // Fallback to canvas coordinates if no transform available
    return {
        x: canvasX,
        y: canvasY,
        canvasX: canvasX,
        canvasY: canvasY
    };
}

function handleFullscreenMouseDown(e) {
    e.preventDefault();
    const pos = getFullscreenMousePos(e);
    
    if (e.button === 2 || e.ctrlKey) {
        // Pan mode - use canvas coordinates
        fullscreenState.isPanning = true;
        fullscreenState.lastX = pos.canvasX;
        fullscreenState.lastY = pos.canvasY;
        fullscreenState.canvas.style.cursor = 'grabbing';
        showPanIndicator();
    } else if (e.button === 0) {
        // Draw mode - start stroke
        if (fullscreenState.image && isPointOnImage(pos.x, pos.y)) {
            fullscreenState.isDrawing = true;
            fullscreenState.currentStroke = [{ x: pos.x, y: pos.y }];
            
            // Start stroke in unified system
            const stroke = unifiedDrawing.startStroke(pos.x, pos.y, fullscreenState.mode);
            
            console.log(`🖊️ Started fullscreen drawing ${fullscreenState.mode} stroke`);
            drawFullscreenCanvas();
            updateFullscreenUI();
        }
    }
}

function handleFullscreenMouseMove(e) {
    e.preventDefault();
    const pos = getFullscreenMousePos(e);
    
    if (fullscreenState.isPanning) {
        // Pan mode - use canvas coordinates
        const dx = pos.canvasX - fullscreenState.lastX;
        const dy = pos.canvasY - fullscreenState.lastY;
        
        fullscreenState.panX += dx;
        fullscreenState.panY += dy;
        
        fullscreenState.lastX = pos.canvasX;
        fullscreenState.lastY = pos.canvasY;
        
        drawFullscreenCanvas();
    } else if (fullscreenState.isDrawing) {
        // Draw mode - continue stroke
        const imagePoint = { x: pos.x, y: pos.y };
        fullscreenState.currentStroke.push(imagePoint);
        
        // Add to unified system
        unifiedDrawing.addPointToStroke(pos.x, pos.y, fullscreenState.mode);
        
        drawFullscreenCanvas();
    }
}

function handleFullscreenMouseUp(e) {
    e.preventDefault();
    
    if (fullscreenState.isPanning) {
        fullscreenState.isPanning = false;
        fullscreenState.canvas.style.cursor = 'crosshair';
        hidePanIndicator();
    } else if (fullscreenState.isDrawing) {
        fullscreenState.isDrawing = false;
        
        // Complete stroke with smart boundary if enabled
        unifiedDrawing.completeStroke(fullscreenState.mode, fullscreenState.canvas).then(completedStroke => {
            // Add to fullscreen state for display
            if (!fullscreenState.annotations[fullscreenState.mode]) {
                fullscreenState.annotations[fullscreenState.mode] = [];
            }
            
            if (completedStroke) {
                fullscreenState.annotations[fullscreenState.mode].push({
                    type: completedStroke.type,
                    points: completedStroke.points,
                    isComplete: completedStroke.isComplete,
                    timestamp: completedStroke.timestamp
                });
                
                console.log(`✅ Completed fullscreen ${fullscreenState.mode} stroke:`, completedStroke);
            }
            
            fullscreenState.currentStroke = [];
            drawFullscreenCanvas();
            updateFullscreenUI();
        }).catch(error => {
            console.error('Error completing fullscreen stroke:', error);
            fullscreenState.currentStroke = [];
            drawFullscreenCanvas();
            updateFullscreenUI();
        });
    }
}

function handleFullscreenWheel(e) {
    e.preventDefault();
    
    const pos = getFullscreenMousePos(e);
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, fullscreenState.zoom * factor));
    
    if (newZoom !== fullscreenState.zoom) {
        // Zoom centered on mouse position
        const ratio = newZoom / fullscreenState.zoom;
        fullscreenState.panX = pos.canvasX - (pos.canvasX - fullscreenState.panX) * ratio;
        fullscreenState.panY = pos.canvasY - (pos.canvasY - fullscreenState.panY) * ratio;
        
        fullscreenState.zoom = newZoom;
        updateFullscreenZoom();
        drawFullscreenCanvas();
    }
}

function handleFullscreenTouchStart(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        handleFullscreenMouseDown({
            preventDefault: () => {},
            clientX: touch.clientX,
            clientY: touch.clientY,
            button: 0
        });
    }
}

function handleFullscreenTouchMove(e) {
    e.preventDefault();
    if (e.touches.length === 1) {
        const touch = e.touches[0];
        handleFullscreenMouseMove({
            preventDefault: () => {},
            clientX: touch.clientX,
            clientY: touch.clientY
        });
    }
}

function handleFullscreenTouchEnd(e) {
    e.preventDefault();
    handleFullscreenMouseUp({ preventDefault: () => {} });
}

function transferFullscreenAnnotations() {
    // Convert fullscreen annotations to main canvas format
    Object.keys(fullscreenState.annotations).forEach(mode => {
        if (!annotations[mode]) annotations[mode] = [];
        
        fullscreenState.annotations[mode].forEach(annotation => {
            if (annotation.type === 'point') {
                // Convert single point
                const mainCanvasAnnotation = {
                    type: 'point',
                    x: annotation.x,
                    y: annotation.y,
                    timestamp: annotation.timestamp,
                    mode: mode
                };
                annotations[mode].push(mainCanvasAnnotation);
            } else if (annotation.points && annotation.points.length > 0) {
                // Convert stroke points
                annotation.points.forEach(point => {
                    const mainCanvasAnnotation = {
                        type: 'point',
                        x: point.x,
                        y: point.y,
                        timestamp: annotation.timestamp,
                        mode: mode
                    };
                    annotations[mode].push(mainCanvasAnnotation);
                });
            }
        });
    });
    
    // Draw annotations on main canvas
    drawAnnotationsOnMainCanvas();
    
    console.log(`✅ Transferred ${Object.values(fullscreenState.annotations).flat().length} annotations to main canvas`);
}

function drawAnnotationsOnMainCanvas() {
    const canvas = document.getElementById('training-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const scaleX = window.canvasScaleX || 1;
    const scaleY = window.canvasScaleY || 1;
    
    // Draw all annotations
    Object.keys(annotations).forEach(mode => {
        const color = getAnnotationColor(mode);
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        
        annotations[mode].forEach(annotation => {
            const canvasX = annotation.x * scaleX;
            const canvasY = annotation.y * scaleY;
            
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
            ctx.fill();
        });
    });
}

function getAnnotationColor(mode) {
    switch (mode) {
        case 'correct': return '#28a745';
        case 'false_positive': return '#dc3545';
        case 'missed': return '#007bff';
        default: return '#6c757d';
    }
}

function updateFullscreenUI() {
    const total = Object.values(fullscreenState.annotations).reduce((sum, arr) => sum + arr.length, 0);
    
    const countEl = document.getElementById('annotationCount');
    if (countEl) {
        countEl.textContent = `${total} annotations`;
    }
    
    const modeEl = document.getElementById('currentDrawingMode');
    if (modeEl) {
        let text = {
            'correct': 'Marking correct cells',
            'false_positive': 'Marking false positives',
            'missed': 'Marking missed cells'
        }[fullscreenState.mode] || 'Drawing mode';
        
        if (fullscreenState.isDrawing) text += ' - Drawing...';
        if (fullscreenState.isPanning) text = 'Panning view';
        
        modeEl.textContent = text;
    }
}

function updateFullscreenZoom() {
    const zoomEl = document.getElementById('zoomLevel');
    if (zoomEl) {
        zoomEl.textContent = Math.round(fullscreenState.zoom * 100) + '%';
    }
}

// Public API functions
function setFullscreenMode(mode) {
    fullscreenState.mode = mode;
    
    // Update button styles
    document.querySelectorAll('.fullscreen-controls .btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    const modeMap = {
        'correct': 'Mark Correct',
        'false_positive': 'Mark False Positive', 
        'missed': 'Mark Missed Cell'
    };
    
    document.querySelectorAll('.fullscreen-controls .btn').forEach(btn => {
        if (btn.textContent.includes(modeMap[mode])) {
            btn.classList.add('active');
        }
    });
    
    updateFullscreenUI();
}

function zoomIn() {
    fullscreenState.zoom = Math.min(10, fullscreenState.zoom * 1.2);
    updateFullscreenZoom();
    drawFullscreenCanvas();
}

function zoomOut() {
    fullscreenState.zoom = Math.max(0.1, fullscreenState.zoom / 1.2);
    updateFullscreenZoom();
    drawFullscreenCanvas();
}

function resetZoom() {
    fullscreenState.zoom = 1;
    fullscreenState.panX = 0;
    fullscreenState.panY = 0;
    updateFullscreenZoom();
    drawFullscreenCanvas();
}

function toggleAutoBorder() {
    // Toggle unified auto-border system
    const isEnabled = unifiedDrawing.toggleAutoBorder();
    
    // Also update fullscreen state for consistency
    fullscreenState.autoBorder = isEnabled;
    
    const statusEl = document.getElementById('autoBorderStatus');
    if (statusEl) {
        statusEl.textContent = isEnabled ? 'ON' : 'OFF';
    }
    
    const button = event?.target?.closest('button');
    if (button) {
        if (isEnabled) {
            button.classList.add('active');
            button.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
        } else {
            button.classList.remove('active');
            button.style.background = '';
        }
    }
    
    console.log(`🎯 Smart Auto-Border: ${isEnabled ? 'ON - Will analyze image to detect cell boundaries' : 'OFF - Simple point annotations'}`);
}

function clearFullscreenAnnotations() {
    // Clear unified annotations
    unifiedDrawing.clearAnnotations();
    
    // Clear local fullscreen annotations
    fullscreenState.annotations = {};
    
    // Clear main canvas annotations for consistency
    annotations = {};
    
    // Redraw everything
    drawFullscreenCanvas();
    updateFullscreenUI();
    
    // Also refresh main canvas if visible
    const mainCanvas = document.getElementById('training-canvas');
    if (mainCanvas) {
        loadTrainingImage();
    }
    
    console.log('🗑️ All annotations cleared from unified system');
}

function showPanIndicator() {
    const indicator = document.getElementById('panIndicator');
    if (indicator) {
        indicator.classList.add('visible');
    }
}

function hidePanIndicator() {
    const indicator = document.getElementById('panIndicator');
    if (indicator) {
        indicator.classList.remove('visible');
    }
}

function isPointOnImage(imageX, imageY) {
    if (!fullscreenState.image) return false;
    
    return imageX >= 0 && imageX <= fullscreenState.image.width &&
           imageY >= 0 && imageY <= fullscreenState.image.height;
}

// Window resize handler
window.addEventListener('resize', () => {
    if (fullscreenState.isActive) {
        setTimeout(() => {
            initializeFullscreenCanvas();
        }, 100);
    }
});

// Note: All fullscreen functions are now implemented above with simple approach

// Enhanced keyboard shortcuts for fullscreen mode
document.addEventListener('keydown', function(e) {
    const modal = document.getElementById('fullscreenAnnotationModal');
    if (modal && modal.style.display === 'block' && fullscreenState.isActive) {
        switch(e.key) {
            case 'Escape':
                e.preventDefault();
                closeFullscreenAnnotator();
                break;
            case '1':
                e.preventDefault();
                setFullscreenMode('correct');
                break;
            case '2':
                e.preventDefault();
                setFullscreenMode('false_positive');
                break;
            case '3':
                e.preventDefault();
                setFullscreenMode('missed');
                break;
            case '+':
            case '=':
                e.preventDefault();
                zoomIn();
                break;
            case '-':
                e.preventDefault();
                zoomOut();
                break;
            case '0':
                e.preventDefault();
                resetZoom();
                break;
            case 'h':
            case 'H':
                e.preventDefault();
                toggleShortcutsHelp();
                break;
            case 'a':
                if (e.ctrlKey) {
                    e.preventDefault();
                    toggleAutoBorder();
                }
                break;
            case 'c':
                if (e.ctrlKey) {
                    e.preventDefault();
                    clearFullscreenAnnotations();
                }
                break;
            case ' ':
                e.preventDefault();
                // Space bar for quick pan mode toggle
                break;
        }
    }
});

// Show/hide keyboard shortcuts help
function toggleShortcutsHelp() {
    const helpElement = document.getElementById('shortcutsHelp');
    console.log('🔍 Toggle shortcuts help:', helpElement ? 'Found element' : 'Element not found');
    
    if (helpElement) {
        const isVisible = helpElement.classList.contains('visible');
        console.log('📱 Help currently visible:', isVisible);
        
        helpElement.classList.toggle('visible');
        
        const nowVisible = helpElement.classList.contains('visible');
        console.log('📱 Help now visible:', nowVisible);
        
        // Auto-hide after 8 seconds if showing
        if (nowVisible) {
            setTimeout(() => {
                if (helpElement.classList.contains('visible')) {
                    helpElement.classList.remove('visible');
                    console.log('⏰ Auto-hiding shortcuts help');
                }
            }, 8000);
        }
    } else {
        console.error('❌ shortcutsHelp element not found!');
    }
}

// Show shortcuts help on first fullscreen entry
let hasShownShortcutsHelp = false;
function showInitialShortcutsHelp() {
    if (!hasShownShortcutsHelp) {
        hasShownShortcutsHelp = true;
        setTimeout(() => {
            toggleShortcutsHelp();
        }, 1000);
    }
}

// ✅ UNIFIED DRAWING SYSTEM
class UnifiedDrawingSystem {
    constructor() {
        this.annotations = unifiedAnnotations;
        this.autoBorderEnabled = false;
        this.currentMode = 'correct';
    }

    // Smart auto-border detection using image analysis
    async detectCellBoundary(imageData, centerX, centerY, radius = 25) {
        if (!imageData) return null;
        
        try {
            // Create a temporary canvas for image analysis
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = imageData.width;
            canvas.height = imageData.height;
            
            // Draw the image
            ctx.putImageData(imageData, 0, 0);
            
            // Get pixel data around the click point
            const searchRadius = radius;
            const startX = Math.max(0, centerX - searchRadius);
            const startY = Math.max(0, centerY - searchRadius);
            const endX = Math.min(imageData.width, centerX + searchRadius);
            const endY = Math.min(imageData.height, centerY + searchRadius);
            
            // Analyze pixels to find cell boundary
            const boundaryPoints = [];
            const imagePixels = ctx.getImageData(startX, startY, endX - startX, endY - startY);
            const data = imagePixels.data;
            
            // Find edges using simple gradient detection
            for (let y = 1; y < imagePixels.height - 1; y++) {
                for (let x = 1; x < imagePixels.width - 1; x++) {
                    const idx = (y * imagePixels.width + x) * 4;
                    
                    // Get green channel intensity (important for cells)
                    const green = data[idx + 1];
                    const greenLeft = data[((y * imagePixels.width) + (x - 1)) * 4 + 1];
                    const greenRight = data[((y * imagePixels.width) + (x + 1)) * 4 + 1];
                    const greenUp = data[(((y - 1) * imagePixels.width) + x) * 4 + 1];
                    const greenDown = data[(((y + 1) * imagePixels.width) + x) * 4 + 1];
                    
                    // Calculate gradient
                    const gradX = greenRight - greenLeft;
                    const gradY = greenDown - greenUp;
                    const gradient = Math.sqrt(gradX * gradX + gradY * gradY);
                    
                    // If gradient is significant, it's likely an edge
                    if (gradient > 20) {
                        const realX = startX + x;
                        const realY = startY + y;
                        const distance = Math.sqrt((realX - centerX) ** 2 + (realY - centerY) ** 2);
                        
                        if (distance <= searchRadius) {
                            boundaryPoints.push({ x: realX, y: realY, gradient });
                        }
                    }
                }
            }
            
            // If we found boundary points, create a smooth boundary
            if (boundaryPoints.length > 5) {
                // Sort points by angle from center to create a closed shape
                boundaryPoints.sort((a, b) => {
                    const angleA = Math.atan2(a.y - centerY, a.x - centerX);
                    const angleB = Math.atan2(b.y - centerY, b.x - centerX);
                    return angleA - angleB;
                });
                
                return boundaryPoints;
            }
            
            // Fallback: create circular boundary
            const fallbackPoints = [];
            const adaptiveRadius = this.estimateCellSize(imageData, centerX, centerY);
            for (let angle = 0; angle < 2 * Math.PI; angle += 0.3) {
                fallbackPoints.push({
                    x: centerX + Math.cos(angle) * adaptiveRadius,
                    y: centerY + Math.sin(angle) * adaptiveRadius
                });
            }
            return fallbackPoints;
            
        } catch (error) {
            console.warn('Smart border detection failed, using circular fallback:', error);
            // Simple circular fallback
            const fallbackPoints = [];
            for (let angle = 0; angle < 2 * Math.PI; angle += 0.5) {
                fallbackPoints.push({
                    x: centerX + Math.cos(angle) * radius,
                    y: centerY + Math.sin(angle) * radius
                });
            }
            return fallbackPoints;
        }
    }

    // Estimate cell size by analyzing local image features
    estimateCellSize(imageData, centerX, centerY) {
        try {
            const ctx = document.createElement('canvas').getContext('2d');
            ctx.canvas.width = imageData.width;
            ctx.canvas.height = imageData.height;
            ctx.putImageData(imageData, 0, 0);
            
            // Sample pixels in expanding circles to find cell boundary
            for (let radius = 5; radius <= 50; radius += 5) {
                let edgeCount = 0;
                const sampleCount = Math.max(8, radius);
                
                for (let i = 0; i < sampleCount; i++) {
                    const angle = (2 * Math.PI * i) / sampleCount;
                    const x = Math.round(centerX + Math.cos(angle) * radius);
                    const y = Math.round(centerY + Math.sin(angle) * radius);
                    
                    if (x >= 0 && x < imageData.width && y >= 0 && y < imageData.height) {
                        const pixel = ctx.getImageData(x, y, 1, 1).data;
                        const intensity = (pixel[0] + pixel[1] + pixel[2]) / 3;
                        
                        // Check if we hit a significant intensity change (edge)
                        if (intensity < 50 || intensity > 200) {
                            edgeCount++;
                        }
                    }
                }
                
                // If we found enough edges, this is likely the cell boundary
                if (edgeCount > sampleCount * 0.3) {
                    return radius;
                }
            }
            
            return 15; // Default size
        } catch (error) {
            return 15; // Fallback size
        }
    }

    // Start a new stroke annotation
    startStroke(x, y, mode) {
        const annotation = {
            type: 'stroke',
            points: [{ x, y }],
            mode: mode,
            timestamp: Date.now(),
            isComplete: false
        };

        this.annotations[mode].push(annotation);
        return annotation;
    }

    // Add point to current stroke
    addPointToStroke(x, y, mode) {
        const modeAnnotations = this.annotations[mode];
        if (modeAnnotations.length > 0) {
            const currentStroke = modeAnnotations[modeAnnotations.length - 1];
            if (!currentStroke.isComplete) {
                currentStroke.points.push({ x, y });
                return currentStroke;
            }
        }
        return null;
    }

    // Complete current stroke with smart boundary completion
    async completeStroke(mode, canvasElement = null) {
        const modeAnnotations = this.annotations[mode];
        if (modeAnnotations.length === 0) return null;

        const currentStroke = modeAnnotations[modeAnnotations.length - 1];
        if (currentStroke.isComplete) return currentStroke;

        currentStroke.isComplete = true;

        // Smart auto-boundary completion
        if (this.autoBorderEnabled && canvasElement && currentStroke.points.length > 2) {
            try {
                const completedBoundary = await this.completeStrokeBoundary(currentStroke, canvasElement);
                if (completedBoundary) {
                    currentStroke.points = completedBoundary;
                    currentStroke.type = 'smart_stroke';
                    console.log(`🧠 Smart boundary completion: ${completedBoundary.length} points`);
                }
            } catch (error) {
                console.warn('Smart boundary completion failed:', error);
            }
        }

        return currentStroke;
    }

    // Complete stroke boundary by analyzing the image
    async completeStrokeBoundary(stroke, canvasElement) {
        try {
            const ctx = canvasElement.getContext('2d');
            const imageData = ctx.getImageData(0, 0, canvasElement.width, canvasElement.height);
            
            // Find the center of the drawn stroke
            const centerX = stroke.points.reduce((sum, p) => sum + p.x, 0) / stroke.points.length;
            const centerY = stroke.points.reduce((sum, p) => sum + p.y, 0) / stroke.points.length;
            
            // Estimate the radius based on stroke extent
            const distances = stroke.points.map(p => Math.sqrt((p.x - centerX) ** 2 + (p.y - centerY) ** 2));
            const avgRadius = distances.reduce((sum, d) => sum + d, 0) / distances.length;
            const searchRadius = Math.max(20, Math.min(50, avgRadius * 1.5));
            
            // Detect complete cell boundary around the stroke
            const boundaryPoints = await this.detectCellBoundary(imageData, centerX, centerY, searchRadius);
            
            if (boundaryPoints && boundaryPoints.length > stroke.points.length) {
                return boundaryPoints;
            }
            
            // If boundary detection fails, try to close the stroke intelligently
            if (stroke.points.length > 3) {
                const closedStroke = [...stroke.points];
                
                // Add intermediate points to close the shape smoothly
                const firstPoint = stroke.points[0];
                const lastPoint = stroke.points[stroke.points.length - 1];
                const distance = Math.sqrt((lastPoint.x - firstPoint.x) ** 2 + (lastPoint.y - firstPoint.y) ** 2);
                
                if (distance > 10) {
                    // Add closing points
                    const steps = Math.max(3, Math.floor(distance / 10));
                    for (let i = 1; i < steps; i++) {
                        const t = i / steps;
                        closedStroke.push({
                            x: lastPoint.x + (firstPoint.x - lastPoint.x) * t,
                            y: lastPoint.y + (firstPoint.y - lastPoint.y) * t
                        });
                    }
                }
                
                return closedStroke;
            }
            
            return stroke.points;
            
        } catch (error) {
            console.warn('Stroke boundary completion failed:', error);
            return stroke.points;
        }
    }

    // Legacy method for backward compatibility
    async addAnnotation(x, y, mode, canvasElement = null) {
        return this.startStroke(x, y, mode);
    }

    // Clear all annotations
    clearAnnotations() {
        this.annotations.correct = [];
        this.annotations.false_positive = [];
        this.annotations.missed = [];
    }

    // Get all annotations
    getAllAnnotations() {
        return this.annotations;
    }

    // Draw annotations on any canvas
    drawOnCanvas(canvas, scaleX = 1, scaleY = 1, offsetX = 0, offsetY = 0) {
        const ctx = canvas.getContext('2d');
        
        Object.keys(this.annotations).forEach(mode => {
            const color = this.getModeColor(mode);
            ctx.fillStyle = color;
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            
            this.annotations[mode].forEach(annotation => {
                if (annotation.type === 'stroke' || annotation.type === 'smart_stroke') {
                    // Draw stroke/path
                    if (annotation.points && annotation.points.length > 0) {
                        ctx.beginPath();
                        annotation.points.forEach((point, i) => {
                            const px = point.x * scaleX + offsetX;
                            const py = point.y * scaleY + offsetY;
                            if (i === 0) ctx.moveTo(px, py);
                            else ctx.lineTo(px, py);
                        });
                        
                        // Close path if it's a completed smart stroke
                        if (annotation.type === 'smart_stroke' && annotation.isComplete) {
                            ctx.closePath();
                        }
                        
                        ctx.stroke();
                    }
                } else if (annotation.type === 'smart_boundary' && annotation.boundaryPoints) {
                    // Draw smart boundary (legacy)
                    ctx.beginPath();
                    annotation.boundaryPoints.forEach((point, i) => {
                        const px = point.x * scaleX + offsetX;
                        const py = point.y * scaleY + offsetY;
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    });
                    ctx.closePath();
                    ctx.stroke();
                } else if (annotation.type === 'point') {
                    // Draw simple point (legacy)
                    const x = annotation.x * scaleX + offsetX;
                    const y = annotation.y * scaleY + offsetY;
                    ctx.beginPath();
                    ctx.arc(x, y, 4, 0, 2 * Math.PI);
                    ctx.fill();
                }
            });
        });
    }

    getModeColor(mode) {
        switch (mode) {
            case 'correct': return '#28a745';
            case 'false_positive': return '#dc3545';
            case 'missed': return '#007bff';
            default: return '#6c757d';
        }
    }

    // Toggle auto-border
    toggleAutoBorder() {
        this.autoBorderEnabled = !this.autoBorderEnabled;
        console.log(`🎯 Smart auto-border: ${this.autoBorderEnabled ? 'ON' : 'OFF'}`);
        return this.autoBorderEnabled;
    }
}

// Create global unified drawing system
const unifiedDrawing = new UnifiedDrawingSystem();

// ✅ IMAGE-SPECIFIC ANNOTATION MANAGEMENT
class ImageAnnotationManager {
    constructor() {
        this.imageAnnotations = new Map();
        this.currentImageId = null;
    }

    // Set current image and load its annotations
    setCurrentImage(imageId) {
        // Save current annotations before switching
        if (this.currentImageId) {
            this.saveCurrentAnnotations();
        }
        
        this.currentImageId = imageId;
        this.loadImageAnnotations(imageId);
        
        console.log(`📸 Switched to image: ${imageId}`);
    }

    // Save current annotations to the current image
    saveCurrentAnnotations() {
        if (!this.currentImageId) return;
        
        const annotationsData = {
            unified: unifiedDrawing.getAllAnnotations(),
            legacy: annotations || {},
            fullscreen: fullscreenState.annotations || {},
            timestamp: Date.now()
        };
        
        this.imageAnnotations.set(this.currentImageId, annotationsData);
        console.log(`💾 Saved annotations for image: ${this.currentImageId}`, annotationsData);
    }

    // Load annotations for specific image
    loadImageAnnotations(imageId) {
        const savedAnnotations = this.imageAnnotations.get(imageId);
        
        if (savedAnnotations) {
            // Restore unified annotations
            unifiedDrawing.annotations = savedAnnotations.unified || {
                correct: [],
                false_positive: [],
                missed: []
            };
            
            // Restore legacy annotations
            annotations = savedAnnotations.legacy || {};
            
            // Restore fullscreen annotations
            fullscreenState.annotations = savedAnnotations.fullscreen || {};
            
            console.log(`📂 Loaded annotations for image: ${imageId}`, savedAnnotations);
        } else {
            // Clear all annotations for new image
            this.clearAllAnnotations();
            console.log(`🆕 New image: ${imageId} - cleared annotations`);
        }
    }

    // Clear all annotation systems
    clearAllAnnotations() {
        unifiedDrawing.clearAnnotations();
        annotations = {};
        fullscreenState.annotations = {};
    }

    // Get annotations for specific image
    getImageAnnotations(imageId) {
        return this.imageAnnotations.get(imageId);
    }

    // Get all images with annotations
    getAllImageAnnotations() {
        return this.imageAnnotations;
    }

    // Force save current annotations
    forceSave() {
        this.saveCurrentAnnotations();
    }

    // Get current image ID
    getCurrentImageId() {
        return this.currentImageId;
    }
}

// Create global annotation manager
const annotationManager = new ImageAnnotationManager();

// ✅ ANNOTATION FORMAT CONVERSION
function convertUnifiedToLegacyFormat(unifiedAnnotations) {
    const legacy = {
        correct: [],
        false_positive: [],
        missed: []
    };
    
    if (!unifiedAnnotations) return legacy;
    
    // Convert each annotation type
    Object.keys(unifiedAnnotations).forEach(mode => {
        if (unifiedAnnotations[mode] && Array.isArray(unifiedAnnotations[mode])) {
            unifiedAnnotations[mode].forEach(annotation => {
                if (annotation.type === 'stroke' || annotation.type === 'smart_stroke') {
                    // Convert stroke to individual points
                    if (annotation.points && annotation.points.length > 0) {
                        annotation.points.forEach(point => {
                            legacy[mode].push({
                                x: Math.round(point.x),
                                y: Math.round(point.y),
                                timestamp: annotation.timestamp || Date.now(),
                                canvas_x: Math.round(point.x), // Same as x for image coordinates
                                canvas_y: Math.round(point.y)  // Same as y for image coordinates
                            });
                        });
                    }
                } else if (annotation.type === 'point') {
                    // Convert single point
                    legacy[mode].push({
                        x: Math.round(annotation.x),
                        y: Math.round(annotation.y),
                        timestamp: annotation.timestamp || Date.now(),
                        canvas_x: Math.round(annotation.x),
                        canvas_y: Math.round(annotation.y)
                    });
                } else if (annotation.type === 'smart_boundary' && annotation.boundaryPoints) {
                    // Convert smart boundary points
                    annotation.boundaryPoints.forEach(point => {
                        legacy[mode].push({
                            x: Math.round(point.x),
                            y: Math.round(point.y),
                            timestamp: annotation.timestamp || Date.now(),
                            canvas_x: Math.round(point.x),
                            canvas_y: Math.round(point.y)
                        });
                    });
                }
            });
        }
    });
    
    console.log('🔄 Converted unified to legacy format:', {
        correct: legacy.correct.length,
        false_positive: legacy.false_positive.length,
        missed: legacy.missed.length
    });
    
    return legacy;
}

console.log('🚀 BIOIMAGIN JavaScript - UNIFIED DRAWING SYSTEM loaded and ready!');
console.log('✅ New Features:');
console.log('   • Class-based fullscreen annotator');
console.log('   • Robust coordinate transformation');
console.log('   • Improved zoom and pan controls');
console.log('   • Real-time drawing feedback');
console.log('   • Keyboard shortcuts help (H key)');
console.log('   • Enhanced visual design');
console.log('   • Mobile touch support');
console.log('   • Memory efficient canvas management');