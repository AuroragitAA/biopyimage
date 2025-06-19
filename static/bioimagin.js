// BIOIMAGIN Professional JavaScript - RESTORED WORKING VERSION
// Handles all web interface interactions and result display

// Global variables
let uploadedFiles = [];
let currentAnalysisId = null;
let currentAnnotationMode = 'correct';
let trainingSession = null;
let currentTrainingImageIndex = 0;
let annotations = {};

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
    const stepCount = pipelineSteps.step_count || 0;
    
    // Define pipeline steps with status
    const pipelineDefinition = [
        { key: 'Original', name: 'Original Image', status: 'success', icon: '📸' },
        { key: 'Denoised', name: 'Denoised Image', status: 'success', icon: '🧹' },
        { key: 'Green_enhanced', name: 'Green Enhancement', status: 'success', icon: '🌿' },
        { key: 'Green_mask', name: 'Green Mask', status: 'success', icon: '🟢' },
        { key: 'Shape_index', name: 'Shape Index', status: 'success', icon: '📐' },
        { key: 'Shape_index_3d', name: 'Shape Index 3D', status: 'success', icon: '📐📐📐' },
        { key: 'Watershed', name: 'Watershed Segmentation', status: 'success', icon: '💧' },
        
        // Legacy or non-watershed steps — still supported if present
        { key: 'Gray', name: 'Grayscale', status: 'success', icon: '⚫' },
        { key: 'Li_foreground', name: 'Li Thresholding', status: 'success', icon: '🎯' },
        { key: 'Plate_removed', name: 'Plate Removal', status: 'success', icon: '🗑️' },
        { key: 'Multi_otsu', name: 'Multi-Otsu', status: 'success', icon: '🔧' },
        { key: 'Combined', name: 'Mask Fusion', status: 'success', icon: '🔀' },
        { key: 'Opened', name: 'Morphological', status: 'success', icon: '🔄' },
        { key: 'Final', name: 'Final Segmentation', status: 'success', icon: '✅' },
        { key: 'Detection_result', name: 'Cell Detection', status: detectionMethod?.includes('AI') ? 'success' : 'warning', icon: '🧬' }
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
    window.location.href = `/api/export_method/${analysisId}/${methodKey}/${format}`;
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
            
            if (statusElement) {
                if (data.model_available && data.model_trained) {
                    statusElement.innerHTML = '✅ Model Ready';
                    statusElement.style.color = 'green';
                } else if (data.model_available) {
                    statusElement.innerHTML = '⚠️ Model Available (Untrained)';
                    statusElement.style.color = 'orange';
                } else {
                    statusElement.innerHTML = '❌ No Model';
                    statusElement.style.color = 'red';
                }
            }
            
            if (sessionsElement) {
                sessionsElement.textContent = `${data.training_sessions_active || 0} active`;
            }
        })
        .catch(error => {
            console.error('Tophat status check error:', error);
            const statusElement = document.getElementById('tophat-model-status');
            if (statusElement) {
                statusElement.innerHTML = '❌ Status Check Failed';
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
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    trainingSession = data.session;
                    currentTrainingImageIndex = 0;
                    document.getElementById('tophat-training-interface').style.display = 'block';
                    loadTrainingImage();
                } else {
                    alert('Training session failed to start: ' + data.error);
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
    annotations = {};
    loadTrainingImage(); // Reload to clear canvas
}

function saveAnnotations() {
    const currentImage = trainingSession.images[currentTrainingImageIndex];
    
    fetch('/api/tophat/save_annotations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: trainingSession.id,
            image_filename: currentImage.filename,
            image_index: currentTrainingImageIndex,
            annotations: annotations,
            annotated_image: '' // Could add canvas data here if needed
        })
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
    currentTrainingImageIndex++;
    
    if (currentTrainingImageIndex >= trainingSession.images.length) {
        alert('Training session completed! You can now run the ML trainer.');
        document.getElementById('tophat-training-interface').style.display = 'none';
        checkTophatStatus();
    } else {
        loadTrainingImage();
    }
}

function finishTraining() {
    if (confirm('Are you sure you want to finish training? This will end the current session.')) {
        document.getElementById('tophat-training-interface').style.display = 'none';
        trainingSession = null;
        currentTrainingImageIndex = 0;
        annotations = {};
        checkTophatStatus();
    }
}

// Canvas drawing for annotations (simplified)
document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('training-canvas');
    if (canvas) {
        canvas.addEventListener('click', function(e) {
            if (!currentAnnotationMode) return;
            
            const rect = canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;
            
            // Convert canvas coordinates to original image coordinates
            const originalX = Math.round(canvasX / (window.canvasScaleX || 1));
            const originalY = Math.round(canvasY / (window.canvasScaleY || 1));
            
            // Store annotation in original image coordinates
            if (!annotations[currentAnnotationMode]) {
                annotations[currentAnnotationMode] = [];
            }
            
            annotations[currentAnnotationMode].push({ 
                x: originalX, 
                y: originalY, 
                canvas_x: canvasX,
                canvas_y: canvasY,
                timestamp: Date.now() 
            });
            
            // Visual feedback on canvas using canvas coordinates
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = currentAnnotationMode === 'correct' ? 'green' : 
                           currentAnnotationMode === 'false_positive' ? 'red' : 'blue';
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
            ctx.fill();
            
            console.log(`Added ${currentAnnotationMode} annotation at original(${originalX}, ${originalY}) canvas(${canvasX}, ${canvasY})`);
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

console.log('🚀 BIOIMAGIN JavaScript - RESTORED WORKING VERSION loaded and ready!');