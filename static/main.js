/**
 * Wolffia Bioimage Analysis Dashboard - Enhanced Frontend
 * Professional implementation with robust error handling and user experience
 */

class WolffiaAnalyzerApp {
    constructor() {
        this.state = {
            currentFiles: [],
            analysisResults: [],
            selectedColorMethod: 'green_wolffia',
            currentAnalysisId: null,
            isAnalyzing: false,
            apiBaseUrl: window.location.origin
        };

        this.config = {
            maxFileSize: 16 * 1024 * 1024, // 16MB
            supportedFormats: ['image/jpeg', 'image/png', 'image/tiff', 'image/bmp'],
            maxFiles: 10
        };

        this.selectors = this.initializeSelectors();
        this.init();
    }

    initializeSelectors() {
        const selectors = {};
        const elements = [
            'uploadArea', 'fileInput', 'analyzeBtn', 'batchBtn', 'exportBtn',
            'analysisMethodSelect', 'colorMethodsDiv', 'progressContainer',
            'progressFill', 'progressText', 'statsGrid', 'imageDisplay',
            'visualizationTab', 'originalCanvas', 'segmentedCanvas'
        ];

        elements.forEach(id => {
            const element = document.getElementById(id);
            if (!element) {
                console.warn(`⚠️ Element not found: ${id}`);
            }
            selectors[id] = element;
        });

        return selectors;
    }

    init() {
        console.log('🚀 Initializing Wolffia Analyzer App...');
        this.setupEventListeners();
        this.setupTabs();
        this.updateUI();
        this.checkServerHealth();
        console.log('✅ Wolffia Analyzer App initialized');
    }

    async checkServerHealth() {
        try {
            const response = await fetch('/api/system/status');
            const health = await response.json();
            
            if (health.status === 'healthy') {
                this.showNotification('🔬 Analysis system ready', 'success');
            } else {
                this.showNotification('⚠️ Some components may not be available', 'warning');
            }
        } catch (error) {
            console.error('Health check failed:', error);
            this.showNotification('❌ Unable to connect to analysis server', 'error');
        }
    }

    setupEventListeners() {
        // File upload handlers
        if (this.selectors.uploadArea) {
            this.selectors.uploadArea.addEventListener('click', () => {
                if (this.selectors.fileInput) this.selectors.fileInput.click();
            });
            
            this.selectors.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            this.selectors.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            this.selectors.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        }

        if (this.selectors.fileInput) {
            this.selectors.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Analysis controls
        if (this.selectors.analysisMethodSelect) {
            this.selectors.analysisMethodSelect.addEventListener('change', this.handleMethodChange.bind(this));
        }

        document.querySelectorAll('.color-method').forEach(method => {
            method.addEventListener('click', this.handleColorMethodSelect.bind(this));
        });

        // Action buttons
        if (this.selectors.analyzeBtn) {
            this.selectors.analyzeBtn.addEventListener('click', this.runSingleAnalysis.bind(this));
        }
        
        if (this.selectors.batchBtn) {
            this.selectors.batchBtn.addEventListener('click', this.runBatchAnalysis.bind(this));
        }
        
        if (this.selectors.exportBtn) {
            this.selectors.exportBtn.addEventListener('click', this.exportResults.bind(this));
        }

        // Global error handler
        window.addEventListener('error', this.handleGlobalError.bind(this));
        window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));
    }

    handleGlobalError(event) {
        console.error('Global error:', event.error);
        this.showNotification(`Unexpected error: ${event.message}`, 'error');
    }

    handleUnhandledRejection(event) {
        console.error('Unhandled promise rejection:', event.reason);
        this.showNotification('Network or processing error occurred', 'error');
    }

    // Event Handlers
    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.selectors.uploadArea?.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.selectors.uploadArea?.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.selectors.uploadArea?.classList.remove('dragover');
        this.processFiles(Array.from(e.dataTransfer.files));
    }

    handleFileSelect(e) {
        this.processFiles(Array.from(e.target.files));
    }

    handleMethodChange() {
        const method = this.selectors.analysisMethodSelect?.value;
        if (this.selectors.colorMethodsDiv) {
            this.selectors.colorMethodsDiv.style.display = method === 'color' ? 'grid' : 'none';
        }
    }

    handleColorMethodSelect(e) {
        const method = e.currentTarget;
        document.querySelectorAll('.color-method').forEach(m => m.classList.remove('selected'));
        method.classList.add('selected');
        this.state.selectedColorMethod = method.dataset.color;
    }

    // File Processing
    validateFile(file) {
        const errors = [];

        // Check file size
        if (file.size > this.config.maxFileSize) {
            errors.push(`File "${file.name}" is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Maximum size is 16MB.`);
        }

        // Check file type
        const isValidType = this.config.supportedFormats.includes(file.type) ||
            /\.(jpg|jpeg|png|tif|tiff|bmp|jfif)$/i.test(file.name);
        
        if (!isValidType) {
            errors.push(`File "${file.name}" has unsupported format. Supported: JPG, PNG, TIF, BMP`);
        }

        return errors;
    }

    async processFiles(files) {
        if (!files || files.length === 0) {
            this.showNotification('No files selected', 'warning');
            return;
        }

        // Validate files
        const allErrors = [];
        const validFiles = [];

        files.forEach(file => {
            const errors = this.validateFile(file);
            if (errors.length > 0) {
                allErrors.push(...errors);
            } else {
                validFiles.push(file);
            }
        });

        // Show validation errors
        if (allErrors.length > 0) {
            this.showNotification(allErrors.join('\n'), 'error');
            if (validFiles.length === 0) return;
        }

        // Check file count limit
        if (validFiles.length > this.config.maxFiles) {
            this.showNotification(`Too many files selected. Maximum is ${this.config.maxFiles}.`, 'warning');
            validFiles.splice(this.config.maxFiles);
        }

        this.state.currentFiles = validFiles;
        this.updateUploadArea();
        this.updateUI();
        
        if (validFiles.length > 0) {
            this.previewImage(validFiles[0]);
            this.showNotification(`✅ ${validFiles.length} image(s) loaded successfully`, 'success');
        }
    }

    // Analysis Methods
    async runSingleAnalysis() {
        if (!this.state.currentFiles.length) {
            this.showNotification('Please select an image first', 'warning');
            return;
        }
        
        await this.analyzeFiles([this.state.currentFiles[0]], false);
    }

        async runBatchAnalysis() {
        const formData = new FormData();
        this.state.currentFiles.forEach(file => formData.append("images", file));

        const res = await fetch("/batch", { method: "POST", body: formData });
        const json = await res.json();
        // handle results like: json.results, json.qc_report, json.export_info
        }


    async analyzeFiles(files, isBatch = false) {
        if (this.state.isAnalyzing) {
            this.showNotification('Analysis already in progress', 'warning');
            return;
        }

        try {
            this.state.isAnalyzing = true;
            this.toggleLoadingState(true);
            
            const results = [];
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                this.updateProgressText(`Analyzing ${file.name} (${i + 1}/${files.length})`);
                
                const result = await this.analyzeSingleFile(file);
                if (result) {
                    results.push({
                        ...result,
                        timestamp: Date.now(),
                        image_name: file.name
                    });
                }
                
                // Update progress
                const progress = ((i + 1) / files.length) * 100;
                this.updateProgress(progress);
            }

            this.handleAnalysisResults(results, isBatch);
            
        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.state.isAnalyzing = false;
            this.toggleLoadingState(false);
        }
    }

    async analyzeSingleFile(file) {
        try {
            const formData = new FormData();
            const params = this.getAnalysisParams();

            // Add file
            formData.append('image', file);

            // Add parameters
            Object.entries(params).forEach(([key, value]) => {
                formData.append(key, value.toString());
            });

            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Server error' }));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            return result;

        } catch (error) {
            console.error(`Error analyzing ${file.name}:`, error);
            this.showNotification(`Failed to analyze ${file.name}: ${error.message}`, 'error');
            return null;
        }
    }

    handleAnalysisResults(results, isBatch) {
        if (!results || results.length === 0) {
            this.showNotification('No successful analyses completed', 'warning');
            return;
        }

        this.state.analysisResults.push(...results);

        if (isBatch) {
            this.updateHistoryPanel();
            this.showNotification(`✅ Batch analysis complete: ${results.length} images processed`, 'success');
        } else {
            this.updateResultsDisplay(results[0]);
            this.updateHistoryPanel();
            this.showNotification('✅ Analysis complete! 🎉', 'success');
        }

        this.updateUI();
    }

    // UI Updates
    updateResultsDisplay(result) {
        if (!result) {
            this.showNotification('No result data to display', 'error');
            return;
        }

        try {
            // Update statistics
            const stats = result.stats || result.summary || {};
            if (this.selectors.statsGrid) {
                this.selectors.statsGrid.innerHTML = this.generateStatsHTML(stats);
            }

            // Update visualizations
            this.drawCanvas(this.selectors.originalCanvas, result.original_image);
            this.drawCanvas(this.selectors.segmentedCanvas, result.visualization);

            // Update details table
            this.populateDetailsTable(result.cell_data || []);

            // Show visualization section
            const visualizations = document.getElementById('visualizations');
            const noVizMessage = document.getElementById('noVisualizationMessage');
            
            if (visualizations) visualizations.style.display = 'block';
            if (noVizMessage) noVizMessage.style.display = 'none';

        } catch (error) {
            console.error('Error updating results display:', error);
            this.showNotification('Error displaying results', 'error');
        }
    }

    drawCanvas(canvas, imageData) {
        if (!canvas || !imageData) {
            console.warn('Canvas or image data missing');
            return;
        }

        try {
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // Set canvas size to match image
                canvas.width = Math.min(img.width, 800);  // Limit max width
                canvas.height = (canvas.width / img.width) * img.height;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };

            img.onerror = (error) => {
                console.error('Failed to load image:', error);
                this.showCanvasError(canvas, 'Failed to load image');
            };

            img.src = `data:image/png;base64,${imageData}`;

        } catch (error) {
            console.error('Error drawing canvas:', error);
            this.showCanvasError(canvas, 'Error displaying image');
        }
    }

    showCanvasError(canvas, message) {
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        canvas.width = 400;
        canvas.height = 200;
        
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.fillStyle = '#6b7280';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(message, canvas.width / 2, canvas.height / 2);
    }

    generateStatsHTML(stats) {
        const formatNumber = (num, decimals = 1) => {
            return typeof num === 'number' ? num.toFixed(decimals) : '0';
        };

        return `
            <div class="stat-card">
                <div class="stat-value">${stats.total_cells || 0}</div>
                <div class="stat-label">Total Cells</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatNumber(stats.avg_area)}</div>
                <div class="stat-label">Avg Area (px²)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatNumber(stats.total_biomass, 2)}</div>
                <div class="stat-label">Biomass Est.</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatNumber(stats.chlorophyll_ratio)}%</div>
                <div class="stat-label">High Chlorophyll</div>
            </div>
        `;
    }

    populateDetailsTable(cellData) {
        try {
            const tbody = document.getElementById('tableBody');
            const resultsTable = document.getElementById('resultsTable');
            const noDetailsMessage = document.getElementById('noDetailsMessage');

            if (!tbody) return;

            if (!cellData || cellData.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #6b7280;">No cell data available</td></tr>';
                if (resultsTable) resultsTable.style.display = 'table';
                if (noDetailsMessage) noDetailsMessage.style.display = 'none';
                return;
            }

            const formatNumber = (num, decimals = 1) => {
                return typeof num === 'number' ? num.toFixed(decimals) : '0';
            };

            tbody.innerHTML = cellData.slice(0, 50).map(cell => `
                <tr>
                    <td>${cell.cell_id || 'N/A'}</td>
                    <td>${formatNumber(cell.area)}</td>
                    <td>${formatNumber(cell.perimeter)}</td>
                    <td>${formatNumber(cell.chlorophyll, 2)}</td>
                    <td>${cell.classification || 'Unknown'}</td>
                </tr>
            `).join('');

            if (resultsTable) resultsTable.style.display = 'table';
            if (noDetailsMessage) noDetailsMessage.style.display = 'none';

            if (cellData.length > 50) {
                tbody.innerHTML += `
                    <tr style="background-color: #f9fafb;">
                        <td colspan="5" style="text-align: center; font-style: italic; color: #6b7280;">
                            Showing first 50 of ${cellData.length} cells
                        </td>
                    </tr>
                `;
            }

        } catch (error) {
            console.error('Error populating details table:', error);
        }
    }

    // Utility Methods
    getAnalysisParams() {
        const getElementValue = (id, defaultValue, type = 'string') => {
            const element = document.getElementById(id);
            if (!element) return defaultValue;
            
            const value = element.value;
            if (type === 'float') return parseFloat(value) || defaultValue;
            if (type === 'int') return parseInt(value) || defaultValue;
            return value || defaultValue;
        };

        return {
            pixel_ratio: getElementValue('pixelRatio', 1.0, 'float'),
            chlorophyll_threshold: getElementValue('chlorophyllThreshold', 0.6, 'float'),
            min_cell_area: getElementValue('minCellArea', 30, 'int'),
            max_cell_area: getElementValue('maxCellArea', 8000, 'int'),
            analysis_method: this.selectors.analysisMethodSelect?.value || 'auto',
            color_method: this.state.selectedColorMethod
        };
    }

    toggleLoadingState(isLoading) {
        // Update progress container
        if (this.selectors.progressContainer) {
            this.selectors.progressContainer.style.display = isLoading ? 'block' : 'none';
        }

        // Update buttons
        const buttons = [this.selectors.analyzeBtn, this.selectors.batchBtn];
        buttons.forEach(btn => {
            if (btn) {
                btn.disabled = isLoading;
                if (isLoading) {
                    btn.textContent = btn === this.selectors.analyzeBtn ? 'Analyzing...' : 'Processing...';
                } else {
                    btn.textContent = btn === this.selectors.analyzeBtn ? 'Analyze Image' : 'Batch Analysis';
                }
            }
        });

        if (isLoading) {
            this.resetProgress();
        }
    }

    resetProgress() {
        this.updateProgress(0);
        this.updateProgressText('Preparing analysis...');
    }

    updateProgress(percentage) {
        if (this.selectors.progressFill) {
            this.selectors.progressFill.style.width = `${Math.min(100, Math.max(0, percentage))}%`;
        }
    }

    updateProgressText(text) {
        if (this.selectors.progressText) {
            this.selectors.progressText.textContent = text;
        }
    }

    showNotification(message, type = 'info') {
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        const notification = document.getElementById('notification');
        if (!notification) {
            // Fallback to console if notification element doesn't exist
            return;
        }

        // Clear any existing timeout
        if (notification.timeout) {
            clearTimeout(notification.timeout);
        }

        notification.textContent = message;
        notification.className = `notification ${type}`;
        notification.style.display = 'block';
        notification.style.opacity = '1';

        // Auto-hide after delay (longer for errors)
        const delay = type === 'error' ? 8000 : type === 'warning' ? 5000 : 3000;
        
        notification.timeout = setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.style.display = 'none';
            }, 300);
        }, delay);
    }

    updateUI() {
        const hasFiles = this.state.currentFiles.length > 0;
        const hasResults = this.state.analysisResults.length > 0;
        const isAnalyzing = this.state.isAnalyzing;

        // Update button states
        if (this.selectors.analyzeBtn) {
            this.selectors.analyzeBtn.disabled = !hasFiles || isAnalyzing;
        }
        
        if (this.selectors.batchBtn) {
            this.selectors.batchBtn.disabled = !hasFiles || this.state.currentFiles.length < 2 || isAnalyzing;
        }
        
        if (this.selectors.exportBtn) {
            this.selectors.exportBtn.disabled = !hasResults || isAnalyzing;
        }
    }

    updateUploadArea() {
        if (!this.selectors.uploadArea) return;

        const files = this.state.currentFiles;
        
        if (files.length === 0) {
            this.selectors.uploadArea.innerHTML = `
                <div class="upload-icon">📷</div>
                <p><strong>Drop images here</strong></p>
                <p>or click to browse</p>
                <p style="font-size: 0.8rem; color: #a1a1aa; margin-top: 10px;">
                    Supported: JPG, PNG, TIF, BMP (Max 16MB each)
                </p>
            `;
        } else {
            const totalSize = files.reduce((sum, file) => sum + file.size, 0);
            const totalSizeMB = (totalSize / 1024 / 1024).toFixed(1);
            
            this.selectors.uploadArea.innerHTML = `
                <div class="upload-icon">✅</div>
                <p><strong>${files.length} image(s) selected</strong></p>
                <p style="font-size: 0.9rem;">${files[0].name}${files.length > 1 ? ` +${files.length - 1} more` : ''}</p>
                <p style="font-size: 0.8rem; color: #a1a1aa; margin-top: 5px;">
                    Total size: ${totalSizeMB} MB
                </p>
                <p style="font-size: 0.8rem; color: #a1a1aa; margin-top: 10px;">
                    Click to select different images
                </p>
            `;
        }
    }

    previewImage(file) {
        if (!this.selectors.imageDisplay || !file) return;

        try {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                this.selectors.imageDisplay.innerHTML = `
                    <img src="${e.target.result}" alt="Preview" style="max-height: 300px; max-width: 100%; border-radius: 8px;">
                    <p style="margin-top: 10px; color: #6b7280; font-size: 0.9rem;">${file.name}</p>
                    <p style="color: #9ca3af; font-size: 0.8rem;">${(file.size / 1024).toFixed(1)} KB</p>
                `;
            };
            
            reader.onerror = () => {
                this.selectors.imageDisplay.innerHTML = `
                    <div style="padding: 20px; text-align: center; color: #ef4444;">
                        ❌ Failed to load image preview
                    </div>
                `;
            };
            
            reader.readAsDataURL(file);
            
        } catch (error) {
            console.error('Error previewing image:', error);
        }
    }

    setupTabs() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                // Remove active class from all tabs and content
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });

                // Add active class to clicked tab and corresponding content
                tab.classList.add('active');
                const tabContent = document.getElementById(`${tab.dataset.tab}Tab`);
                if (tabContent) {
                    tabContent.classList.add('active');
                }
            });
        });
    }

    updateHistoryPanel() {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        if (this.state.analysisResults.length === 0) {
            historyList.innerHTML = `
                <p style="text-align: center; color: #a1a1aa; padding: 20px;">
                    📝 Your analysis history will appear here
                </p>
            `;
            return;
        }

        try {
            historyList.innerHTML = this.state.analysisResults
                .slice(-10) // Show last 10 results
                .reverse() // Most recent first
                .map((result, index) => {
                    const stats = result.stats || result.summary || {};
                    const date = result.timestamp ? new Date(result.timestamp).toLocaleDateString() : 'Unknown';
                    
                    return `
                        <div class="history-item" data-index="${index}">
                            <div class="history-header">
                                <span class="history-date">${date}</span>
                                <span class="history-name">${result.image_name || 'Unknown Image'}</span>
                            </div>
                            <div class="history-stats">
                                <span>${stats.total_cells || 0} cells</span>
                                <span>${(stats.avg_area || 0).toFixed(1)} px²</span>
                            </div>
                        </div>
                    `;
                }).join('');

            // Add click handlers for history items
            historyList.querySelectorAll('.history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const index = parseInt(item.dataset.index);
                    const result = this.state.analysisResults[this.state.analysisResults.length - 1 - index];
                    this.updateResultsDisplay(result);
                });
            });

        } catch (error) {
            console.error('Error updating history panel:', error);
            historyList.innerHTML = `
                <p style="text-align: center; color: #ef4444; padding: 20px;">
                    ❌ Error loading history
                </p>
            `;
        }
    }

    async exportResults() {
        if (this.state.analysisResults.length === 0) {
            this.showNotification('No results to export', 'warning');
            return;
        }

        try {
            this.showNotification('Preparing export...', 'info');
            
            // For now, create a simple CSV export
            const csvData = this.generateCSVExport();
            this.downloadCSV(csvData, `wolffia_analysis_${Date.now()}.csv`);
            
            this.showNotification('✅ Results exported successfully', 'success');
            
        } catch (error) {
            console.error('Export error:', error);
            this.showNotification(`Export failed: ${error.message}`, 'error');
        }
    }

    generateCSVExport() {
        const headers = ['Image Name', 'Date', 'Total Cells', 'Avg Area', 'Total Biomass', 'Chlorophyll Ratio'];
        const rows = this.state.analysisResults.map(result => {
            const stats = result.stats || result.summary || {};
            return [
                result.image_name || 'Unknown',
                result.timestamp ? new Date(result.timestamp).toISOString() : '',
                stats.total_cells || 0,
                (stats.avg_area || 0).toFixed(2),
                (stats.total_biomass || 0).toFixed(3),
                (stats.chlorophyll_ratio || 0).toFixed(1)
            ];
        });

        return [headers, ...rows].map(row => 
            row.map(cell => `"${cell}"`).join(',')
        ).join('\n');
    }

    downloadCSV(csvContent, filename) {
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        
        if (link.download !== undefined) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌱 Starting Wolffia Analyzer...');
    try {
        window.wolffiaApp = new WolffiaAnalyzerApp();
    } catch (error) {
        console.error('❌ Failed to initialize Wolffia Analyzer:', error);
        
        // Show fallback error message
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #ef4444; color: white; padding: 15px; border-radius: 8px; z-index: 1000; max-width: 300px;';
        errorDiv.innerHTML = `
            <strong>⚠️ Application Error</strong><br>
            Failed to initialize. Please check the console and refresh the page.
        `;
        document.body.appendChild(errorDiv);
    }
});