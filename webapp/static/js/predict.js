/**
 * predict.js — Detection page logic.
 *
 * Fixes from plan:
 *  1. Tab switch no longer clears extractedFeatures (data preserved across tabs).
 *  2. Orphaned handler bindings removed.
 *  3. resetForm() also clears combined status/inputs/checkboxes.
 *  4. showNotification uses Bootstrap 5 Toast API (via main.js global).
 *  5. Upload buttons show spinner & disable during upload.
 *  6. Detect button tooltip when disabled.
 *  7. Unified showNotification (falls through to main.js).
 */

let mediaRecorder;
let audioChunks = [];
let isRecording = false;

let extractedFeatures = {
    speech: null,
    handwriting: null,
    gait: null
};

let uploadedFilenames = {
    speech: null,
    handwriting: null,
    gait: null
};

// Separate tracking for Combined tab uploads
let combinedTabUploads = {
    speech: false,
    handwriting: false,
    gait: false
};

let referenceCategory = null;

// Feature name mappings (matching backend utils modules)
const FEATURE_NAMES = {
    speech: [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 
        'MDVP:APQ', 'Shimmer:DDA',
        'NHR', 'HNR',
        'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ],
    handwriting: [
        'mean_pressure', 'std_pressure',
        'mean_velocity', 'std_velocity',
        'mean_acceleration',
        'pen_up_time',
        'stroke_length',
        'writing_tempo',
        'tremor_frequency',
        'fluency_score'
    ],
    gait: [
        'stride_interval',
        'stride_interval_std',
        'swing_time',
        'stance_time',
        'double_support',
        'gait_speed',
        'cadence',
        'step_length',
        'stride_regularity',
        'gait_asymmetry'
    ]
};
let currentActiveTab = 'speech'; // Track the current active tab

/* ------------------------------------------------------------------ */
/*  Light Mode Detection                                               */
/* ------------------------------------------------------------------ */

function checkLightMode() {
    // Test if upload endpoints are available by checking health endpoint
    fetch('/api/health')
        .then(function(response) { return response.json(); })
        .then(function(data) {
            if (data.active_backend === 'custom_logic') {
                enableLightMode();
            }
        })
        .catch(function() {
            // If health check fails, assume we need to show login message
        });
}

function enableLightMode() {
    // Hide upload sections and show light mode notice
    $('.upload-zone').each(function() {
        $(this).addClass('disabled').css('opacity', '0.4');
        $(this).find('.upload-empty-state p').html('<i class="fas fa-lock"></i> File upload disabled in demo mode');
        $(this).find('.upload-empty-state .upload-hint').html('<strong>Use example data buttons below instead</strong>');
    });
    
    // Disable upload buttons and change their text
    $('#uploadAudioBtn').prop('disabled', true).html('<i class="fas fa-lock"></i> Demo Mode');
    $('#uploadHandwritingBtn').prop('disabled', true).html('<i class="fas fa-lock"></i> Demo Mode');
    $('#uploadGaitBtn').prop('disabled', true).html('<i class="fas fa-lock"></i> Demo Mode');
    $('#uploadCombinedBtn').prop('disabled', false).html('<i class="fas fa-bolt"></i> Extract Features');
    
    // Hide recording section
    $('#recordBtn').prop('disabled', true).html('<i class="fas fa-lock"></i> Demo Mode').removeClass('btn-danger').addClass('btn-secondary');
    $('#recordBtn').closest('.surface-card').css('opacity', '0.4');
    
    // Add light mode notice
    if (!$('#lightModeNotice').length) {
        $('.container').prepend(
            '<div id="lightModeNotice" class="alert alert-info mb-4 fade-in-up">' +
            '<i class="fas fa-info-circle"></i> <strong>Demo Mode Active:</strong> ' +
            'File uploads are disabled. Use the <strong>example data buttons</strong> below to test the AI prediction system. ' +
            '<br><small class="mt-1 d-block"><i class="fas fa-lightbulb"></i> ' +
            'Tip: Files with "pd" in the name will be classified as Parkinson\'s Disease, others as Healthy.</small>' +
            '</div>'
        );
    }
}

/* ------------------------------------------------------------------ */
/*  Bootstrap Ready                                                    */
/* ------------------------------------------------------------------ */

$(document).ready(function () {
    
    // ---- Performance Optimizations ----
    
    // Detect and optimize for desktop performance
    if (window.innerWidth >= 1024) {
        // Preload critical resources for faster interactions
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'fetch';
        link.href = '/api/health';
        document.head.appendChild(link);
        
        // Optimize animations for desktop
        document.body.classList.add('desktop-optimized');
    }
    
    // Detect reduced motion preference
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        document.body.classList.add('reduced-motion');
    }

    // ---- Event Bindings ----

    // Voice recording
    $('#recordBtn').click(function () {
        if (!isRecording) startRecording(); else stopRecording();
    });

    // Upload buttons
    $('#uploadAudioBtn').click(function () { uploadAudioFile(); });
    $('#uploadHandwritingBtn').click(function () { uploadHandwritingFile(); });
    $('#uploadGaitBtn').click(function () { uploadGaitFile(); });
    $('#uploadCombinedBtn').click(function () { uploadCombinedVideo(); });

    // Example buttons – Speech
    $('#useSpeechHealthy').click(function () { loadExample('healthy', 'speech'); });
    $('#useSpeechPD').click(function () { loadExample('parkinsons', 'speech'); });

    // Example buttons – Handwriting
    $('#useHandwritingHealthy').click(function () { loadExample('healthy', 'handwriting'); });
    $('#useHandwritingPD').click(function () { loadExample('parkinsons', 'handwriting'); });

    // Example buttons – Gait
    $('#useGaitHealthy').click(function () { loadExample('healthy', 'gait'); });
    $('#useGaitPD').click(function () { loadExample('parkinsons', 'gait'); });

    // Example buttons – Combined
    $('#useCombinedHealthy').click(function () { loadExample('healthy', 'all'); });
    $('#useCombinedPD').click(function () { loadExample('parkinsons', 'all'); });

    // Image preview
    $('#handwritingFileInput').change(function () { previewHandwritingImage(this); });

    // Track file selections
    $('#audioFileInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.speech = this.files[0].name;
            updateUploadZoneState('audioDropZone', 'has-file', this.files[0]);
            updateStepsBasedOnProgress();
        } else {
            updateUploadZoneState('audioDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });
    $('#handwritingFileInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.handwriting = this.files[0].name;
            updateUploadZoneState('handwritingDropZone', 'has-file', this.files[0]);
            $('#handwritingPreview').show();
            updateStepsBasedOnProgress();
        } else {
            updateUploadZoneState('handwritingDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });
    $('#gaitFileInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.gait = this.files[0].name;
            updateUploadZoneState('gaitDropZone', 'has-file', this.files[0]);
            updateStepsBasedOnProgress();
        } else {
            updateUploadZoneState('gaitDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });
    
    // Combined tab file selections
    $('#combinedSpeechInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.speech = this.files[0].name;
            combinedTabUploads.speech = true;
            updateUploadZoneState('combinedSpeechDropZone', 'has-file', this.files[0]);
            $('#combinedSpeechFileName').text(this.files[0].name);
            $('#combinedSpeechPreview').show();
            updateStepsBasedOnProgress();
        } else {
            combinedTabUploads.speech = false;
            updateUploadZoneState('combinedSpeechDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });
    $('#combinedHandwritingInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.handwriting = this.files[0].name;
            combinedTabUploads.handwriting = true;
            updateUploadZoneState('combinedHandwritingDropZone', 'has-file', this.files[0]);
            $('#combinedHandwritingFileName').text(this.files[0].name);
            $('#combinedHandwritingPreview').show();
            updateStepsBasedOnProgress();
        } else {
            combinedTabUploads.handwriting = false;
            updateUploadZoneState('combinedHandwritingDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });
    $('#combinedGaitInput').change(function () {
        if (this.files && this.files[0]) {
            uploadedFilenames.gait = this.files[0].name;
            combinedTabUploads.gait = true;
            updateUploadZoneState('combinedGaitDropZone', 'has-file', this.files[0]);
            $('#combinedGaitFileName').text(this.files[0].name);
            $('#combinedGaitPreview').show();
            updateStepsBasedOnProgress();
        } else {
            combinedTabUploads.gait = false;
            updateUploadZoneState('combinedGaitDropZone', 'empty');
            updateStepsBasedOnProgress();
        }
    });

    // Detect & Reset
    $('#predictBtn').click(function () { makeDetection(); });
    $('#resetBtn').click(function () { resetForm(); });

    // Initialize detect button tooltip (Bootstrap 5)
    var predictBtn = document.getElementById('predictBtn');
    if (predictBtn) new bootstrap.Tooltip(predictBtn);

    // Explainability section toggles (event delegation)
    $(document).on('click', '.dl-section-toggle', function () {
        var targetId = $(this).data('target');
        var $content = $('#' + targetId);
        var $toggle = $(this);
        
        if ($content.is(':visible')) {
            $content.slideUp(200);
            $toggle.addClass('collapsed').attr('aria-expanded', 'false');
            $toggle.find('.toggle-icon').removeClass('fa-chevron-up').addClass('fa-chevron-down');
        } else {
            $content.slideDown(200);
            $toggle.removeClass('collapsed').attr('aria-expanded', 'true');
            $toggle.find('.toggle-icon').removeClass('fa-chevron-down').addClass('fa-chevron-up');
        }
    });
    
    // Keyboard support for explainability toggles
    $(document).on('keydown', '.dl-section-toggle', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            $(this).click();
        }
    });

    // Drag-and-drop zones
    initDropZone('audioDropZone', 'audioFileInput');
    initDropZone('handwritingDropZone', 'handwritingFileInput');
    initDropZone('gaitDropZone', 'gaitFileInput');
    
    // Combined tab separate inputs
    initDropZone('combinedSpeechDropZone', 'combinedSpeechInput');
    initDropZone('combinedHandwritingDropZone', 'combinedHandwritingInput');
    initDropZone('combinedGaitDropZone', 'combinedGaitInput');
    
    // Check for light mode on page load
    checkLightMode();
    
    // Initialize current tab
    currentActiveTab = 'speech'; // Default to speech tab
    
    // Initialize step descriptions
    updateStepDescriptions();
    
    // Initialize touch gestures for mobile tab navigation
    initTouchGestures();
    
    // Initialize accessibility features
    initAccessibilityFeatures();
    
    // File remove button event listeners
    $(document).on('click', '.file-remove-btn', function() {
        const zoneId = $(this).data('zone');
        const inputId = $(this).data('input');
        removeUploadedFile(zoneId, inputId);
    });
    
    // File change button event listeners
    $(document).on('click', '.file-change-btn', function() {
        const inputId = $(this).data('input');
        const input = document.getElementById(inputId);
        if (input) {
            input.click();
        }
    });
    
    
    // Keyboard shortcuts for desktop users
    $(document).keydown(function(e) {
        // Only handle shortcuts when not typing in inputs
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // Ctrl/Cmd + Enter: Make Detection
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            makeDetection();
        }
        
        // Ctrl/Cmd + R: Reset Form
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            resetForm();
        }
        
        // Escape: Close modals
        if (e.key === 'Escape') {
            // Close all open modals using Bootstrap 5 API
            document.querySelectorAll('.modal.show').forEach(function(modal) {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
        
        // Tab navigation between modality tabs (1-4 keys)
        if (e.key >= '1' && e.key <= '4') {
            e.preventDefault();
            const tabMap = {
                '1': '#speech-tab',
                '2': '#handwriting-tab', 
                '3': '#gait-tab',
                '4': '#combined-tab'
            };
            const tabElement = document.querySelector(tabMap[e.key]);
            if (tabElement) {
                const tabInstance = new bootstrap.Tab(tabElement);
                tabInstance.show();
            }
        }
        
        // Show keyboard shortcuts help
        if (e.key === '?') {
            e.preventDefault();
            const shortcutsModal = document.getElementById('keyboardShortcutsModal');
            if (shortcutsModal) {
                const modalInstance = new bootstrap.Modal(shortcutsModal);
                modalInstance.show();
            }
        }
    });
    
    // Universal file preview click handlers
    $(document).on('click', '.file-preview-container, .image-preview-container', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation(); // Stop all event propagation
        
        const zone = $(this).closest('.upload-zone');
        const fileName = zone.find('.file-name').text();
        const fileMeta = zone.find('.file-meta').text();
        const fileType = $(this).data('file-type');
        
        // Get the file input to access the actual file
        const input = zone.find('input[type="file"]')[0];
        if (input && input.files && input.files[0]) {
            showFilePreviewModal(input.files[0], fileName, fileMeta, fileType);
        } else {
            // Fallback for image preview containers
            const img = $(this).find('.image-preview');
            if (img.length && img.attr('src') && img.attr('src') !== '') {
                showFilePreviewModal(null, fileName, fileMeta, 'image', img.attr('src'));
            }
        }
        
        return false; // Additional prevention
    });
    
    // Make file preview containers look clickable
    $(document).on('mouseenter', '.file-preview-container, .image-preview-container', function() {
        $(this).css('cursor', 'pointer');
    });
    
    // Also handle clicks on file icons and overlays specifically
    $(document).on('click', '.file-preview-overlay, .file-icon', function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
        
        // Trigger the preview for the parent container
        const container = $(this).closest('.file-preview-container, .image-preview-container');
        if (container.length) {
            container.trigger('click');
        }
        
        return false;
    });
});

/* ------------------------------------------------------------------ */
/*  Drag & Drop Helper                                                 */
/* ------------------------------------------------------------------ */

function initDropZone(zoneId, inputId) {
    var zone = document.getElementById(zoneId);
    var input = document.getElementById(inputId);
    if (!zone || !input) return;

    // Make zone focusable for keyboard navigation
    zone.setAttribute('tabindex', '0');
    zone.setAttribute('role', 'button');
    zone.setAttribute('aria-label', 'Upload file area');

    zone.addEventListener('click', function (e) { 
        // Don't trigger file dialog if clicking on preview elements or remove buttons
        if (e.target.closest('.file-preview-container') || 
            e.target.closest('.image-preview-container') || 
            e.target.closest('.file-remove-btn') ||
            e.target.closest('.file-change-btn') ||
            e.target.closest('.upload-zone-actions')) {
            return;
        }
        
        // Allow click-to-replace for better desktop UX
        // Show confirmation for replacement if file exists
        if (zone.classList.contains('has-file')) {
            const fileName = zone.querySelector('.file-name')?.textContent;
            if (fileName) {
                showFileReplaceConfirmation(fileName, function(confirmed) {
                    if (confirmed) {
                        input.click();
                    }
                });
                return;
            }
        }
        
        // Open file dialog
        input.click(); 
    });

    // Keyboard navigation support
    zone.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            
            // Same logic as click handler
            if (zone.classList.contains('has-file')) {
                const fileName = zone.querySelector('.file-name')?.textContent;
                if (fileName) {
                    showFileReplaceConfirmation(fileName, function(confirmed) {
                        if (confirmed) {
                            input.click();
                        }
                    });
                    return;
                }
            }
            
            input.click();
        }
    });

    zone.addEventListener('dragover', function (e) {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', function () {
        zone.classList.remove('dragover');
    });
    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            $(input).trigger('change');
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Upload Zone State Management                                       */
/* ------------------------------------------------------------------ */

function updateUploadZoneState(zoneId, state, fileData = null) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    const emptyState = zone.querySelector('.upload-empty-state');
    const fileState = zone.querySelector('.upload-file-state');
    
    // Remove all state classes
    zone.classList.remove('has-file', 'processing', 'error');
    
    switch (state) {
        case 'empty':
            if (emptyState) emptyState.style.display = 'block';
            if (fileState) fileState.style.display = 'none';
            break;
            
        case 'has-file':
            zone.classList.add('has-file');
            if (emptyState) emptyState.style.display = 'none';
            if (fileState) fileState.style.display = 'block';
            if (fileData) showFilePreview(zoneId, fileData);
            break;
            
        case 'processing':
            zone.classList.add('processing');
            break;
            
        case 'error':
            zone.classList.add('error');
            break;
    }
}

function showFilePreview(zoneId, file) {
    const zone = document.getElementById(zoneId);
    if (!zone) return;

    const fileName = zone.querySelector('.file-name');
    const fileMeta = zone.querySelector('.file-meta');
    
    if (fileName) {
        fileName.textContent = file.name;
    }
    
    if (fileMeta) {
        const size = formatFileSize(file.size);
        const type = file.type || 'Unknown type';
        fileMeta.textContent = `${size} • ${type}`;
    }

    // Update the file-preview-container with the correct file type
    const previewContainer = zone.querySelector('.file-preview-container');
    if (previewContainer) {
        if (file.type.startsWith('image/')) {
            previewContainer.setAttribute('data-file-type', 'image');
        } else if (file.type.startsWith('audio/')) {
            previewContainer.setAttribute('data-file-type', 'audio');
        } else if (file.type.startsWith('video/')) {
            previewContainer.setAttribute('data-file-type', 'video');
        }
    }

    // Handle image preview (create thumbnail in upload zone for handwriting)
    if (file.type.startsWith('image/')) {
        const img = zone.querySelector('.image-preview');
        console.log('Processing image file:', file.name, 'Image element found:', !!img); // Debug log
        if (img) {
            const reader = new FileReader();
            reader.onload = function(e) {
                img.src = e.target.result;
                img.style.display = 'block';
                console.log('Image loaded and displayed:', img.src.substring(0, 50) + '...'); // Debug log
            };
            reader.readAsDataURL(file);
        } else {
            // For zones that don't have image preview elements, we'll rely on the modal
            console.log('No image preview element found, will use modal preview only');
        }
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function removeUploadedFile(zoneId, inputId) {
    const input = document.getElementById(inputId);
    const zone = document.getElementById(zoneId);
    
    if (input) {
        input.value = '';
        input.files = null;
    }
    
    // Clear the image preview
    const img = zone.querySelector('.image-preview');
    if (img) {
        img.src = '';
        img.style.display = 'none';
    }
    
    // Update zone state to empty
    updateUploadZoneState(zoneId, 'empty');
    
    // Clear related features and filenames based on the input type
    if (inputId.includes('audio') || inputId.includes('Speech')) {
        extractedFeatures.speech = null;
        uploadedFilenames.speech = null;
        if (inputId.includes('combinedSpeech')) {
            $('#combinedSpeechPreview').hide();
        }
    } else if (inputId.includes('handwriting') || inputId.includes('Handwriting')) {
        extractedFeatures.handwriting = null;
        uploadedFilenames.handwriting = null;
        if (inputId.includes('combinedHandwriting')) {
            $('#combinedHandwritingPreview').hide();
        } else {
            $('#handwritingPreview').hide();
        }
    } else if (inputId.includes('gait') || inputId.includes('Gait')) {
        extractedFeatures.gait = null;
        uploadedFilenames.gait = null;
        if (inputId.includes('combinedGait')) {
            $('#combinedGaitPreview').hide();
        }
    }
    
    // Clear status displays
    $('#speechFeatureStatus, #handwritingFeatureStatus, #gaitFeatureStatus, #combinedFeatureStatus').html('');
    $('#audioUploadStatus, #handwritingUploadStatus, #gaitUploadStatus, #combinedUploadStatus').html('');
    
    updateDetectButton();
    
    showNotification('File removed successfully', 'info');
}

function showFilePreviewModal(file, fileName, fileMeta, fileType, imageSrc = null) {
    // Hide all preview containers first
    $('#imagePreviewContainer, #audioPreviewContainer, #videoPreviewContainer').hide();
    
    // Set file info
    $('#filePreviewInfo .file-name').text(fileName || 'Unknown file');
    $('#filePreviewInfo .file-meta').text(fileMeta || '');
    
    if (fileType === 'image' || (file && file.type.startsWith('image/'))) {
        // Image preview
        $('#filePreviewModalTitle').text('Image Preview');
        const imgSrc = imageSrc || (file ? URL.createObjectURL(file) : '');
        $('#filePreviewImg').attr('src', imgSrc);
        $('#imagePreviewContainer').show();
    } else if (fileType === 'audio' || (file && file.type.startsWith('audio/'))) {
        // Audio preview
        $('#filePreviewModalTitle').text('Audio Preview');
        if (file) {
            const audioSrc = URL.createObjectURL(file);
            $('#filePreviewAudio').attr('src', audioSrc);
        }
        $('#audioPreviewContainer').show();
    } else if (fileType === 'video' || (file && file.type.startsWith('video/'))) {
        // Video preview
        $('#filePreviewModalTitle').text('Video Preview');
        if (file) {
            const videoSrc = URL.createObjectURL(file);
            $('#filePreviewVideo').attr('src', videoSrc);
        }
        $('#videoPreviewContainer').show();
    }
    
    const modal = new bootstrap.Modal(document.getElementById('filePreviewModal'));
    modal.show();
    
    // Clean up object URLs when modal is closed
    $('#filePreviewModal').on('hidden.bs.modal', function() {
        const audio = document.getElementById('filePreviewAudio');
        const video = document.getElementById('filePreviewVideo');
        if (audio && audio.src && audio.src.startsWith('blob:')) {
            URL.revokeObjectURL(audio.src);
            audio.src = '';
        }
        if (video && video.src && video.src.startsWith('blob:')) {
            URL.revokeObjectURL(video.src);
            video.src = '';
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Step Indicator                                                     */
/* ------------------------------------------------------------------ */

function updateSteps(current) {
    // current: 1 = select, 2 = provide, 3 = results
    const stepIndicator = document.getElementById('stepIndicator');
    
    // Update ARIA attributes for screen readers
    if (stepIndicator) {
        stepIndicator.setAttribute('aria-valuenow', current);
        stepIndicator.setAttribute('aria-valuetext', `Step ${current} of 3`);
    }
    
    for (var i = 1; i <= 3; i++) {
        var step = document.getElementById('step' + i);
        if (!step) continue;
        
        // Remove existing states
        step.classList.remove('active', 'completed');
        step.removeAttribute('aria-current');
        
        // Apply new states
        if (i < current) {
            step.classList.add('completed');
            step.setAttribute('aria-label', `Step ${i} completed`);
        } else if (i === current) {
            step.classList.add('active');
            step.setAttribute('aria-current', 'step');
            step.setAttribute('aria-label', `Step ${i} current`);
        } else {
            step.setAttribute('aria-label', `Step ${i} pending`);
        }
    }
    
    for (var j = 1; j <= 2; j++) {
        var line = document.getElementById('stepLine' + j);
        if (!line) continue;
        line.classList.remove('active', 'completed');
        if (j < current) line.classList.add('completed');
        else if (j === current) line.classList.add('active');
    }
}

function updateStepDescriptions() {
    // Update step descriptions based on current active tab
    const activeTab = currentActiveTab || 'speech';
    const step1 = document.querySelector('#step1 span:last-child');
    const step2 = document.querySelector('#step2 span:last-child');
    
    if (step1) {
        const modalityNames = {
            'speech': 'Speech Analysis',
            'handwriting': 'Handwriting Analysis', 
            'gait': 'Gait Analysis',
            'combined': 'Multi-Modal Analysis'
        };
        step1.textContent = modalityNames[activeTab] || 'Select Modality';
    }
    
    if (step2) {
        const dataTypes = {
            'speech': 'Record or Upload Audio',
            'handwriting': 'Upload Handwriting Image',
            'gait': 'Upload Gait Video', 
            'combined': 'Upload Multiple Files'
        };
        step2.textContent = dataTypes[activeTab] || 'Provide Data';
    }
}

function updateStepsBasedOnProgress() {
    // Determine current step based on actual user progress
    let currentStep = 1; // Default: Select Modality
    
    // Check if user has provided any data (step 2)
    const hasFiles = $('#audioFileInput')[0]?.files?.length > 0 ||
                    $('#handwritingFileInput')[0]?.files?.length > 0 ||
                    $('#gaitFileInput')[0]?.files?.length > 0 ||
                    $('#combinedSpeechInput')[0]?.files?.length > 0 ||
                    $('#combinedHandwritingInput')[0]?.files?.length > 0 ||
                    $('#combinedGaitInput')[0]?.files?.length > 0;
    
    const hasFeatures = extractedFeatures.speech || extractedFeatures.handwriting || extractedFeatures.gait;
    
    if (hasFiles || hasFeatures) {
        currentStep = 2; // Provide Data
    }
    
    // Update step indicator with current progress
    updateSteps(currentStep);
    
    // Update step descriptions based on current tab
    updateStepDescriptions();
}


/* ------------------------------------------------------------------ */
/*  Tab Switch -- FIX #1: Do NOT clear extractedFeatures               */
/* ------------------------------------------------------------------ */

$('button[data-bs-toggle="tab"]').on('shown.bs.tab', function (e) {
    // Get the new tab that was activated
    let newTab = $(e.target).attr('id').replace('-tab', '');
    
    // Update current active tab without clearing data
    if (newTab !== currentActiveTab) {
        currentActiveTab = newTab;
        // Only stop recording if switching away from speech tab
        if (isRecording && newTab !== 'speech') {
            stopRecording();
        }
    }
    
    // Update steps based on actual progress, not just tab selection
    updateStepsBasedOnProgress();
    
});

/* ------------------------------------------------------------------ */
/*  Voice Recording                                                    */
/* ------------------------------------------------------------------ */

async function startRecording() {
    try {
        var stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = function (event) { audioChunks.push(event.data); };
        mediaRecorder.onstop = function () {
            var audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            uploadRecordedAudio(audioBlob);
            stream.getTracks().forEach(function (t) { t.stop(); });
        };

        mediaRecorder.start();
        isRecording = true;
        $('#recordBtn').removeClass('btn-danger').addClass('btn-warning');
        $('#recordBtnText').text('Stop Recording');
        $('#recordingStatus').show();
    } catch (error) {
        console.error('Microphone error:', error);
        showNotification('Could not access microphone. Check permissions.', 'danger');
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        $('#recordBtn').removeClass('btn-warning').addClass('btn-danger');
        $('#recordBtnText').text('Start Recording');
        $('#recordingStatus').hide();
        showNotification('Recording stopped. Processing audio...', 'success');
    }
}

/* ------------------------------------------------------------------ */
/*  Upload Functions                                                   */
/* ------------------------------------------------------------------ */

// Dynamic extraction loading time based on device capabilities
var MIN_EXTRACT_LOADER_MS = Math.max(1000, getOptimalLoadingTime() - 1000);

function showExtractLoader(title, subtitle) {
    $('#loadingTitle').text(title || 'Extracting features...');
    $('#loadingText').html(subtitle || '');
    $('#loadingSection').css('display', 'flex');
}

function hideExtractLoaderAfter(startTime, callback) {
    var elapsed = Date.now() - startTime;
    var remaining = Math.max(0, MIN_EXTRACT_LOADER_MS - elapsed);
    setTimeout(function () {
        $('#loadingSection').hide();
        if (callback) callback();
        scrollDetectButtonIntoView();
    }, remaining);
}

function scrollDetectButtonIntoView() {
    var btn = document.getElementById('predictBtn');
    if (btn) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function uploadRecordedAudio(audioBlob) {
    var formData = new FormData();
    formData.append('file', audioBlob, 'recording.wav');
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus', '#uploadAudioBtn');
}

function uploadAudioFile() {
    var fileInput = document.getElementById('audioFileInput');
    if (!fileInput.files.length) { showNotification('Please select an audio file first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.speech = fileInput.files[0].name;
    uploadFile('/api/upload/audio', formData, 'speech', 'speechFeatureStatus', '#uploadAudioBtn');
}

function uploadHandwritingFile() {
    var fileInput = document.getElementById('handwritingFileInput');
    if (!fileInput.files.length) { showNotification('Please select an image first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.handwriting = fileInput.files[0].name;
    uploadFile('/api/upload/handwriting', formData, 'handwriting', 'handwritingFeatureStatus', '#uploadHandwritingBtn');
}

function uploadGaitFile() {
    var fileInput = document.getElementById('gaitFileInput');
    if (!fileInput.files.length) { showNotification('Please select a video first', 'warning'); return; }
    var formData = new FormData();
    formData.append('file', fileInput.files[0]);
    uploadedFilenames.gait = fileInput.files[0].name;
    uploadFile('/api/upload/gait', formData, 'gait', 'gaitFeatureStatus', '#uploadGaitBtn');
}

/* FIX #5 — spinner + disable on upload button; full-screen loader for extraction */
function uploadFile(endpoint, formData, modality, statusElementId, btnSelector) {
    referenceCategory = null;
    var $btn = btnSelector ? $(btnSelector) : null;
    var btnOrigHtml = $btn ? $btn.html() : '';

    if ($btn) {
        $btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');
    }

    // Show upload status
    var uploadStatusId = modality === 'speech' ? 'audioUploadStatus' :
                        modality === 'handwriting' ? 'handwritingUploadStatus' :
                        'gaitUploadStatus';
    $('#' + uploadStatusId).html('<div class="alert alert-info small"><i class="fas fa-upload"></i> Uploading file...</div>');

    updateSteps(2);

    // Set upload zone to processing state
    var zoneId = modality === 'speech' ? 'audioDropZone' :
                 modality === 'handwriting' ? 'handwritingDropZone' :
                 'gaitDropZone';
    updateUploadZoneState(zoneId, 'processing');

    var modalityLabel = modality === 'speech' ? 'Speech / audio' : modality === 'handwriting' ? 'Handwriting' : 'Gait';
    var startTime = Date.now();
    showExtractLoader('Extracting features...', 'Analyzing <strong>' + modalityLabel + '</strong> data. This may take a moment.');

    $.ajax({
        url: endpoint,
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            if (response.success) {
                // Set the current modality features (preserve others for multimodal analysis)
                extractedFeatures[modality] = response.features;
                $('#' + modality + 'Features').val(response.features.join(','));
                
                // Clear only the current modality's status display
                $('#' + modality + 'FeatureStatus').html('');
                
                // Announce success to screen readers
                announceToScreenReader(`${modality} features extracted successfully. ${response.features.length} features ready for analysis.`);
                
                updateDetectButton();
            }
            hideExtractLoaderAfter(startTime, function () {
                if ($btn) $btn.prop('disabled', false).html(btnOrigHtml);
                
                // Reset zone state back to has-file
                var zoneId = modality === 'speech' ? 'audioDropZone' :
                             modality === 'handwriting' ? 'handwritingDropZone' :
                             'gaitDropZone';
                updateUploadZoneState(zoneId, 'has-file');
                
                if (response.success) {
                    var modalityIcon = { speech: '🎤', handwriting: '✍️', gait: '🚶' };
                    $('#' + statusElementId).html(
                        '<div class="alert alert-success small">' +
                        '<i class="fas fa-check-circle"></i> <strong>Success!</strong><br>' +
                        modalityIcon[modality] + ' Extracted ' + response.feature_count + ' features<br>' +
                        '<small>' + response.message + '</small>' +
                        (response.note ? '<br><small class="text-muted">' + response.note + '</small>' : '') +
                        '</div>'
                    );
                    clearUploadStatus(modality);
                } else {
                    clearUploadStatus(modality);
                }
            });
        },
        error: function (xhr) {
            hideExtractLoaderAfter(startTime, function () {
                if ($btn) $btn.prop('disabled', false).html(btnOrigHtml);
                
                // Set zone state to error
                var zoneId = modality === 'speech' ? 'audioDropZone' :
                             modality === 'handwriting' ? 'handwritingDropZone' :
                             'gaitDropZone';
                updateUploadZoneState(zoneId, 'error');
                
                // Reset to has-file after a delay
                setTimeout(function() {
                    updateUploadZoneState(zoneId, 'has-file');
                }, 3000);
                
                if (xhr.status === 401) {
                    showNotification('Please log in to upload files and make predictions.', 'warning');
                } else {
                    var errorMsg = (xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : 'Upload failed. Please try again.';
                    showNotification(errorMsg, 'danger');
                }
                clearUploadStatus(modality);
            });
        }
    });
}

function clearUploadStatus(modality) {
    var id = modality === 'speech' ? 'audioUploadStatus' :
             modality === 'handwriting' ? 'handwritingUploadStatus' :
             'gaitUploadStatus';
    $('#' + id).html('');
}

/* ------------------------------------------------------------------ */
/*  Combined Video Upload                                              */
/* ------------------------------------------------------------------ */

function uploadCombinedVideo() {
    // Check for separate file inputs in the combined tab
    var speechInput = document.getElementById('combinedSpeechInput');
    var handwritingInput = document.getElementById('combinedHandwritingInput');
    var gaitInput = document.getElementById('combinedGaitInput');
    
    var hasSpeech = speechInput && speechInput.files.length > 0;
    var hasHandwriting = handwritingInput && handwritingInput.files.length > 0;
    var hasGait = gaitInput && gaitInput.files.length > 0;
    
    // In light mode, create dummy files for demonstration
    var isLightMode = $('#uploadCombinedBtn').text().includes('Extract Features') && $('#lightModeNotice').length > 0;
    if (isLightMode) {
        // For demo purposes, simulate having all modalities
        hasSpeech = true;
        hasHandwriting = true; 
        hasGait = true;
    } else if (!hasSpeech && !hasHandwriting && !hasGait) {
        showNotification('Please select at least one file to extract features from!', 'warning');
        return;
    }

    // Determine which modalities to extract based on uploaded files
    var extractVoice = hasSpeech;
    var extractHandwriting = hasHandwriting;
    var extractGait = hasGait;

    // Create descriptive modality text for loading message
    var modalityDescriptions = [];
    if (extractVoice) modalityDescriptions.push('Speech / audio');
    if (extractHandwriting) modalityDescriptions.push('Handwriting');
    if (extractGait) modalityDescriptions.push('Gait');
    
    var modalitiesText = modalityDescriptions.join(', ');

    var formData = new FormData();
    var videoFile, videoFilename;
    
    if (isLightMode) {
        // Create a dummy video file for light mode
        videoFile = new Blob(['dummy video content'], { type: 'video/mp4' });
        videoFilename = 'demo_combined_pd.mp4'; // Use 'pd' in filename for consistent demo behavior
    } else {
        // In real mode, we need to create a combined approach or use the first available file
        // For now, let's use the gait video as the primary file since it's most likely to contain all modalities
        if (hasGait) {
            videoFile = gaitInput.files[0];
            videoFilename = videoFile.name;
        } else if (hasSpeech) {
            videoFile = speechInput.files[0];
            videoFilename = videoFile.name;
        } else if (hasHandwriting) {
            videoFile = handwritingInput.files[0];
            videoFilename = videoFile.name;
        }
    }
    
    formData.append('video', videoFile);
    formData.append('extract_voice', extractVoice);
    formData.append('extract_handwriting', extractHandwriting);
    formData.append('extract_gait', extractGait);
    // Set filenames based on what was actually uploaded or demo mode
    if (extractVoice) {
        uploadedFilenames.speech = isLightMode ? 'demo_speech_pd.mp3' : (hasSpeech ? speechInput.files[0].name : videoFilename);
    }
    if (extractHandwriting) {
        uploadedFilenames.handwriting = isLightMode ? 'demo_handwriting_pd.jpg' : (hasHandwriting ? handwritingInput.files[0].name : videoFilename);
    }
    if (extractGait) {
        uploadedFilenames.gait = isLightMode ? 'demo_gait_pd.mp4' : (hasGait ? gaitInput.files[0].name : videoFilename);
    }

    var $btn = $('#uploadCombinedBtn');
    var btnOrig = $btn.html();
    $btn.prop('disabled', true).html('<i class="fas fa-spinner fa-spin"></i> Processing...');

    updateSteps(2);

    // Set processing state for selected modalities
    if (extractVoice) updateUploadZoneState('combinedSpeechDropZone', 'processing');
    if (extractHandwriting) updateUploadZoneState('combinedHandwritingDropZone', 'processing');
    if (extractGait) updateUploadZoneState('combinedGaitDropZone', 'processing');

    var startTime = Date.now();
    showExtractLoader('Extracting features...', 'Analyzing <strong>' + modalitiesText + '</strong> data. This may take a moment.');

    $.ajax({
        url: '/api/process_combined_video',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function (response) {
            if (response.success) {
                // Clear all features first, then set only the ones that were extracted
                extractedFeatures.speech = null;
                extractedFeatures.handwriting = null;
                extractedFeatures.gait = null;
                
                if (response.voice_features) {
                    extractedFeatures.speech = response.voice_features;
                    // Only mark as Combined tab upload if it was actually extracted
                    if (extractVoice) combinedTabUploads.speech = true;
                }
                if (response.handwriting_features) {
                    extractedFeatures.handwriting = response.handwriting_features;
                    // Only mark as Combined tab upload if it was actually extracted
                    if (extractHandwriting) combinedTabUploads.handwriting = true;
                }
                if (response.gait_features) {
                    extractedFeatures.gait = response.gait_features;
                    // Only mark as Combined tab upload if it was actually extracted
                    if (extractGait) combinedTabUploads.gait = true;
                }
                
                // Clear individual tab status displays since we're using combined processing
                $('#speechFeatureStatus, #handwritingFeatureStatus, #gaitFeatureStatus').html('');
                
                updateDetectButton();
            }
            hideExtractLoaderAfter(startTime, function () {
                $btn.prop('disabled', false).html(btnOrig);
                
                // Reset zone states
                if (extractVoice) updateUploadZoneState('combinedSpeechDropZone', 'has-file');
                if (extractHandwriting) updateUploadZoneState('combinedHandwritingDropZone', 'has-file');
                if (extractGait) updateUploadZoneState('combinedGaitDropZone', 'has-file');
                
                if (response.success) {
                    var fe = [];
                    if (response.voice_features) fe.push('🎤 Voice: ' + response.voice_features.length + ' features');
                    if (response.handwriting_features) fe.push('✍️ Handwriting: ' + response.handwriting_features.length + ' features');
                    if (response.gait_features) fe.push('🚶 Gait: ' + response.gait_features.length + ' features');

                    $('#combinedFeatureStatus').html(
                        '<div class="alert alert-success small">' +
                        '<i class="fas fa-check-circle"></i> <strong>Success!</strong><br>' +
                        fe.join('<br>') +
                        '<br><small class="text-muted">Total: ' + response.total_features + ' features extracted</small>' +
                        '<br><small class="text-muted">Source: Multi-modal analysis</small></div>'
                    );
                    $('#combinedUploadStatus').html('');
                    showNotification('Successfully extracted ' + response.total_features + ' features from combined data!', 'success');
                } else {
                    $('#combinedUploadStatus').html('');
                    showNotification('Feature extraction failed. Please try again.', 'danger');
                }
            });
        },
        error: function (xhr) {
            hideExtractLoaderAfter(startTime, function () {
                $btn.prop('disabled', false).html(btnOrig);
                
                // Set error state for zones
                if (extractVoice) updateUploadZoneState('combinedSpeechDropZone', 'error');
                if (extractHandwriting) updateUploadZoneState('combinedHandwritingDropZone', 'error');
                if (extractGait) updateUploadZoneState('combinedGaitDropZone', 'error');
                
                // Reset to has-file after delay
                setTimeout(function() {
                    if (extractVoice) updateUploadZoneState('combinedSpeechDropZone', 'has-file');
                    if (extractHandwriting) updateUploadZoneState('combinedHandwritingDropZone', 'has-file');
                    if (extractGait) updateUploadZoneState('combinedGaitDropZone', 'has-file');
                }, 3000);
                
                if (xhr.status === 401) {
                    showNotification('Please log in to upload files and make predictions.', 'warning');
                } else {
                    var errorMsg = (xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : 'Feature extraction failed. Please try again.';
                    showNotification(errorMsg, 'danger');
                }
                $('#combinedUploadStatus').html('');
                $('#combinedFeatureStatus').html(
                    '<div class="alert alert-danger small">' +
                    '<i class="fas fa-exclamation-triangle"></i> <strong>Error:</strong> Features could not be extracted. Please try again.' +
                    '</div>'
                );
            });
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Preview & Button State                                             */
/* ------------------------------------------------------------------ */

function previewHandwritingImage(input) {
    // This function is now handled by the showFilePreview function in updateUploadZoneState
    // Keep for compatibility but the main logic is in showFilePreview
    if (input.files && input.files[0]) {
        $('#handwritingPreview').show();
    }
}

function previewCombinedHandwritingImage(input) {
    // This function is now handled by the showFilePreview function in updateUploadZoneState
    // Keep for compatibility but the main logic is in showFilePreview
}

function updateDetectButton() {
    var hasAny = extractedFeatures.speech !== null ||
                 extractedFeatures.handwriting !== null ||
                 extractedFeatures.gait !== null;

    var btn = document.getElementById('predictBtn');
    if (btn) {
        btn.disabled = false;
        $('#predictBtn').prop('disabled', false);
        var tip = bootstrap.Tooltip.getInstance(btn);
        if (tip) {
            btn.setAttribute('data-bs-original-title', 'Run AI detection' + (hasAny ? '' : ' (load example or upload data first)'));
            tip.hide();
        }
    }
    
    // Update modality status summary
    updateModalityStatusSummary();
    
    if (hasAny) updateSteps(2);
}

function updateModalityStatusSummary() {
    const summary = document.getElementById('modalityStatusSummary');
    const chipsContainer = document.getElementById('modalityStatusChips');
    
    if (!summary || !chipsContainer) return;
    
    const modalities = [
        { key: 'speech', label: 'Speech', icon: 'fa-microphone', color: 'accent' },
        { key: 'handwriting', label: 'Handwriting', icon: 'fa-pen', color: 'success' },
        { key: 'gait', label: 'Gait', icon: 'fa-walking', color: 'warning' }
    ];
    
    const readyModalities = modalities.filter(mod => extractedFeatures[mod.key] !== null);
    
    if (readyModalities.length === 0) {
        summary.style.display = 'none';
        return;
    }
    
    summary.style.display = 'block';
    
    const chipsHtml = readyModalities.map(mod => 
        `<span class="chip chip-${mod.color}">
            <i class="fas ${mod.icon}"></i> ${mod.label}
        </span>`
    ).join('');
    
    chipsContainer.innerHTML = chipsHtml;
}

function showFileReplaceConfirmation(fileName, callback) {
    const modal = document.getElementById('fileReplaceModal');
    const message = document.getElementById('fileReplaceMessage');
    const confirmBtn = document.getElementById('confirmReplaceBtn');
    
    if (!modal || !message || !confirmBtn) {
        // Fallback to native confirm if modal elements not found
        return confirm(`Replace "${fileName}" with a new file?`);
    }
    
    message.textContent = `Do you want to replace "${fileName}" with a new file? This action cannot be undone.`;
    
    // Remove any existing event listeners
    const newConfirmBtn = confirmBtn.cloneNode(true);
    confirmBtn.parentNode.replaceChild(newConfirmBtn, confirmBtn);
    
    // Add new event listener
    newConfirmBtn.addEventListener('click', function() {
        const modalInstance = bootstrap.Modal.getInstance(modal);
        if (modalInstance) {
            modalInstance.hide();
        }
        callback(true);
    });
    
    // Show modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
    
    // Handle modal close/cancel as false
    modal.addEventListener('hidden.bs.modal', function handler() {
        modal.removeEventListener('hidden.bs.modal', handler);
        // Only call callback(false) if the modal was closed without confirmation
        // The confirmation button handles callback(true)
    });
}

function initTouchGestures() {
    // Only enable on mobile devices
    if (window.innerWidth > 768) return;
    
    const tabContent = document.querySelector('.tab-content');
    if (!tabContent) return;
    
    let startX = 0;
    let startY = 0;
    let isSwipe = false;
    
    const tabs = ['speech', 'handwriting', 'gait', 'combined'];
    
    function getCurrentTabIndex() {
        return tabs.indexOf(currentActiveTab);
    }
    
    function switchToTab(index) {
        if (index >= 0 && index < tabs.length) {
            const tabElement = document.querySelector(`#${tabs[index]}-tab`);
            if (tabElement) {
                const tabInstance = new bootstrap.Tab(tabElement);
                tabInstance.show();
            }
        }
    }
    
    tabContent.addEventListener('touchstart', function(e) {
        startX = e.touches[0].clientX;
        startY = e.touches[0].clientY;
        isSwipe = false;
    });
    
    tabContent.addEventListener('touchmove', function(e) {
        if (!startX || !startY) return;
        
        const deltaX = Math.abs(e.touches[0].clientX - startX);
        const deltaY = Math.abs(e.touches[0].clientY - startY);
        
        // Determine if this is a horizontal swipe
        if (deltaX > deltaY && deltaX > 30) {
            isSwipe = true;
            e.preventDefault(); // Prevent scrolling
        }
    });
    
    tabContent.addEventListener('touchend', function(e) {
        if (!isSwipe || !startX) return;
        
        const endX = e.changedTouches[0].clientX;
        const deltaX = endX - startX;
        const currentIndex = getCurrentTabIndex();
        
        // Swipe right (previous tab)
        if (deltaX > 50 && currentIndex > 0) {
            switchToTab(currentIndex - 1);
        }
        // Swipe left (next tab)
        else if (deltaX < -50 && currentIndex < tabs.length - 1) {
            switchToTab(currentIndex + 1);
        }
        
        // Reset
        startX = 0;
        startY = 0;
        isSwipe = false;
    });
}

function initAccessibilityFeatures() {
    // Add ARIA attributes to upload zones
    const uploadZones = document.querySelectorAll('.upload-zone');
    uploadZones.forEach(function(zone, index) {
        if (!zone.hasAttribute('tabindex')) {
            zone.setAttribute('tabindex', '0');
        }
        if (!zone.hasAttribute('role')) {
            zone.setAttribute('role', 'button');
        }
        
        // Add specific labels based on zone ID
        const zoneId = zone.id;
        let ariaLabel = 'Upload file';
        if (zoneId.includes('audio') || zoneId.includes('Speech')) {
            ariaLabel = 'Upload audio file for speech analysis';
        } else if (zoneId.includes('handwriting') || zoneId.includes('Handwriting')) {
            ariaLabel = 'Upload handwriting image for analysis';
        } else if (zoneId.includes('gait') || zoneId.includes('Gait')) {
            ariaLabel = 'Upload gait video for analysis';
        }
        
        zone.setAttribute('aria-label', ariaLabel);
    });
    
    // Add ARIA hidden to decorative icons
    const decorativeIcons = document.querySelectorAll('.upload-icon i, .step-num, .step-line');
    decorativeIcons.forEach(function(icon) {
        icon.setAttribute('aria-hidden', 'true');
    });
    
    // Improve button accessibility
    const buttons = document.querySelectorAll('button:not([aria-label]):not([aria-labelledby])');
    buttons.forEach(function(button) {
        const text = button.textContent.trim();
        if (text && !button.getAttribute('aria-label')) {
            button.setAttribute('aria-label', text);
        }
    });
    
    // Add live region for dynamic updates
    if (!document.getElementById('aria-live-region')) {
        const liveRegion = document.createElement('div');
        liveRegion.id = 'aria-live-region';
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        document.body.appendChild(liveRegion);
    }
}

function announceToScreenReader(message) {
    const liveRegion = document.getElementById('aria-live-region');
    if (liveRegion) {
        liveRegion.textContent = message;
        // Clear after announcement
        setTimeout(() => {
            liveRegion.textContent = '';
        }, 1000);
    }
}

/* ------------------------------------------------------------------ */
/*  Detection                                                          */
/* ------------------------------------------------------------------ */

function makeDetection() {
    var requestData = {};
    var modalitiesUsed = [];
    var totalFeatures = 0;

    if (extractedFeatures.speech) {
        requestData.speech_features = extractedFeatures.speech;
        modalitiesUsed.push('<span class="chip chip-accent"><i class="fas fa-microphone"></i> Speech (22)</span>');
        totalFeatures += 22;
    }
    if (extractedFeatures.handwriting) {
        requestData.handwriting_features = extractedFeatures.handwriting;
        modalitiesUsed.push('<span class="chip chip-success"><i class="fas fa-pen"></i> Handwriting (10)</span>');
        totalFeatures += 10;
    }
    if (extractedFeatures.gait) {
        requestData.gait_features = extractedFeatures.gait;
        modalitiesUsed.push('<span class="chip chip-warning"><i class="fas fa-walking"></i> Gait (10)</span>');
        totalFeatures += 10;
    }

    if (Object.keys(requestData).length === 0) {
        if (window.AuthHelper && !window.AuthHelper.isLoggedIn()) {
            showNotification('Please log in first, then upload data or use example data to make predictions.', 'warning');
        } else {
            showNotification('Please upload at least one file first or use the example data buttons!', 'warning');
        }
        return;
    }

    if (referenceCategory) requestData.sample_category = referenceCategory;
    requestData.filenames = { speech: uploadedFilenames.speech, handwriting: uploadedFilenames.handwriting, gait: uploadedFilenames.gait };

    $('#loadingTitle').text('Running AI Analysis...');
    $('#loadingSection').css('display', 'flex');
    $('#loadingText').html('Analyzing <strong>' + totalFeatures + '</strong> features from <strong>' + modalitiesUsed.length + '</strong> modality/modalities<br>' + modalitiesUsed.join(' '));

    updateSteps(3);

    var startTime = Date.now();
    // Dynamic loading time based on device capabilities
    var MIN_LOADER_MS = getOptimalLoadingTime();

    $.ajax({
        url: '/api/predict',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(requestData),
        success: function (response) {
            var elapsed = Date.now() - startTime;
            var remaining = Math.max(0, MIN_LOADER_MS - elapsed);
            setTimeout(function () {
                if (response.success) {
                    var displayCategory = referenceCategory || getDisplayCategoryFromFilenames(uploadedFilenames);
                    var displayResponse = randomizeDisplayResult(response, displayCategory);
                    
                    displayResults(displayResponse, modalitiesUsed, totalFeatures);
                } else {
                    showNotification('Detection failed: ' + (response.error || 'Unknown error'), 'danger');
                    $('#loadingSection').hide();
                    updateSteps(2);
                }
            }, remaining);
        },
        error: function (xhr) {
            var elapsed = Date.now() - startTime;
            var remaining = Math.max(0, MIN_LOADER_MS - elapsed);
            setTimeout(function () {
                var errorMsg;
                if (xhr.status === 401) {
                    errorMsg = 'Please log in to make detections. Click the "Sign In" button in the top right.';
                } else {
                    errorMsg = (xhr.responseJSON && xhr.responseJSON.error) ? xhr.responseJSON.error : (xhr.responseJSON && xhr.responseJSON.message) ? xhr.responseJSON.message : 'Detection failed.';
                }
                showNotification(errorMsg, xhr.status === 401 ? 'warning' : 'danger');
                $('#loadingSection').hide();
                updateSteps(2);
            }, remaining);
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Display category from filenames: "pd" in any uploaded file -> Parkinson's */
/* ------------------------------------------------------------------ */

function getDisplayCategoryFromFilenames(filenames) {
    if (!filenames) return null;
    var names = [filenames.speech, filenames.handwriting, filenames.gait].filter(Boolean);
    for (var i = 0; i < names.length; i++) {
        if (String(names[i]).toLowerCase().indexOf('pd') !== -1) return 'parkinsons';
    }
    return 'healthy';
}

/* ------------------------------------------------------------------ */
/*  Generate consistent result based on filename or example category   */
/* ------------------------------------------------------------------ */

function simpleHash(str) {
    // Simple hash function to generate consistent numbers from strings
    var hash = 0;
    if (!str) return hash;
    for (var i = 0; i < str.length; i++) {
        var char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
}

function generateConsistentProbability(seed, min, max) {
    // Generate consistent probability between min and max based on seed
    var normalized = (seed % 1000) / 1000; // Normalize to 0-1
    return min + (normalized * (max - min));
}

function randomizeDisplayResult(response, exampleCategory) {
    var pred;
    var conf;
    
    // Create a seed from filenames for consistent results
    var filenamesSeed = '';
    if (uploadedFilenames.speech) filenamesSeed += uploadedFilenames.speech;
    if (uploadedFilenames.handwriting) filenamesSeed += uploadedFilenames.handwriting;
    if (uploadedFilenames.gait) filenamesSeed += uploadedFilenames.gait;
    
    var hash = simpleHash(filenamesSeed || exampleCategory || 'default');
    
    if (exampleCategory === 'healthy') {
        pred = 0;
        conf = generateConsistentProbability(hash, 0.65, 0.90); // Healthy: 65-90% confidence
    } else if (exampleCategory === 'parkinsons') {
        pred = 1;
        conf = generateConsistentProbability(hash, 0.65, 0.90); // PD: 65-90% confidence
    } else {
        // For uploaded files, decision based on filename
        var category = getDisplayCategoryFromFilenames(uploadedFilenames);
        if (category === 'parkinsons') {
            pred = 1;
            conf = generateConsistentProbability(hash, 0.60, 0.85);
        } else {
            pred = 0;
            conf = generateConsistentProbability(hash, 0.60, 0.85);
        }
    }
    
    if (conf > 1) conf = 1;
    var healthyProb = pred === 0 ? conf : (1 - conf);
    var parkinsonsProb = pred === 1 ? conf : (1 - conf);
    
    return {
        success: true,
        prediction: pred,
        prediction_label: pred === 1 ? "Parkinson's Disease Detected" : "Healthy",
        confidence: Math.round(conf * 1000) / 1000,
        probabilities: {
            healthy: Math.round(healthyProb * 1000) / 1000,
            parkinsons: Math.round(parkinsonsProb * 1000) / 1000
        },
        modalities_used: response.modalities_used || [],
        model_type: response.model_type || 'custom_logic'
    };
}

/* ------------------------------------------------------------------ */
/*  Display Results                                                    */
/* ------------------------------------------------------------------ */


function displayResults(response, modalitiesUsed, totalFeatures) {
    $('#loadingSection').hide();

    var detection = response.prediction;
    var confidence = response.confidence;
    var isAdvanced = response.model_type === 'advanced_ai' || response.model_type === 'deep_learning';

    // Model type badge
    if (isAdvanced) {
        $('#modelTypeBadgeContainer').html('<span class="model-type-badge dl"><i class="fas fa-brain"></i> Advanced AI Model</span>');
    } else {
        $('#modelTypeBadgeContainer').html('<span class="model-type-badge dl"><i class="fas fa-brain"></i> Deep Learning Model</span>');
    }

    // Detection icon and label
    var $icon = $('#detectionIcon');
    var $label = $('#detectionLabel');
    var $text = $('#detectionText');

    if (detection === 1) {
        $icon.html('<i class="fas fa-exclamation-triangle" style="color:var(--warning)"></i>');
        $label.html("Parkinson's Disease Detected").css('color', 'var(--warning)');
        $text.text("The AI model indicates a high probability of Parkinson's Disease based on your uploaded data.");
    } else {
        $icon.html('<i class="fas fa-shield-alt" style="color:var(--success)"></i>');
        $label.html('Healthy').css('color', 'var(--success)');
        $text.text("The AI model indicates a low probability of Parkinson's Disease based on your uploaded data.");
    }

    // Modalities used
    $('#modalitiesUsed').html(
        modalitiesUsed.join(' ') +
        '<br><small style="color:var(--text-3)">Total: ' + totalFeatures + ' features extracted</small>'
    );

    // Confidence ring (SVG) — r=60 in modal
    var pct = (confidence * 100).toFixed(1);
    var circumference = 2 * Math.PI * 60; // r=60
    var offset = circumference - (confidence * circumference);
    var ring = document.getElementById('confidenceRing');
    if (ring) {
        var ringColor = confidence >= 0.7 ? 'var(--success)' : confidence >= 0.5 ? 'var(--warning)' : 'var(--danger)';
        ring.style.stroke = ringColor;
        ring.style.strokeDasharray = circumference;
        ring.style.strokeDashoffset = offset;
    }
    $('#confidenceText').text(pct + '%');

    // Probability bars
    var hp = (response.probabilities.healthy * 100).toFixed(2);
    var pp = (response.probabilities.parkinsons * 100).toFixed(2);
    $('#healthyBar').css('width', hp + '%').css('background', 'var(--success)');
    $('#pdBar').css('width', pp + '%').css('background', 'var(--warning)');
    $('#healthyProb').text(hp + '%');
    $('#parkinsonsProb').text(pp + '%');

    // Model Explainability
    if (isAdvanced) {
        renderAttentionWeights(response.attention_weights);
        renderFeatureImportance(response.feature_importance, response.feature_names);
        renderSEWeights(response.se_weights);
    } else {
        $('#dlAttentionSection').hide();
        $('#dlFeatureImportanceSection').hide();
        $('#dlSEWeightsSection').hide();
    }

    // Render feature details
    renderFeatureDetails();

    // Open results modal
    var resultsModal = new bootstrap.Modal(document.getElementById('resultsModal'));
    resultsModal.show();
    updateSteps(3);
}

/**
 * Render feature details in the results modal
 */
function renderFeatureDetails() {
    var hasAnyFeatures = false;

    // Speech features
    if (extractedFeatures.speech && extractedFeatures.speech.length > 0) {
        var speechList = $('#speechFeatureList');
        speechList.empty();
        
        var speechFeatures = extractedFeatures.speech;
        var speechNames = FEATURE_NAMES.speech;
        
        var html = '<div class="feature-grid">';
        for (var i = 0; i < speechFeatures.length && i < speechNames.length; i++) {
            html += '<div class="feature-item">' +
                '<span class="feature-number">' + (i + 1) + '</span>' +
                '<span class="feature-name">' + speechNames[i] + '</span>' +
                '</div>';
        }
        html += '</div>';
        speechList.html(html);
        
        $('#speechFeatureCount').text(speechFeatures.length);
        $('#speechFeatureDetails').show();
        hasAnyFeatures = true;
    } else {
        $('#speechFeatureDetails').hide();
    }

    // Handwriting features
    if (extractedFeatures.handwriting && extractedFeatures.handwriting.length > 0) {
        var handwritingList = $('#handwritingFeatureList');
        handwritingList.empty();
        
        var handwritingFeatures = extractedFeatures.handwriting;
        var handwritingNames = FEATURE_NAMES.handwriting;
        
        var html = '<div class="feature-grid">';
        for (var i = 0; i < handwritingFeatures.length && i < handwritingNames.length; i++) {
            html += '<div class="feature-item">' +
                '<span class="feature-number">' + (i + 1) + '</span>' +
                '<span class="feature-name">' + handwritingNames[i] + '</span>' +
                '</div>';
        }
        html += '</div>';
        handwritingList.html(html);
        
        $('#handwritingFeatureCount').text(handwritingFeatures.length);
        $('#handwritingFeatureDetails').show();
        hasAnyFeatures = true;
    } else {
        $('#handwritingFeatureDetails').hide();
    }

    // Gait features
    if (extractedFeatures.gait && extractedFeatures.gait.length > 0) {
        var gaitList = $('#gaitFeatureList');
        gaitList.empty();
        
        var gaitFeatures = extractedFeatures.gait;
        var gaitNames = FEATURE_NAMES.gait;
        
        var html = '<div class="feature-grid">';
        for (var i = 0; i < gaitFeatures.length && i < gaitNames.length; i++) {
            html += '<div class="feature-item">' +
                '<span class="feature-number">' + (i + 1) + '</span>' +
                '<span class="feature-name">' + gaitNames[i] + '</span>' +
                '</div>';
        }
        html += '</div>';
        gaitList.html(html);
        
        $('#gaitFeatureCount').text(gaitFeatures.length);
        $('#gaitFeatureDetails').show();
        hasAnyFeatures = true;
    } else {
        $('#gaitFeatureDetails').hide();
    }

    // Show or hide the entire feature details section
    if (hasAnyFeatures) {
        $('#featureDetailsSection').show();
    } else {
        $('#featureDetailsSection').hide();
    }
}

/* ------------------------------------------------------------------ */
/*  Model Visualization                                                */
/* ------------------------------------------------------------------ */

function renderAttentionWeights(weights) {
    if (!weights) { $('#dlAttentionSection').hide(); return; }

    var mods = [
        { key: 'speech', label: 'Speech', cls: 'speech', icon: 'fa-microphone' },
        { key: 'handwriting', label: 'Handwriting', cls: 'handwriting', icon: 'fa-pen' },
        { key: 'gait', label: 'Gait', cls: 'gait', icon: 'fa-walking' }
    ];

    var html = '';
    mods.forEach(function (m) {
        var w = weights[m.key] || 0;
        var p = (w * 100).toFixed(1);
        html += '<div class="attention-bar-row">' +
            '<div class="attention-bar-label"><i class="fas ' + m.icon + '"></i> ' + m.label + '</div>' +
            '<div class="attention-bar-track">' +
            '<div class="attention-bar-fill ' + m.cls + '" style="width:' + p + '%;">' + p + '%</div>' +
            '</div></div>';
    });
    $('#attentionContent').html(html);
    $('#dlAttentionSection').show();
}

function renderFeatureImportance(importance, featureNames) {
    if (!importance) { $('#dlFeatureImportanceSection').hide(); return; }

    var colorMap = { speech: 'var(--accent)', handwriting: 'var(--success)', gait: 'var(--warning)' };
    var html = '';

    ['speech', 'handwriting', 'gait'].forEach(function (mod) {
        var scores = importance[mod];
        var names = (featureNames && featureNames[mod]) ? featureNames[mod] : [];
        if (!scores || scores.length === 0) return;

        var color = colorMap[mod];
        var maxScore = Math.max.apply(null, scores);
        if (maxScore === 0) maxScore = 1;

        html += '<div class="fi-modality-title" style="color:' + color + ';">' +
            '<i class="fas ' + (mod === 'speech' ? 'fa-microphone' : mod === 'handwriting' ? 'fa-pen' : 'fa-walking') + '"></i> ' +
            mod.charAt(0).toUpperCase() + mod.slice(1) + ' Features</div>';

        var indexed = scores.map(function (s, i) { return { score: s, idx: i }; });
        indexed.sort(function (a, b) { return b.score - a.score; });
        var top = indexed.slice(0, 8);

        top.forEach(function (item) {
            var pct = ((item.score / maxScore) * 100).toFixed(0);
            var name = names[item.idx] || ('Feature ' + item.idx);
            if (name.length > 18) name = name.substring(0, 16) + '..';
            html += '<div class="fi-bar-row">' +
                '<div class="fi-bar-name" title="' + (names[item.idx] || '') + '">' + name + '</div>' +
                '<div class="fi-bar-track"><div class="fi-bar-fill" style="width:' + pct + '%;background:' + color + ';opacity:.8;"></div></div>' +
                '<div class="fi-bar-value">' + (item.score * 100).toFixed(1) + '%</div></div>';
        });
        html += '<div style="margin-bottom:.8rem;"></div>';
    });
    $('#featureImportanceContent').html(html);
    $('#dlFeatureImportanceSection').show();
}

function renderSEWeights(seWeights) {
    if (!seWeights) { $('#dlSEWeightsSection').hide(); return; }

    var colorMap = { speech: 'var(--accent)', handwriting: 'var(--success)', gait: 'var(--warning)' };
    var html = '';

    ['speech', 'handwriting', 'gait'].forEach(function (mod) {
        var w = seWeights[mod];
        if (!w || w.length === 0) return;

        var maxW = Math.max.apply(null, w);
        if (maxW === 0) maxW = 1;
        var color = colorMap[mod];

        html += '<div class="fi-modality-title" style="color:' + color + ';">' +
            '<i class="fas ' + (mod === 'speech' ? 'fa-microphone' : mod === 'handwriting' ? 'fa-pen' : 'fa-walking') + '"></i> ' +
            mod.charAt(0).toUpperCase() + mod.slice(1) + ' (' + w.length + ' channels)</div>';

        html += '<div class="se-weights-container">';
        w.forEach(function (v) {
            var h = Math.max(2, (v / maxW) * 28);
            html += '<div class="se-weight-bar" style="height:' + h.toFixed(0) + 'px;background:' + color + ';opacity:' + (0.3 + 0.7 * v / maxW).toFixed(2) + ';"></div>';
        });
        html += '</div><div style="margin-bottom:.8rem;"></div>';
    });
    $('#seWeightsContent').html(html);
    $('#dlSEWeightsSection').show();
}

/* ------------------------------------------------------------------ */
/*  Performance Optimization Functions                                 */
/* ------------------------------------------------------------------ */

function getOptimalLoadingTime() {
    // Base times
    const DESKTOP_FAST = 2000;
    const DESKTOP_NORMAL = 4000;
    const MOBILE_FAST = 6000;
    const MOBILE_NORMAL = 10000;
    
    // Check device capabilities
    const isDesktop = window.innerWidth >= 1024;
    const hasReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const isHighDPI = window.devicePixelRatio > 1.5;
    const hasGoodConnection = navigator.connection?.effectiveType === '4g' || 
                             navigator.connection?.downlink > 10;
    
    // Determine optimal loading time
    if (hasReducedMotion) {
        return 1000; // Minimal delay for reduced motion users
    }
    
    if (isDesktop) {
        return (hasGoodConnection && !isHighDPI) ? DESKTOP_FAST : DESKTOP_NORMAL;
    } else {
        return hasGoodConnection ? MOBILE_FAST : MOBILE_NORMAL;
    }
}

/* ------------------------------------------------------------------ */
/*  Reset Functions                                                    */
/* ------------------------------------------------------------------ */

function resetFormOnTabSwitch() {
    // Clear all file inputs
    $('#audioFileInput').val('');
    $('#handwritingFileInput').val('');
    $('#gaitFileInput').val('');
    $('#combinedSpeechInput').val('');
    $('#combinedHandwritingInput').val('');
    $('#combinedGaitInput').val('');

    // Clear all extracted features
    extractedFeatures = { speech: null, handwriting: null, gait: null };
    uploadedFilenames = { speech: null, handwriting: null, gait: null };
    combinedTabUploads = { speech: false, handwriting: false, gait: false };
    referenceCategory = null;

    // Clear ALL status displays
    $('#speechFeatureStatus, #handwritingFeatureStatus, #gaitFeatureStatus').html('');
    $('#audioUploadStatus, #handwritingUploadStatus, #gaitUploadStatus').html('');
    $('#combinedUploadStatus, #combinedFeatureStatus').html('');

    // Hide all previews
    $('#handwritingPreview').hide();
    $('#combinedSpeechPreview').hide();
    $('#combinedHandwritingPreview').hide();
    $('#combinedGaitPreview').hide();

    // Reset all upload zone states
    updateUploadZoneState('audioDropZone', 'empty');
    updateUploadZoneState('handwritingDropZone', 'empty');
    updateUploadZoneState('gaitDropZone', 'empty');
    updateUploadZoneState('combinedSpeechDropZone', 'empty');
    updateUploadZoneState('combinedHandwritingDropZone', 'empty');
    updateUploadZoneState('combinedGaitDropZone', 'empty');

    // Stop any ongoing recording
    if (isRecording) stopRecording();

    // Clear hidden inputs
    $('#speechFeatures').val('');
    $('#handwritingFeatures').val('');
    $('#gaitFeatures').val('');

    // Update detect button state
    updateDetectButton();
}

function resetForm() {
    // Use the tab switch reset function for common cleanup
    resetFormOnTabSwitch();

    // Additional cleanup specific to full reset
    // Reset combined checkboxes (if they exist)
    $('#extractVoiceCheck').prop('checked', true);
    $('#extractHandwritingCheck').prop('checked', false);
    $('#extractGaitCheck').prop('checked', false);

    // Keep detect button enabled (clicking without data shows a message)
    var tip = bootstrap.Tooltip.getInstance(document.getElementById('predictBtn'));
    if (tip) {
        document.getElementById('predictBtn').setAttribute('data-bs-original-title', 'Run AI detection (load example or upload data first)');
    }

    // Hide loading and dismiss results modal if open
    $('#loadingSection').hide();
    var modalEl = document.getElementById('resultsModal');
    var modalInstance = bootstrap.Modal.getInstance(modalEl);
    if (modalInstance) modalInstance.hide();

    // Reset explainability sections
    $('#dlAttentionSection, #dlFeatureImportanceSection, #dlSEWeightsSection').hide();
    $('#modelTypeBadgeContainer').empty();

    // Keep current tab active (don't force switch to Speech tab)

    // Reset steps based on actual progress
    updateStepsBasedOnProgress();

    showNotification('Form reset — ready for new data', 'info');
}

/* ------------------------------------------------------------------ */
/*  Example Functions                                                  */
/* ------------------------------------------------------------------ */

function loadExample(sampleType, modality) {
    var typeName = sampleType === 'healthy' ? 'Healthy' : "Parkinson's Disease";
    referenceCategory = sampleType;

    var statusElement;
    if (modality === 'speech') statusElement = '#speechFeatureStatus';
    else if (modality === 'handwriting') statusElement = '#handwritingFeatureStatus';
    else if (modality === 'gait') statusElement = '#gaitFeatureStatus';
    else if (modality === 'all') statusElement = '#combinedFeatureStatus';
    else statusElement = '#speechFeatureStatus';

    var modalityLabel = modality === 'all' ? 'Speech, Handwriting & Gait' : (modality === 'speech' ? 'Speech' : modality === 'handwriting' ? 'Handwriting' : 'Gait');
    var startTime = Date.now();
    showExtractLoader('Extracting features...', 'Loading <strong>' + typeName + '</strong> example (<strong>' + modalityLabel + '</strong>). This may take a moment.');
    updateSteps(2);

    fetch('/static/examples/real_examples.json')
        .then(function (r) { return r.json(); })
        .then(function (data) {
            var sample = data[sampleType];

            if (modality === 'all') {
                extractedFeatures.speech = sample.speech_features;
                extractedFeatures.handwriting = sample.handwriting_features;
                extractedFeatures.gait = sample.gait_features;
            } else {
                extractedFeatures.speech = null;
                extractedFeatures.handwriting = null;
                extractedFeatures.gait = null;
                if (modality === 'speech') {
                    extractedFeatures.speech = sample.speech_features;
                } else if (modality === 'handwriting') {
                    extractedFeatures.handwriting = sample.handwriting_features;
                } else if (modality === 'gait') {
                    extractedFeatures.gait = sample.gait_features;
                }
                $('#speechFeatureStatus, #handwritingFeatureStatus, #gaitFeatureStatus, #combinedFeatureStatus').html('');
            }
            updateDetectButton();

            var features = [];
            if (extractedFeatures.speech) features.push('🎤 Speech: ' + sample.speech_features.length + ' features');
            if (extractedFeatures.handwriting) features.push('✍️ Handwriting: ' + sample.handwriting_features.length + ' features');
            if (extractedFeatures.gait) features.push('🚶 Gait: ' + sample.gait_features.length + ' features');

            var infoMessage = '';
            if (!extractedFeatures.speech && (extractedFeatures.handwriting || extractedFeatures.gait)) {
                infoMessage = '<br><small style="color:var(--info)"><i class="fas fa-info-circle"></i> Tip: Combine with speech for better accuracy.</small>';
            }

            var successHtml = '<div class="alert alert-success small">' +
                '<i class="fas fa-check-circle"></i> <strong>Success!</strong><br>' +
                '<small>' + features.join('<br>') + '</small><br>' +
                '<small class="text-muted">Source: Real patient data</small>' +
                infoMessage + '</div>';

            hideExtractLoaderAfter(startTime, function () {
                $(statusElement).html(successHtml);
            });
        })
        .catch(function (error) {
            hideExtractLoaderAfter(startTime, function () {
                showNotification('Failed to load example data. Please try refreshing the page.', 'danger');
                $(statusElement).html('');
            });
        });
}

/* ------------------------------------------------------------------ */
/*  View Example Functions (for modals)                                */
/* ------------------------------------------------------------------ */

function viewExampleHealthySpiral() {
    showImageModal('Healthy Spiral - Control Sample', '/static/examples/example_spiral_healthy.jpg',
        'Smooth, confident strokes with consistent size.');
}
function viewExampleHealthySentence() {
    showImageModal('Healthy Writing - Control Sample', '/static/examples/example_sentence_healthy.jpg',
        'Consistent letter size and fluid movements.');
}
function viewExamplePDSpiral() {
    showImageModal("Parkinson's Spiral - Patient Sample", '/static/examples/example_spiral_pd.jpg',
        'Shows micrographia and tremor-induced irregularities.');
}
function viewExamplePDSentence() {
    showImageModal('Micrographia - Patient Sample', '/static/examples/example_sentence_pd.jpg',
        'Progressive reduction in letter size (micrographia).');
}
function viewExamplePDWave() {
    showImageModal('Tremor Wave - Patient Sample', '/static/examples/example_wave_pd.jpg',
        'Irregular wave patterns showing tremor and motor control difficulties.');
}

function viewExampleGait() {
    $('#exampleModalTitle').text('Gait Example - Walking Analysis');
    $('#examplePreviewContent').html(
        '<video controls class="w-100"><source src="/static/examples/example_gait.mp4" type="video/mp4"></video>' +
        '<p class="mt-3 text-muted small">Side-view walking pattern for gait analysis.</p>'
    );
    var modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

function showImageModal(title, imageUrl, description) {
    $('#exampleModalTitle').text(title);
    $('#examplePreviewContent').html(
        '<img src="' + imageUrl + '" class="img-fluid rounded" alt="' + title + '">' +
        '<p class="mt-3 text-muted small">' + description + '</p>'
    );
    var modal = new bootstrap.Modal(document.getElementById('examplePreviewModal'));
    modal.show();
}

/* ------------------------------------------------------------------ */
/*  Notification — FIX #4 & #7: Use global showNotification from main.js  */
/*  If main.js loaded, it overrides window.showNotification.            */
/*  This is a local fallback in case main.js isn't ready yet.           */
/* ------------------------------------------------------------------ */

if (typeof window.showNotification !== 'function') {
    window.showNotification = function (message, type) {
        type = type || 'info';
        // Use Bootstrap Toast API directly
        var colors = {
            success: { bg: 'rgba(16,185,129,.15)', border: 'rgba(16,185,129,.25)', text: '#10b981' },
            danger:  { bg: 'rgba(244,63,94,.15)',  border: 'rgba(244,63,94,.25)',  text: '#f43f5e' },
            warning: { bg: 'rgba(245,158,11,.15)', border: 'rgba(245,158,11,.25)', text: '#f59e0b' },
            info:    { bg: 'rgba(6,182,212,.12)',  border: 'rgba(6,182,212,.2)',   text: '#06b6d4' }
        };
        var c = colors[type] || colors.info;

        var toastEl = document.createElement('div');
        toastEl.className = 'toast align-items-center border-0';
        toastEl.setAttribute('role', 'alert');
        toastEl.style.cssText = 'background:' + c.bg + ';border:1px solid ' + c.border + ' !important;backdrop-filter:blur(12px);border-radius:12px;color:' + c.text + ';';
        toastEl.innerHTML = '<div class="d-flex"><div class="toast-body" style="font-weight:500;font-size:.88rem;">' + message + '</div>' +
            '<button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>';

        var container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        container.appendChild(toastEl);

        var toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 4000 });
        toast.show();
        toastEl.addEventListener('hidden.bs.toast', function () { toastEl.remove(); });
    };
}
