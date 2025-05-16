// js/ui.js

function showToast(message, type = 'info', duration = 3500) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        console.error("Toast container not found.");
        alert(`Toast Fallback: ${message}`); // Fallback if container is missing
        return;
    }
    if (typeof bootstrap === 'undefined' || typeof bootstrap.Toast === 'undefined') {
        console.error("Bootstrap Toast JS not loaded or not ready.");
        alert(`Bootstrap Toast Fallback: ${message}`); // Fallback if Bootstrap JS is an issue
        return;
    }

    const toastId = 'toast-' + Date.now();
    
    // Map simple types to Aura theme toast background/text classes
    let toastBgClass = 'bg-aura-sky-light'; // Default to info
    let textClass = 'text-aura-text-light-primary'; // Default text for light BGs
    let iconHtml = '<i class="fas fa-info-circle me-2"></i>';

    switch (type) {
        case 'success':
        case 'success-ultra':
            toastBgClass = 'bg-aura-green-light';
            textClass = 'text-white'; // Green is dark enough for white text
            iconHtml = '<i class="fas fa-check-circle me-2"></i>';
            break;
        case 'danger':
        case 'danger-ultra':
            toastBgClass = 'bg-aura-red-light';
            textClass = 'text-white';
            iconHtml = '<i class="fas fa-exclamation-triangle me-2"></i>';
            break;
        case 'warning':
        case 'warning-ultra':
            toastBgClass = 'bg-aura-amber-light';
            textClass = 'text-aura-text-light-primary'; // Amber is light
            iconHtml = '<i class="fas fa-exclamation-circle me-2"></i>';
            break;
        case 'info':
        case 'info-ultra':
            toastBgClass = 'bg-aura-sky-light'; // Already default
            textClass = 'text-white';
            break;
        default: // Default to info style
            toastBgClass = 'bg-aura-sky-light';
            textClass = 'text-white';
    }

    // If dark theme is active, use dark theme equivalents
    if (document.body.classList.contains('dark-theme')) {
        switch (type) {
            case 'success':
            case 'success-ultra':
                toastBgClass = 'bg-aura-green-dark';
                textClass = 'text-aura-bg-dark-primary'; // Dark text on lighter green
                break;
            case 'danger':
            case 'danger-ultra':
                toastBgClass = 'bg-aura-red-dark';
                textClass = 'text-white';
                break;
            case 'warning':
            case 'warning-ultra':
                toastBgClass = 'bg-aura-amber-dark';
                textClass = 'text-aura-bg-dark-primary';
                break;
            case 'info':
            case 'info-ultra':
            default:
                toastBgClass = 'bg-aura-sky-dark';
                textClass = 'text-aura-bg-dark-primary'; // Dark text on lighter blue
                break;
        }
    }


    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center ${textClass} ${toastBgClass} border-0 shadow-lg" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="${duration}">
            <div class="d-flex">
                <div class="toast-body">
                    ${iconHtml}
                    ${message}
                </div>
                <button type="button" class="btn-close ${textClass === 'text-white' || textClass === 'text-aura-text-dark-primary' ? 'btn-close-white' : ''} me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;
    toastContainer.insertAdjacentHTML('beforeend', toastHTML);
    const toastElement = document.getElementById(toastId);
    
    const toast = bootstrap.Toast.getOrCreateInstance(toastElement, { autohide: true, delay: duration });
    toast.show();
    toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
}


function populateDummyFastaSelect() {
    const selectElement = document.getElementById('dummy-fasta-select');
    if (!selectElement) return;
    selectElement.innerHTML = '';
    const examples = getDummyFastaExamples();
    examples.forEach((example, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = example.name;
        selectElement.appendChild(option);
    });
}

function showSpinner(message = "Working...") {
    let overlay = document.getElementById('loading-spinner-overlay-main');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-spinner-overlay-main';
        overlay.className = 'spinner-overlay-enhanced'; // Uses Aura CSS
        overlay.innerHTML = `
            <div class="text-center">
                <div class="spinner-border text-primary-gradient spinner-border-enhanced" role="status"></div>
                <p class="spinner-text mt-2 mb-0 small">${message}</p>
            </div>`;
        document.body.appendChild(overlay);
    } else {
        overlay.querySelector('p.spinner-text').textContent = message;
    }
    // Add small delay to ensure class is added after potential display:none
    setTimeout(() => overlay.classList.add('show'), 10);
}

function hideSpinner() {
    const overlay = document.getElementById('loading-spinner-overlay-main');
    if (overlay) {
        overlay.classList.remove('show');
    }
}

function toggleThemeAndSave() {
    // currentConfig is global, defined in config.js
    const newTheme = currentConfig.currentTheme === 'dark' ? 'light' : 'dark';
    saveConfig({ currentTheme: newTheme }); // saveConfig is from config.js
    applyConfigToUI(); // applyConfigToUI is from config.js, handles body class and icon

    // Re-render plots if any, as their styling might need to adapt
    if (window.currentEmbeddingDataForPlot && typeof plotEmbeddings === 'function') {
         plotEmbeddings(
                window.currentEmbeddingDataForPlot,
                'embedding-plot-container',
                document.getElementById('color-by-select').value,
                document.getElementById('embedding-plot-type').value
            );
    }
    // Ensure settings modal theme selector is also updated
    const themeSelectInModal = document.getElementById('setting-theme-select');
    if (themeSelectInModal) {
        themeSelectInModal.value = newTheme;
    }
}


function updateClassificationProgress(current, total, isComplete = false) {
    const progressBarContainer = document.getElementById('classification-progress-bar-container');
    const progressBar = document.getElementById('classification-progress-bar');
    if (!progressBar || !progressBarContainer) return;

    if (total > 0 && current <= total && !isComplete) {
        progressBarContainer.style.display = 'flex';
        const percentage = (current / total) * 100;
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.textContent = `${Math.round(percentage)}%`;
    } else {
        if(isComplete && total > 0) { // Only show "Done" if there was work
            progressBar.style.width = `100%`;
            progressBar.textContent = `Done`;
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                progressBar.style.width = `0%`; 
                progressBar.textContent = `0%`;
            }, 1500);
        } else { // No work or not complete, hide immediately
            progressBarContainer.style.display = 'none';
            progressBar.style.width = `0%`; 
            progressBar.textContent = `0%`;
        }
    }
}

function displayEmbeddingFileInfo(file, numPoints, numDimensions) {
    const infoDiv = document.getElementById('embedding-file-info');
    if (!infoDiv) return;
    infoDiv.innerHTML = `
        <strong class="d-block mb-1"><i class="fas fa-file-alt me-1 text-primary-gradient"></i> ${file.name}</strong>
        <span class="badge bg-light-subtle text-dark-emphasis me-1">${(file.size / 1024).toFixed(1)} KB</span>
        <span class="badge bg-info-subtle text-info-emphasis me-1">${numPoints} points</span>
        <span class="badge bg-info-subtle text-info-emphasis">${numDimensions} dims</span>
    `;
    infoDiv.style.display = 'block';
}

function clearEmbeddingFileInfo() {
     const infoDiv = document.getElementById('embedding-file-info');
    if (infoDiv) {
        infoDiv.style.display = 'none';
        infoDiv.innerHTML = '';
    }
    const statsDiv = document.getElementById('embedding-stats');
    if (statsDiv) statsDiv.innerHTML = '';
     const plotContainer = document.getElementById('embedding-plot-container');
    if(plotContainer) plotContainer.innerHTML = '<div class="plot-placeholder"><i class="fas fa-atom fa-3x text-placeholder mb-2"></i><p>Visualize your high-dimensional biological data.</p></div>';
    window.currentEmbeddingDataForPlot = null;
}

function updateTotalSequencesClassified(count) {
    const badge = document.getElementById('total-sequences-classified');
    if (badge) {
        badge.textContent = `${count} sequence${count !== 1 ? 's' : ''}`;
    }
}