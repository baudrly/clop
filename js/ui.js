// js/ui.js

function showToast(message, type = 'info', duration = 3500) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        console.error("Toast container not found.");
        alert(`Toast Fallback: ${message}`); 
        return;
    }
    if (typeof bootstrap === 'undefined' || typeof bootstrap.Toast === 'undefined') {
        console.error("Bootstrap Toast JS not loaded or not ready for toast.");
        alert(`Bootstrap Toast Fallback (JS missing/not ready): ${message}`);
        return;
    }

    const toastId = 'toast-' + Date.now();
    
    let toastBgClass = 'bg-aura-sky-light'; 
    let textClass = 'text-white'; 
    let iconHtml = '<i class="fas fa-info-circle me-2"></i>';

    const isDarkTheme = document.body.classList.contains('dark-theme');

    switch (type) {
        case 'success': case 'success-ultra':
            toastBgClass = isDarkTheme ? 'bg-aura-green-dark' : 'bg-aura-green-light';
            textClass = isDarkTheme ? 'text-aura-bg-dark-primary' : 'text-white';
            iconHtml = '<i class="fas fa-check-circle me-2"></i>';
            break;
        case 'danger': case 'danger-ultra':
            toastBgClass = isDarkTheme ? 'bg-aura-red-dark' : 'bg-aura-red-light';
            textClass = 'text-white';
            iconHtml = '<i class="fas fa-exclamation-triangle me-2"></i>';
            break;
        case 'warning': case 'warning-ultra':
            toastBgClass = isDarkTheme ? 'bg-aura-amber-dark' : 'bg-aura-amber-light';
            textClass = 'text-aura-text-light-primary'; 
            if (isDarkTheme) textClass = 'text-aura-bg-dark-primary';
            iconHtml = '<i class="fas fa-exclamation-circle me-2"></i>';
            break;
        case 'info': case 'info-ultra': default:
            toastBgClass = isDarkTheme ? 'bg-aura-sky-dark' : 'bg-aura-sky-light';
            textClass = isDarkTheme ? 'text-aura-bg-dark-primary' : 'text-white';
            break;
    }

    const toastHTML = `
        <div id="${toastId}" class="toast align-items-center ${textClass} ${toastBgClass} border-0 shadow-lg" role="alert" aria-live="assertive" aria-atomic="true" data-bs-delay="${duration}">
            <div class="d-flex">
                <div class="toast-body">
                    ${iconHtml}
                    ${message}
                </div>
                <button type="button" class="btn-close ${textClass === 'text-white' ? 'btn-close-white' : ''} me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
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

// Spinner functions (showSpinner, hideSpinner) are now REMOVED.

function toggleThemeAndSave() { 
    const newTheme = currentConfig.currentTheme === 'dark' ? 'light' : 'dark';
    saveConfig({ currentTheme: newTheme }); 
    applyConfigToUI(); 

    if (window.currentEmbeddingDataForPlot && typeof plotEmbeddings === 'function') {
         plotEmbeddings(
                window.currentEmbeddingDataForPlot,
                'embedding-plot-container',
                document.getElementById('color-by-select').value,
                document.getElementById('embedding-plot-type').value
            ).catch(console.error); 
    }
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
        if(isComplete && total > 0) { 
            progressBar.style.width = `100%`;
            progressBar.textContent = `Done`;
            setTimeout(() => {
                progressBarContainer.style.display = 'none';
                progressBar.style.width = `0%`; 
                progressBar.textContent = `0%`;
            }, 1500);
        } else { 
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