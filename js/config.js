// js/config.js

const DEFAULT_CONFIG = {
    // General
    currentTheme: 'light', // 'light' or 'dark'

    // Zero-Shot Classification
    kmerSize: 6,
    maxDnaLen: 128,
    maxTextLen: 64,
    defaultBiotypes: "gene, protein_coding, lncRNA, enhancer, promoter, tRNA, rRNA, repeat, pseudogene",
    defaultSpecies: "Homo sapiens, Mus musculus, Danio rerio, Drosophila melanogaster, Saccharomyces cerevisiae, Arabidopsis thaliana",
    classificationTopN: 5, // Number of top results to show
    onnxInputNameDnaIds: "dna_input_ids",
    onnxInputNameDnaMask: "dna_attention_mask",
    onnxOutputNameDnaEmbedding: "dna_embedding",
    onnxInputNameTextIds: "text_input_ids",
    onnxInputNameTextMask: "text_attention_mask",
    onnxOutputNameTextEmbedding: "text_embedding",

    // Embedding Plot
    plotMarkerSize: 8,
    plotMarkerOpacity: 0.75,
    plotDefaultColorBy: "biotype",
    plotDefaultReduction: "pca",
    plotShowLegend: true,
    plotColorScheme: "Plotly" // e.g., "Plotly", "D3", "Category10" - placeholder, actual palette chosen in viz
};

let currentConfig = { ...DEFAULT_CONFIG };

function loadConfig() {
    const savedConfig = localStorage.getItem('bioClipExplorerProConfig'); // Changed key for new version
    if (savedConfig) {
        try {
            const parsedConfig = JSON.parse(savedConfig);
            currentConfig = { ...DEFAULT_CONFIG, ...parsedConfig };
        } catch (e) {
            console.error("Error parsing saved config, using defaults.", e);
            currentConfig = { ...DEFAULT_CONFIG };
        }
    } else {
        currentConfig = { ...DEFAULT_CONFIG };
    }
    applyConfigToUI(); // Apply loaded/default config to all relevant UI elements
    return currentConfig;
}

function saveConfig(newConfigPartial) {
    currentConfig = { ...currentConfig, ...newConfigPartial };
    try {
        localStorage.setItem('bioClipExplorerProConfig', JSON.stringify(currentConfig));
        showToast('Settings saved successfully!', 'success', 2000);
    } catch (e) {
        console.error("Error saving config to localStorage.", e);
        showToast('Failed to save settings. Storage might be full.', 'danger');
    }
}

function applyConfigToUI() {
    // Theme (handled separately by toggleThemeAndSave or initial load)
    document.body.classList.toggle('dark-theme', currentConfig.currentTheme === 'dark');
    const themeBtn = document.getElementById('theme-toggle-btn');
    if(themeBtn) themeBtn.innerHTML = currentConfig.currentTheme === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';


    // Settings Modal Inputs
    document.getElementById('setting-theme-select').value = currentConfig.currentTheme;
    document.getElementById('setting-kmer-size').value = currentConfig.kmerSize;
    document.getElementById('setting-max-dna-len').value = currentConfig.maxDnaLen;
    document.getElementById('setting-max-text-len').value = currentConfig.maxTextLen;
    document.getElementById('setting-default-biotypes').value = currentConfig.defaultBiotypes;
    document.getElementById('setting-default-species').value = currentConfig.defaultSpecies;
    document.getElementById('setting-classification-top-n').value = currentConfig.classificationTopN;

    document.getElementById('setting-onnx-dna-ids').value = currentConfig.onnxInputNameDnaIds;
    document.getElementById('setting-onnx-dna-mask').value = currentConfig.onnxInputNameDnaMask;
    document.getElementById('setting-onnx-dna-emb').value = currentConfig.onnxOutputNameDnaEmbedding;
    document.getElementById('setting-onnx-text-ids').value = currentConfig.onnxInputNameTextIds;
    document.getElementById('setting-onnx-text-mask').value = currentConfig.onnxInputNameTextMask;
    document.getElementById('setting-onnx-text-emb').value = currentConfig.onnxOutputNameTextEmbedding;

    document.getElementById('setting-plot-marker-size').value = currentConfig.plotMarkerSize;
    document.getElementById('setting-plot-marker-opacity').value = currentConfig.plotMarkerOpacity;
    document.getElementById('setting-plot-show-legend').checked = currentConfig.plotShowLegend;


    // Main UI elements that should reflect config (e.g., on page load)
    // Kmer size input in the classification section
    const kmerSizeInputMain = document.getElementById('kmer-size-input');
    if (kmerSizeInputMain) kmerSizeInputMain.value = currentConfig.kmerSize;

    const biotypeLabelsInputMain = document.getElementById('biotype-labels-input');
    if(biotypeLabelsInputMain) biotypeLabelsInputMain.value = currentConfig.defaultBiotypes;

    const speciesLabelsInputMain = document.getElementById('species-labels-input');
    if(speciesLabelsInputMain) speciesLabelsInputMain.value = currentConfig.defaultSpecies;

    // Embedding plot defaults (these influence dropdowns if they exist)
    const colorBySelectMain = document.getElementById('color-by-select');
    if (colorBySelectMain) colorBySelectMain.value = currentConfig.plotDefaultColorBy;
    const plotTypeSelectMain = document.getElementById('embedding-plot-type');
    if (plotTypeSelectMain) plotTypeSelectMain.value = currentConfig.plotDefaultReduction;
}

function resetToDefaults() {
    if (confirm("Are you sure you want to reset all settings to their default values?")) {
        currentConfig = { ...DEFAULT_CONFIG }; // Deep copy defaults
        saveConfig({}); // Save the reset state (effectively overwriting with defaults)
        applyConfigToUI();
        showToast('Settings have been reset to defaults.', 'info');
    }
}

// Helper to get a specific config value, ensuring it's valid or default
function getConfigValue(key) {
    return currentConfig[key] !== undefined ? currentConfig[key] : DEFAULT_CONFIG[key];
}