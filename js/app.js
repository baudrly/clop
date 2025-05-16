// js/app.js

let currentONNXSession = null;
let currentDnaTokenizer = new DNATokenizerClient();
let currentTextTokenizer = new TextTokenizerClient();
window.currentEmbeddingDataForPlot = null;
window.lastPlotLayout = null; 
window.lastPlotTraces = null; 
// window.parquetWasmModule is now set directly by the ESM script in index.html

async function initializeApp() {
    loadConfig(); 
    initEventListeners();
    populateDummyFastaSelect();

    if (window.parquetWasm && typeof window.parquetWasm.readParquet === 'function') {
        console.log("parquet-wasm API (readParquet) is available on window.parquetWasm.");
    } else {
        console.warn("parquet-wasm API not available on window.parquetWasm after ESM script execution. Parquet uploads might fail or use fallback initialization in parser.");
    }

    if (typeof Plotly === 'undefined') {
        console.error("Plotly.js is not loaded!");
        showToast("Plotting library (Plotly) failed to load. Visualizations will be affected.", "danger-ultra", 10000);
    }
    if (typeof sk === 'undefined') {
        console.warn("scikit-js (for t-SNE) not loaded. t-SNE will not be available.");
    }
    if (typeof UMAP === 'undefined') {
        console.warn("UMAP-JS not loaded. UMAP will not be available.");
    }
}

document.addEventListener('DOMContentLoaded', initializeApp);

function initEventListeners() {
    document.getElementById('theme-toggle-btn').addEventListener('click', toggleThemeAndSave);
    document.getElementById('settings-btn').addEventListener('click', () => { /* Modal toggled by Bootstrap */ });
    document.getElementById('save-settings-btn').addEventListener('click', handleSaveSettings);
    document.getElementById('reset-settings-btn').addEventListener('click', resetToDefaults);
    document.getElementById('data-clear-btn').addEventListener('click', handleClearAllData);

    // Embedding Tab Listeners
    document.getElementById('embedding-file-input').addEventListener('change', handleEmbeddingFileUpload);
    document.getElementById('load-dummy-embeddings-btn').addEventListener('click', loadDummyEmbeddings);
    document.getElementById('embedding-plot-type').addEventListener('change', rePlotEmbeddingsWithCurrentData);
    document.getElementById('color-by-select').addEventListener('change', rePlotEmbeddingsWithCurrentData);
    document.getElementById('export-plot-png-btn').addEventListener('click', () => exportCurrentPlot('png'));
    document.getElementById('export-plot-svg-btn').addEventListener('click', () => exportCurrentPlot('svg'));
    document.getElementById('reset-plot-view-btn').addEventListener('click', resetPlotView);

    // Zero-Shot Tab Listeners
    document.getElementById('fasta-file-input').addEventListener('change', handleFastaFileUpload);
    document.getElementById('fetch-fasta-url-btn').addEventListener('click', handleFastaUrlFetch);
    document.getElementById('load-dummy-fasta-btn').addEventListener('click', loadSelectedDummyFasta);
    document.getElementById('process-fasta-btn').addEventListener('click', processAndClassifyFasta);

    document.getElementById('onnx-model-input').addEventListener('change', handleONNXModelUpload);
    document.getElementById('dna-vocab-input').addEventListener('change', (e) => handleVocabUpload(e, 'dna'));
    document.getElementById('text-vocab-input').addEventListener('change', (e) => handleVocabUpload(e, 'text'));
    document.getElementById('load-dummy-vocabs-btn').addEventListener('click', loadDummyVocabularies);

    document.getElementById('setting-theme-select').addEventListener('change', (e) => {
        saveConfig({ currentTheme: e.target.value });
        applyConfigToUI(); 
        rePlotEmbeddingsWithCurrentData(); 
    });
}

function handleSaveSettings() {
    const newSettings = {
        currentTheme: document.getElementById('setting-theme-select').value,
        kmerSize: parseInt(document.getElementById('setting-kmer-size').value, 10),
        maxDnaLen: parseInt(document.getElementById('setting-max-dna-len').value, 10),
        maxTextLen: parseInt(document.getElementById('setting-max-text-len').value, 10),
        defaultBiotypes: document.getElementById('setting-default-biotypes').value,
        defaultSpecies: document.getElementById('setting-default-species').value,
        classificationTopN: parseInt(document.getElementById('setting-classification-top-n').value, 10),
        
        onnxInputNameDnaIds: document.getElementById('setting-onnx-dna-ids').value.trim(),
        onnxInputNameDnaMask: document.getElementById('setting-onnx-dna-mask').value.trim(),
        onnxOutputNameDnaEmbedding: document.getElementById('setting-onnx-dna-emb').value.trim(),
        onnxInputNameTextIds: document.getElementById('setting-onnx-text-ids').value.trim(),
        onnxInputNameTextMask: document.getElementById('setting-onnx-text-mask').value.trim(),
        onnxOutputNameTextEmbedding: document.getElementById('setting-onnx-text-emb').value.trim(),

        plotMarkerSize: parseInt(document.getElementById('setting-plot-marker-size').value, 10),
        plotMarkerOpacity: parseFloat(document.getElementById('setting-plot-marker-opacity').value),
        plotShowLegend: document.getElementById('setting-plot-show-legend').checked
    };
    saveConfig(newSettings);
    applyConfigToUI(); 
    rePlotEmbeddingsWithCurrentData(); 

    const settingsModalElement = document.getElementById('settingsModal');
    const settingsModal = bootstrap.Modal.getInstance(settingsModalElement);
    if (settingsModal) settingsModal.hide();
}

async function rePlotEmbeddingsWithCurrentData() { 
    if (window.currentEmbeddingDataForPlot) {
        try {
            // No spinner here
            await plotEmbeddings( 
                window.currentEmbeddingDataForPlot,
                'embedding-plot-container',
                document.getElementById('color-by-select').value,
                document.getElementById('embedding-plot-type').value
            );
        } catch (err) {
            console.error("Re-plotting error:", err);
            showToast("Error during re-plotting. Check console.", "danger-ultra");
            const plotContainer = document.getElementById('embedding-plot-container');
            if (plotContainer) {
                 plotContainer.innerHTML = `<div class="plot-placeholder"><i class="fas fa-exclamation-circle fa-3x text-danger-ultra mb-2"></i><p>Failed to render plot. ${err.message}</p></div>`;
            }
        }
    }
}

async function handleEmbeddingFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    // showSpinner('Parsing embedding file...'); // REMOVED
    const loadingIndicator = document.getElementById('embedding-loading-indicator'); // Optional: use a more local indicator
    if(loadingIndicator) loadingIndicator.style.display = 'block';


    const format = document.getElementById('embedding-format-select').value;

    try {
        let data;
        if (format === 'json') {
            const fileContent = await readFileAsText(file); data = JSON.parse(fileContent);
        } else if (format === 'csv') {
            const fileContent = await readFileAsText(file); data = parseCSVEmbeddings(fileContent);
        } else if (format === 'parquet') {
            const arrayBuffer = await readFileAsArrayBuffer(file); data = await parseParquetEmbeddings(arrayBuffer);
        } else { throw new Error("Unsupported format."); }
        
        if (!Array.isArray(data) || data.length === 0 || !data.every(item => Array.isArray(item.embeddings) && item.embeddings.length > 0)) {
            throw new Error("Data format error or embeddings missing/empty.");
        }
        window.currentEmbeddingDataForPlot = data;
        ensureSequenceMetrics(window.currentEmbeddingDataForPlot); 
        displayEmbeddingFileInfo(file, data.length, data[0].embeddings.length);
        await plotEmbeddings( 
            data,
            'embedding-plot-container',
            document.getElementById('color-by-select').value,
            document.getElementById('embedding-plot-type').value
        );
        showToast('Embeddings loaded!', 'success-ultra');
    } catch (error) {
        console.error("Embedding File Error:", error);
        showToast(`Embedding Error: ${error.message}`, 'danger-ultra', 7000);
        clearEmbeddingFileInfo();
    } finally {
        // hideSpinner(); // REMOVED
        if(loadingIndicator) loadingIndicator.style.display = 'none';
        event.target.value = null;
    }
}

async function loadDummyEmbeddings() {
    // showSpinner('Loading example embeddings...'); // REMOVED
    const loadingIndicator = document.getElementById('embedding-loading-indicator');
    if(loadingIndicator) loadingIndicator.style.display = 'block';
    try {
        const response = await fetch(dummyEmbeddingsPath);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        window.currentEmbeddingDataForPlot = data;
        ensureSequenceMetrics(window.currentEmbeddingDataForPlot);
        displayEmbeddingFileInfo({name: "Example Embeddings", size: 0}, data.length, data[0].embeddings.length); 
        await plotEmbeddings( 
            data,
            'embedding-plot-container',
            document.getElementById('color-by-select').value,
            document.getElementById('embedding-plot-type').value
        );
        showToast('Example embeddings loaded!', 'success-ultra');
    } catch (error) {
        showToast(`Load Dummy Error: ${error.message}`, 'danger-ultra');
        clearEmbeddingFileInfo();
    } finally { 
        // hideSpinner(); // REMOVED
        if(loadingIndicator) loadingIndicator.style.display = 'none';
    }
}

async function handleFastaFileUpload(event) { 
    const file = event.target.files[0];
    if (!file) return;
    // showSpinner('Reading FASTA...'); // REMOVED
    try {
        const fastaContent = await readFileAsText(file);
        document.getElementById('fasta-text-input').value = fastaContent;
        showToast('FASTA file loaded.', 'info');
    } catch (error) { showToast(`FASTA Read Error: ${error.message}`, 'danger-ultra');
    } finally { /* hideSpinner(); */ event.target.value = null; } // REMOVED
}
async function handleFastaUrlFetch() { 
    const urlInput = document.getElementById('fasta-url-input');
    const url = urlInput.value;
    if (!url) { showToast('Please enter a URL.', 'warning'); return; }
    // showSpinner('Fetching FASTA...'); // REMOVED
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const fastaContent = await response.text();
        document.getElementById('fasta-text-input').value = fastaContent;
        showToast('FASTA fetched successfully.', 'info');
    } catch (error) { showToast(`FASTA Fetch Error: ${error.message}. Check CORS.`, 'danger-ultra', 7000);
    } finally { /* hideSpinner(); */ } // REMOVED
}
function loadSelectedDummyFasta() { 
    const selectElement = document.getElementById('dummy-fasta-select');
    const idx = parseInt(selectElement.value, 10);
    const ex = getDummyFastaExamples()[idx];
    if (ex) {
        document.getElementById('fasta-text-input').value = ex.data;
        showToast(`Example FASTA "${ex.name}" loaded.`, 'info');
    }
}

async function handleONNXModelUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    // showSpinner('Loading ONNX model...'); // REMOVED
    try {
        const modelBuffer = await readFileAsArrayBuffer(file);
        currentONNXSession = await loadONNXModel(modelBuffer);
        showToast('ONNX model loaded.', 'success-ultra');
    } catch (error) { showToast(`ONNX Load Error: ${error.message}`, 'danger-ultra', 7000); currentONNXSession = null;
    } finally { /* hideSpinner(); */ event.target.value = null; } // REMOVED
}
async function handleVocabUpload(event, vocabType) { 
    const file = event.target.files[0];
    if (!file) return;
    // showSpinner(`Loading ${vocabType} vocabulary...`); // REMOVED
    try {
        const vocabContent = await readFileAsText(file);
        const vocabData = JSON.parse(vocabContent);
        if (Object.keys(vocabData).length === 0) throw new Error("Vocab empty.");
        if (vocabType === 'dna') {
            currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'), vocabData);
            showToast('DNA vocabulary loaded.', 'success-ultra');
        } else if (vocabType === 'text') {
            currentTextTokenizer = new TextTokenizerClient(vocabData);
            showToast('Text vocabulary loaded.', 'success-ultra');
        }
    } catch (error) { showToast(`Vocab Load Error (${vocabType}): ${error.message}`, 'danger-ultra', 7000);
    } finally { /* hideSpinner(); */ event.target.value = null; } // REMOVED
}
async function loadDummyVocabularies() { 
    // showSpinner('Loading example vocabularies...'); // REMOVED
    try {
        const [dnaRes, textRes] = await Promise.all([fetch(dummyDnaVocabPath), fetch(dummyTextVocabPath)]);
        if (!dnaRes.ok) throw new Error(`DNA Vocab HTTP ${dnaRes.status}`);
        if (!textRes.ok) throw new Error(`Text Vocab HTTP ${textRes.status}`);
        const dnaVocabData = await dnaRes.json(); const textVocabData = await textRes.json();
        currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'), dnaVocabData); 
        currentTextTokenizer = new TextTokenizerClient(textVocabData);
        showToast(`Example vocabs (DNA k=${getConfigValue('kmerSize')}) loaded.`, 'success-ultra');
    } catch (error) { showToast(`Dummy Vocab Error: ${error.message}`, 'danger-ultra');
    } finally { /* hideSpinner(); */ } // REMOVED
}

async function processAndClassifyFasta() {
    const fastaText = document.getElementById('fasta-text-input').value;
    if (!fastaText.trim()) { showToast('FASTA input is empty.', 'warning'); return; }

    // showSpinner('Parsing FASTA...'); await delay(50); // REMOVED
    document.getElementById('process-fasta-btn').disabled = true; // Disable button during processing
    await delay(50); // Allow UI to update (button disable)

    const parsedSequences = parseFasta(fastaText);
    if (parsedSequences.length === 0) { 
        // hideSpinner(); // REMOVED
        document.getElementById('process-fasta-btn').disabled = false;
        showToast('No valid FASTA sequences found.', 'warning'); return; 
    }

    const biotypeLabels = document.getElementById('biotype-labels-input').value.split(',').map(s => s.trim()).filter(s => s);
    const speciesLabels = document.getElementById('species-labels-input').value.split(',').map(s => s.trim()).filter(s => s);
    const customPrompts = document.getElementById('custom-prompts-input').value.split('\n').map(s => s.trim()).filter(s => s);
    
    if (currentDnaTokenizer.k !== getConfigValue('kmerSize')) { 
        currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'), currentDnaTokenizer.vocab);
        showToast(`DNA K-mer size set to ${getConfigValue('kmerSize')}.`, 'info', 2000);
    }

    const resultsContainer = document.getElementById('classification-results-container');
    resultsContainer.innerHTML = '';
    const tokenizationPreviewContainer = document.getElementById('tokenization-preview-container');
    tokenizationPreviewContainer.innerHTML = '';
    updateTotalSequencesClassified(0);

    // showSpinner(`Classifying ${parsedSequences.length} sequences...`); // REMOVED
    let sequencesProcessed = 0;
    updateClassificationProgress(0, parsedSequences.length);

    try { 
        for (const seqData of parsedSequences) {
            try {
                const classificationPayload = { biotype: [], species: [], custom: [] };
                let firstDnaTokenData = null, firstTextTokenData = null;
                const classificationConfig = { ...currentConfig }; 

                if (biotypeLabels.length > 0) {
                    const r = await zeroShotClassifySingle(seqData.sequence, biotypeLabels, currentONNXSession, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                    classificationPayload.biotype = r.results; if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData; if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                }
                if (speciesLabels.length > 0) {
                    const r = await zeroShotClassifySingle(seqData.sequence, speciesLabels, currentONNXSession, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                    classificationPayload.species = r.results; if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData; if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                }
                if (customPrompts.length > 0) {
                    const r = await zeroShotClassifySingle(seqData.sequence, customPrompts, currentONNXSession, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                    classificationPayload.custom = r.results; if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData; if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                }

                displaySingleSequenceClassification(seqData, classificationPayload, resultsContainer);

                if (sequencesProcessed === 0) { 
                    if(firstDnaTokenData) displayTokenizationPreview(firstDnaTokenData.type, firstDnaTokenData.input, firstDnaTokenData.tokens, firstDnaTokenData.ids, tokenizationPreviewContainer);
                    if(firstTextTokenData) {
                        const textTokenDiv = document.createElement('div'); textTokenDiv.className = 'mt-2'; 
                        displayTokenizationPreview(firstTextTokenData.type, firstTextTokenData.input, firstTextTokenData.tokens, firstTextTokenData.ids, textTokenDiv);
                        tokenizationPreviewContainer.appendChild(textTokenDiv);
                    }
                }
                sequencesProcessed++;
                updateClassificationProgress(sequencesProcessed, parsedSequences.length);
                if(sequencesProcessed % 5 === 0 && parsedSequences.length > 10) await delay(10); 

            } catch (error) {
                console.error(`Classification Error (Seq ${seqData.header}):`, error);
                resultsContainer.insertAdjacentHTML('beforeend', `<div class="sequence-result-card-enhanced text-danger-ultra p-2 small"><strong>${seqData.header}</strong>: Failed. ${error.message.substring(0,100)}</div>`);
            }
        }
    } finally {
        // hideSpinner(); // REMOVED
        document.getElementById('process-fasta-btn').disabled = false; // Re-enable button
    }

    updateTotalSequencesClassified(sequencesProcessed);
    updateClassificationProgress(sequencesProcessed, parsedSequences.length, true); 

    if (sequencesProcessed > 0) showToast(`Classification complete for ${sequencesProcessed} seqs.`, 'success-ultra');
    else showToast('No sequences processed. Check input/console.', 'danger-ultra');
    if (resultsContainer.innerHTML.trim() === '') resultsContainer.innerHTML = '<div class="plot-placeholder"><i class="fas fa-folder-open fa-3x text-placeholder mb-2"></i><p>No results to display. Try classifying some sequences.</p></div>';
}

function exportCurrentPlot(format = 'png') {
    const plotContainer = document.getElementById('embedding-plot-container');
    if (!plotContainer || !plotContainer.children.length || !window.Plotly) {
        showToast('No plot to export.', 'warning'); return;
    }
    const plotType = document.getElementById('embedding-plot-type').value || 'plot';
    const colorBy = document.getElementById('color-by-select').value || 'data';
    const filename = `clop_embedding_${plotType}_${colorBy}.${format}`;

    Plotly.downloadImage(plotContainer, {format: format, width: 1200, height: 900, filename: filename})
        .then(() => showToast(`Plot exported as ${format.toUpperCase()}.`, 'success-ultra'))
        .catch(err => { showToast('Plot Export Error. See console.', 'danger-ultra'); console.error('Plotly export error:', err); });
}

function resetPlotView() {
    const plotContainer = document.getElementById('embedding-plot-container');
    if (window.lastPlotLayout && window.lastPlotTraces && plotContainer && Plotly) {
        Plotly.react(plotContainer, window.lastPlotTraces, window.lastPlotLayout); 
        showToast('Plot view reset.', 'info');
    } else if (window.currentEmbeddingDataForPlot) {
        rePlotEmbeddingsWithCurrentData(); 
    } else {
        showToast('No plot data to reset view for.', 'warning');
    }
}

function handleClearAllData() {
    if (!confirm("Are you sure you want to clear all loaded data (embeddings, models, vocabularies) and FASTA input? This cannot be undone.")) return;

    clearEmbeddingFileInfo(); 

    document.getElementById('fasta-text-input').value = '';
    document.getElementById('classification-results-container').innerHTML = '<div class="plot-placeholder"><i class="fas fa-tags fa-3x text-placeholder mb-2"></i><p>Classification results appear here.</p></div>';
    document.getElementById('tokenization-preview-container').innerHTML = '';
    updateTotalSequencesClassified(0);

    document.getElementById('embedding-file-input').value = null;
    document.getElementById('fasta-file-input').value = null;
    document.getElementById('onnx-model-input').value = null;
    document.getElementById('dna-vocab-input').value = null;
    document.getElementById('text-vocab-input').value = null;

    currentONNXSession = null;
    currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize')); 
    currentTextTokenizer = new TextTokenizerClient();
    
    showToast('All loaded data has been cleared.', 'success-ultra');
}