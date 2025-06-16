// js/app.js

let dnaOnnxSession = null;
let textOnnxSession = null;
let currentDnaTokenizer = new DNATokenizerClient();
let currentTextTokenizer = new TextTokenizerClient();
window.currentEmbeddingDataForPlot = null;
window.lastPlotLayout = null;
window.lastPlotTraces = null;

async function initializeApp() {
    loadConfig();
    initEventListeners();
    populateDummyFastaSelect();

    // Add a small delay to ensure module scripts have completed
    await new Promise(resolve => setTimeout(resolve, 100));

    if (window.parquetWasmModule && typeof window.parquetWasmModule.readParquet === 'function') {
        console.log("parquet-wasm API (readParquet) is available on window.parquetWasmModule.");
    } else {
        console.warn("parquet-wasm API not available on window.parquetWasmModule after ESM script execution. Parquet uploads might fail.");
    }

    if (typeof Plotly === 'undefined') {
        console.error("Plotly.js is not loaded!");
        showToast("Plotting library (Plotly) failed to load. Visualizations will be affected.", "danger-ultra", 10000);
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

    // Updated model and vocab listeners
    document.getElementById('dna-model-input').addEventListener('change', (e) => handleONNXModelUpload(e, 'dna'));
    document.getElementById('text-model-input').addEventListener('change', (e) => handleONNXModelUpload(e, 'text'));
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

/**
 * Extracts unique biotypes and species from embedding data and populates the
 * corresponding textareas in the UI.
 * @param {Array<Object>} embeddingData The array of loaded embedding objects.
 */
function updateClassificationLabelsFromData(embeddingData) {
    if (!embeddingData || embeddingData.length === 0) return;

    const biotypes = [...new Set(embeddingData.map(d => d.biotype).filter(b => b && b !== 'Unknown'))].sort();
    const species = [...new Set(embeddingData.map(d => d.species).filter(s => s && s !== 'Unknown'))].sort();

    const biotypeInput = document.getElementById('biotype-labels-input');
    if (biotypeInput && biotypes.length > 0) {
        biotypeInput.value = biotypes.join(', ');
        showToast(`Updated biotype labels from loaded data.`, 'info', 2500);
    }

    const speciesInput = document.getElementById('species-labels-input');
    if (speciesInput && species.length > 0) {
        speciesInput.value = species.join(', ');
         showToast(`Updated species labels from loaded data.`, 'info', 2500);
    }
}

async function handleEmbeddingFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const loadingIndicator = document.getElementById('embedding-loading-indicator');
    const fileInput = document.getElementById('embedding-file-input');

    // Show loading state
    if(loadingIndicator) loadingIndicator.style.display = 'block';
    fileInput.classList.add('border-warning');

    // Add loading visual cue
    const fileInputParent = fileInput.parentElement;
    const loadingBadge = document.createElement('span');
    loadingBadge.className = 'badge bg-warning ms-2';
    loadingBadge.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing...';
    loadingBadge.id = 'upload-loading-badge';
    fileInputParent.appendChild(loadingBadge);

    const format = document.getElementById('embedding-format-select').value;

    try {
        let data;
        if (format === 'json') {
            const fileContent = await readFileAsText(file);
            data = JSON.parse(fileContent);
        } else if (format === 'csv') {
            const fileContent = await readFileAsText(file);
            data = parseCSVEmbeddings(fileContent);
        } else if (format === 'parquet') {
            const arrayBuffer = await readFileAsArrayBuffer(file);
            data = await parseParquetEmbeddings(arrayBuffer);
        } else {
            throw new Error("Unsupported format.");
        }

        if (!Array.isArray(data) || data.length === 0 || !data.every(item => Array.isArray(item.embeddings) && item.embeddings.length > 0)) {
            throw new Error("Data format error or embeddings missing/empty.");
        }

        window.currentEmbeddingDataForPlot = data;
        ensureSequenceMetrics(window.currentEmbeddingDataForPlot);
        displayEmbeddingFileInfo(file, data.length, data[0].embeddings.length);

        updateEmbeddingClassificationIndicator(data);
        updateClassificationLabelsFromData(data);

        await plotEmbeddings(
            data,
            'embedding-plot-container',
            document.getElementById('color-by-select').value,
            document.getElementById('embedding-plot-type').value
        );

        // Success visual cues
        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-success');
        showToast('Embeddings loaded successfully!', 'success-ultra');

    } catch (error) {
        console.error("Embedding File Error:", error);
        showToast(`Embedding Error: ${error.message}`, 'danger-ultra', 7000);
        clearEmbeddingFileInfo();
        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-danger');
        setTimeout(() => fileInput.classList.remove('border-danger'), 3000);
    } finally {
        if(loadingIndicator) loadingIndicator.style.display = 'none';
        const loadingBadgeToRemove = document.getElementById('upload-loading-badge');
        if (loadingBadgeToRemove) loadingBadgeToRemove.remove();
        fileInput.classList.remove('border-warning');
        event.target.value = null;
    }
}

function updateEmbeddingClassificationIndicator(embeddingData) {
    const indicator = document.getElementById('embedding-classification-indicator');
    const info = document.getElementById('embedding-db-info');

    if (!indicator || !info) return;

    if (!embeddingData || embeddingData.length === 0 || !(embeddingData[0].dna_embeddings || embeddingData[0].embeddings)) {
        indicator.style.display = 'none';
        return;
    }

    // Count unique biotypes and species
    const biotypes = new Set(embeddingData.map(d => d.biotype).filter(b => b && b !== 'Unknown'));
    const species = new Set(embeddingData.map(d => d.species).filter(s => s && s !== 'Unknown'));

    info.innerHTML = `${embeddingData.length} sequences with ${biotypes.size} biotypes, ${species.size} species`;
    indicator.style.display = 'block';
}

async function loadDummyEmbeddings() {
    const loadingIndicator = document.getElementById('embedding-loading-indicator');
    if(loadingIndicator) loadingIndicator.style.display = 'block';
    try {
        const response = await fetch(dummyEmbeddingsPath);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        window.currentEmbeddingDataForPlot = data;
        ensureSequenceMetrics(window.currentEmbeddingDataForPlot);
        displayEmbeddingFileInfo({name: "Example Embeddings", size: 0}, data.length, data[0].embeddings.length);

        updateEmbeddingClassificationIndicator(data);
        updateClassificationLabelsFromData(data);

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
        if(loadingIndicator) loadingIndicator.style.display = 'none';
    }
}

async function handleFastaFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const fileInput = event.target;
    fileInput.classList.add('border-warning');

    const fileInputParent = fileInput.parentElement;
    const loadingBadge = document.createElement('span');
    loadingBadge.className = 'badge bg-warning ms-2';
    loadingBadge.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    loadingBadge.id = 'fasta-loading-badge';
    fileInputParent.appendChild(loadingBadge);

    try {
        const fastaContent = await readFileAsText(file);
        document.getElementById('fasta-text-input').value = fastaContent;

        // Success visual cues
        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-success');
        showToast(`FASTA file loaded successfully: ${file.name}`, 'success');
        setTimeout(() => fileInput.classList.remove('border-success'), 3000);

    } catch (error) {
        showToast(`FASTA Read Error: ${error.message}`, 'danger-ultra');
        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-danger');
        setTimeout(() => fileInput.classList.remove('border-danger'), 3000);
    } finally {
        const loadingBadgeToRemove = document.getElementById('fasta-loading-badge');
        if (loadingBadgeToRemove) loadingBadgeToRemove.remove();
        fileInput.classList.remove('border-warning');
        event.target.value = null;
    }
}

async function handleFastaUrlFetch() {
    const urlInput = document.getElementById('fasta-url-input');
    const url = urlInput.value;
    if (!url) { showToast('Please enter a URL.', 'warning'); return; }
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const fastaContent = await response.text();
        document.getElementById('fasta-text-input').value = fastaContent;
        showToast('FASTA fetched successfully.', 'info');
    } catch (error) { showToast(`FASTA Fetch Error: ${error.message}. Check CORS.`, 'danger-ultra', 7000);
    }
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

async function handleONNXModelUpload(event, modelType) {
    const file = event.target.files[0];
    if (!file) return;

    const fileInput = event.target;
    fileInput.classList.add('border-warning');

    try {
        const modelBuffer = await readFileAsArrayBuffer(file);
        const session = await ort.InferenceSession.create(modelBuffer);

        if (modelType === 'dna') {
            dnaOnnxSession = session;
        } else {
            textOnnxSession = session;
        }

        // Add a success badge
        const parent = fileInput.parentElement;
        // Remove old badge if it exists
        const oldBadge = parent.querySelector('.badge');
        if (oldBadge) oldBadge.remove();

        const successBadge = document.createElement('span');
        successBadge.className = 'badge bg-success ms-2';
        successBadge.innerHTML = `<i class="fas fa-check me-1"></i>${modelType.toUpperCase()} Model Loaded`;
        parent.appendChild(successBadge);

        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-success');
        setTimeout(() => fileInput.classList.remove('border-success'), 3000);

        showToast(`${modelType.toUpperCase()} ONNX model loaded successfully.`, 'success-ultra');
    } catch (error) {
        showToast(`ONNX Load Error (${modelType}): ${error.message}`, 'danger-ultra', 7000);
        if (modelType === 'dna') dnaOnnxSession = null;
        else textOnnxSession = null;

        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-danger');
    } finally {
        event.target.value = null;
    }
}

async function handleVocabUpload(event, vocabType) {
    const file = event.target.files[0];
    if (!file) return;

    const fileInput = event.target;
    fileInput.classList.add('border-warning');

    // Add loading badge
    const fileInputParent = fileInput.parentElement;
    const loadingBadge = document.createElement('span');
    loadingBadge.className = 'badge bg-warning ms-2';
    loadingBadge.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Loading...';
    loadingBadge.id = `${vocabType}-vocab-loading-badge`;
    fileInputParent.appendChild(loadingBadge);

    try {
        const vocabContent = await readFileAsText(file);
        const vocabData = JSON.parse(vocabContent);

        // Check if this is a vocab file from clop.py (has 'vocab' and possibly 'k' keys)
        let actualVocab = vocabData;
        let kmerSize = null;

        if (vocabData.vocab && typeof vocabData.vocab === 'object') {
            actualVocab = vocabData.vocab;
            if (vocabData.k) {
                kmerSize = vocabData.k;
            }
        }

        if (Object.keys(actualVocab).length === 0) throw new Error("Vocab empty.");

        if (vocabType === 'dna') {
            currentDnaTokenizer = new DNATokenizerClient(kmerSize || getConfigValue('kmerSize'), actualVocab);
            if (kmerSize) {
                // Update k-mer size in config if it was in the vocab file
                saveConfig({ kmerSize: kmerSize });
                applyConfigToUI();
            }

            // Success visual cues
            fileInput.classList.remove('border-warning');
            fileInput.classList.add('border-success');

            // Add success badge
            const successBadge = document.createElement('span');
            successBadge.className = 'badge bg-success ms-2';
            successBadge.innerHTML = `<i class="fas fa-check me-1"></i>DNA Vocab (${Object.keys(actualVocab).length} tokens${kmerSize ? `, k=${kmerSize}` : ''})`;
            successBadge.id = 'dna-vocab-success-badge';
            fileInputParent.appendChild(successBadge);

            showToast(`DNA vocabulary loaded (${Object.keys(actualVocab).length} tokens${kmerSize ? `, k=${kmerSize}` : ''}).`, 'success-ultra');

        } else if (vocabType === 'text') {
            currentTextTokenizer = new TextTokenizerClient(actualVocab);

            // Success visual cues
            fileInput.classList.remove('border-warning');
            fileInput.classList.add('border-success');

            // Add success badge
            const successBadge = document.createElement('span');
            successBadge.className = 'badge bg-success ms-2';
            successBadge.innerHTML = `<i class="fas fa-check me-1"></i>Text Vocab (${Object.keys(actualVocab).length} tokens)`;
            successBadge.id = 'text-vocab-success-badge';
            fileInputParent.appendChild(successBadge);

            showToast(`Text vocabulary loaded (${Object.keys(actualVocab).length} tokens).`, 'success-ultra');
        }

        setTimeout(() => fileInput.classList.remove('border-success'), 3000);

    } catch (error) {
        showToast(`Vocab Load Error (${vocabType}): ${error.message}`, 'danger-ultra', 7000);
        fileInput.classList.remove('border-warning');
        fileInput.classList.add('border-danger');
        setTimeout(() => fileInput.classList.remove('border-danger'), 3000);
    } finally {
        const loadingBadgeToRemove = document.getElementById(`${vocabType}-vocab-loading-badge`);
        if (loadingBadgeToRemove) loadingBadgeToRemove.remove();
        fileInput.classList.remove('border-warning');
        event.target.value = null;
    }
}

async function loadDummyVocabularies() {
    const dnaInput = document.getElementById('dna-vocab-input');
    const textInput = document.getElementById('text-vocab-input');

    // Add loading visual cues
    dnaInput.classList.add('border-warning');
    textInput.classList.add('border-warning');

    try {
        const [dnaRes, textRes] = await Promise.all([fetch(dummyDnaVocabPath), fetch(dummyTextVocabPath)]);
        if (!dnaRes.ok) throw new Error(`DNA Vocab HTTP ${dnaRes.status}`);
        if (!textRes.ok) throw new Error(`Text Vocab HTTP ${textRes.status}`);
        const dnaVocabData = await dnaRes.json();
        const textVocabData = await textRes.json();
        currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'), dnaVocabData);
        currentTextTokenizer = new TextTokenizerClient(textVocabData);

        // Success visual cues
        dnaInput.classList.remove('border-warning');
        dnaInput.classList.add('border-success');
        textInput.classList.remove('border-warning');
        textInput.classList.add('border-success');

        // Add success badges
        const dnaParent = dnaInput.parentElement;
        const textParent = textInput.parentElement;

        // Remove any existing badges
        const existingDnaBadge = document.getElementById('dna-vocab-success-badge');
        const existingTextBadge = document.getElementById('text-vocab-success-badge');
        if (existingDnaBadge) existingDnaBadge.remove();
        if (existingTextBadge) existingTextBadge.remove();

        const dnaSuccessBadge = document.createElement('span');
        dnaSuccessBadge.className = 'badge bg-success ms-2';
        dnaSuccessBadge.innerHTML = `<i class="fas fa-check me-1"></i>Example DNA Vocab (k=${getConfigValue('kmerSize')})`;
        dnaSuccessBadge.id = 'dna-vocab-success-badge';
        dnaParent.appendChild(dnaSuccessBadge);

        const textSuccessBadge = document.createElement('span');
        textSuccessBadge.className = 'badge bg-success ms-2';
        textSuccessBadge.innerHTML = `<i class="fas fa-check me-1"></i>Example Text Vocab`;
        textSuccessBadge.id = 'text-vocab-success-badge';
        textParent.appendChild(textSuccessBadge);

        showToast(`Example vocabs (DNA k=${getConfigValue('kmerSize')}) loaded.`, 'success-ultra');

        setTimeout(() => {
            dnaInput.classList.remove('border-success');
            textInput.classList.remove('border-success');
        }, 3000);

    } catch (error) {
        showToast(`Dummy Vocab Error: ${error.message}`, 'danger-ultra');
        dnaInput.classList.remove('border-warning');
        dnaInput.classList.add('border-danger');
        textInput.classList.remove('border-warning');
        textInput.classList.add('border-danger');
        setTimeout(() => {
            dnaInput.classList.remove('border-danger');
            textInput.classList.remove('border-danger');
        }, 3000);
    }
}
async function processAndClassifyFasta() {
    const fastaText = document.getElementById('fasta-text-input').value;
    if (!fastaText.trim()) { showToast('FASTA input is empty.', 'warning'); return; }

    document.getElementById('process-fasta-btn').disabled = true;
    await delay(50);

    const parsedSequences = parseFasta(fastaText);
    if (parsedSequences.length === 0) {
        document.getElementById('process-fasta-btn').disabled = false;
        showToast('No valid FASTA sequences found.', 'warning'); return;
    }

    // Check if we have embeddings loaded for similarity-based classification
    const useEmbeddingClassification = window.currentEmbeddingDataForPlot &&
                                     window.currentEmbeddingDataForPlot.length > 0 &&
                                     window.currentEmbeddingDataForPlot.some(d => d.dna_embeddings && d.dna_embeddings.length > 0);

    // Get the custom text labels from the input fields
    const biotypeLabels = document.getElementById('biotype-labels-input').value.split(',').map(s => s.trim()).filter(s => s);
    const speciesLabels = document.getElementById('species-labels-input').value.split(',').map(s => s.trim()).filter(s => s);
    const customPrompts = document.getElementById('custom-prompts-input').value.split('\n').map(s => s.trim()).filter(s => s);

    // For embedding-based classification, we don't need labels
    if (!useEmbeddingClassification && biotypeLabels.length === 0 && speciesLabels.length === 0 && customPrompts.length === 0) {
        document.getElementById('process-fasta-btn').disabled = false;
        showToast('Please provide classification labels OR load an embedding database for similarity-based classification.', 'warning', 7000);
        return;
    }

    // Check for both models if using zero-shot ONNX classification
    if (!useEmbeddingClassification && (!dnaOnnxSession || !textOnnxSession)) {
        document.getElementById('process-fasta-btn').disabled = false;
        showToast('Please upload both a DNA and a Text ONNX model before classifying.', 'warning');
        return;
    }

    if (currentDnaTokenizer.k !== getConfigValue('kmerSize')) {
        currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'), currentDnaTokenizer.vocab);
        showToast(`DNA K-mer size set to ${getConfigValue('kmerSize')}.`, 'info', 2000);
    }

    const resultsContainer = document.getElementById('classification-results-container');
    resultsContainer.innerHTML = '';
    const tokenizationPreviewContainer = document.getElementById('tokenization-preview-container');
    tokenizationPreviewContainer.innerHTML = '';
    updateTotalSequencesClassified(0);

    let sequencesProcessed = 0;
    updateClassificationProgress(0, parsedSequences.length);

    try {
        for (const seqData of parsedSequences) {
            try {
                let classificationPayload = { biotype: [], species: [], custom: [], similar: [] };
                let firstDnaTokenData = null;

                if (useEmbeddingClassification) {
                    // Get embedding for the query sequence using the loaded embedding database
                    let queryEmbedding;

                    if (dnaOnnxSession) { // Use the correct session variable
                        const dnaEncoded = currentDnaTokenizer.encode(seqData.sequence, getConfigValue('maxDnaLen'));
                        const dnaInputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(dnaEncoded.ids.map(BigInt)), [1, dnaEncoded.ids.length]);
                        const dnaLengthsTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(dnaEncoded.length)]), [1]);

                        // Hardcode the input names to match the model export
                        const dnaFeeds = {
                            'dna_tokens': dnaInputIdsTensor,
                            'dna_lengths': dnaLengthsTensor
                        };

                        try {
                            const dnaResults = await dnaOnnxSession.run(dnaFeeds);
                            queryEmbedding = Array.from(dnaResults['dna_embedding'].data);
                        } catch (e) {
                            console.error("ONNX DNA encoding error:", e);
                            queryEmbedding = null;
                        }

                        firstDnaTokenData = {
                            type: 'DNA',
                            input: dnaEncoded.original_input,
                            tokens: dnaEncoded.original_tokens,
                            ids: dnaEncoded.ids.slice(0, dnaEncoded.length)
                        };
                    }

                    // If no ONNX model or ONNX failed, use the embedding database for similarity
                    if (!queryEmbedding) {
                        // Find the most similar sequence using cosine similarity of k-mer frequencies
                        const querySeq = seqData.sequence.toUpperCase();
                        let bestMatch = null;
                        let bestScore = -1;

                        const kmerSize = currentDnaTokenizer.k || getConfigValue('kmerSize') || 4;

                        console.log(`Finding best sequence match using k-mer cosine similarity (k=${kmerSize})`);

                        for (const dbEntry of window.currentEmbeddingDataForPlot) {
                            if (dbEntry.sequence || dbEntry.sequence_snippet) {
                                const dbSeq = (dbEntry.sequence || dbEntry.sequence_snippet).toUpperCase();
                                // Use cosine similarity based on k-mer frequencies
                                const similarity = calculateSequenceCosineSimilarity(querySeq, dbSeq, kmerSize);
                                if (similarity > bestScore) {
                                    bestScore = similarity;
                                    bestMatch = dbEntry;
                                }
                            }
                        }

                        if (bestMatch && (bestMatch.dna_embeddings || bestMatch.embeddings)) {
                            queryEmbedding = bestMatch.dna_embeddings || bestMatch.embeddings;
                            console.log(`Using embedding from most similar sequence (${bestScore.toFixed(4)} k-mer cosine similarity): ${bestMatch.id}`);
                        } else {
                            // Fallback: create a weighted average embedding based on k-mer similarities
                            const allValidEntries = window.currentEmbeddingDataForPlot.filter(entry =>
                                (entry.dna_embeddings || entry.embeddings) &&
                                (entry.sequence || entry.sequence_snippet)
                            );

                            if (allValidEntries.length > 0) {
                                const similarities = allValidEntries.map(entry => {
                                    const dbSeq = (entry.sequence || entry.sequence_snippet).toUpperCase();
                                    const similarity = calculateSequenceCosineSimilarity(querySeq, dbSeq, kmerSize);
                                    return { entry, similarity };
                                });

                                // Sort by similarity and take top 5 for weighted average
                                const topSimilar = similarities
                                    .sort((a, b) => b.similarity - a.similarity)
                                    .slice(0, 5)
                                    .filter(item => item.similarity > 0);

                                if (topSimilar.length > 0) {
                                    const firstEmbedding = topSimilar[0].entry.dna_embeddings || topSimilar[0].entry.embeddings;
                                    const embeddingDim = firstEmbedding.length;
                                    queryEmbedding = Array(embeddingDim).fill(0);

                                    let totalWeight = 0;
                                    for (const { entry, similarity } of topSimilar) {
                                        const embedding = entry.dna_embeddings || entry.embeddings;
                                        if (embedding && embedding.length === embeddingDim) {
                                            for (let i = 0; i < embeddingDim; i++) {
                                                queryEmbedding[i] += embedding[i] * similarity;
                                            }
                                            totalWeight += similarity;
                                        }
                                    }

                                    if (totalWeight > 0) {
                                        for (let i = 0; i < embeddingDim; i++) {
                                            queryEmbedding[i] /= totalWeight;
                                        }
                                        console.log(`Using weighted average embedding from top ${topSimilar.length} similar sequences`);
                                    }
                                }
                            }
                        }
                    }

                    if (!queryEmbedding) {
                        console.error('Could not generate query embedding for classification');
                        classificationPayload.biotype = [];
                        classificationPayload.species = [];
                        classificationPayload.similar = [];
                    } else {
                        // Debug: log the query embedding and database info
                        console.log('Query embedding dimension:', queryEmbedding.length);
                        console.log('Database size:', window.currentEmbeddingDataForPlot.length);

                        // Classify by similarity using the embedding database
                        const biotypeResults = classifyByEmbeddingSimilarity(queryEmbedding, window.currentEmbeddingDataForPlot, 'biotype', 10);
                        const speciesResults = classifyByEmbeddingSimilarity(queryEmbedding, window.currentEmbeddingDataForPlot, 'species', 10);

                        console.log('Biotype classification results:', biotypeResults);
                        console.log('Species classification results:', speciesResults);

                        classificationPayload.biotype = biotypeResults;
                        classificationPayload.species = speciesResults;

                        // Find most similar sequences
                        classificationPayload.similar = findSimilarSequences(queryEmbedding, window.currentEmbeddingDataForPlot, 5);
                    }


                } else {
                    // Original zero-shot classification
                    let firstTextTokenData = null;
                    const classificationConfig = { ...currentConfig };
                    const sessions = { dna: dnaOnnxSession, text: textOnnxSession };

                    if (biotypeLabels.length > 0) {
                        const r = await zeroShotClassifySingle(seqData.sequence, biotypeLabels, sessions, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                        classificationPayload.biotype = r.results;
                        if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData;
                        if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                    }
                    if (speciesLabels.length > 0) {
                        const r = await zeroShotClassifySingle(seqData.sequence, speciesLabels, sessions, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                        classificationPayload.species = r.results;
                        if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData;
                        if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                    }
                    if (customPrompts.length > 0) {
                        const r = await zeroShotClassifySingle(seqData.sequence, customPrompts, sessions, currentDnaTokenizer, currentTextTokenizer, classificationConfig);
                        classificationPayload.custom = r.results;
                        if (!firstDnaTokenData) firstDnaTokenData = r.dnaTokenData;
                        if (!firstTextTokenData && r.textTokenData.length > 0) firstTextTokenData = r.textTokenData[0];
                    }

                    if (sequencesProcessed === 0 && firstTextTokenData) {
                        const textTokenDiv = document.createElement('div');
                        textTokenDiv.className = 'mt-2';
                        displayTokenizationPreview(firstTextTokenData.type, firstTextTokenData.input, firstTextTokenData.tokens, firstTextTokenData.ids, textTokenDiv);
                        tokenizationPreviewContainer.appendChild(textTokenDiv);
                    }
                }

                displaySingleSequenceClassification(seqData, classificationPayload, resultsContainer);

                if (sequencesProcessed === 0 && firstDnaTokenData) {
                    displayTokenizationPreview(firstDnaTokenData.type, firstDnaTokenData.input, firstDnaTokenData.tokens, firstDnaTokenData.ids, tokenizationPreviewContainer);
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
        document.getElementById('process-fasta-btn').disabled = false;
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
    document.getElementById('dna-model-input').value = null;
    document.getElementById('text-model-input').value = null;
    document.getElementById('dna-vocab-input').value = null;
    document.getElementById('text-vocab-input').value = null;

    dnaOnnxSession = null;
    textOnnxSession = null;

    currentDnaTokenizer = new DNATokenizerClient(getConfigValue('kmerSize'));
    currentTextTokenizer = new TextTokenizerClient();

    showToast('All loaded data has been cleared.', 'success-ultra');
}