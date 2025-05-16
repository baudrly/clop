// js/parsers.js

// parseFasta (from "Ultra Fix 1")
function parseFasta(fastaString) {
    if (!fastaString || typeof fastaString !== 'string') return [];
    
    const sequences = [];
    const lines = fastaString.split(/\r?\n/);
    let currentHeaderLine = null;
    let currentSequenceChars = [];

    function processCurrentSequence() {
        if (currentHeaderLine) {
            const originalHeader = currentHeaderLine.substring(1);
            const headerParts = originalHeader.split(/\s+/); 
            const primaryId = headerParts[0];
            const metadata = {};
            
            const metadataString = originalHeader.substring(primaryId.length).trim();
            const metaParts = metadataString.split('|');
            metaParts.forEach(part => {
                const kv = part.split('=');
                if (kv.length === 2 && kv[0].trim() && kv[1].trim()) {
                    metadata[kv[0].trim().toLowerCase()] = kv[1].trim();
                } else if (kv.length === 1 && kv[0].trim()) {
                    if (!metadata.description) metadata.description = kv[0].trim();
                    else metadata.description += " " + kv[0].trim();
                }
            });
            if (!metadata.id) metadata.id = primaryId;

            sequences.push({
                id: metadata.id || `seq_${sequences.length + 1}`,
                header: primaryId,
                originalHeader: originalHeader,
                sequence: expandCompactFASTASequence(currentSequenceChars.join('').toUpperCase()),
                metadata: metadata
            });
        }
    }

    for (const line of lines) {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith('>')) {
            processCurrentSequence(); 
            currentHeaderLine = trimmedLine;
            currentSequenceChars = [];
        } else if (currentHeaderLine && trimmedLine) {
            currentSequenceChars.push(trimmedLine.replace(/\s+/g, ''));
        }
    }
    processCurrentSequence(); 
    return sequences;
}

// parseCSVEmbeddings (from "Ultra Fix 1")
function parseCSVEmbeddings(csvString) {
    const lines = csvString.trim().split('\n');
    if (lines.length < 2) return [];

    const rawHeaders = lines[0].split(',').map(h => h.trim().toLowerCase());
    const data = [];

    const idCol = rawHeaders.find(h => ['id', 'name', 'header'].includes(h)) || rawHeaders[0];
    const biotypeCol = rawHeaders.find(h => ['biotype', 'type', 'gene_biotype'].includes(h));
    const speciesCol = rawHeaders.find(h => ['species', 'organism'].includes(h));
    const seqSnippetCol = rawHeaders.find(h => ['sequence_snippet', 'seq_snippet', 'raw_sequence', 'sequence'].includes(h));
    const lengthCol = rawHeaders.find(h => ['length', 'len', 'sequence_length'].includes(h));
    const gcCol = rawHeaders.find(h => ['gc_content', 'gc', 'gc_fraction'].includes(h));

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.length === rawHeaders.length) {
            const entry = { embeddings: [] };
            rawHeaders.forEach((header, index) => {
                const value = values[index].trim();
                if (header.startsWith('emb_') || header.startsWith('embedding_') || /^\d+$/.test(header) || header.startsWith('dim_')) {
                    entry.embeddings.push(parseFloat(value));
                } else if (header === idCol) entry.id = value;
                else if (header === biotypeCol) entry.biotype = value;
                else if (header === speciesCol) entry.species = value;
                else if (header === seqSnippetCol) entry.sequence_snippet = value;
                else if (header === lengthCol) entry.length = parseFloat(value);
                else if (header === gcCol) entry.gc_content = parseFloat(value);
                else {
                    if(!entry.metadata) entry.metadata = {};
                    entry.metadata[header] = isNaN(parseFloat(value)) || !isFinite(value) ? value : parseFloat(value);
                }
            });
            if (!entry.id) entry.id = `csv_item_${i}`;
            if (!entry.biotype) entry.biotype = 'Unknown';
            if (!entry.species) entry.species = 'Unknown';
            data.push(entry);
        }
    }
    return data;
}


async function parseParquetEmbeddings(arrayBuffer) {
    // Use window.parquetWasmModule which should be initialized by app.js
    if (!window.parquetWasmModule || typeof window.parquetWasmModule.readParquet !== 'function') {
        console.error("Parquet library (parquetWasmModule) is not ready or readParquet function missing in parsers.js.");
        // Attempt a last-ditch init if the global window.parquetWasm (init function) exists
        if (window.parquetWasm && typeof window.parquetWasm === 'function' && !window.parquetWasm.readParquet) {
            try {
                await window.parquetWasm(); // Call the init function
                window.parquetWasmModule = window.parquetWasm; // Assume it modifies itself
                if (!window.parquetWasmModule || typeof window.parquetWasmModule.readParquet !== 'function') {
                    throw new Error("readParquet still not available after re-init attempt.");
                }
                 console.log("parquet-wasm lazy-initialized successfully from parser.");
            } catch (initErr) {
                console.error("Failed to lazy-initialize parquet-wasm from parser:", initErr);
                throw new Error("Parquet library could not be initialized properly.");
            }
        } else if (window.parquetWasm && window.parquetWasm.default && typeof window.parquetWasm.default === 'function' && !window.parquetWasm.default.readParquet ) {
             try {
                await window.parquetWasm.default(); // Call the init function
                window.parquetWasmModule = window.parquetWasm.default; // Assume it modifies itself
                if (!window.parquetWasmModule || typeof window.parquetWasmModule.readParquet !== 'function') {
                    throw new Error("readParquet still not available after re-init attempt from default.");
                }
                 console.log("parquet-wasm lazy-initialized successfully from parser (default).");
            } catch (initErr) {
                console.error("Failed to lazy-initialize parquet-wasm from parser (default):", initErr);
                throw new Error("Parquet library could not be initialized properly from default.");
            }
        }
        else {
            throw new Error("Parquet library (parquet-wasm) is not available or not an object with readParquet.");
        }
    }
    
    const arr = new Uint8Array(arrayBuffer);
    const arrowTable = window.parquetWasmModule.readParquet(arr); 
    const rawData = arrowTable.toArray().map(row => row.toJSON());
    
    return rawData.map((row, i) => {
        const entry = { embeddings: [] };
        const lowerCaseRow = {};
        for (const key in row) {
            lowerCaseRow[key.toLowerCase()] = row[key];
        }

        entry.id = lowerCaseRow.id || lowerCaseRow.name || lowerCaseRow.header || `pq_item_${i}`;
        entry.biotype = lowerCaseRow.biotype || lowerCaseRow.type || lowerCaseRow.gene_biotype || 'Unknown';
        entry.species = lowerCaseRow.species || lowerCaseRow.organism || 'Unknown';
        entry.sequence_snippet = lowerCaseRow.sequence_snippet || lowerCaseRow.seq_snippet || lowerCaseRow.raw_sequence || lowerCaseRow.sequence; // For metrics
        entry.sequence = lowerCaseRow.sequence || lowerCaseRow.raw_sequence || lowerCaseRow.sequence_snippet; // Prefer full sequence for metrics

        // Ensure length and gc_content are parsed as numbers if they exist
        if (lowerCaseRow.hasOwnProperty('length') || lowerCaseRow.hasOwnProperty('len') || lowerCaseRow.hasOwnProperty('sequence_length')) {
            entry.length = parseFloat(lowerCaseRow.length || lowerCaseRow.len || lowerCaseRow.sequence_length);
        }
        if (lowerCaseRow.hasOwnProperty('gc_content') || lowerCaseRow.hasOwnProperty('gc') || lowerCaseRow.hasOwnProperty('gc_fraction')) {
            entry.gc_content = parseFloat(lowerCaseRow.gc_content || lowerCaseRow.gc || lowerCaseRow.gc_fraction);
        }


        for (const key in lowerCaseRow) {
            if (key.startsWith('emb_') || key.startsWith('embedding_') || key.startsWith('dna_emb_') || key.startsWith('text_emb_') || /^\d+$/.test(key) || key.startsWith('dim_')) {
                 if (typeof lowerCaseRow[key] === 'number') {
                    entry.embeddings.push(lowerCaseRow[key]);
                } else if (Array.isArray(lowerCaseRow[key])) { 
                    entry.embeddings = lowerCaseRow[key].filter(v => typeof v === 'number');
                    break; 
                }
            }
        }
        return entry;
    });
}

// loadONNXModel (from "Ultra Fix 1")
async function loadONNXModel(modelData) {
    if (!window.ort) {
        throw new Error("ONNX Runtime Web (ort) is not loaded.");
    }
    try {
        const session = await ort.InferenceSession.create(modelData, { executionProviders: ['wasm'] });
        console.log("ONNX model loaded successfully with WASM provider.");
        return session;
    } catch (e) {
        console.error("Error loading ONNX model:", e);
        throw e;
    }
}