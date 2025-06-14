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

function parseCSVEmbeddings(csvString) {
    const lines = csvString.trim().split(/\r?\n/);
    if (lines.length < 2) return [];

    // Improved CSV parsing function
    const parseCSVLine = (line) => {
        const result = [];
        let inQuotes = false;
        let current = '';
        
        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            if (char === '"') {
                inQuotes = !inQuotes;
            } else if (char === ',' && !inQuotes) {
                result.push(current.trim());
                current = '';
            } else {
                current += char;
            }
        }
        result.push(current.trim()); // Add last field
        return result;
    };

    // Parse headers
    const rawHeaders = parseCSVLine(lines[0]);
    const data = [];

    // Column detection
    const idCol = rawHeaders.findIndex(h => ['id', 'name', 'header'].includes(h.toLowerCase()));
    const biotypeCol = rawHeaders.findIndex(h => ['biotype', 'type', 'gene_biotype'].includes(h.toLowerCase()));
    const speciesCol = rawHeaders.findIndex(h => ['species', 'organism'].includes(h.toLowerCase()));
    const seqCol = rawHeaders.findIndex(h => ['raw_sequence', 'sequence'].includes(h.toLowerCase()));
    const annCol = rawHeaders.findIndex(h => ['raw_annotation', 'annotation'].includes(h.toLowerCase()));

    // Find all embedding columns
    const dnaEmbeddingCols = [];
    const textEmbeddingCols = [];
    
    rawHeaders.forEach((header, index) => {
        if (header.startsWith('dna_emb_')) {
            const embIndex = parseInt(header.replace('dna_emb_', ''));
            if (!isNaN(embIndex)) {
                dnaEmbeddingCols[embIndex] = index;
            }
        } else if (header.startsWith('text_emb_')) {
            const embIndex = parseInt(header.replace('text_emb_', ''));
            if (!isNaN(embIndex)) {
                textEmbeddingCols[embIndex] = index;
            }
        }
    });

    // Process data rows
    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue; // skip empty lines
        
        const values = parseCSVLine(lines[i]);
        if (values.length >= rawHeaders.length) { // Use >= to be more permissive
            const entry = { 
                embeddings: [],
                dna_embeddings: [],
                text_embeddings: [],
                metadata: {}
            };
            
            // Extract DNA embeddings
            dnaEmbeddingCols.forEach((colIndex, embIndex) => {
                if (colIndex !== undefined && values[colIndex]) {
                    const numValue = parseFloat(values[colIndex]);
                    if (!isNaN(numValue)) {
                        entry.dna_embeddings[embIndex] = numValue;
                    }
                }
            });
            
            // Extract text embeddings
            textEmbeddingCols.forEach((colIndex, embIndex) => {
                if (colIndex !== undefined && values[colIndex]) {
                    const numValue = parseFloat(values[colIndex]);
                    if (!isNaN(numValue)) {
                        entry.text_embeddings[embIndex] = numValue;
                    }
                }
            });
            
            // Filter out undefined values
            entry.dna_embeddings = entry.dna_embeddings.filter(v => v !== undefined);
            entry.text_embeddings = entry.text_embeddings.filter(v => v !== undefined);
            
            // Use text embeddings for visualization by default
            entry.embeddings = entry.text_embeddings.length > 0 ? entry.text_embeddings : entry.dna_embeddings;

            // Extract other fields
            if (idCol >= 0) entry.id = values[idCol] || `csv_item_${i}`;
            if (biotypeCol >= 0) entry.biotype = values[biotypeCol] || 'Unknown';
            if (speciesCol >= 0) entry.species = values[speciesCol] || 'Unknown';
            if (seqCol >= 0) {
                entry.sequence = values[seqCol];
                entry.sequence_snippet = values[seqCol];
            }
            if (annCol >= 0) entry.annotation = values[annCol];

            // Add metadata for all non-embedding columns
            rawHeaders.forEach((header, index) => {
                if (!dnaEmbeddingCols.includes(index) && 
                    !textEmbeddingCols.includes(index) &&
                    index !== idCol && 
                    index !== biotypeCol && 
                    index !== speciesCol && 
                    index !== seqCol &&
                    index !== annCol) {
                    entry.metadata[header] = values[index];
                }
            });

            if (entry.embeddings.length > 0) {
                data.push(entry);
            }
        }
    }

    return data;
}

async function parseParquetEmbeddings(arrayBuffer) {
    try {
        // Try using a different approach with hyparquet
        const { parquetRead } = await import('https://cdn.jsdelivr.net/npm/hyparquet@1.4.1/src/hyparquet.js');
        
        let data;
        try {
            // First try reading with all columns
            data = await parquetRead({
                file: arrayBuffer,
                compressors: {
                    // Provide fallback for unsupported compression
                    ZSTD: null, // Skip ZSTD compression
                    SNAPPY: null,
                    GZIP: null,
                    LZ4: null
                }
            });
        } catch (compressionError) {
            console.warn('Failed with compression handling, trying simpler approach:', compressionError);
            // Fallback: try reading without compression specification
            data = await parquetRead({
                file: arrayBuffer
            });
        }
        
        if (!data || data.length === 0) {
            throw new Error('No data returned from parquet file');
        }
        
        return data.map((row, i) => {
            const entry = { 
                embeddings: [],
                metadata: {}
            };
            
            const dnaEmbeddings = [];
            const textEmbeddings = [];
            
            for (const key in row) {
                if (key.startsWith('dna_emb_')) {
                    const index = parseInt(key.replace('dna_emb_', ''));
                    if (!isNaN(index) && typeof row[key] === 'number') {
                        dnaEmbeddings[index] = row[key];
                    }
                } else if (key.startsWith('text_emb_')) {
                    const index = parseInt(key.replace('text_emb_', ''));
                    if (!isNaN(index) && typeof row[key] === 'number') {
                        textEmbeddings[index] = row[key];
                    }
                } else {
                    entry.metadata[key] = row[key];
                }
            }
            
            entry.embeddings = textEmbeddings.length > 0 ? textEmbeddings.filter(v => v !== undefined) : dnaEmbeddings.filter(v => v !== undefined);
            entry.dna_embeddings = dnaEmbeddings.filter(v => v !== undefined);
            entry.text_embeddings = textEmbeddings.filter(v => v !== undefined);
            
            entry.id = row.id || row.name || row.header || `item_${i}`;
            entry.biotype = row.biotype || row.type || row.gene_biotype || 'Unknown';
            entry.species = row.species || row.organism || 'Unknown';
            entry.sequence = row.raw_sequence || row.sequence || '';
            entry.sequence_snippet = entry.sequence;
            entry.annotation = row.raw_annotation || row.annotation || '';
            
            return entry;
        });
    } catch (error) {
        console.error("Parquet parsing error:", error);
        
        // If parquet fails completely, suggest using CSV format
        if (error.message.includes('compression') || error.message.includes('ZSTD')) {
            throw new Error(`Parquet compression not supported (${error.message}). Please convert your file to CSV format or use uncompressed Parquet.`);
        }
        
        throw new Error(`Failed to parse Parquet: ${error.message}`);
    }
}
// Helper functions
function isEmbeddingColumn(columnName) {
    return columnName.startsWith('emb_') || 
           columnName.startsWith('embedding_') || 
           columnName.startsWith('dna_emb_') || 
           columnName.startsWith('text_emb_') || 
           /^\d+$/.test(columnName) || 
           columnName.startsWith('dim_');
}

function getField(row, possibleKeys, defaultValue = undefined) {
    for (const key of possibleKeys) {
        if (row[key] !== undefined && row[key] !== null && row[key] !== '') {
            return row[key];
        }
    }
    return defaultValue;
}

function getNumericField(row, possibleKeys) {
    const value = getField(row, possibleKeys);
    if (value === undefined) return undefined;
    const num = parseFloat(value);
    return isNaN(num) ? undefined : num;
}

function extractEmbeddings(row, entry) {
    // First check for array-type embeddings (common in some formats)
    for (const key in row) {
        if ((key === 'embeddings' || key === 'embedding') && Array.isArray(row[key])) {
            entry.embeddings = row[key].filter(v => typeof v === 'number');
            return;
        }
    }
    
    // Otherwise collect all embedding columns
    for (const key in row) {
        if (isEmbeddingColumn(key) && typeof row[key] === 'number') {
            entry.embeddings.push(row[key]);
        }
    }
    
    // Sort embeddings if they appear to be ordered (like text_emb_0, text_emb_1)
    if (entry.embeddings.length > 0 && 
        Object.keys(row).some(k => k.startsWith('text_emb_0'))) {
        entry.embeddings.sort((a, b) => a - b);
    }
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