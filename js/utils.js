// js/utils.js

/**
 * Expands a FASTA sequence string that might contain compact repeat notations.
 * Example: "{A*10}{G*5}NNN" becomes "AAAAAAAAAAGGGGGNNN"
 * @param {string} sequenceString The potentially compacted sequence string.
 * @returns {string} The expanded sequence string.
 */
function expandCompactFASTASequence(sequenceString) {
    if (!sequenceString) return "";
    return sequenceString.replace(/\{(\w)\*(\d+)\}/g, (match, base, count) => {
        return base.repeat(parseInt(count, 10));
    });
}

/**
 * Calculates the cosine similarity between two vectors.
 * @param {number[]} vecA Array of numbers.
 * @param {number[]} vecB Array of numbers.
 * @returns {number} Cosine similarity, or 0 if input is invalid.
 */
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
        return 0;
    }
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += (vecA[i] || 0) * (vecB[i] || 0); 
        normA += (vecA[i] || 0) * (vecA[i] || 0);
        normB += (vecB[i] || 0) * (vecB[i] || 0);
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}


/**
 * Performs Principal Component Analysis (PCA) on a dataset.
 * This is a simplified implementation.
 * @param {number[][]} data Matrix of data (samples x features).
 * @param {number} nComponents Number of principal components to return.
 * @returns {number[][]} Data projected onto the principal components.
 */
function simplePca(data, nComponents = 2) {
    if (!data || data.length === 0 || !data[0] || data[0].length === 0) return [];
    const numSamples = data.length;
    const numFeatures = data[0].length;

    if (numFeatures <= nComponents) { 
        return data.map(sample => sample.slice(0, nComponents));
    }
    if (numSamples < 2) { 
        console.warn("PCA requires at least 2 samples.");
        return data.map(sample => sample.slice(0, nComponents));
    }

    const means = Array(numFeatures).fill(0);
    for (let j = 0; j < numFeatures; j++) {
        for (let i = 0; i < numSamples; i++) {
            means[j] += data[i][j];
        }
        means[j] /= numSamples;
    }
    const centeredData = data.map(sample => sample.map((val, j) => val - means[j]));
    
    // For true PCA, you'd compute covariance and eigenvectors.
    // Using scikit-js for a more robust PCA if available, otherwise fallback.
    if (typeof sk === 'object' && sk.decomposition && sk.decomposition.PCA) {
        try {
            const pca = new sk.decomposition.PCA({ nComponents: nComponents });
            pca.fit(data); // data or centeredData
            return pca.transform(data);
        } catch (e) {
            console.warn("scikit-js PCA failed, using simplified projection:", e);
            return centeredData.map(sample => sample.slice(0, nComponents));
        }
    }
    return centeredData.map(sample => sample.slice(0, nComponents));
}

/**
 * Reads a file as text.
 */
function readFileAsText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsText(file);
    });
}

/**
 * Reads a file as an ArrayBuffer.
 */
function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(file);
    });
}

/**
 * Delays execution for a specified number of milliseconds.
 */
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// --- Sequence Metrics ---

/**
 * Calculates GC content of a DNA sequence.
 * @param {string} sequence DNA sequence string.
 * @returns {number} GC content as a fraction (0 to 1), or 0 if invalid.
 */
function calculateGcContent(sequence) {
    if (!sequence || typeof sequence !== 'string') return 0;
    const upperSeq = sequence.toUpperCase();
    let gcCount = 0;
    let validBaseCount = 0;
    for (let i = 0; i < upperSeq.length; i++) {
        const char = upperSeq[i];
        if (char === 'G' || char === 'C') {
            gcCount++;
            validBaseCount++;
        } else if (char === 'A' || char === 'T' || char === 'N') { // Count N as valid for length but not for GC
            if (char !== 'N') validBaseCount++;
        }
    }
    return validBaseCount > 0 ? gcCount / validBaseCount : 0;
}

/**
 * Calculates Shannon entropy for a DNA sequence.
 * H = -sum(pi * log2(pi)) for i in {A, C, G, T}
 * Ignores 'N's for probability calculation.
 * @param {string} sequence DNA sequence string.
 * @returns {number} Shannon entropy (0 to 2), or 0 if invalid/empty.
 */
function calculateShannonEntropy(sequence) {
    if (!sequence || typeof sequence !== 'string') return 0;
    const upperSeq = sequence.toUpperCase().replace(/N/g, ''); // Remove Ns
    if (upperSeq.length === 0) return 0;

    const counts = { A: 0, C: 0, G: 0, T: 0 };
    for (const base of upperSeq) {
        if (counts[base] !== undefined) {
            counts[base]++;
        }
    }

    let entropy = 0;
    for (const base in counts) {
        if (counts[base] > 0) {
            const probability = counts[base] / upperSeq.length;
            entropy -= probability * Math.log2(probability);
        }
    }
    return entropy;
}

/**
 * Calculates CpG Observed/Expected ratio.
 * CpG O/E = (Number of CpG * TotalLength) / (Number of C * Number of G)
 * @param {string} sequence DNA sequence string.
 * @returns {number} CpG O/E ratio, or 0 if counts are insufficient.
 */
function calculateCpGObsExp(sequence) {
    if (!sequence || typeof sequence !== 'string' || sequence.length < 2) return 0;
    const upperSeq = sequence.toUpperCase();
    
    let cpgCount = 0;
    let cCount = 0;
    let gCount = 0;
    
    for (let i = 0; i < upperSeq.length; i++) {
        const base = upperSeq[i];
        if (base === 'C') cCount++;
        else if (base === 'G') gCount++;
        
        if (i < upperSeq.length - 1) {
            if (base === 'C' && upperSeq[i+1] === 'G') {
                cpgCount++;
            }
        }
    }
    
    if (cCount === 0 || gCount === 0) return 0; // Avoid division by zero
    
    const totalLength = upperSeq.length;
    const cpgObsExp = (cpgCount * totalLength) / (cCount * gCount);
    
    return isNaN(cpgObsExp) ? 0 : cpgObsExp;
}

/**
 * Pre-calculates or retrieves sequence metrics for embedding data.
 * Modifies the embeddingData array in place by adding/updating metric fields.
 * @param {Array<Object>} embeddingData Array of embedding objects.
 *                                     Each object should have a 'sequence_snippet' or 'sequence' field.
 */
function ensureSequenceMetrics(embeddingData) {
    if (!embeddingData || embeddingData.length === 0) return;

    embeddingData.forEach(item => {
        const sequence = item.sequence || item.sequence_snippet || ""; // Prefer full sequence if available

        if (item.length === undefined && sequence) {
            item.length = sequence.length;
        }
        if (item.gc_content === undefined && sequence) {
            item.gc_content = calculateGcContent(sequence);
        }
        if (item.shannon_entropy === undefined && sequence) {
            item.shannon_entropy = calculateShannonEntropy(sequence);
        }
        if (item.cpg_oe === undefined && sequence) {
            item.cpg_oe = calculateCpGObsExp(sequence);
        }
    });
}