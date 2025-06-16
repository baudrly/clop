// js/classification.js

const PAD_TOKEN = "<PAD>";
const UNK_TOKEN = "<UNK>";

class DNATokenizerClient {
    constructor(k = 6, vocab = {[PAD_TOKEN]: 0, [UNK_TOKEN]: 1}) {
        this.k = k;
        this.vocab = vocab || {[PAD_TOKEN]: 0, [UNK_TOKEN]: 1}; // Ensure vocab is not null/undefined
        this.pad_token_id = this.vocab[PAD_TOKEN] !== undefined ? this.vocab[PAD_TOKEN] : 0;
        this.unk_token_id = this.vocab[UNK_TOKEN] !== undefined ? this.vocab[UNK_TOKEN] : 1;
    }

    tokenize(sequence) {
        sequence = (sequence || "").toUpperCase();
        const kmers = [];
        if (!sequence || sequence.length < this.k) return { tokens: [UNK_TOKEN], original: sequence };

        for (let i = 0; i < sequence.length - this.k + 1; i++) {
            const kmer = sequence.substring(i, i + this.k);
            if (/^[ACGTN]+$/.test(kmer)) { // Basic validation for DNA chars
                kmers.push(kmer);
            } else {
                kmers.push(UNK_TOKEN); // Represent invalid k-mer as UNK
            }
        }
        return { tokens: kmers.length > 0 ? kmers : [UNK_TOKEN], original: sequence };
    }

    encode(sequence, maxLength = null) {
        const { tokens, original } = this.tokenize(sequence);
        let encodedIds = tokens.map(kmer => this.vocab[kmer] !== undefined ? this.vocab[kmer] : this.unk_token_id);
        const originalLength = tokens.length;

        if (maxLength) {
            if (encodedIds.length > maxLength) {
                encodedIds = encodedIds.slice(0, maxLength);
            } else {
                encodedIds = encodedIds.concat(Array(maxLength - encodedIds.length).fill(this.pad_token_id));
            }
        }
        return {
            ids: encodedIds,
            // ATTENTION: The 'attention_mask' key is removed as it's not used by the model.
            // The 'length' is the crucial value.
            length: Math.min(originalLength, maxLength || originalLength),
            original_tokens: tokens, // For preview
            original_input: original // For preview
        };
    }
}

class TextTokenizerClient {
    constructor(vocab = {[PAD_TOKEN]: 0, [UNK_TOKEN]: 1}) {
        this.vocab = vocab || {[PAD_TOKEN]: 0, [UNK_TOKEN]: 1};
        this.pad_token_id = this.vocab[PAD_TOKEN] !== undefined ? this.vocab[PAD_TOKEN] : 0;
        this.unk_token_id = this.vocab[UNK_TOKEN] !== undefined ? this.vocab[UNK_TOKEN] : 1;
    }

    _cleanText(text) {
        text = (text || "").toLowerCase();
        text = text.replace(/[^a-z0-9\s_'-]/g, "");
        text = text.replace(/\s+/g, " ").trim();
        return text;
    }

    tokenize(text) {
        const original = text;
        if (!text || !text.trim()) return { tokens: [UNK_TOKEN], original: original };
        const cleanedText = this._cleanText(text);
        const tokens = cleanedText.split(/\s+/);
        return { tokens: tokens.length > 0 ? tokens : [UNK_TOKEN], original: original };
    }

    encode(text, maxLength = null) {
        const { tokens, original } = this.tokenize(text);
        let encodedIds = tokens.map(token => this.vocab[token] !== undefined ? this.vocab[token] : this.unk_token_id);
        const originalLength = tokens.length;

        if (maxLength) {
            if (encodedIds.length > maxLength) {
                encodedIds = encodedIds.slice(0, maxLength);
            } else {
                encodedIds = encodedIds.concat(Array(maxLength - encodedIds.length).fill(this.pad_token_id));
            }
        }
        return {
            ids: encodedIds,
            // ATTENTION: The 'attention_mask' key is removed.
            length: Math.min(originalLength, maxLength || originalLength),
            original_tokens: tokens, // For preview
            original_input: original // For preview
        };
    }
}


/**
 * Performs zero-shot classification for a given DNA sequence against a list of text labels.
 * @param {string} dnaSequence The input DNA sequence.
 * @param {Array<string>} textLabels Array of text labels to classify against.
 * @param {ort.InferenceSession | null} onnxSessions Pre-loaded ONNX sessions. Null for simulated mode.
 * @param {DNATokenizerClient} dnaTokenizer Instance of DNA tokenizer.
 * @param {TextTokenizerClient} textTokenizer Instance of Text tokenizer.
 * @param {Object} config Configuration object with maxDnaLen, maxTextLen.
 * @returns {Promise<{results: Array<{label: string, score: number}>, dnaTokenData: Object, textTokenData: Array<Object>}>}
 *          Results, and tokenization data for the first DNA and all text labels.
 */

async function zeroShotClassifySingle(dnaSequence, textLabels, onnxSessions, dnaTokenizer, textTokenizer, config) {
    if (!textLabels || textLabels.length === 0) {
        console.debug("No text labels provided for classification.");
        return { results: [], dnaTokenData: null, textTokenData: [] };
    }

    const { dna: dnaOnnxSession, text: textOnnxSession } = onnxSessions;
    
    const { maxDnaLen, maxTextLen } = config;

    // These names MUST match the names used in the Python ONNX export script.
    const onnxInputNameDnaIds = 'dna_tokens';
    const onnxInputNameDnaMask = 'dna_lengths';
    const onnxOutputNameDnaEmbedding = 'dna_embedding';
    const onnxInputNameTextIds = 'text_tokens';
    const onnxInputNameTextMask = 'text_lengths';
    const onnxOutputNameTextEmbedding = 'text_embedding';

    let dnaEmbedding;
    const dnaEncoded = dnaTokenizer.encode(dnaSequence, maxDnaLen);
    const dnaTokenData = {
        type: 'DNA',
        input: dnaEncoded.original_input,
        tokens: dnaEncoded.original_tokens,
        ids: dnaEncoded.ids.slice(0, dnaEncoded.length)
    };

    if (dnaOnnxSession) {
        try {
            const dnaInputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(dnaEncoded.ids.map(BigInt)), [1, dnaEncoded.ids.length]);
            const dnaLengthsTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(dnaEncoded.length)]), [1]);
            
            const dnaFeeds = {};
            dnaFeeds[onnxInputNameDnaIds] = dnaInputIdsTensor;
            dnaFeeds[onnxInputNameDnaMask] = dnaLengthsTensor;
            
            console.debug("DNA Feeds for ONNX:", dnaFeeds);
            const dnaResults = await dnaOnnxSession.run(dnaFeeds);
            console.debug("DNA Embedding Results:", dnaResults);
            dnaEmbedding = Array.from(dnaResults[onnxOutputNameDnaEmbedding].data);

        } catch (e) {
            console.error("ONNX DNA encoding error:", e);
            showToast(`ONNX DNA Error: ${e.message.substring(0,100)}...`, 'danger-ultra', 7000);
            dnaEmbedding = simulateEmbedding(dnaSequence, 'dna', dnaTokenizer.k, 128);
        }
    } else {
        dnaEmbedding = simulateEmbedding(dnaSequence, 'dna', dnaTokenizer.k, 128);
    }

    const results = [];
    const allTextTokenData = [];

    for (const label of textLabels) {
        let textEmbedding;
        const textEncoded = textTokenizer.encode(label, maxTextLen);
        allTextTokenData.push({
            type: 'Text Label',
            input: textEncoded.original_input,
            tokens: textEncoded.original_tokens,
            ids: textEncoded.ids.slice(0, textEncoded.length)
        });

        if (textOnnxSession) {
            try {
                const textInputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(textEncoded.ids.map(BigInt)), [1, textEncoded.ids.length]);
                const textLengthsTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(textEncoded.length)]), [1]);
                
                const textFeeds = {};
                textFeeds[onnxInputNameTextIds] = textInputIdsTensor;
                textFeeds[onnxInputNameTextMask] = textLengthsTensor;

                console.debug(`Text Feeds for ONNX for label "${label}":`, textFeeds);
                const textResults = await textOnnxSession.run(textFeeds);
                console.debug(`Text Embedding Results for label "${label}":`, textResults);
                textEmbedding = Array.from(textResults[onnxOutputNameTextEmbedding].data);
            } catch (e) {
                console.error(`ONNX Text encoding error for "${label}":`, e);
                showToast(`ONNX Text Error ("${label}"): ${e.message.substring(0,80)}...`, 'danger-ultra', 7000);
                textEmbedding = simulateEmbedding(label, 'text', undefined, 128);
            }
        } else {
            textEmbedding = simulateEmbedding(label, 'text', undefined, 128);
        }
        
        console.debug(`DNA Embedding:`, dnaEmbedding);
        console.debug(`Text Embedding for label "${label}":`, textEmbedding);
        const similarity = cosineSimilarity(dnaEmbedding, textEmbedding);
        if (!isNaN(similarity)) {
             results.push({ label: label, score: (similarity + 1) / 2 });
        }
    }

    return {
        results: results.sort((a, b) => b.score - a.score),
        dnaTokenData: dnaTokenData,
        textTokenData: allTextTokenData
    };
}

/**
 * Simulates an embedding. Uses the actual embedding dimension from the model.
 */
function simulateEmbedding(input, type, k = 6, dim = 64) {
    const embedding = Array(dim).fill(0);
    let charsToProcess;
    if (type === 'dna') {
        const { tokens } = (new DNATokenizerClient(k)).tokenize(input);
        charsToProcess = tokens.join('').split('');
    } else {
        charsToProcess = input.toLowerCase().split('');
    }
    if (charsToProcess.length === 0) return embedding;
    
    // Create a more realistic simulation based on sequence content
    for (let i = 0; i < charsToProcess.length; i++) {
        const charCode = charsToProcess[i].charCodeAt(0);
        embedding[i % dim] += (charCode % 100) / 100.0 + (i * 0.01);
    }
    
    // Add some sequence-specific variation
    for (let i = 0; i < dim; i++) {
        embedding[i] += (Math.sin(i + input.length) * 0.1);
    }
    
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return norm > 0 ? embedding.map(val => val / norm) : embedding;
}

/**
 * Performs embedding-based classification using pre-computed embeddings.
 * Finds the k-nearest neighbors in the embedding space.
 * @param {Array<number>} queryEmbedding The embedding of the query sequence
 * @param {Array<Object>} embeddingDatabase Array of objects with embeddings and metadata
 * @param {string} attributeType 'biotype' or 'species' to classify
 * @param {number} k Number of nearest neighbors to consider
 * @returns {Array<{label: string, score: number, count: number}>} Sorted results
 */
function classifyByEmbeddingSimilarity(queryEmbedding, embeddingDatabase, attributeType = 'biotype', k = 10) {
    if (!queryEmbedding || queryEmbedding.length === 0 || !embeddingDatabase || embeddingDatabase.length === 0) {
        console.warn('Missing query embedding or database for classification');
        return [];
    }
    
    console.log(`Starting classification for ${attributeType} with query embedding dim: ${queryEmbedding.length}`);
    
    // Calculate similarities to all database entries
    const similarities = embeddingDatabase.map((entry, index) => {
        // Try different embedding sources in order of preference
        let targetEmbedding = entry.dna_embeddings || entry.text_embeddings || entry.embeddings;
        
        if (!targetEmbedding || targetEmbedding.length === 0) {
            return null;
        }
        
        // Check dimension compatibility
        if (targetEmbedding.length !== queryEmbedding.length) {
            // Try to find a compatible embedding
            if (entry.dna_embeddings && entry.dna_embeddings.length === queryEmbedding.length) {
                targetEmbedding = entry.dna_embeddings;
            } else if (entry.text_embeddings && entry.text_embeddings.length === queryEmbedding.length) {
                targetEmbedding = entry.text_embeddings;
            } else {
                return null; // Skip incompatible entries
            }
        }
        
        const similarity = cosineSimilarity(queryEmbedding, targetEmbedding);
        if (isNaN(similarity)) {
            return null;
        }
        
        return { index, similarity, entry };
    }).filter(item => item !== null);
    
    console.log(`Calculated similarities for ${similarities.length} entries out of ${embeddingDatabase.length} total`);
    
    if (similarities.length === 0) {
        console.warn('No valid similarities calculated');
        return [];
    }
    
    // Log similarity distribution for debugging
    const simValues = similarities.map(s => s.similarity);
    const minSim = Math.min(...simValues);
    const maxSim = Math.max(...simValues);
    const avgSim = simValues.reduce((sum, val) => sum + val, 0) / simValues.length;
    console.log(`Similarity range: min=${minSim.toFixed(4)}, max=${maxSim.toFixed(4)}, avg=${avgSim.toFixed(4)}`);
    
    // Sort by similarity and take top k
    const topK = similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);
    
    console.log(`Top ${Math.min(k, similarities.length)} similarities for ${attributeType}:`, 
        topK.map(s => ({ 
            similarity: s.similarity.toFixed(4), 
            label: s.entry[attributeType],
            id: s.entry.id 
        }))
    );
    
    // Aggregate scores by attribute
    const scoresByLabel = {};
    const countsByLabel = {};
    const similaritiesByLabel = {};
    
    topK.forEach(({ entry, similarity }) => {
        const label = entry[attributeType] || 'Unknown';
        if (!scoresByLabel[label]) {
            scoresByLabel[label] = [];
            countsByLabel[label] = 0;
            similaritiesByLabel[label] = [];
        }
        scoresByLabel[label].push(similarity);
        countsByLabel[label]++;
        similaritiesByLabel[label].push(similarity);
    });
    
    // Convert to array and calculate more nuanced scores
    const results = Object.entries(scoresByLabel).map(([label, similarities]) => {
        const count = countsByLabel[label];
        const avgSimilarity = similarities.reduce((sum, sim) => sum + sim, 0) / similarities.length;
        const maxSimilarity = Math.max(...similarities);
        
        // Use raw similarity as score (don't artificially inflate)
        // Weight by count (more occurrences = higher confidence)
        const countWeight = Math.min(1.0, count / 3); // Cap the count bonus
        const rawScore = avgSimilarity * 0.8 + maxSimilarity * 0.2;
        const finalScore = rawScore * (0.7 + 0.3 * countWeight);
        
        return {
            label,
            score: Math.max(0, Math.min(1, finalScore)), // Ensure 0-1 range
            count,
            avgSimilarity: avgSimilarity,
            maxSimilarity: maxSimilarity,
            rawScore: rawScore
        };
    });
    
    const sortedResults = results.sort((a, b) => b.score - a.score);
    console.log(`Final classification results for ${attributeType}:`, sortedResults.map(r => ({
        label: r.label,
        score: r.score.toFixed(4),
        rawScore: r.rawScore.toFixed(4),
        avgSim: r.avgSimilarity.toFixed(4),
        count: r.count
    })));
    
    return sortedResults;
}
/**
 * Gets the most similar sequences from the embedding database
 * @param {Array<number>} queryEmbedding The embedding of the query sequence
 * @param {Array<Object>} embeddingDatabase Array of objects with embeddings and metadata
 * @param {number} topN Number of similar sequences to return
 * @returns {Array<Object>} Top similar sequences with similarity scores
 */
function findSimilarSequences(queryEmbedding, embeddingDatabase, topN = 5) {
    if (!queryEmbedding || queryEmbedding.length === 0 || !embeddingDatabase || embeddingDatabase.length === 0) {
        return [];
    }
    
    const similarities = embeddingDatabase.map(entry => {
        const targetEmbedding = entry.dna_embeddings || entry.embeddings;
        const similarity = cosineSimilarity(queryEmbedding, targetEmbedding);
        
        if (isNaN(similarity)) {
            return null;
        }
        
        return { ...entry, similarity };
    });
    
    return similarities
        .filter(s => s !== null)
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topN);
}