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
            attention_mask: encodedIds.map(id => id === this.pad_token_id ? 0 : 1), 
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
            attention_mask: encodedIds.map(id => id === this.pad_token_id ? 0 : 1), 
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
 * @param {ort.InferenceSession | null} onnxSession Pre-loaded ONNX session. Null for simulated mode.
 * @param {DNATokenizerClient} dnaTokenizer Instance of DNA tokenizer.
 * @param {TextTokenizerClient} textTokenizer Instance of Text tokenizer.
 * @param {Object} config Configuration object with maxDnaLen, maxTextLen.
 * @returns {Promise<{results: Array<{label: string, score: number}>, dnaTokenData: Object, textTokenData: Array<Object>}>}
 *          Results, and tokenization data for the first DNA and all text labels.
 */

async function zeroShotClassifySingle(dnaSequence, textLabels, onnxSession, dnaTokenizer, textTokenizer, config) {
    if (!textLabels || textLabels.length === 0) return { results: [], dnaTokenData: null, textTokenData: [] };

    // Get ONNX names from config
    const { 
        maxDnaLen, maxTextLen,
        onnxInputNameDnaIds, onnxInputNameDnaMask, onnxOutputNameDnaEmbedding,
        onnxInputNameTextIds, onnxInputNameTextMask, onnxOutputNameTextEmbedding
    } = config; // config is now currentConfig passed from app.js

    let dnaEmbedding;
    const dnaEncoded = dnaTokenizer.encode(dnaSequence, maxDnaLen);
    const dnaTokenData = {
        type: 'DNA',
        input: dnaEncoded.original_input,
        tokens: dnaEncoded.original_tokens,
        ids: dnaEncoded.ids.slice(0, dnaEncoded.length) // Show effective IDs
    };

    if (onnxSession) {
        try {
            const dnaInputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(dnaEncoded.ids.map(BigInt)), [1, dnaEncoded.ids.length]);
            const dnaAttentionMaskTensor = new ort.Tensor('int64', BigInt64Array.from(dnaEncoded.attention_mask.map(BigInt)), [1, dnaEncoded.attention_mask.length]);
            
            const dnaFeeds = {};
            dnaFeeds[onnxInputNameDnaIds] = dnaInputIdsTensor;
            dnaFeeds[onnxInputNameDnaMask] = dnaAttentionMaskTensor;
            
            const dnaResults = await onnxSession.run(dnaFeeds, [onnxOutputNameDnaEmbedding]);
            dnaEmbedding = Array.from(dnaResults[onnxOutputNameDnaEmbedding].data);

        } catch (e) {
            console.error("ONNX DNA encoding error:", e);
            showToast(`ONNX DNA Error: ${e.message.substring(0,100)}...`, 'danger-ultra', 7000);
            dnaEmbedding = simulateEmbedding(dnaSequence, 'dna', dnaTokenizer.k, 16);
        }
    } else {
        dnaEmbedding = simulateEmbedding(dnaSequence, 'dna', dnaTokenizer.k, 16);
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

        if (onnxSession) {
            try {
                const textInputIdsTensor = new ort.Tensor('int64', BigInt64Array.from(textEncoded.ids.map(BigInt)), [1, textEncoded.ids.length]);
                const textAttentionMaskTensor = new ort.Tensor('int64', BigInt64Array.from(textEncoded.attention_mask.map(BigInt)), [1, textEncoded.attention_mask.length]);
                
                const textFeeds = {};
                textFeeds[onnxInputNameTextIds] = textInputIdsTensor;
                textFeeds[onnxInputNameTextMask] = textAttentionMaskTensor;

                const textResults = await onnxSession.run(textFeeds, [onnxOutputNameTextEmbedding]);
                textEmbedding = Array.from(textResults[onnxOutputNameTextEmbedding].data);
            } catch (e) {
                console.error(`ONNX Text encoding error for "${label}":`, e);
                showToast(`ONNX Text Error ("${label}"): ${e.message.substring(0,80)}...`, 'danger-ultra', 7000);
                textEmbedding = simulateEmbedding(label, 'text', undefined, 16);
            }
        } else {
            textEmbedding = simulateEmbedding(label, 'text', undefined, 16);
        }
        
        const similarity = cosineSimilarity(dnaEmbedding, textEmbedding);
        results.push({ label: label, score: (similarity + 1) / 2 });
    }

    return {
        results: results.sort((a, b) => b.score - a.score),
        dnaTokenData: dnaTokenData,
        textTokenData: allTextTokenData
    };
}

/**
 * Simulates an embedding. Uses a fixed dimension for consistency.
 */
function simulateEmbedding(input, type, k = 6, dim = 16) {
    const embedding = Array(dim).fill(0);
    let charsToProcess;
    if (type === 'dna') {
        const { tokens } = (new DNATokenizerClient(k)).tokenize(input);
        charsToProcess = tokens.join('').split('');
    } else {
        charsToProcess = input.toLowerCase().split('');
    }
    if (charsToProcess.length === 0) return embedding;
    for (let i = 0; i < charsToProcess.length; i++) {
        embedding[i % dim] += (charsToProcess[i].charCodeAt(0) % 100) / 100.0 + (i * 0.01); // Add variance
    }
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return norm > 0 ? embedding.map(val => val / norm) : embedding;
}