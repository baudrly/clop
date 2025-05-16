// js/visualizations.js

const FALLBACK_CATEGORY_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
];

async function plotEmbeddings(embeddingData, plotElementId, colorByAttribute, plotType) {
    const plotContainer = document.getElementById(plotElementId);
    const reductionWarningDiv = document.getElementById('reduction-warning');
    if (!plotContainer) { console.error(`Element ${plotElementId} not found.`); return; }

    if (typeof Plotly !== 'undefined' && typeof Plotly.purge === 'function') {
        try {
            Plotly.purge(plotElementId);
        } catch (e) {
            console.warn("Error purging previous Plotly instance:", e);
            while (plotContainer.firstChild) {
                plotContainer.removeChild(plotContainer.firstChild);
            }
        }
    }
    
    plotContainer.innerHTML = '<div class="plot-placeholder"><div class="spinner-border text-primary-gradient" role="status"></div><p class="mt-2">Processing embeddings...</p></div>';
    if (reductionWarningDiv) reductionWarningDiv.style.display = 'none';


    if (!embeddingData || embeddingData.length === 0 || !embeddingData[0] || !embeddingData[0].embeddings || embeddingData[0].embeddings.length === 0) {
        plotContainer.innerHTML = '<div class="plot-placeholder"><i class="fas fa-exclamation-triangle fa-3x text-warning-ultra mb-2"></i><p>No valid embedding data. Ensure "embeddings" array exists.</p></div>';
        return;
    }

    ensureSequenceMetrics(embeddingData); 

    let rawEmbeddings = embeddingData.map(d => d.embeddings);
    let processedData; 
    const numSamples = rawEmbeddings.length;
    const numDimensions = rawEmbeddings[0].length;

    const statsDiv = document.getElementById('embedding-stats');
    if(statsDiv) {
        statsDiv.innerHTML = `Displaying <strong>${numSamples}</strong> points, original dim: <strong>${numDimensions}</strong>.`;
    }
    
    document.getElementById('load-dummy-embeddings-btn').disabled = true;
    document.getElementById('embedding-file-input').disabled = true;
    await delay(10); 

    try {
        if (numDimensions <= 2 && (plotType === 'pca' || plotType === 'none')) {
            processedData = rawEmbeddings.map(emb => [(emb[0] || 0), (emb.length > 1 ? emb[1] : 0) || 0]);
        } else {
            switch (plotType) {
                case 'pca':
                    processedData = simplePca(rawEmbeddings, 2);
                    break;
                case 'tsne':
                    if (typeof sk !== 'object' || !sk.manifold || !sk.manifold.TSNE) {
                        throw new Error("scikit-js (for t-SNE) is not loaded.");
                    }
                    if (numSamples < 5) throw new Error("t-SNE requires at least 5 samples.");
                    if (reductionWarningDiv) reductionWarningDiv.style.display = 'block';
                    const tsne = new sk.manifold.TSNE({
                        nComponents: 2,
                        perplexity: Math.min(30, numSamples - 1), 
                        nIter: Math.max(250, numSamples * 3), 
                        randomState: 42
                    });
                    processedData = tsne.fitTransform(rawEmbeddings);
                    break;
                case 'umap':
                    if (typeof UMAP === 'undefined') {
                        throw new Error("UMAP-JS is not loaded.");
                    }
                    if (numSamples < 3) throw new Error("UMAP requires at least 3 samples.");
                     if (reductionWarningDiv) reductionWarningDiv.style.display = 'block';
                    const umap = new UMAP({
                        nComponents: 2,
                        nNeighbors: Math.min(15, numSamples - 1),
                        minDist: 0.1,
                        randomState: 42 
                    });
                    processedData = umap.fitTransform(rawEmbeddings);
                    break;
                case 'none':
                default:
                    processedData = rawEmbeddings.map(emb => [(emb[0] || 0), (emb.length > 1 ? emb[1] : 0) || 0]);
                    break;
            }
        }
    } catch (e) {
        console.error(`Error during ${plotType.toUpperCase()} reduction:`, e);
        showToast(`Error in ${plotType.toUpperCase()}: ${e.message}`, 'danger-ultra', 7000);
        plotContainer.innerHTML = `<div class="plot-placeholder"><i class="fas fa-times-circle fa-3x text-danger-ultra mb-2"></i><p>Failed ${plotType.toUpperCase()} reduction. ${e.message}</p></div>`;
        document.getElementById('load-dummy-embeddings-btn').disabled = false;
        document.getElementById('embedding-file-input').disabled = false;
        return;
    }
    
    plotContainer.innerHTML = ''; 

    await new Promise(resolve => requestAnimationFrame(resolve));

    const labels = embeddingData.map(d => {
        let val = d[colorByAttribute];
        if (typeof val === 'number' && !isNaN(val)) { 
             if (colorByAttribute === 'gc_content' || colorByAttribute === 'shannon_entropy' || colorByAttribute === 'cpg_oe') return val.toFixed(2);
             if (colorByAttribute === 'length') return val > 1000 ? ">1kb" : (val > 500 ? "0.5-1kb" : (val > 100 ? "0.1-0.5kb" : "<0.1kb"));
        }
        return String(val || 'Unknown');
    });

    const hoverTexts = embeddingData.map(d => {
        let text = `<b>ID:</b> ${d.id || 'N/A'}`;
        if(d.biotype) text += `<br><b>Biotype:</b> ${d.biotype}`;
        if(d.species) text += `<br><b>Species:</b> ${d.species}`;
        if(d.length !== undefined) text += `<br><b>Length:</b> ${d.length}bp`;
        if(typeof d.gc_content === 'number' && !isNaN(d.gc_content)) text += `<br><b>GC:</b> ${(d.gc_content * 100).toFixed(1)}%`;
        if(typeof d.shannon_entropy === 'number' && !isNaN(d.shannon_entropy)) text += `<br><b>Entropy:</b> ${d.shannon_entropy.toFixed(3)}`;
        if(typeof d.cpg_oe === 'number' && !isNaN(d.cpg_oe)) text += `<br><b>CpG O/E:</b> ${d.cpg_oe.toFixed(3)}`;
        if(d.sequence_snippet) text += `<br><b>Seq:</b> ${d.sequence_snippet.substring(0,30)}${d.sequence_snippet.length > 30 ? '...' : ''}`;
        return text;
    });
    
    const uniqueLabels = [...new Set(labels)];
    const traces = [];
    const markerConfig = {
        size: getConfigValue('plotMarkerSize'),
        opacity: getConfigValue('plotMarkerOpacity'),
        line: { width: 0.5, color: getConfigValue('currentTheme') === 'dark' ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)'}
    };

    const isContinuousColoring = (['gc_content', 'length', 'shannon_entropy', 'cpg_oe'].includes(colorByAttribute)) && 
                               embeddingData.length > 0 && 
                               typeof embeddingData[0][colorByAttribute] === 'number' &&
                               !isNaN(embeddingData[0][colorByAttribute]) && 
                               uniqueLabels.length > 5; 

    const defaultMarkerColorLight = '#3B82F6'; 
    const defaultMarkerColorDark = '#60A5FA';  

    if (colorByAttribute === 'none' || uniqueLabels.length === 1) {
        traces.push({
            x: processedData.map(d => d[0]), y: processedData.map(d => d[1]),
            mode: 'markers', type: 'scattergl', text: hoverTexts, hoverinfo: 'text',
            marker: { ...markerConfig, color: getConfigValue('currentTheme') === 'dark' ? defaultMarkerColorDark : defaultMarkerColorLight }
        });
    } else if (isContinuousColoring) {
        const colorValues = embeddingData.map(d => d[colorByAttribute]);
        traces.push({
            x: processedData.map(d => d[0]), y: processedData.map(d => d[1]),
            mode: 'markers', type: 'scattergl', text: hoverTexts, hoverinfo: 'text',
            marker: {
                ...markerConfig, color: colorValues,
                colorscale: getConfigValue('currentTheme') === 'dark' ? 'Cividis' : 'Viridis',
                showscale: true,
                colorbar: { title: colorByAttribute.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), thickness:15, len:0.75, x: 1.02, yanchor:'middle', y:0.5 }
            }
        });
    } else { 
        const palette = (window.Plotly && window.Plotly.Palettes && window.Plotly.Palettes.Category10) 
                        ? window.Plotly.Palettes.Category10 
                        : FALLBACK_CATEGORY_PALETTE;
                        
        uniqueLabels.forEach((label, i) => {
            const indices = labels.map((l, idxL) => l === label ? idxL : -1).filter(idxL => idxL !== -1);
            traces.push({
                x: indices.map(idx => processedData[idx][0]),
                y: indices.map(idx => processedData[idx][1]),
                mode: 'markers', type: 'scattergl', name: label,
                text: indices.map(idx => hoverTexts[idx]), hoverinfo: 'text',
                marker: { ...markerConfig, color: palette[i % palette.length] }
            });
        });
    }

    const layout = {
        title: false, 
        autosize: true, 
        xaxis: { 
            title: `${plotType.toUpperCase()} Dim 1`, 
            zeroline: false, 
            automargin: true 
        },
        yaxis: { 
            title: `${plotType.toUpperCase()} Dim 2`, 
            zeroline: false, 
            automargin: true
        },
        hovermode: 'closest',
        margin: { l: 60, r: 30, t: 30, b: 70, pad: 5 }, 
        legend: {
            showlegend: getConfigValue('plotShowLegend') && uniqueLabels.length > 1 && !isContinuousColoring,
            orientation: "v", yanchor: "top", y: 0.98, xanchor: "right", x: 0.98, 
            bgcolor: 'rgba(255,255,255,0.8)', bordercolor: '#ccc', borderwidth: 1,
            font: {size: 9}
        },
        paper_bgcolor: 'transparent', 
        plot_bgcolor: 'transparent',
    };

     if (getConfigValue('currentTheme') === 'dark') {
        layout.font = { color: 'var(--aura-text-dark-primary)', family: 'var(--font-family-sans-serif)' };
        layout.xaxis.gridcolor = 'rgba(255,255,255,0.07)';
        layout.yaxis.gridcolor = 'rgba(255,255,255,0.07)';
        layout.xaxis.linecolor = 'var(--aura-border-dark)';
        layout.yaxis.linecolor = 'var(--aura-border-dark)';
        layout.xaxis.zerolinecolor = 'var(--aura-border-dark)';
        layout.yaxis.zerolinecolor = 'var(--aura-border-dark)';
        layout.xaxis.title.font = {color: 'var(--aura-text-dark-secondary)'};
        layout.yaxis.title.font = {color: 'var(--aura-text-dark-secondary)'};
        layout.xaxis.tickfont = {color: 'var(--aura-text-dark-secondary)'};
        layout.yaxis.tickfont = {color: 'var(--aura-text-dark-secondary)'};
        layout.legend.bgcolor = 'rgba(33, 38, 45, 0.9)'; 
        layout.legend.bordercolor = 'var(--aura-border-dark)';
        layout.legend.font.color = 'var(--aura-text-dark-secondary)';
        if (traces[0] && traces[0].marker && traces[0].marker.colorbar) {
             traces[0].marker.colorbar.tickfont = { color: 'var(--aura-text-dark-secondary)' };
             traces[0].marker.colorbar.title.font = { color: 'var(--aura-text-dark-primary)' };
             traces[0].marker.colorbar.outlinecolor = 'var(--aura-border-dark)';
             traces[0].marker.colorbar.bgcolor = 'rgba(33, 38, 45, 0.7)';
        }
    } else { 
        layout.font = { color: 'var(--aura-text-light-primary)', family: 'var(--font-family-sans-serif)' };
        layout.xaxis.gridcolor = 'rgba(0,0,0,0.05)';
        layout.yaxis.gridcolor = 'rgba(0,0,0,0.05)';
        layout.xaxis.linecolor = 'var(--aura-border-light)';
        layout.yaxis.linecolor = 'var(--aura-border-light)';
        layout.xaxis.zerolinecolor = 'var(--aura-border-light)';
        layout.yaxis.zerolinecolor = 'var(--aura-border-light)';
        layout.xaxis.title.font = {color: 'var(--aura-text-light-secondary)'};
        layout.yaxis.title.font = {color: 'var(--aura-text-light-secondary)'};
        layout.xaxis.tickfont = {color: 'var(--aura-text-light-secondary)'};
        layout.yaxis.tickfont = {color: 'var(--aura-text-light-secondary)'};
        layout.legend.font.color = 'var(--aura-text-light-secondary)';
         if (traces[0] && traces[0].marker && traces[0].marker.colorbar) {
             traces[0].marker.colorbar.tickfont = { color: 'var(--aura-text-light-secondary)' };
             traces[0].marker.colorbar.title.font = { color: 'var(--aura-text-light-primary)' };
        }
    }
    
    const plotlyConfig = {
        responsive: true, 
        displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud', 'toggleSpikelines'] 
    };

    try {
        await Plotly.react(plotElementId, traces, layout, plotlyConfig);
    } catch(plotError) {
        console.error("Plotly.react error:", plotError);
        plotContainer.innerHTML = `<div class="plot-placeholder"><i class="fas fa-chart-area fa-3x text-danger-ultra mb-2"></i><p>Error rendering plot. ${plotError.message}</p></div>`;
    } finally {
        document.getElementById('load-dummy-embeddings-btn').disabled = false;
        document.getElementById('embedding-file-input').disabled = false;
    }
    
    window.lastPlotLayout = layout; 
    window.lastPlotTraces = traces; 

    await delay(50); 
    const plotDivElement = document.getElementById(plotElementId);
    if (plotDivElement && typeof Plotly !== 'undefined' && Plotly.Plots && Plotly.Plots.resize) {
        try {
            Plotly.Plots.resize(plotDivElement);
        } catch (resizeError) {
            console.warn("Plotly resize error after react:", resizeError);
        }
    }
}

// displaySingleSequenceClassification (from "Ultra Fix 1")
function displaySingleSequenceClassification(sequenceData, classificationResults, containerElement) {
    const resultCard = document.createElement('div');
    resultCard.className = 'sequence-result-card-enhanced';

    let content = `<div class="sequence-header-enhanced" title="${sequenceData.originalHeader || sequenceData.header}">${sequenceData.header} (ID: ${sequenceData.id || 'N/A'})</div>`;
    if(sequenceData.sequence) {
        content += `<div class="sequence-snippet-enhanced">${sequenceData.sequence.substring(0, 80)}${sequenceData.sequence.length > 80 ? '...' : ''}</div>`;
    }

    const topN = getConfigValue('classificationTopN');

    function renderScores(title, scores) {
        if (scores && scores.length > 0) {
            content += `<div class="classification-title-enhanced">${title}:</div>`;
            scores.slice(0, topN).forEach(res => {
                const percentage = (res.score * 100);
                const displayPercentage = percentage.toFixed(1);
                content += `<div class="small">${res.label}: 
                    <div class="score-bar-container-enhanced">
                        <div class="score-bar-value" style="width: ${Math.max(2, percentage)}%;" title="${displayPercentage}%">${percentage > 10 ? displayPercentage + '%' : ''}</div>
                    </div>
                </div>`;
            });
        }
    }

    renderScores('Biotype', classificationResults.biotype);
    renderScores('Species', classificationResults.species);
    renderScores('Custom Prompts', classificationResults.custom);
    
    const hasResults = (classificationResults.biotype && classificationResults.biotype.length > 0) ||
                       (classificationResults.species && classificationResults.species.length > 0) ||
                       (classificationResults.custom && classificationResults.custom.length > 0);

    if (!hasResults) {
        content += `<p class="text-muted-ultra small mt-2">No classification scores above threshold or an error occurred.</p>`;
    }

    resultCard.innerHTML = content;
    containerElement.appendChild(resultCard);
}

// displayTokenizationPreview (from "Ultra Fix 1")
function displayTokenizationPreview(sequenceType, originalInput, tokens, encodedIds, containerElement) {
    if (!containerElement) return;
    let previewHtml = `<div class="tokenization-preview-ultra">
        <h6><i class="fas fa-stream me-1"></i>Tokenization (${sequenceType})</h6>
        <p class="mb-1"><small><strong>Input:</strong> <span class="font-monospace">${originalInput.substring(0,60)}${originalInput.length > 60 ? '...' : ''}</span></small></p>
        <p class="mb-1"><small><strong>Tokens:</strong> ${tokens.slice(0,8).map(t => `<code>${t}</code>`).join(' ')}${tokens.length > 8 ? ' ... (' + tokens.length + ' total)' : ''}</small></p>
        <p class="mb-0"><small><strong>IDs:</strong> ${encodedIds.slice(0,8).map(id => `<span class="token-id">${id}</span>`).join(' ')}${encodedIds.length > 8 ? ' ...' : ''}</small></p>
    </div>`;
    containerElement.innerHTML = previewHtml;
}