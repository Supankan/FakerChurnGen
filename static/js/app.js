const state = {
    selectedPreset: null,
    presetConfig: null,
    columns: [],
    relationships: [],
    numRows: 10000,
    previewRows: 50,
    targetChurnRate: 0.265,
    noiseLevel: 0.3,
    seed: null,
    searchFilter: '',
    ruleStats: []
};

document.addEventListener('DOMContentLoaded', () => {
    loadPresets();
    setupEventListeners();
});

function setupEventListeners() {
    const presetSelect = document.getElementById('preset-select');
    presetSelect.addEventListener('change', (e) => onPresetChange(e.target.value));

    // Row Count input & Preset Pills
    const numRowsInput = document.getElementById('num-rows');
    numRowsInput.addEventListener('input', (e) => {
        const val = parseInt(e.target.value) || 1000;
        state.numRows = val;
        syncRowPills(val);
    });

    const pillButtons = document.querySelectorAll('#row-presets-pills .btn-pill');
    pillButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const rows = parseInt(btn.dataset.rows);
            state.numRows = rows;
            numRowsInput.value = rows;
            pillButtons.forEach(b => b.classList.toggle('active', b === btn));
        });
    });

    // Preview Rows Select
    const previewRowsSelect = document.getElementById('preview-rows-select');
    previewRowsSelect.addEventListener('change', (e) => {
        state.previewRows = parseInt(e.target.value) || 50;
    });

    // Churn & Noise Sliders
    const churnSlider = document.getElementById('target-churn');
    churnSlider.addEventListener('input', (e) => {
        document.getElementById('target-churn-val').innerText = e.target.value;
        state.targetChurnRate = parseFloat(e.target.value) / 100;
    });

    const noiseSlider = document.getElementById('noise-level');
    noiseSlider.addEventListener('input', (e) => {
        document.getElementById('noise-level-val').innerText = parseFloat(e.target.value).toFixed(2);
        state.noiseLevel = parseFloat(e.target.value);
    });

    // Seed
    document.getElementById('seed').addEventListener('change', (e) => {
        state.seed = e.target.value ? parseInt(e.target.value) : null;
    });

    // Column Search Filter
    const searchInput = document.getElementById('column-search');
    searchInput.addEventListener('input', (e) => {
        state.searchFilter = e.target.value.toLowerCase().trim();
        renderColumns();
    });

    // Global Select All / Deselect All
    document.getElementById('btn-select-all-cols').addEventListener('click', () => {
        state.columns.forEach(c => c.enabled = true);
        renderColumns();
        populateModalColumnSelects();
    });
    document.getElementById('btn-deselect-all-cols').addEventListener('click', () => {
        state.columns.forEach(c => c.enabled = false);
        renderColumns();
        populateModalColumnSelects();
    });

    // Action Buttons
    document.getElementById('btn-preview').addEventListener('click', preview);
    document.getElementById('btn-download').addEventListener('click', download);
    const btnTrainMl = document.getElementById('btn-train-ml');
    if (btnTrainMl) {
        btnTrainMl.addEventListener('click', trainBaselineModel);
    }
    
    // Modal controls
    const modal = document.getElementById('rule-modal');
    document.getElementById('btn-add-rule').addEventListener('click', () => {
        populateModalColumnSelects();
        ensureInitialConditionRow();
        modal.classList.remove('hidden');
    });
    
    const closeModal = () => modal.classList.add('hidden');
    document.getElementById('modal-close').addEventListener('click', closeModal);
    window.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) closeModal();
    });

    // Rule type switch
    document.getElementById('rule-type').addEventListener('change', (e) => {
        const type = e.target.value;
        document.getElementById('form-group-interaction').classList.toggle('hidden', type !== 'interaction');
        document.getElementById('form-group-correlation').classList.toggle('hidden', type !== 'correlation');
        document.getElementById('form-group-conditional').classList.toggle('hidden', type !== 'conditional');
        document.getElementById('form-group-nonlinear').classList.toggle('hidden', type !== 'nonlinear');
        if (type === 'interaction') {
            ensureInitialConditionRow();
        }
    });

    // Add condition row button
    document.getElementById('btn-add-condition-row').addEventListener('click', () => {
        addInteractionConditionRow();
    });

    document.getElementById('btn-save-rule').addEventListener('click', addRelationship);
    
    // Rule templates
    const templateButtons = document.querySelectorAll('.btn-template');
    templateButtons.forEach(btn => {
        btn.addEventListener('click', () => applyRuleTemplate(btn.dataset.template));
    });

    // JSON Toggle
    document.getElementById('json-toggle').addEventListener('change', toggleJsonEditor);
    document.getElementById('btn-apply-json').addEventListener('click', applyJsonRules);
}

function syncRowPills(currentVal) {
    const pillButtons = document.querySelectorAll('#row-presets-pills .btn-pill');
    pillButtons.forEach(btn => {
        const rows = parseInt(btn.dataset.rows);
        btn.classList.toggle('active', rows === currentVal);
    });
}

function populateModalColumnSelects() {
    const activeCols = state.columns.filter(c => c.enabled);
    const selects = document.querySelectorAll('.modal-col-select');
    selects.forEach(select => {
        const curVal = select.value;
        select.innerHTML = '';
        activeCols.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c.name;
            opt.textContent = `${c.name} (${c.type || 'feat'})`;
            select.appendChild(opt);
        });
        if (curVal && activeCols.some(c => c.name === curVal)) {
            select.value = curVal;
        }
    });
}

function ensureInitialConditionRow() {
    const container = document.getElementById('interaction-conditions-list');
    if (container.children.length === 0) {
        addInteractionConditionRow();
    }
}

function addInteractionConditionRow(defaultCol = '', defaultOp = '==', defaultVal = '') {
    const container = document.getElementById('interaction-conditions-list');
    const row = document.createElement('div');
    row.className = 'condition-row';

    const colSelect = document.createElement('select');
    colSelect.className = 'modal-col-select condition-col-select';
    
    const activeCols = state.columns.filter(c => c.enabled);
    activeCols.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = c.name;
        colSelect.appendChild(opt);
    });
    if (defaultCol && activeCols.some(c => c.name === defaultCol)) {
        colSelect.value = defaultCol;
    }

    const opSelect = document.createElement('select');
    opSelect.className = 'condition-op-select';
    ['==', '!=', '>', '<', '>=', '<='].forEach(op => {
        const opt = document.createElement('option');
        opt.value = op;
        opt.textContent = op;
        opSelect.appendChild(opt);
    });
    if (defaultOp) opSelect.value = defaultOp;

    const valInput = document.createElement('input');
    valInput.type = 'text';
    valInput.className = 'condition-val-input';
    valInput.placeholder = 'Value (e.g. 50 or True)';
    if (defaultVal) valInput.value = defaultVal;

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'btn-remove-condition';
    removeBtn.innerHTML = '&times;';
    removeBtn.title = 'Remove condition';
    removeBtn.addEventListener('click', () => {
        if (container.children.length > 1) {
            row.remove();
        } else {
            valInput.value = '';
        }
    });

    row.appendChild(colSelect);
    row.appendChild(opSelect);
    row.appendChild(valInput);
    row.appendChild(removeBtn);
    container.appendChild(row);
}

async function loadPresets() {
    try {
        const res = await fetch('/api/presets');
        const presets = await res.json();
        
        const select = document.getElementById('preset-select');
        select.innerHTML = '';
        presets.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = p.name.charAt(0).toUpperCase() + p.name.slice(1);
            select.appendChild(opt);
        });
        
        if (presets.length > 0) {
            onPresetChange(presets[0].name);
        }
    } catch (err) {
        console.error("Failed to load presets", err);
    }
}

async function onPresetChange(name) {
    try {
        const res = await fetch(`/api/presets/${name}`);
        const config = await res.json();
        
        state.selectedPreset = name;
        state.presetConfig = config;
        
        document.getElementById('preset-description').innerText = config.description;
        
        // Setup slider values
        state.targetChurnRate = config.target_churn_rate;
        const churnPercent = (config.target_churn_rate * 100).toFixed(1);
        document.getElementById('target-churn').value = churnPercent;
        document.getElementById('target-churn-val').innerText = churnPercent;
        
        if (config.noise_level !== undefined) {
            state.noiseLevel = config.noise_level;
            document.getElementById('noise-level').value = config.noise_level;
            document.getElementById('noise-level-val').innerText = parseFloat(config.noise_level).toFixed(2);
        }
        
        // Setup columns
        state.columns = config.columns.map(c => ({...c, enabled: true}));
        renderColumns();
        
        // Setup rules (with enabled: true by default)
        state.relationships = JSON.parse(JSON.stringify(config.relationships)).map(r => ({
            ...r,
            enabled: r.enabled !== false
        }));
        renderRelationships();
        
        populateModalColumnSelects();
    } catch (err) {
        console.error(`Failed to load preset ${name}`, err);
    }
}

function updateColumnCounter() {
    const total = state.columns.length;
    const enabled = state.columns.filter(c => c.enabled).length;
    document.getElementById('column-counter').innerText = `${enabled} / ${total} Selected`;
}

function renderColumns() {
    const container = document.getElementById('categories-container');
    container.innerHTML = '';
    
    // Group columns by category
    const categoriesMap = {};
    state.columns.forEach((col, idx) => {
        const cat = col.category || 'General';
        if (!categoriesMap[cat]) categoriesMap[cat] = [];
        categoriesMap[cat].push({ col, idx });
    });

    const filter = state.searchFilter;
    let anyVisible = false;

    Object.entries(categoriesMap).forEach(([categoryName, items]) => {
        const visibleItems = items.filter(({ col }) => {
            if (!filter) return true;
            return col.name.toLowerCase().includes(filter) || categoryName.toLowerCase().includes(filter);
        });

        if (visibleItems.length === 0) return;
        anyVisible = true;

        const catCard = document.createElement('div');
        catCard.className = 'category-card';

        const allEnabled = visibleItems.every(({ col }) => col.enabled);

        const header = document.createElement('div');
        header.className = 'category-header';
        
        const titleWrap = document.createElement('div');
        titleWrap.className = 'category-title-wrap';
        titleWrap.innerHTML = `
            <span class="category-title">${categoryName}</span>
            <span class="category-count">(${visibleItems.filter(i => i.col.enabled).length}/${visibleItems.length})</span>
        `;

        const toggleBtn = document.createElement('button');
        toggleBtn.type = 'button';
        toggleBtn.className = 'category-toggle-all';
        toggleBtn.innerText = allEnabled ? 'Deselect All' : 'Select All';
        toggleBtn.addEventListener('click', () => {
            const targetState = !allEnabled;
            visibleItems.forEach(({ col }) => col.enabled = targetState);
            renderColumns();
            populateModalColumnSelects();
        });

        header.appendChild(titleWrap);
        header.appendChild(toggleBtn);
        catCard.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'category-items-grid';

        visibleItems.forEach(({ col, idx }) => {
            const div = document.createElement('div');
            div.className = 'checkbox-item';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = `col-${idx}`;
            checkbox.checked = col.enabled;
            checkbox.addEventListener('change', (e) => {
                state.columns[idx].enabled = e.target.checked;
                updateColumnCounter();
                populateModalColumnSelects();
                titleWrap.querySelector('.category-count').innerText = 
                    `(${visibleItems.filter(i => i.col.enabled).length}/${visibleItems.length})`;
                toggleBtn.innerText = visibleItems.every(i => i.col.enabled) ? 'Deselect All' : 'Select All';
            });

            const label = document.createElement('label');
            label.htmlFor = `col-${idx}`;
            label.innerText = col.name;
            label.title = col.name;

            const badge = document.createElement('span');
            badge.className = `badge ${col.type ? col.type.toLowerCase() : 'categorical'}`;
            badge.innerText = col.type || 'feat';

            div.appendChild(checkbox);
            div.appendChild(label);
            div.appendChild(badge);
            grid.appendChild(div);
        });

        catCard.appendChild(grid);
        container.appendChild(catCard);
    });

    if (!anyVisible) {
        container.innerHTML = '<p class="help-text" style="text-align: center; padding: 1.5rem;">No features matching search filter.</p>';
    }

    updateColumnCounter();
}

function renderRelationships() {
    const list = document.getElementById('rules-list');
    list.innerHTML = '';
    
    if (state.relationships.length === 0) {
        list.innerHTML = '<p class="help-text">No causal relationship rules defined.</p>';
        return;
    }
    
    state.relationships.forEach((rule, idx) => {
        const card = document.createElement('div');
        const isEnabled = rule.enabled !== false;
        
        let naturalText = '';
        let codeDetails = '';

        if (rule.type === 'correlation') {
            const cols = rule.columns ? rule.columns.join(' ↔ ') : 'N/A';
            const r = rule.matrix ? rule.matrix[0][1] : (rule.parameter || '0.5');
            naturalText = `Impose Pearson correlation (r = ${r}) between features`;
            codeDetails = `Columns: ${cols}`;
        } else if (rule.type === 'conditional') {
            const cond = `${rule.condition_col || ''} ${rule.condition_val !== undefined ? rule.condition_val : ''}`;
            naturalText = `When ${cond}, generate ${rule.target} using ${rule.dist} distribution`;
            codeDetails = rule.params ? JSON.stringify(rule.params) : '';
        } else if (rule.type === 'interaction') {
            let condParts = [];
            if (Array.isArray(rule.conditions)) {
                condParts = rule.conditions.map(c => `${c.col} ${c.op || '=='} ${c.val}`);
            } else if (typeof rule.conditions === 'object') {
                condParts = Object.entries(rule.conditions).map(([k, v]) => `${k} ${v}`);
            }
            const boost = rule.churn_boost !== undefined ? rule.churn_boost : 0.0;
            const sign = boost > 0 ? '+' : '';
            naturalText = `WHEN (${condParts.join(' AND ')}) THEN Churn Risk ${sign}${boost}`;
            codeDetails = `Composite multi-condition trigger`;
        } else if (rule.type === 'nonlinear') {
            naturalText = `Apply ${rule.transform} curve to ${rule.column}`;
            codeDetails = rule.params ? JSON.stringify(rule.params) : '';
        }

        // Header elements
        const headerRow = document.createElement('div');
        headerRow.className = 'rule-header-row';

        const toggleWrap = document.createElement('div');
        toggleWrap.className = 'rule-toggle-wrap';

        const chk = document.createElement('input');
        chk.type = 'checkbox';
        chk.id = `rule-enable-${idx}`;
        chk.checked = isEnabled;

        const lbl = document.createElement('label');
        lbl.htmlFor = `rule-enable-${idx}`;
        lbl.textContent = isEnabled ? 'Active' : 'Muted';

        toggleWrap.appendChild(chk);
        toggleWrap.appendChild(lbl);

        const actionsDiv = document.createElement('div');
        actionsDiv.style.display = 'flex';
        actionsDiv.style.alignItems = 'center';
        actionsDiv.style.gap = '0.5rem';

        const typeBadge = document.createElement('span');
        typeBadge.className = 'badge rule';
        typeBadge.textContent = rule.type;
        actionsDiv.appendChild(typeBadge);

        // Telemetry badges (from latest preview generation)
        const ruleStat = state.ruleStats && state.ruleStats.find(s => s.index === idx);
        let isRuleActive = isEnabled;
        if (ruleStat && isEnabled) {
            if (ruleStat.active === false) {
                isRuleActive = false;
                const warnBadge = document.createElement('span');
                warnBadge.className = 'badge rule-warning';
                warnBadge.style.background = 'rgba(196, 93, 76, 0.2)';
                warnBadge.style.color = 'var(--accent-clay)';
                warnBadge.style.border = '1px solid var(--accent-clay)';
                warnBadge.textContent = ruleStat.warning || 'Inactive: Missing Column';
                actionsDiv.appendChild(warnBadge);
            } else if (ruleStat.rows_matched === 0 && rule.type !== 'correlation') {
                const zeroBadge = document.createElement('span');
                zeroBadge.className = 'badge rule-warning';
                zeroBadge.style.background = 'rgba(217, 119, 6, 0.15)';
                zeroBadge.style.color = 'var(--accent-amber)';
                zeroBadge.style.border = '1px solid var(--accent-amber)';
                zeroBadge.textContent = '0 Matches';
                actionsDiv.appendChild(zeroBadge);
            } else if (ruleStat.rows_matched > 0 && rule.type !== 'correlation') {
                const matchBadge = document.createElement('span');
                matchBadge.className = 'badge rule-match';
                matchBadge.style.background = 'rgba(78, 128, 93, 0.2)';
                matchBadge.style.color = 'var(--accent-sage)';
                matchBadge.style.border = '1px solid var(--accent-sage)';
                matchBadge.textContent = `Matched ${ruleStat.rows_matched.toLocaleString()} rows`;
                actionsDiv.appendChild(matchBadge);
            }

            if (rule.type === 'correlation' && ruleStat.target_r !== undefined && ruleStat.realized_r !== undefined) {
                const corrBadge = document.createElement('span');
                corrBadge.className = 'badge rule-match';
                corrBadge.style.background = 'rgba(217, 119, 6, 0.15)';
                corrBadge.style.color = 'var(--accent-amber)';
                corrBadge.style.border = '1px solid var(--accent-amber)';
                corrBadge.textContent = `Target r: ${ruleStat.target_r} | Realized r: ${ruleStat.realized_r}`;
                actionsDiv.appendChild(corrBadge);
            }
        }

        card.className = `rule-card ${isRuleActive ? '' : 'disabled'}`;

        const delBtn = document.createElement('button');
        delBtn.className = 'btn-delete';
        delBtn.title = 'Delete rule';
        delBtn.textContent = '×';
        actionsDiv.appendChild(delBtn);

        headerRow.appendChild(toggleWrap);
        headerRow.appendChild(actionsDiv);

        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'rule-natural-summary';
        summaryDiv.textContent = naturalText;

        const codeDiv = document.createElement('div');
        codeDiv.className = 'rule-param-code';
        codeDiv.textContent = codeDetails;

        card.appendChild(headerRow);
        card.appendChild(summaryDiv);
        card.appendChild(codeDiv);

        // Toggle checkbox listener
        chk.addEventListener('change', (e) => {
            state.relationships[idx].enabled = e.target.checked;
            renderRelationships();
        });

        // Delete button listener
        delBtn.addEventListener('click', () => {
            removeRelationship(idx);
        });

        list.appendChild(card);
    });
}

function removeRelationship(index) {
    state.relationships.splice(index, 1);
    renderRelationships();
}

function addRelationship() {
    const type = document.getElementById('rule-type').value;
    let newRule = null;
    
    if (type === 'interaction') {
        const rows = document.querySelectorAll('#interaction-conditions-list .condition-row');
        const conditions = [];
        rows.forEach(r => {
            const col = r.querySelector('.condition-col-select').value;
            const op = r.querySelector('.condition-op-select').value;
            const val = r.querySelector('.condition-val-input').value.trim();
            if (col && val) {
                conditions.push({ col, op, val });
            }
        });

        if (conditions.length === 0) {
            alert("Please specify at least one condition.");
            return;
        }

        const rawBoost = document.getElementById('inter-boost').value;
        const boost = (rawBoost !== '' && !isNaN(parseFloat(rawBoost))) ? parseFloat(rawBoost) : 1.0;
        newRule = {
            type: "interaction",
            conditions: conditions,
            churn_boost: boost,
            enabled: true
        };
    } else if (type === 'correlation') {
        const colA = document.getElementById('corr-col-a').value;
        const colB = document.getElementById('corr-col-b').value;
        const rawCoef = document.getElementById('corr-coef').value;
        const coef = (rawCoef !== '' && !isNaN(parseFloat(rawCoef))) ? parseFloat(rawCoef) : 0.7;
        
        if (colA === colB) {
            alert("Please select two distinct columns for correlation.");
            return;
        }
        newRule = {
            type: "correlation",
            columns: [colA, colB],
            matrix: [[1.0, coef], [coef, 1.0]],
            enabled: true
        };
    } else if (type === 'conditional') {
        const target = document.getElementById('cond-target').value;
        const source = document.getElementById('cond-source').value;
        const op = document.getElementById('cond-op').value;
        const rawVal = document.getElementById('cond-val').value.trim();
        const dist = document.getElementById('cond-dist').value;
        let params = {};
        try {
            const pText = document.getElementById('cond-params').value.trim();
            if (pText) params = JSON.parse(pText);
        } catch (e) {
            alert("Invalid JSON in Distribution Parameters.");
            return;
        }
        
        const condVal = (op === '==' ? rawVal : `${op}${rawVal}`);
        newRule = {
            type: "conditional",
            target: target,
            condition_col: source,
            condition_val: condVal,
            dist: dist,
            params: params,
            enabled: true
        };
    } else if (type === 'nonlinear') {
        const col = document.getElementById('nonlin-col').value;
        const transform = document.getElementById('nonlin-type').value;
        let params = {};
        try {
            const pText = document.getElementById('nonlin-params').value.trim();
            if (pText) params = JSON.parse(pText);
        } catch (e) {
            alert("Invalid JSON in Parameters.");
            return;
        }
        newRule = {
            type: "nonlinear",
            column: col,
            transform: transform,
            params: params,
            enabled: true
        };
    }
    
    if (newRule) {
        state.relationships.push(newRule);
        renderRelationships();
        document.getElementById('rule-modal').classList.add('hidden');
    }
}

function applyRuleTemplate(templateKey) {
    const activeCols = state.columns.filter(c => c.enabled).map(c => c.name);
    let newRule = null;

    if (templateKey === 'bathtub') {
        const timeCol = activeCols.find(c => c.includes('tenure') || c.includes('age') || c.includes('days')) || activeCols[0];
        newRule = {
            type: "nonlinear",
            column: timeCol,
            transform: "u_curve",
            params: { center: 36, scale: 0.002 },
            enabled: true
        };
    } else if (templateKey === 'price-shock') {
        const chargeCol = activeCols.find(c => c.includes('charge') || c.includes('fee') || c.includes('surge') || c.includes('price')) || activeCols[0];
        newRule = {
            type: "interaction",
            conditions: [{ col: chargeCol, op: ">=", val: "80" }],
            churn_boost: 1.2,
            enabled: true
        };
    } else if (templateKey === 'inactivity') {
        const activeCol = activeCols.find(c => c.includes('login') || c.includes('visit') || c.includes('session') || c.includes('rides') || c.includes('hours')) || activeCols[0];
        newRule = {
            type: "interaction",
            conditions: [{ col: activeCol, op: "<=", val: "2" }],
            churn_boost: 1.4,
            enabled: true
        };
    } else if (templateKey === 'loyalty-anchor') {
        const anchorCol = activeCols.find(c => c.includes('contract') || c.includes('partner') || c.includes('clan') || c.includes('tier') || c.includes('annual')) || activeCols[0];
        newRule = {
            type: "interaction",
            conditions: [{ col: anchorCol, op: "==", val: "Two year" }],
            churn_boost: -1.0,
            enabled: true
        };
    }

    if (newRule) {
        state.relationships.unshift(newRule);
        renderRelationships();
    }
}

function toggleJsonEditor(e) {
    const showJson = e.target.checked;
    if (showJson) {
        document.getElementById('rules-ui-view').classList.add('hidden');
        document.getElementById('rules-json-view').classList.remove('hidden');
        document.getElementById('rules-json-textarea').value = JSON.stringify(state.relationships, null, 2);
    } else {
        document.getElementById('rules-ui-view').classList.remove('hidden');
        document.getElementById('rules-json-view').classList.add('hidden');
        renderRelationships();
    }
}

function applyJsonRules() {
    try {
        const val = document.getElementById('rules-json-textarea').value;
        const parsed = JSON.parse(val);
        if (Array.isArray(parsed)) {
            state.relationships = parsed;
            alert("Rules applied successfully.");
            document.getElementById('json-toggle').click();
        } else {
            alert("Rules must be a JSON array.");
        }
    } catch(err) {
        alert("Invalid JSON: " + err.message);
    }
}

function getConfig() {
    return {
        preset_name: state.selectedPreset,
        num_rows: state.numRows,
        preview_rows: state.previewRows,
        columns: state.columns.filter(c => c.enabled),
        relationships: state.relationships,
        target_churn_rate: state.targetChurnRate,
        noise_level: state.noiseLevel,
        seed: state.seed
    };
}

function showLoading() {
    document.getElementById('loading-spinner').classList.remove('hidden');
    document.getElementById('table-container').classList.add('hidden');
    document.getElementById('stats-container').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('loading-spinner').classList.add('hidden');
}

async function preview() {
    showLoading();
    try {
        const config = getConfig();
        config.num_rows = state.previewRows;
        
        const res = await fetch('/api/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!res.ok) {
            const errJson = await res.json().catch(() => null);
            throw new Error((errJson && errJson.error) ? errJson.error : await res.text());
        }
        
        const data = await res.json();
        
        state.ruleStats = (data.stats && data.stats.rule_stats) || [];
        renderRelationships();

        document.getElementById('stats-container').classList.remove('hidden');
        document.getElementById('stat-churn-rate').innerText = (data.stats.actual_churn_rate * 100).toFixed(1) + '%';
        document.getElementById('stat-cols').innerText = data.columns.length;
        document.getElementById('stat-rows').innerText = data.stats.num_rows;
        
        if (data.stats && data.stats.invariants_report) {
            const invEl = document.getElementById('stat-invariants');
            if (invEl) {
                invEl.innerText = `${data.stats.invariants_report.compliant_records_pct}% Verified`;
                invEl.style.color = data.stats.invariants_report.is_compliant ? 'var(--accent-sage)' : 'var(--accent-clay)';
            }
        }
        
        renderPreviewTable(data);
    } catch (err) {
        alert("Preview failed: " + err.message);
    } finally {
        hideLoading();
    }
}

function renderPreviewTable(data) {
    const container = document.getElementById('table-container');
    const thead = document.getElementById('table-head-row');
    const tbody = document.getElementById('table-body');
    
    thead.innerHTML = '';
    tbody.innerHTML = '';
    
    data.columns.forEach(col => {
        const th = document.createElement('th');
        th.innerText = col;
        thead.appendChild(th);
    });
    
    data.data.forEach(row => {
        const tr = document.createElement('tr');
        data.columns.forEach(col => {
            const td = document.createElement('td');
            td.innerText = row[col] !== null && row[col] !== undefined ? row[col] : '';
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    
    container.classList.remove('hidden');
}

async function download() {
    showLoading();
    try {
        const config = getConfig();
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (!res.ok) {
            const errJson = await res.json().catch(() => null);
            throw new Error((errJson && errJson.error) ? errJson.error : await res.text());
        }
        
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${config.preset_name || 'custom'}_churn_dataset.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
    } catch (err) {
        alert("Download failed: " + err.message);
    } finally {
        hideLoading();
    }
}

async function trainBaselineModel() {
    const spinner = document.getElementById('ml-loading-spinner');
    const resultsContainer = document.getElementById('ml-results-container');
    const btnTrain = document.getElementById('btn-train-ml');
    const badge = document.getElementById('ml-model-badge');

    const modelType = document.getElementById('ml-model-type').value;
    const trainRows = parseInt(document.getElementById('ml-train-rows').value, 10) || 2500;

    btnTrain.disabled = true;
    spinner.classList.remove('hidden');
    resultsContainer.classList.add('hidden');
    badge.innerText = 'Training in-memory...';
    badge.style.color = 'var(--accent-amber)';

    try {
        const config = getConfig();
        config.train_rows = trainRows;
        config.model_type = modelType;

        const res = await fetch('/api/train_baseline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!res.ok) {
            const errJson = await res.json().catch(() => null);
            throw new Error((errJson && errJson.error) ? errJson.error : await res.text());
        }

        const data = await res.json();
        renderMLResults(data);
    } catch (err) {
        alert("Baseline Training failed: " + err.message);
        badge.innerText = 'Training Failed';
        badge.style.color = 'var(--accent-clay)';
    } finally {
        btnTrain.disabled = false;
        spinner.classList.add('hidden');
    }
}

function renderMLResults(data) {
    const badge = document.getElementById('ml-model-badge');
    const modelLabel = data.model_type === 'hist_gb' ? 'HistGradientBoosting' : 'LogisticRegression';
    badge.innerText = `${modelLabel} (${data.sample_size.toLocaleString()} samples)`;
    badge.style.color = 'var(--accent-sage)';

    // Metrics
    document.getElementById('ml-roc-auc').innerText = Number(data.metrics.roc_auc).toFixed(3);
    document.getElementById('ml-pr-auc').innerText = Number(data.metrics.pr_auc).toFixed(3);
    document.getElementById('ml-f1').innerText = Number(data.metrics.f1).toFixed(3);
    document.getElementById('ml-acc').innerText = (Number(data.metrics.accuracy) * 100).toFixed(1) + '%';

    // Confusion Matrix
    const cm = data.confusion_matrix;
    document.getElementById('cm-tn').innerText = cm.tn.toLocaleString();
    document.getElementById('cm-tn-pct').innerText = `${cm.tn_pct}%`;
    document.getElementById('cm-fp').innerText = cm.fp.toLocaleString();
    document.getElementById('cm-fp-pct').innerText = `${cm.fp_pct}%`;
    document.getElementById('cm-fn').innerText = cm.fn.toLocaleString();
    document.getElementById('cm-fn-pct').innerText = `${cm.fn_pct}%`;
    document.getElementById('cm-tp').innerText = cm.tp.toLocaleString();
    document.getElementById('cm-tp-pct').innerText = `${cm.tp_pct}%`;

    // Diagnostic narrative
    document.getElementById('ml-diagnostic-text').innerText = data.diagnostic.narrative;

    // Feature Importances
    const listContainer = document.getElementById('feature-importance-list');
    listContainer.innerHTML = '';

    if (!data.feature_importances || data.feature_importances.length === 0) {
        listContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.85rem; padding: 0.5rem 0;">No feature importances computed.</div>';
    } else {
        data.feature_importances.forEach(item => {
            const row = document.createElement('div');
            row.className = 'feature-bar-item';
            
            const header = document.createElement('div');
            header.className = 'feature-bar-header';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'feature-bar-name';
            nameSpan.innerText = item.feature;
            
            const pctSpan = document.createElement('span');
            pctSpan.className = 'feature-bar-pct';
            pctSpan.innerText = `${item.pct}%`;
            
            header.appendChild(nameSpan);
            header.appendChild(pctSpan);
            
            const track = document.createElement('div');
            track.className = 'feature-bar-track';
            
            const fill = document.createElement('div');
            fill.className = 'feature-bar-fill';
            const fillPct = Math.min(Math.max(item.pct, 0), 100);
            fill.style.width = `${fillPct}%`;
            
            track.appendChild(fill);
            row.appendChild(header);
            row.appendChild(track);
            listContainer.appendChild(row);
        });
    }

    document.getElementById('ml-results-container').classList.remove('hidden');
}

