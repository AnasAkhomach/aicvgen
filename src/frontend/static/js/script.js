// Global state
const appState = {
    sessionId: null,
    cvStructure: null,
    selectedItems: new Set(),
    currentStep: 1
};

// DOM elements
const elements = {
    initialForm: document.getElementById('initial-form'),
    loadingSpinner: document.getElementById('loading-spinner'),
    cvEditor: document.getElementById('cv-editor'),
    cvPreview: document.getElementById('cv-preview'),
    cvForm: document.getElementById('cv-form'),
    jobDescription: document.getElementById('job-description'),
    cvText: document.getElementById('cv-text'),
    useExistingCv: document.getElementById('use-existing-cv'),
    startFromScratch: document.getElementById('start-from-scratch'),
    existingCvInput: document.getElementById('existing-cv-input'),
    cvSections: document.getElementById('cv-sections'),
    regenerateSelected: document.getElementById('regenerate-selected'),
    exportCv: document.getElementById('export-cv'),
    markdownPreview: document.getElementById('markdown-preview'),
    downloadMd: document.getElementById('download-md'),
    backToEditor: document.getElementById('back-to-editor'),
    sessionInfo: document.getElementById('session-info'),
    sessionIdSpan: document.getElementById('session-id')
};

// Templates
const templates = {
    section: document.getElementById('section-template'),
    subsection: document.getElementById('subsection-template'),
    item: document.getElementById('item-template')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Set up event listeners
    setupEventListeners();
    
    // Check for existing session in URL or local storage
    const urlParams = new URLSearchParams(window.location.search);
    const sessionParam = urlParams.get('session');
    
    if (sessionParam) {
        loadExistingSession(sessionParam);
    }
});

// Set up all event listeners
function setupEventListeners() {
    // Form submission
    elements.cvForm.addEventListener('submit', handleFormSubmit);
    
    // CV input type toggle
    elements.useExistingCv.addEventListener('change', toggleCvInputType);
    elements.startFromScratch.addEventListener('change', toggleCvInputType);
    
    // Regenerate selected items
    elements.regenerateSelected.addEventListener('click', handleRegenerateSelected);
    
    // Export CV
    elements.exportCv.addEventListener('click', handleExportCv);
    
    // Download Markdown
    elements.downloadMd.addEventListener('click', handleDownloadMd);
    
    // Back to editor
    elements.backToEditor.addEventListener('click', handleBackToEditor);
}

// Toggle CV input type (existing vs from scratch)
function toggleCvInputType() {
    if (elements.useExistingCv.checked) {
        elements.existingCvInput.style.display = 'block';
    } else {
        elements.existingCvInput.style.display = 'none';
    }
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Show loading spinner
    setAppStep(0);
    
    try {
        // Prepare request data
        const requestData = {
            job_description: elements.jobDescription.value,
            cv_text: elements.useExistingCv.checked ? elements.cvText.value : '',
            start_from_scratch: elements.startFromScratch.checked
        };
        
        // Call API to parse CV and job description
        const response = await fetch('/api/cv/parse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save session ID
        appState.sessionId = data.session_id;
        updateSessionDisplay();
        
        // Load CV structure
        await loadCvStructure(data.session_id);
        
        // Show CV editor
        setAppStep(2);
        
    } catch (error) {
        console.error('Error submitting form:', error);
        alert('An error occurred while processing your request. Please try again.');
        setAppStep(1);
    }
}

// Load existing session
async function loadExistingSession(sessionId) {
    try {
        // Show loading spinner
        setAppStep(0);
        
        // Call API to load session
        const response = await fetch('/api/session/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Save session ID
        appState.sessionId = data.session_id;
        updateSessionDisplay();
        
        // Load CV structure
        await loadCvStructure(data.session_id);
        
        // Show CV editor
        setAppStep(2);
        
    } catch (error) {
        console.error('Error loading session:', error);
        alert('An error occurred while loading your session. Please try again.');
        setAppStep(1);
    }
}

// Load CV structure from API
async function loadCvStructure(sessionId) {
    try {
        const response = await fetch(`/api/cv/structure/${sessionId}`);
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        appState.cvStructure = data.sections;
        
        // Render CV structure
        renderCvStructure(data.sections);
        
    } catch (error) {
        console.error('Error loading CV structure:', error);
        throw error;
    }
}

// Render CV structure in the editor
function renderCvStructure(sections) {
    // Clear existing content
    elements.cvSections.innerHTML = '';
    
    // Sort sections by order
    const sortedSections = [...sections].sort((a, b) => a.order - b.order);
    
    // Create and append each section
    sortedSections.forEach(section => {
        const sectionElement = createSectionElement(section);
        elements.cvSections.appendChild(sectionElement);
    });
    
    // Add event listeners for section elements
    addSectionEventListeners();
}

// Create section element from template
function createSectionElement(section) {
    const template = templates.section.content.cloneNode(true);
    const sectionElement = template.querySelector('.cv-section');
    
    sectionElement.dataset.sectionId = section.id;
    sectionElement.querySelector('.section-name').textContent = section.name;
    sectionElement.querySelector('.section-checkbox').value = section.id;
    
    const itemsContainer = sectionElement.querySelector('.section-items');
    
    // Add direct items
    section.items.forEach(item => {
        const itemElement = createItemElement(item);
        itemsContainer.appendChild(itemElement);
    });
    
    // Add subsections
    section.subsections.forEach(subsection => {
        const subsectionElement = createSubsectionElement(subsection);
        itemsContainer.appendChild(subsectionElement);
    });
    
    return sectionElement;
}

// Create subsection element from template
function createSubsectionElement(subsection) {
    const template = templates.subsection.content.cloneNode(true);
    const subsectionElement = template.querySelector('.cv-subsection');
    
    subsectionElement.dataset.subsectionId = subsection.id;
    subsectionElement.querySelector('.subsection-name').textContent = subsection.name;
    
    const itemsContainer = subsectionElement.querySelector('.subsection-items');
    
    // Add items
    subsection.items.forEach(item => {
        const itemElement = createItemElement(item);
        itemsContainer.appendChild(itemElement);
    });
    
    return subsectionElement;
}

// Create item element from template
function createItemElement(item) {
    const template = templates.item.content.cloneNode(true);
    const itemElement = template.querySelector('.cv-item');
    
    itemElement.dataset.itemId = item.id;
    itemElement.querySelector('.item-content').value = item.content || '';
    
    // Set status badge
    const statusBadge = itemElement.querySelector('.item-status');
    statusBadge.textContent = formatStatus(item.status);
    statusBadge.classList.add(`status-${item.status.toLowerCase()}`);
    
    // Set feedback if available
    if (item.user_feedback) {
        itemElement.querySelector('.feedback-input').value = item.user_feedback;
    }
    
    return itemElement;
}

// Format status for display
function formatStatus(status) {
    const statusMap = {
        'TO_REGENERATE': 'Needs Regeneration',
        'REGENERATED': 'Regenerated',
        'GENERATED': 'Generated',
        'EDITED': 'Edited'
    };
    
    return statusMap[status] || status;
}

// Add event listeners to section elements
function addSectionEventListeners() {
    // Section checkboxes
    document.querySelectorAll('.section-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', handleSectionCheckboxChange);
    });
    
    // Item content textareas
    document.querySelectorAll('.item-content').forEach(textarea => {
        textarea.addEventListener('change', handleItemContentChange);
    });
    
    // Regenerate item buttons
    document.querySelectorAll('.regenerate-item').forEach(button => {
        button.addEventListener('click', handleRegenerateItem);
    });
    
    // Feedback inputs
    document.querySelectorAll('.feedback-input').forEach(input => {
        input.addEventListener('change', handleFeedbackChange);
    });
}

// Handle section checkbox change
function handleSectionCheckboxChange(event) {
    const checkbox = event.target;
    const sectionId = checkbox.value;
    const itemCheckboxes = document.querySelectorAll(`.cv-section[data-section-id="${sectionId}"] .item-checkbox`);
    
    if (checkbox.checked) {
        // Get all item IDs in this section
        const section = appState.cvStructure.find(s => s.id === sectionId);
        if (section) {
            // Add direct items
            section.items.forEach(item => {
                appState.selectedItems.add(item.id);
            });
            
            // Add subsection items
            section.subsections.forEach(subsection => {
                subsection.items.forEach(item => {
                    appState.selectedItems.add(item.id);
                });
            });
        }
    } else {
        // Get all item IDs in this section
        const section = appState.cvStructure.find(s => s.id === sectionId);
        if (section) {
            // Remove direct items
            section.items.forEach(item => {
                appState.selectedItems.delete(item.id);
            });
            
            // Remove subsection items
            section.subsections.forEach(subsection => {
                subsection.items.forEach(item => {
                    appState.selectedItems.delete(item.id);
                });
            });
        }
    }
    
    updateRegenerateButton();
}

// Handle item content change
async function handleItemContentChange(event) {
    const textarea = event.target;
    const itemElement = textarea.closest('.cv-item');
    const itemId = itemElement.dataset.itemId;
    const content = textarea.value;
    
    try {
        const response = await fetch('/api/cv/item/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                item_id: itemId,
                content: content,
                status: 'EDITED'
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        // Update status badge
        const statusBadge = itemElement.querySelector('.item-status');
        statusBadge.textContent = 'Edited';
        statusBadge.className = 'item-status badge status-edited';
        
    } catch (error) {
        console.error('Error updating item content:', error);
        alert('An error occurred while updating the item. Please try again.');
    }
}

// Handle regenerate item button click
async function handleRegenerateItem(event) {
    const button = event.target.closest('.regenerate-item');
    const itemElement = button.closest('.cv-item');
    const itemId = itemElement.dataset.itemId;
    const feedbackInput = itemElement.querySelector('.feedback-input');
    const feedback = feedbackInput.value;
    
    try {
        // Update item status to TO_REGENERATE
        await fetch('/api/cv/item/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                item_id: itemId,
                status: 'TO_REGENERATE',
                feedback: feedback
            })
        });
        
        // Regenerate the item
        const response = await fetch('/api/cv/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                item_ids: [itemId]
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        // Reload CV structure
        await loadCvStructure(appState.sessionId);
        
    } catch (error) {
        console.error('Error regenerating item:', error);
        alert('An error occurred while regenerating the item. Please try again.');
    }
}

// Handle feedback change
async function handleFeedbackChange(event) {
    const input = event.target;
    const itemElement = input.closest('.cv-item');
    const itemId = itemElement.dataset.itemId;
    const feedback = input.value;
    
    try {
        const response = await fetch('/api/cv/item/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                item_id: itemId,
                feedback: feedback
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
    } catch (error) {
        console.error('Error updating feedback:', error);
        alert('An error occurred while updating feedback. Please try again.');
    }
}

// Handle regenerate selected button click
async function handleRegenerateSelected() {
    if (appState.selectedItems.size === 0) {
        return;
    }
    
    try {
        // Show loading spinner
        setAppStep(0);
        
        const response = await fetch('/api/cv/regenerate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                item_ids: Array.from(appState.selectedItems)
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        // Clear selected items
        appState.selectedItems.clear();
        updateRegenerateButton();
        
        // Reload CV structure
        await loadCvStructure(appState.sessionId);
        
        // Show CV editor
        setAppStep(2);
        
    } catch (error) {
        console.error('Error regenerating items:', error);
        alert('An error occurred while regenerating items. Please try again.');
        setAppStep(2);
    }
}

// Handle export CV button click
async function handleExportCv() {
    try {
        // Show loading spinner
        setAppStep(0);
        
        const response = await fetch('/api/cv/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: appState.sessionId,
                format: 'markdown'
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Render markdown preview
        elements.markdownPreview.innerHTML = marked.parse(data.content);
        
        // Store content for download
        appState.exportedContent = data.content;
        
        // Show preview
        setAppStep(3);
        
    } catch (error) {
        console.error('Error exporting CV:', error);
        alert('An error occurred while exporting the CV. Please try again.');
        setAppStep(2);
    }
}

// Handle download markdown button click
function handleDownloadMd() {
    if (!appState.exportedContent) {
        return;
    }
    
    const blob = new Blob([appState.exportedContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `tailored_cv_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    
    // Cleanup
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
}

// Handle back to editor button click
function handleBackToEditor() {
    setAppStep(2);
}

// Update regenerate button state
function updateRegenerateButton() {
    elements.regenerateSelected.disabled = appState.selectedItems.size === 0;
}

// Update session display
function updateSessionDisplay() {
    if (appState.sessionId) {
        elements.sessionInfo.classList.remove('d-none');
        elements.sessionIdSpan.textContent = appState.sessionId;
        
        // Update URL with session ID without reloading
        const url = new URL(window.location);
        url.searchParams.set('session', appState.sessionId);
        window.history.pushState({}, '', url);
    } else {
        elements.sessionInfo.classList.add('d-none');
    }
}

// Set application step
function setAppStep(step) {
    appState.currentStep = step;
    
    // Hide all steps
    elements.initialForm.classList.add('d-none');
    elements.loadingSpinner.classList.add('d-none');
    elements.cvEditor.classList.add('d-none');
    elements.cvPreview.classList.add('d-none');
    
    // Show current step
    switch (step) {
        case 0: // Loading
            elements.loadingSpinner.classList.remove('d-none');
            break;
        case 1: // Initial Form
            elements.initialForm.classList.remove('d-none');
            break;
        case 2: // CV Editor
            elements.cvEditor.classList.remove('d-none');
            break;
        case 3: // CV Preview
            elements.cvPreview.classList.remove('d-none');
            break;
    }
} 