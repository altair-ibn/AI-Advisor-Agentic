document.addEventListener('DOMContentLoaded', function() {
    // Navigation handling
    const navItems = document.querySelectorAll('.list-group-item');
    const contentSections = document.querySelectorAll('.content-section');
    
    // Function to show a specific section
    function showSection(sectionId) {
        // Hide all sections
        contentSections.forEach(section => {
            section.classList.add('d-none');
        });
        
        // Show the selected section
        document.getElementById(sectionId).classList.remove('d-none');
        
        // Update active nav item
        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.id === 'nav-' + sectionId.replace('-section', '')) {
                item.classList.add('active');
            }
        });
    }
    
    // Add click handlers to nav items
    navItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.id.replace('nav-', '') + '-section';
            showSection(sectionId);
            
            // Load document list when navigating to documents section
            if (sectionId === 'documents-section') {
                loadDocuments();
            }
            
            // Load document dropdown for query section
            if (sectionId === 'query-section') {
                populateDocumentDropdown();
            }
        });
    });
    
    // Document list loading and handling
    const documentsTable = document.getElementById('documents-table-body');
    const refreshDocumentsBtn = document.getElementById('refresh-documents');
    
    // Load documents from API
    function loadDocuments() {
        documentsTable.innerHTML = '<tr><td colspan="5" class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></td></tr>';
        
        fetch('/api/documents')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    if (data.documents.length === 0) {
                        documentsTable.innerHTML = '<tr><td colspan="5" class="text-center">No documents found. Upload or scan for documents.</td></tr>';
                    } else {
                        let tableContent = '';
                        data.documents.forEach(doc => {
                            tableContent += `
                                <tr>
                                    <td>${doc.file_name}</td>
                                    <td>${doc.report_type || 'Unknown'}</td>
                                    <td>${doc.report_period || 'Unknown'}</td>
                                    <td>${doc.client_name || 'Unknown'}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary view-document" data-id="${doc.id}">
                                            <i class="fas fa-eye"></i>
                                        </button>
                                        <button class="btn btn-sm btn-outline-secondary query-document" data-id="${doc.id}">
                                            <i class="fas fa-question"></i>
                                        </button>
                                        ${doc.csv_path ? `
                                            <a href="/api/csv/${doc.csv_path.split('/').pop()}" class="btn btn-sm btn-outline-success" target="_blank">
                                                <i class="fas fa-table"></i>
                                            </a>
                                        ` : ''}
                                    </td>
                                </tr>
                            `;
                        });
                        documentsTable.innerHTML = tableContent;
                        
                        // Add event listeners to view document buttons
                        document.querySelectorAll('.view-document').forEach(button => {
                            button.addEventListener('click', function() {
                                viewDocument(this.dataset.id);
                            });
                        });
                        
                        // Add event listeners to query document buttons
                        document.querySelectorAll('.query-document').forEach(button => {
                            button.addEventListener('click', function() {
                                const docId = this.dataset.id;
                                showSection('query-section');
                                document.getElementById('document-select').value = docId;
                            });
                        });
                    }
                } else {
                    documentsTable.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Error: ${data.message}</td></tr>`;
                }
            })
            .catch(error => {
                documentsTable.innerHTML = `<tr><td colspan="5" class="text-center text-danger">Error: ${error.message}</td></tr>`;
            });
    }
    
    // Refresh documents button handler
    if (refreshDocumentsBtn) {
        refreshDocumentsBtn.addEventListener('click', loadDocuments);
    }
    
    // View document details
    function viewDocument(docId) {
        const modalBody = document.getElementById('document-modal-body');
        modalBody.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        
        // Show the modal
        const documentModal = new bootstrap.Modal(document.getElementById('document-modal'));
        documentModal.show();
        
        // Fetch document details
        fetch(`/api/documents/${docId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    const doc = data.document;
                    let content = `
                        <div class="row">
                            <div class="col-md-6">
                                <p><strong>File Name:</strong> ${doc.file_name}</p>
                                <p><strong>Report Type:</strong> ${doc.report_type || 'Unknown'}</p>
                                <p><strong>Report Period:</strong> ${doc.report_period || 'Unknown'}</p>
                                <p><strong>Client Name:</strong> ${doc.client_name || 'Unknown'}</p>
                                <p><strong>Entity:</strong> ${doc.entity || 'Unknown'}</p>
                            </div>
                            <div class="col-md-6">
                                <p><strong>File Path:</strong> ${doc.file_path}</p>
                                <p><strong>File Size:</strong> ${formatFileSize(doc.file_size)}</p>
                                <p><strong>Created:</strong> ${formatDate(doc.created_at)}</p>
                                <p><strong>Modified:</strong> ${formatDate(doc.modified_at)}</p>
                                ${doc.csv_path ? `<p><strong>CSV:</strong> <a href="/api/csv/${doc.csv_path.split('/').pop()}" target="_blank">${doc.csv_path.split('/').pop()}</a></p>` : ''}
                            </div>
                        </div>
                        <hr>
                        <div class="row">
                            <div class="col-12">
                                <p><strong>Description:</strong> ${doc.description || 'No description available'}</p>
                                <p><strong>Information Present:</strong></p>
                                <ul>
                                    ${doc.information_present && doc.information_present.length > 0 ? 
                                        doc.information_present.map(info => `<li>${info}</li>`).join('') : 
                                        '<li>No information available</li>'}
                                </ul>
                            </div>
                        </div>
                    `;
                    modalBody.innerHTML = content;
                } else {
                    modalBody.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                modalBody.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            });
    }
    
    // Format file size for display
    function formatFileSize(bytes) {
        if (!bytes) return 'Unknown';
        const units = ['B', 'KB', 'MB', 'GB'];
        let size = bytes;
        let unitIndex = 0;
        
        while (size >= 1024 && unitIndex < units.length - 1) {
            size /= 1024;
            unitIndex++;
        }
        
        return `${size.toFixed(2)} ${units[unitIndex]}`;
    }
    
    // Format date for display
    function formatDate(dateStr) {
        if (!dateStr) return 'Unknown';
        const date = new Date(dateStr);
        return date.toLocaleString();
    }
    
    // Upload form handling
    const uploadForm = document.getElementById('upload-form');
    const uploadStatus = document.getElementById('upload-status');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('file-upload');
            if (!fileInput.files.length) {
                uploadStatus.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> Please select a file to upload</div>';
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            // Update status
            uploadStatus.innerHTML = '<div class="alert alert-info"><div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div> Uploading and processing file...</div>';
            
            // Submit form
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    uploadStatus.innerHTML = `<div class="alert alert-success"><i class="fas fa-check-circle"></i> ${data.message}</div>`;
                    // Reset form
                    uploadForm.reset();
                } else {
                    uploadStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                uploadStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${error.message}</div>`;
            });
        });
    }
    
    // Scan directory form handling
    const scanForm = document.getElementById('scan-form');
    const scanStatus = document.getElementById('scan-status');
    
    if (scanForm) {
        scanForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const directoryPath = document.getElementById('directory-path').value;
            if (!directoryPath) {
                scanStatus.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> Please enter a directory path</div>';
                return;
            }
            
            const recursive = document.getElementById('recursive-scan').checked;
            
            // Update status
            scanStatus.innerHTML = '<div class="alert alert-info"><div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div> Scanning directory...</div>';
            
            // Submit request
            fetch('/api/scan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    directory_path: directoryPath,
                    recursive: recursive
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    scanStatus.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> Scan completed
                            <p class="mb-0 mt-2">Found ${data.count || 0} files</p>
                        </div>
                    `;
                    
                    // If files were found, show a list
                    if (data.files_found && data.files_found.length > 0) {
                        let fileList = '<div class="mt-3"><strong>Files found:</strong><ul class="list-group mt-2">';
                        data.files_found.forEach(file => {
                            fileList += `<li class="list-group-item d-flex justify-content-between align-items-center">
                                ${file.file_name}
                                <span class="badge bg-primary rounded-pill">${formatFileSize(file.file_size)}</span>
                            </li>`;
                        });
                        fileList += '</ul></div>';
                        scanStatus.innerHTML += fileList;
                    }
                } else {
                    scanStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                scanStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${error.message}</div>`;
            });
        });
    }
    
    // Query form handling
    const queryForm = document.getElementById('query-form');
    const queryStatus = document.getElementById('query-status');
    const queryResult = document.getElementById('query-result');
    
    // Populate document dropdown
    function populateDocumentDropdown() {
        const documentSelect = document.getElementById('document-select');
        
        // Clear existing options except the first one
        while (documentSelect.options.length > 1) {
            documentSelect.remove(1);
        }
        
        // Fetch documents
        fetch('/api/documents')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.documents.length > 0) {
                    data.documents.forEach(doc => {
                        const option = document.createElement('option');
                        option.value = doc.id;
                        option.textContent = `${doc.file_name} (${doc.report_type || 'Unknown'})`;
                        documentSelect.appendChild(option);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading documents for dropdown:', error);
            });
    }
    
    if (queryForm) {
        queryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const queryText = document.getElementById('query-text').value;
            if (!queryText) {
                queryStatus.innerHTML = '<div class="alert alert-warning"><i class="fas fa-exclamation-triangle"></i> Please enter a query</div>';
                return;
            }
            
            const docId = document.getElementById('document-select').value;
            
            // Update status
            queryStatus.innerHTML = '<div class="alert alert-info"><div class="spinner-border spinner-border-sm text-primary me-2" role="status"></div> Processing query...</div>';
            queryResult.innerHTML = '<p class="text-muted">Processing your query...</p>';
            
            // Submit request
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: queryText,
                    doc_id: docId || null
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    queryStatus.innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Query processed successfully</div>';
                    
                    // Format and display the result
                    if (data.result) {
                        // If there's document context, show it
                        let resultHtml = '';
                        if (data.document) {
                            resultHtml += `<div class="alert alert-secondary mb-3">
                                <p class="mb-1"><strong>Source:</strong> ${data.document.file_name}</p>
                                <p class="mb-1"><strong>Type:</strong> ${data.document.report_type || 'Unknown'}</p>
                                <p class="mb-0"><strong>Period:</strong> ${data.document.report_period || 'Unknown'}</p>
                            </div>`;
                        }
                        
                        // Add the answer
                        resultHtml += `<div class="answer-content">${formatAnswer(data.result)}</div>`;
                        queryResult.innerHTML = resultHtml;
                    } else {
                        queryResult.innerHTML = '<p class="text-muted">No result returned from query.</p>';
                    }
                } else {
                    queryStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${data.message}</div>`;
                    queryResult.innerHTML = '<p class="text-danger">Failed to process query. Please try again.</p>';
                }
            })
            .catch(error => {
                queryStatus.innerHTML = `<div class="alert alert-danger"><i class="fas fa-times-circle"></i> Error: ${error.message}</div>`;
                queryResult.innerHTML = '<p class="text-danger">An error occurred while processing your query.</p>';
            });
        });
    }
    
    // Format answer with basic Markdown-like syntax
    function formatAnswer(text) {
        if (!text) return '';
        
        // Replace newlines with <br>
        let formatted = text.replace(/\n/g, '<br>');
        
        // Bold text between ** or __
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        formatted = formatted.replace(/__(.*?)__/g, '<strong>$1</strong>');
        
        // Italic text between * or _
        formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
        formatted = formatted.replace(/_(.*?)_/g, '<em>$1</em>');
        
        // Lists
        formatted = formatted.replace(/^\s*\*\s+(.*)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');
        
        // Headers
        formatted = formatted.replace(/^# (.*?)$/gm, '<h1>$1</h1>');
        formatted = formatted.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
        formatted = formatted.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
        
        return formatted;
    }
    
    // Initialize the app by showing the documents section and loading documents
    loadDocuments();
}); 