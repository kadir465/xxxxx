// DOM Elements
let csvFileInput;
let fileNameDisplay;
let runModelButton;
let predictionLoading;
let avgPredictionElement;
let accuracyRateElement;
let predictionTableBody;
let uploadArea;
let exportResultsButton;

// Graphic Elements
let corrMatrix;
let confusionMatrix

// Refresh Buttons
let refreshHistogramButton;
let refreshScatterButton;
let refreshLineButton;
let refreshBarButton;

// Statics Elements
let fileCountElement;
let modelCountElement;
let accuracyRateStatElement;
let processingTimeElement;

// Hyper-Parameters
let n_estimatorsInput;
let learning_rateInput;
let max_depthInput;
let min_child_weightInput;
let subsampleInput;
let colsample_bytreeInput;

// App State
let appState;
let stateSwitchButton;
let tuneCard;
let predictCard;


document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    csvFileInput = document.getElementById('csvFile');
    fileNameDisplay = document.getElementById('fileName');
    runModelButton = document.getElementById('runModel');
    predictionLoading = document.getElementById('predictionLoading');
    avgPredictionElement = document.getElementById('avgPrediction');
    accuracyRateElement = document.getElementById('accuracyRate');
    predictionTableBody = document.getElementById('predictionTableBody');
    uploadArea = document.getElementById('uploadArea');
    exportResultsButton = document.getElementById('exportResults');

    // Graphic Elements
    corrMatrix = document.getElementById('corrMatrix');
    confusionMatrix = document.getElementById('confusionMatrix');

    // Refresh Buttons
    refreshHistogramButton = document.getElementById('refreshHistogram');
    refreshScatterButton = document.getElementById('refreshScatter');
    refreshLineButton = document.getElementById('refreshLine');
    refreshBarButton = document.getElementById('refreshBar');

    // Hyper-Parameters
    n_estimatorsInput = document.getElementById('n_estimators');
    learning_rateInput = document.getElementById('learning_rate');
    max_depthInput = document.getElementById('max_depth');
    min_child_weightInput = document.getElementById('min_child_weight');
    subsampleInput = document.getElementById('subsample');
    colsample_bytreeInput = document.getElementById('colsample_bytree');

    // Statics Elements
    fileCountElement = document.getElementById('fileCount');
    modelCountElement = document.getElementById('modelCount');
    accuracyRateStatElement = document.getElementById('accuracyRate');
    processingTimeElement = document.getElementById('processingTime');


    // App State
    appState = {
        uploadedFile: null,
        modelResults: null,
        chartsInitialized: false,
        isPredict: true,
    };
    stateSwitchButton = document.querySelector("#stateSwitcher");
    tuneCard = document.querySelector("#tuneCard");
    predictCard = document.querySelector("#predictCard");
});

document.addEventListener('DOMContentLoaded', () => {
    // CSV Download Operations
    exportResultsButton.addEventListener('click', function () {
        if (!appState.modelResults) {
            showNotification('You need to run a model first.', 'warning');
            return;
        }
        exportResults();
    });

    // CSV Upload Operations
    csvFileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        handleFileUpload(file);
    });

    // Drag-Drop Functionality
    uploadArea.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', function () {
        uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file && (file.type === 'text/csv' || file.name.endsWith('.csv'))) {
            handleFileUpload(file);
        } else {
            showNotification('Pls upload a CSV file.', 'warning');
        }

    });

    runModelButton.addEventListener('click', function () {
        if (!appState.uploadedFile) {
            showNotification('Pls upload a CSV file first.', 'warning');
            return;
        }
        if (runModelButton.innerText === 'Run Model') {
            getPredictions();
        } else if (runModelButton.innerText === 'Train Model') {
            trainModel();
        } else {
            alert('Something went wrong...');
        }


    });

    // Initial Notification
    setTimeout(() => {
        showNotification('Welcome to Dataway Analytics Dashboard!', 'info');
    }, 1000);
    stateSwitchButton.addEventListener('click', function () {
        if (appState.isPredict) {
            stateSwitchButton.innerText = 'Make Prediction';
            runModelButton.innerText = 'Train Model'
            predictCard.classList.add('hidden');
            tuneCard.classList.remove('hidden');
        } else {
            stateSwitchButton.innerText = 'Hyper-Parameter Tweaking';
            runModelButton.innerText = 'Run Model'
            predictCard.classList.remove('hidden');
            tuneCard.classList.add('hidden');
        }
        appState.isPredict = !appState.isPredict;
    })

    fetch('/get_header_info', {
        method: 'GET',
    }).then((response) => {
        if (!response.ok) {
            alert('Something went wrong!');
        }
        return response.json();
    }).then((data) => {
        if (!data.success) {
            alert(data.message ? data.message : 'Something went wrong!');
            return;
        }
        fileCountElement.innerText = data['file_count'];
    });
});


// File Upload Operations
function handleFileUpload(file) {
    if (file) {
        appState.uploadedFile = file;
        fileNameDisplay.textContent = file.name;

        showNotification(`"${file.name}" has been successfully uploaded!`, 'success');
    }
}

// Model Prediction
async function getPredictions() {
    predictionLoading.style.display = 'block';
    runModelButton.disabled = true;

    start = Date.now();
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", file);

    let resp = await fetch('/predict', {
        method: 'POST',
        body: formData,
    });
    if (!resp.ok) {
        alert('Something went wrong!');
        return;
    }
    let data = await resp.json();
    if (!data.success) {
        alert(data.message ? data.message : 'Something went wrong!');
        return;
    }
    let tableHTML = '';
    for (let i = 1; i <= data.predictions.length; i++) {
        const prediction = data.predictions[i - 1];
        tableHTML += `
            <tr>
                <td>${i}</td>
                <td style="text-align: right">${prediction}</td>
            </tr>
        `;
    }
    predictionTableBody.innerHTML = tableHTML;

    // Update App State
    appState.modelResults = {
        predictions: tableHTML
    };
    predictionLoading.style.display = 'None';
    runModelButton.disabled = false;
    processingTimeElement.innerText = (Date.now() - start) / 1000;
}

// Model Tweaking
async function trainModel() {
    predictionLoading.style.display = 'block';
    runModelButton.disabled = true;

    start = Date.now();
    const fileInput = document.getElementById("csvFile");
    const file = fileInput.files[0];

    const jsonData = JSON.stringify({
        n_estimators: n_estimatorsInput.value,
        learning_rate: learning_rateInput.value,
        max_depth: max_depthInput.value,
        min_child_weight: min_child_weightInput.value,
        subsample: subsampleInput.value,
        colsample_bytree: colsample_bytreeInput.value,
    });

    const formData = new FormData();
    formData.append("file", file);
    formData.append("hyper-parameters",jsonData);

    let resp = await fetch('/tweak', {
        method: 'POST',
        body: formData,
    });
    if (!resp.ok) {
        alert('Something went wrong!');
        return;
    }
    let data = await resp.json();
    if (!data.success) {
        alert(data.message ? data.message : 'Something went wrong!');
        return;
    }
    console.log("Fuck Yeaaaah");
    updateHeatmapChart(data['confusionMatrixInfo']['values'], confusionMatrix);
    return;
    updateHeatmapChart(data['correlationMatrixInfo']['values'], corrMatrix);
}


function exportResults() {
    let csvContent = "ID,Prediction\n";

    const rows = predictionTableBody.querySelectorAll('tr');
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        const rowData = Array.from(cells).map(cell => {
            return cell.textContent;
        });
        csvContent += rowData.join(',') + '\n';
    });

    // Blob oluştur ve indirme linki yarat
    const blob = new Blob([csvContent], {type: 'text/csv;charset=utf-8;'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'model_prediction_results.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    showNotification('Results exported successfully!', 'success');
}

// Notification System
function showNotification(message, type = 'info') {
    // Mevcut bildirimleri temizle
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => {
        notification.remove();
    });

    // Yeni bildirim oluştur
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close">
            <i class="fas fa-times"></i>
        </button>
    `;

    // Stil ekle (eğer henüz eklenmediyse)
    if (!document.querySelector('#notification-styles')) {
        const style = document.createElement('style');
        style.id = 'notification-styles';
        style.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-left: 4px solid;
                border-radius: 12px;
                padding: 16px;
                max-width: 400px;
                box-shadow: var(--shadow);
                z-index: 1000;
                animation: slideIn 0.3s ease-out;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            
            .notification-success {
                border-left-color: var(--secondary);
            }
            
            .notification-warning {
                border-left-color: var(--warning);
            }
            
            .notification-info {
                border-left-color: var(--primary);
            }
            
            .notification-content {
                display: flex;
                align-items: center;
                gap: 12px;
                flex: 1;
            }
            
            .notification-close {
                background: none;
                border: none;
                color: var(--gray);
                cursor: pointer;
                padding: 4px;
                border-radius: 6px;
                transition: var(--transition);
            }
            
            .notification-close:hover {
                background: rgba(255,255,255,0.1);
                color: var(--light);
            }
            
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(notification);

    // Kapatma butonuna event listener ekle
    notification.querySelector('.notification-close').addEventListener('click', () => {
        notification.remove();
    });

    // 5 saniye sonra otomatik kapat
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Define Notification Icons
function getNotificationIcon(type) {
    switch (type) {
        case 'success':
            return 'check-circle';
        case 'warning':
            return 'exclamation-triangle';
        case 'info':
        default:
            return 'info-circle';
    }
}

function updateHeatmapChart(zValues, chart) {

    const trace = {
        z: zValues,
        type: 'heatmap',
        colorscale: [
            [0, '#1e3a8a'],   // dark blue
            [0.5, '#6366f1'], // indigo
            [1, '#a78bfa']    // light purple
        ],
        showscale: true,
        hoverongaps: false,
        hovertemplate: 'Correlation Value: %{z}<extra></extra>',
        hoverlabel: {
            bgcolor: 'rgba(30, 41, 59, 0.9)',
            font: {color: '#f8fafc'},
            hovertemplate: 'x=%{x}<br>y=%{y}<br>val=%{z}<extra></extra>'
        },
    };

    const layout = {
        title: '',
        xaxis: {
            title: 'X Axis',
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.3)'
        },
        yaxis: {
            title: 'Y Axis',
            gridcolor: 'rgba(255,255,255,0.1)',
            zerolinecolor: 'rgba(255,255,255,0.3)'
        },
        font: {color: '#f8fafc'},
        margin: {t: 10, r: 20, b: 50, l: 50},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot(chart, [trace], layout, {
        responsive: true,
        displayModeBar: false
    });
}