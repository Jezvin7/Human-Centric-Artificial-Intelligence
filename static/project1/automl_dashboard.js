const uploadForm = document.getElementById("upload-form");
const uploadBtn = document.getElementById("upload-btn");
const resetBtn = document.getElementById("reset-btn");
const msgArea = document.getElementById("message-area");
const dataSections = document.getElementById("data-sections");

function showMessage(text, type) {
    msgArea.innerHTML = `<div class="message ${type}">${text}</div>`;
    setTimeout(() => { msgArea.innerHTML = ""; }, 4000);
}

function buildTableHTML(columns, rows) {
    const ths = columns.map(c => `<th>${c}</th>`).join("");
    const trs = rows.map(r =>
        `<tr>${r.map(v => `<td>${v}</td>`).join("")}</tr>`
    ).join("");

    return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
}

function buildPipelineHTML() {
    return `
        <section class="card pipeline-section">
            <div class="section-header">
                <h2>ML Pipeline Configuration</h2>
                <p>Select multiple machine learning models and compare their performance.</p>
            </div>

            <div class="pipeline-content">
                <div class="model-selection-card">
                    <h3>Select Models</h3>

                    <div class="model-grid">
                        <label>
                            <input type="checkbox" value="Logistic Regression">
                            Logistic Regression
                        </label>

                        <label>
                            <input type="checkbox" value="Random Forest">
                            Random Forest
                        </label>

                        <label>
                            <input type="checkbox" value="KNN">
                            KNN
                        </label>

                        <label>
                            <input type="checkbox" value="SVM">
                            SVM
                        </label>

                        <label>
                            <input type="checkbox" value="Decision Tree">
                            Decision Tree
                        </label>
                    </div>
                </div>

                <div class="settings-card">
                    <h3>Pipeline Settings</h3>

                    <div class="slider-group">
                        <label>
                            Test Size:
                            <span id="splitValue">20%</span>
                        </label>

                        <input type="range" id="testSizeSlider" min="10" max="40" value="20">
                    </div>

                    <div class="hyperparameter-group">
                        <label>
                            KNN K values (comma-separated)
                            <input type="text" id="knnValues" value="1,3,5,7">
                        </label>

                        <label>
                            Random Forest trees (comma-separated)
                            <input type="text" id="rfValues" value="50,100,200">
                        </label>

                        <label>
                            Decision Tree depths (comma-separated)
                            <input type="text" id="dtDepths" value="2,4,6">
                        </label>
                    </div>

                    <button type="button" id="runModelBtn">Run ML Pipeline</button>
                </div>
            </div>
        </section>

        <section class="card">
            <h2>Model Results</h2>
            <div id="resultBox"></div>
            <div class="chart-wrapper">
                <canvas id="comparisonChart"></canvas>
            </div>
        </section>
    `;
}

function renderScatterChart(datasets, chartX, chartY) {
    const scatterCanvas = document.getElementById("scatterChart");

    if (!scatterCanvas || !datasets) {
        return;
    }

    if (window.activeChart) {
        window.activeChart.destroy();
    }

    window.activeChart = new Chart(scatterCanvas, {
        type: "scatter",
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: chartX
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: chartY
                    }
                }
            }
        }
    });
}

function validateCommaSeparatedIntegers(value, fieldName) {
    if (!value.trim()) {
        return `${fieldName} cannot be empty.`;
    }

    const values = value.split(",");

    for (const item of values) {
        const number = Number(item.trim());

        if (!Number.isInteger(number) || number <= 0) {
            return `${fieldName} must contain positive integers separated by commas.`;
        }
    }

    return null;
}

function attachPipelineEvents() {
    const slider = document.getElementById("testSizeSlider");

    if (slider) {
        slider.addEventListener("input", function () {
            document.getElementById("splitValue").textContent = this.value + "%";
        });
    }

    const runBtn = document.getElementById("runModelBtn");

    if (runBtn) {
        runBtn.addEventListener("click", runPipeline);
    }
}

function showPipelineLoader() {
    document.getElementById("resultBox").innerHTML = `
        <div class="pipeline-loader">
            <div class="spinner"></div>
            <p>Training selected models...</p>
        </div>
    `;
}

async function runPipeline(e) {
    e.preventDefault();

    const selectedModels = [];

    document.querySelectorAll(".model-grid input:checked").forEach(cb => {
        selectedModels.push(cb.value);
    });

    if (selectedModels.length === 0) {
        document.getElementById("resultBox").innerHTML =
            `<div class="message error">Please select at least one model.</div>`;
        return;
    }

    const testSize = parseInt(document.getElementById("testSizeSlider").value) / 100;
    const knnValues = document.getElementById("knnValues").value;
    const rfValues = document.getElementById("rfValues").value;
    const dtDepths = document.getElementById("dtDepths").value;

    let validationError = null;

    if (selectedModels.includes("KNN")) {
        validationError = validateCommaSeparatedIntegers(
            knnValues,
            "KNN K values"
        );
    }

    if (!validationError && selectedModels.includes("Random Forest")) {
        validationError = validateCommaSeparatedIntegers(
            rfValues,
            "Random Forest trees"
        );
    }

    if (!validationError && selectedModels.includes("Decision Tree")) {
        validationError = validateCommaSeparatedIntegers(
            dtDepths,
            "Decision Tree depths"
        );
    }

    if (validationError) {
        document.getElementById("resultBox").innerHTML =
            `<div class="message error">${validationError}</div>`;
        return;
    }
    const runBtn = document.getElementById("runModelBtn");
    runBtn.disabled = true;
    runBtn.textContent = "Running...";

    showPipelineLoader();
    try {
        const response = await fetch(window.project1Config.runModelUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRFToken": window.project1Config.csrfToken
            },
            body: JSON.stringify({
                models: selectedModels,
                test_size: testSize,
                hyperparameters: {
                    knn_values: knnValues,
                    rf_values: rfValues,
                    dt_depths: dtDepths
                }
            })
        });

        const data = await response.json();

        if (!data.ok) {
            document.getElementById("resultBox").innerHTML =
                `<div class="message error">${data.error}</div>`;
            return;
        }

        let html = `<div class="results-grid">`;
        const chartLabels = [];
        const chartScores = [];

        data.results.forEach(result => {
            html += `
                <div class="result-card">
                    <h3>${result.model}</h3>
                    <div class="metric-list">
            `;

            let mainScore = 0;

            for (const key in result.metrics) {
                if (key === "error") {
                    html += `
                        <div class="metric-row">
                            <span>Error</span>
                            <strong>${result.metrics[key]}</strong>
                        </div>
                    `;
                    continue;
                }

                html += `
                    <div class="metric-row">
                        <span>${key}</span>
                        <strong>${result.metrics[key]}</strong>
                    </div>
                `;

                if (key === "accuracy" || key === "r2_score") {
                    mainScore = result.metrics[key];
                }
            }

            chartLabels.push(result.model);
            chartScores.push(mainScore);

            html += `</div></div>`;
        });

        html += `</div>`;

        document.getElementById("resultBox").innerHTML = html;

        if (window.modelChart) {
            window.modelChart.destroy();
        }

        const ctx = document.getElementById("comparisonChart").getContext("2d");
        ctx.canvas.height = 170;

        window.modelChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: chartLabels,
                datasets: [
                    {
                        label: data.task_type === "classification" ? "Accuracy" : "R² Score",
                        data: chartScores,
                        backgroundColor: [
                            "#4F46E5",
                            "#10B981",
                            "#F59E0B",
                            "#EF4444",
                            "#06B6D4"
                        ],
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        suggestedMax: 1
                    }
                }
            }
        });

    } catch (error) {
        document.getElementById("resultBox").innerHTML =
            `<div class="message error">${error}</div>`;
    }
    finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run ML Pipeline";
    }
}

if (uploadForm) {
    uploadForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        uploadBtn.disabled = true;
        uploadBtn.textContent = "Uploading…";

        try {
            const response = await fetch(window.project1Config.uploadUrl, {
                method: "POST",
                headers: {
                    "X-CSRFToken": window.project1Config.csrfToken
                },
                body: new FormData(uploadForm)
            });

            const data = await response.json();

            if (!data.ok) {
                showMessage(data.error || "Upload failed.", "error");
                return;
            }

            showMessage("CSV uploaded successfully.", "success");

            let html = `
                <section class="card">
                    <h2>Dataset Overview</h2>
                    <p><strong>Rows:</strong> ${data.row_count}</p>
                    <p><strong>Columns:</strong> ${data.col_count}</p>
                    <p><strong>Target column:</strong> ${data.target_col}</p>
                    <p><strong>Detected task type:</strong> ${data.detected_task_type}</p>
                </section>

                <section class="card">
                    <h2>Data Preview</h2>
                    <div class="table-wrapper">
                        ${buildTableHTML(data.columns, data.rows)}
                    </div>
                </section>
            `;

            if (data.scatter_datasets) {
                html += `
                    <section class="card">
                        <h2>Scatter Plot</h2>
                        <p><strong>X-axis:</strong> ${data.chart_x}</p>
                        <p><strong>Y-axis:</strong> ${data.chart_y}</p>
                        <canvas id="scatterChart"></canvas>
                    </section>
                `;
            } else {
                html += `
                    <section class="card">
                        <h2>Scatter Plot</h2>
                        <p>At least two numeric feature columns are needed for plotting.</p>
                    </section>
                `;
            }

            html += buildPipelineHTML();
            dataSections.innerHTML = html;

            renderScatterChart(
                data.scatter_datasets,
                data.chart_x,
                data.chart_y
            );

            attachPipelineEvents();

        } catch (error) {
            showMessage("Network error: " + error.message, "error");
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.textContent = "Upload";
        }
    });
}

if (resetBtn) {
    resetBtn.addEventListener("click", async function () {
        try {
            const response = await fetch(window.project1Config.resetUrl, {
                method: "POST",
                headers: {
                    "X-CSRFToken": window.project1Config.csrfToken
                }
            });

            const data = await response.json();

            if (!data.ok) {
                showMessage("Reset failed.", "error");
                return;
            }

            if (window.activeChart) {
                window.activeChart.destroy();
                window.activeChart = null;
            }

            if (window.modelChart) {
                window.modelChart.destroy();
                window.modelChart = null;
            }

            dataSections.innerHTML = "";
            uploadForm.reset();

            const fileLabel = document.getElementById("file-label-text");

            if (fileLabel) {
                fileLabel.textContent = "Upload File";
            }

            showMessage("Session cleared.", "info");

        } catch (error) {
            showMessage("Network error: " + error.message, "error");
        }
    });
}

const fileInput = document.getElementById("id_csv_file");

if (fileInput) {
    fileInput.addEventListener("change", function () {
        const name = this.files[0] ? this.files[0].name : "Upload File";
        document.getElementById("file-label-text").textContent = name;
    });
}

if (window.project1Config.hasScatterData === "true" && window.project1Config.scatterDatasetsJson) {
    renderScatterChart(
        JSON.parse(window.project1Config.scatterDatasetsJson),
        window.project1Config.chartX,
        window.project1Config.chartY
    );
}

attachPipelineEvents();