<?php
/**
 * index.php
 *
 * Main entry point for PHP-based referee app
 * Provides two-tab interface for Referee Report and Section Evaluator
 */

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/includes/session.php';
require_once __DIR__ . '/includes/file_handler.php';
require_once __DIR__ . '/includes/python_bridge.php';

SessionManager::init();

// Handle file uploads
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['files'])) {
    $result = FileHandler::handleUpload($_FILES['files']);
    if (!empty($result['uploaded'])) {
        $success_message = count($result['uploaded']) . ' file(s) uploaded successfully';
    }
    if (!empty($result['errors'])) {
        $error_message = implode('<br>', $result['errors']);
    }
}

// Handle pasted text
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['pasted_text']) && !empty(trim($_POST['pasted_text']))) {
    $format = $_POST['paste_format'] ?? 'text';
    $filename = FileHandler::handlePastedText($_POST['pasted_text'], $format);
    $success_message = "Pasted text added as $filename";
}

// Handle clear files
if (isset($_GET['action']) && $_GET['action'] === 'clear_files') {
    FileHandler::clearFiles();
    SessionManager::clearFiles();
    SessionManager::clearResults();
    header('Location: index.php');
    exit;
}

// Handle referee evaluation (AJAX)
if (isset($_GET['action']) && $_GET['action'] === 'run_referee' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');

    $selected_file = $_POST['selected_file'] ?? null;
    $paper_type = $_POST['paper_type'] ?? null;
    $custom_context = $_POST['custom_context'] ?? null;
    $persona_mode = $_POST['persona_mode'] ?? 'auto';
    $personas = $_POST['personas'] ?? [];
    $use_cache = isset($_POST['use_cache']);

    if (!$selected_file) {
        echo json_encode(['error' => 'No file selected']);
        exit;
    }

    $file_path = FileHandler::getTempPath($selected_file);
    if (!$file_path) {
        echo json_encode(['error' => 'File not found']);
        exit;
    }

    // Prepare arguments
    $args = [
        'file' => $file_path,
        'session-id' => SessionManager::getId(),
        'use-cache' => $use_cache,
    ];

    if ($paper_type && $paper_type !== '') {
        $args['paper-type'] = $paper_type;
    }

    if ($custom_context && trim($custom_context) !== '') {
        $args['custom-context'] = trim($custom_context);
    }

    if ($persona_mode === 'manual' && !empty($personas)) {
        $args['personas'] = implode(',', $personas);
    }

    // Execute Python script
    try {
        $results = PythonBridge::execute('run_referee.py', $args);
        SessionManager::set('referee_results', $results);
        SessionManager::set('referee_file', $selected_file);
        echo json_encode(['success' => true, 'redirect' => 'index.php?tab=referee&view=results']);
    } catch (Exception $e) {
        echo json_encode(['error' => $e->getMessage()]);
    }
    exit;
}

// Handle section evaluator (AJAX)
if (isset($_GET['action']) && $_GET['action'] === 'run_section_eval' && $_SERVER['REQUEST_METHOD'] === 'POST') {
    header('Content-Type: application/json');

    $selected_file = $_POST['selected_file'] ?? null;
    $paper_type = $_POST['paper_type'] ?? null;

    if (!$selected_file || !$paper_type) {
        echo json_encode(['error' => 'File and paper type required']);
        exit;
    }

    $file_path = FileHandler::getTempPath($selected_file);
    if (!$file_path) {
        echo json_encode(['error' => 'File not found']);
        exit;
    }

    $args = [
        'file' => $file_path,
        'paper-type' => $paper_type,
        'mode' => 'auto',
    ];

    try {
        $results = PythonBridge::execute('run_section_eval.py', $args);
        SessionManager::set('section_eval_results', $results);
        SessionManager::set('section_eval_file', $selected_file);
        echo json_encode(['success' => true, 'redirect' => 'index.php?tab=section_eval&view=results']);
    } catch (Exception $e) {
        echo json_encode(['error' => $e->getMessage()]);
    }
    exit;
}

// Get progress (AJAX)
if (isset($_GET['action']) && $_GET['action'] === 'get_progress') {
    header('Content-Type: application/json');
    $progress = PythonBridge::readProgress(SessionManager::getId());
    echo json_encode($progress ?? ['progress' => 0, 'message' => 'Initializing...']);
    exit;
}

// Get active tab and view
$active_tab = $_GET['tab'] ?? SessionManager::get('active_tab', 'referee');
$view = $_GET['view'] ?? 'form';
SessionManager::set('active_tab', $active_tab);

$files = FileHandler::getSessionFiles();
$paper_types = [
    'empirical' => 'Empirical',
    'theoretical' => 'Theoretical',
    'policy' => 'Policy',
    'finance' => 'Finance',
    'macro' => 'Macroeconomics',
    'systematic_review' => 'Systematic Review'
];
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Co-Economist: Referee Report - PHP Version</title>
    <link rel="stylesheet" href="assets/css/main.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎓 Co-Economist: Referee Report</h1>
            <p class="subtitle">AI-powered academic paper evaluation system</p>
        </header>

        <?php if (isset($success_message)): ?>
            <div class="alert alert-success"><?= htmlspecialchars($success_message) ?></div>
        <?php endif; ?>

        <?php if (isset($error_message)): ?>
            <div class="alert alert-error"><?= htmlspecialchars($error_message) ?></div>
        <?php endif; ?>

        <!-- File Upload Section -->
        <div class="upload-section">
            <h3>📄 Document Upload</h3>

            <form method="POST" enctype="multipart/form-data" class="upload-form">
                <div class="form-group">
                    <label>Upload PDF, LaTeX, or text files:</label>
                    <input type="file" name="files[]" multiple accept=".pdf,.txt,.tex,.docx" class="file-input">
                </div>
                <button type="submit" class="btn btn-primary">Upload Files</button>
            </form>

            <hr>

            <div class="paste-section">
                <h4>Or paste text directly:</h4>
                <form method="POST">
                    <div class="radio-group">
                        <label>
                            <input type="radio" name="paste_format" value="text" checked> Plain text
                        </label>
                        <label>
                            <input type="radio" name="paste_format" value="latex"> LaTeX source
                        </label>
                    </div>
                    <textarea name="pasted_text" rows="8" placeholder="Paste your manuscript here..." class="text-area"></textarea>
                    <button type="submit" class="btn btn-secondary">Add Pasted Text</button>
                </form>
            </div>

            <?php if (!empty($files)): ?>
                <div class="files-list">
                    <strong>📁 Available files:</strong>
                    <?php foreach (array_keys($files) as $idx => $filename): ?>
                        <span class="file-tag"><?= htmlspecialchars($filename) ?></span><?= $idx < count($files) - 1 ? ', ' : '' ?>
                    <?php endforeach; ?>
                    <a href="?action=clear_files" class="btn-link" onclick="return confirm('Clear all files?')">Clear files</a>
                </div>
            <?php endif; ?>
        </div>

        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn <?= $active_tab === 'referee' ? 'active' : '' ?>" onclick="location.href='?tab=referee'">
                🏛️ Referee Report
            </button>
            <button class="tab-btn <?= $active_tab === 'section_eval' ? 'active' : '' ?>" onclick="location.href='?tab=section_eval'">
                📊 Section Evaluator
            </button>
        </div>

        <div class="tab-content">
            <?php if ($active_tab === 'referee'): ?>
                <!-- Referee Report Tab -->
                <?php if ($view === 'results' && SessionManager::has('referee_results')): ?>
                    <?php
                    $results = SessionManager::get('referee_results');
                    $filename = SessionManager::get('referee_file', 'Unknown');
                    ?>
                    <div class="results-section">
                        <h2>🏛️ Referee Report Results</h2>
                        <p><strong>File:</strong> <?= htmlspecialchars($filename) ?></p>

                        <?php if (isset($results['error'])): ?>
                            <div class="alert alert-error">
                                <strong>Error:</strong> <?= htmlspecialchars($results['error']) ?>
                            </div>
                        <?php else: ?>
                            <!-- Round 0: Selected Personas -->
                            <?php if (isset($results['round_0'])): ?>
                                <div class="result-card">
                                    <h3>Round 0: Persona Selection</h3>
                                    <p><strong>Selected Personas:</strong>
                                        <?= implode(', ', $results['round_0']['selected_personas'] ?? []) ?>
                                    </p>
                                    <p><strong>Weights:</strong>
                                        <?php foreach ($results['round_0']['weights'] ?? [] as $persona => $weight): ?>
                                            <span class="tag"><?= $persona ?>: <?= number_format($weight, 2) ?></span>
                                        <?php endforeach; ?>
                                    </p>
                                </div>
                            <?php endif; ?>

                            <!-- Final Decision -->
                            <?php if (isset($results['consensus'])): ?>
                                <div class="result-card decision-card">
                                    <h3>Final Decision</h3>
                                    <p class="decision-badge decision-<?= strtolower(str_replace(' ', '-', $results['consensus']['decision'])) ?>">
                                        <?= htmlspecialchars($results['consensus']['decision']) ?>
                                    </p>
                                    <p><strong>Weighted Score:</strong> <?= number_format($results['consensus']['weighted_score_categorical'], 2) ?></p>
                                    <p><strong>Individual Verdicts:</strong></p>
                                    <ul>
                                        <?php foreach ($results['consensus']['verdicts'] as $persona => $verdict): ?>
                                            <li><?= $persona ?>: <strong><?= $verdict ?></strong></li>
                                        <?php endforeach; ?>
                                    </ul>
                                </div>
                            <?php endif; ?>

                            <!-- Editor Report -->
                            <?php if (isset($results['final_decision'])): ?>
                                <div class="result-card">
                                    <h3>Editor's Report</h3>
                                    <div class="markdown-content">
                                        <?= nl2br(htmlspecialchars($results['final_decision'])) ?>
                                    </div>
                                </div>
                            <?php endif; ?>

                            <!-- Metadata -->
                            <?php if (isset($results['metadata'])): ?>
                                <div class="result-card metadata-card">
                                    <h3>Evaluation Metadata</h3>
                                    <ul>
                                        <li><strong>Duration:</strong> <?= $results['metadata']['duration_seconds'] ?? 'N/A' ?> seconds</li>
                                        <li><strong>Model:</strong> <?= htmlspecialchars($results['metadata']['model'] ?? 'N/A') ?></li>
                                        <li><strong>Total Tokens:</strong> <?= number_format($results['metadata']['token_usage']['total_tokens'] ?? 0) ?></li>
                                        <li><strong>Est. Cost:</strong> $<?= number_format($results['metadata']['token_usage']['cost_usd']['total'] ?? 0, 2) ?></li>
                                    </ul>
                                </div>
                            <?php endif; ?>
                        <?php endif; ?>

                        <div class="actions">
                            <button onclick="location.href='?tab=referee'" class="btn btn-secondary">Run Another Evaluation</button>
                        </div>
                    </div>
                <?php else: ?>
                    <!-- Evaluation Form -->
                    <?php if (empty($files)): ?>
                        <div class="notice">
                            <p>👆 Please upload a file or paste text above to begin evaluation.</p>
                        </div>
                    <?php else: ?>
                        <form id="referee-form" class="evaluation-form">
                            <h2>🏛️ Referee Report Configuration</h2>

                            <div class="form-group">
                                <label>Select File:</label>
                                <select name="selected_file" required class="select-input">
                                    <?php foreach (array_keys($files) as $filename): ?>
                                        <option value="<?= htmlspecialchars($filename) ?>"><?= htmlspecialchars($filename) ?></option>
                                    <?php endforeach; ?>
                                </select>
                            </div>

                            <div class="form-group">
                                <label>Paper Type (Optional):</label>
                                <select name="paper_type" class="select-input">
                                    <option value="">Auto-detect</option>
                                    <?php foreach ($paper_types as $value => $label): ?>
                                        <?php if (in_array($value, ['empirical', 'theoretical', 'policy'])): ?>
                                            <option value="<?= $value ?>"><?= $label ?></option>
                                        <?php endif; ?>
                                    <?php endforeach; ?>
                                </select>
                            </div>

                            <div class="form-group">
                                <label>Custom Evaluation Context (Optional):</label>
                                <textarea name="custom_context" rows="4" class="text-area" placeholder="E.g., 'Focus on causal identification and robustness'"></textarea>
                            </div>

                            <div class="form-group">
                                <label>Persona Selection Mode:</label>
                                <select name="persona_mode" id="persona-mode" class="select-input">
                                    <option value="auto">Automatic (LLM selects 3 best personas)</option>
                                    <option value="manual">Manual Selection</option>
                                </select>
                            </div>

                            <div id="manual-personas" class="persona-grid" style="display: none;">
                                <?php
                                $personas = [
                                    'Theorist', 'Econometrician', 'ML_Expert', 'Data_Scientist',
                                    'CS_Expert', 'Historian', 'Visionary', 'Policymaker',
                                    'Ethicist', 'Perspective'
                                ];
                                foreach ($personas as $persona):
                                ?>
                                    <label class="checkbox-label">
                                        <input type="checkbox" name="personas[]" value="<?= $persona ?>">
                                        <?= $persona ?>
                                    </label>
                                <?php endforeach; ?>
                            </div>

                            <div class="form-group">
                                <label class="checkbox-label">
                                    <input type="checkbox" name="use_cache" value="1" checked>
                                    Use caching (faster, cheaper for repeated evaluations)
                                </label>
                            </div>

                            <button type="submit" class="btn btn-primary btn-large">▶️ Run Evaluation</button>
                        </form>

                        <div id="progress-container" style="display: none;">
                            <h3>⏳ Evaluation in Progress</h3>
                            <div class="progress-bar">
                                <div id="progress-fill"></div>
                            </div>
                            <p id="progress-message">Initializing...</p>
                        </div>
                    <?php endif; ?>
                <?php endif; ?>

            <?php else: ?>
                <!-- Section Evaluator Tab -->
                <?php if ($view === 'results' && SessionManager::has('section_eval_results')): ?>
                    <?php
                    $results = SessionManager::get('section_eval_results');
                    $filename = SessionManager::get('section_eval_file', 'Unknown');
                    ?>
                    <div class="results-section">
                        <h2>📊 Section Evaluation Results</h2>
                        <p><strong>File:</strong> <?= htmlspecialchars($filename) ?></p>

                        <?php if (isset($results['error'])): ?>
                            <div class="alert alert-error">
                                <strong>Error:</strong> <?= htmlspecialchars($results['error']) ?>
                            </div>
                        <?php else: ?>
                            <!-- Overall Score -->
                            <?php if (isset($results['overall'])): ?>
                                <div class="result-card overall-card">
                                    <h3>Overall Assessment</h3>
                                    <p><strong>Overall Score:</strong> <?= number_format($results['overall']['overall_score'], 2) ?> / 5.0</p>
                                    <p><strong>Publication Readiness:</strong> <?= htmlspecialchars($results['overall']['publication_readiness']) ?></p>
                                    <p><strong>Paper Type:</strong> <?= htmlspecialchars($results['paper_type']) ?></p>
                                </div>
                            <?php endif; ?>

                            <!-- Section Results -->
                            <?php if (isset($results['sections'])): ?>
                                <h3>Section-by-Section Results</h3>
                                <?php foreach ($results['sections'] as $section_name => $section_data): ?>
                                    <div class="result-card section-card">
                                        <h4><?= htmlspecialchars($section_name) ?></h4>

                                        <?php if (isset($section_data['error'])): ?>
                                            <p class="error-text">Error: <?= htmlspecialchars($section_data['error']) ?></p>
                                        <?php else: ?>
                                            <?php if (isset($section_data['section_score'])): ?>
                                                <p><strong>Score:</strong> <?= number_format($section_data['section_score']['raw_score'], 2) ?> / 5.0
                                                    (adjusted: <?= number_format($section_data['section_score']['adjusted_score'], 2) ?>)</p>
                                            <?php endif; ?>

                                            <?php if (isset($section_data['qualitative_assessment'])): ?>
                                                <p><strong>Assessment:</strong> <?= nl2br(htmlspecialchars($section_data['qualitative_assessment'])) ?></p>
                                            <?php endif; ?>

                                            <?php if (isset($section_data['improvements']) && !empty($section_data['improvements'])): ?>
                                                <details>
                                                    <summary><strong>Improvement Suggestions (<?= count($section_data['improvements']) ?>)</strong></summary>
                                                    <ul>
                                                        <?php foreach ($section_data['improvements'] as $improvement): ?>
                                                            <li><strong>P<?= $improvement['priority'] ?>:</strong> <?= htmlspecialchars($improvement['suggestion']) ?></li>
                                                        <?php endforeach; ?>
                                                    </ul>
                                                </details>
                                            <?php endif; ?>
                                        <?php endif; ?>
                                    </div>
                                <?php endforeach; ?>
                            <?php endif; ?>
                        <?php endif; ?>

                        <div class="actions">
                            <button onclick="location.href='?tab=section_eval'" class="btn btn-secondary">Run Another Evaluation</button>
                        </div>
                    </div>
                <?php else: ?>
                    <!-- Evaluation Form -->
                    <?php if (empty($files)): ?>
                        <div class="notice">
                            <p>👆 Please upload a file or paste text above to begin evaluation.</p>
                        </div>
                    <?php else: ?>
                        <form id="section-eval-form" class="evaluation-form">
                            <h2>📊 Section Evaluator Configuration</h2>

                            <div class="form-group">
                                <label>Select File:</label>
                                <select name="selected_file" required class="select-input">
                                    <?php foreach (array_keys($files) as $filename): ?>
                                        <option value="<?= htmlspecialchars($filename) ?>"><?= htmlspecialchars($filename) ?></option>
                                    <?php endforeach; ?>
                                </select>
                            </div>

                            <div class="form-group">
                                <label>Paper Type (Required):</label>
                                <select name="paper_type" required class="select-input">
                                    <option value="">— Select paper type —</option>
                                    <?php foreach ($paper_types as $value => $label): ?>
                                        <option value="<?= $value ?>"><?= $label ?></option>
                                    <?php endforeach; ?>
                                </select>
                            </div>

                            <button type="submit" class="btn btn-primary btn-large">▶️ Run Evaluation</button>
                        </form>

                        <div id="section-progress-container" style="display: none;">
                            <h3>⏳ Evaluation in Progress</h3>
                            <div class="progress-bar">
                                <div id="section-progress-fill"></div>
                            </div>
                            <p id="section-progress-message">Initializing...</p>
                        </div>
                    <?php endif; ?>
                <?php endif; ?>
            <?php endif; ?>
        </div>

        <footer>
            <p>Co-Economist Referee Report System - PHP Version | Powered by Claude 4.5 Sonnet</p>
        </footer>
    </div>

    <script src="assets/js/progress.js"></script>
</body>
</html>
