<?php
/**
 * config.php
 *
 * Configuration file for PHP referee app
 * Loads environment variables from .env file and sets up paths
 */

// Load .env file from parent directory (app_system/.env)
$dotenv_path = __DIR__ . '/../.env';
if (file_exists($dotenv_path)) {
    $lines = file($dotenv_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    foreach ($lines as $line) {
        // Skip comments and empty lines
        if (empty($line) || $line[0] === '#') {
            continue;
        }

        // Parse KEY=VALUE format
        if (strpos($line, '=') !== false) {
            list($key, $value) = explode('=', $line, 2);
            $key = trim($key);
            $value = trim($value);

            // Remove quotes if present
            if (($value[0] === '"' && $value[strlen($value)-1] === '"') ||
                ($value[0] === "'" && $value[strlen($value)-1] === "'")) {
                $value = substr($value, 1, -1);
            }

            putenv("$key=$value");
            $_ENV[$key] = $value;
        }
    }
}

// Application root paths
define('APP_ROOT', __DIR__);
define('APP_SYSTEM_ROOT', realpath(__DIR__ . '/..'));

// Python environment paths
define('PYTHON_VENV', realpath(__DIR__ . '/../../venv/bin/python'));
define('PYTHON_VENV_DIR', realpath(__DIR__ . '/../../venv'));

// Directory paths
define('UPLOAD_DIR', APP_ROOT . '/uploads');
define('CACHE_DIR', APP_ROOT . '/cache');
define('PYTHON_SCRIPTS_DIR', APP_ROOT . '/python_scripts');
define('TEMPLATES_DIR', APP_ROOT . '/templates');
define('ASSETS_DIR', APP_ROOT . '/assets');

// Ensure directories exist
foreach ([UPLOAD_DIR, CACHE_DIR] as $dir) {
    if (!is_dir($dir)) {
        mkdir($dir, 0755, true);
    }
}

// API Configuration (from .env)
define('API_KEY', getenv('API_KEY') ?: '');
define('API_BASE', getenv('API_BASE') ?: '');
define('MODEL_PRIMARY', getenv('MODEL_PRIMARY') ?: 'anthropic.claude-sonnet-4-5-20250929-v1:0');
define('MODEL_SECONDARY', getenv('MODEL_SECONDARY') ?: MODEL_PRIMARY);
define('MODEL_TERTIARY', getenv('MODEL_TERTIARY') ?: MODEL_PRIMARY);

// Session configuration
ini_set('session.gc_maxlifetime', 3600); // 1 hour
ini_set('session.cookie_lifetime', 3600);
ini_set('session.cookie_httponly', 1);

// Error reporting (disable in production)
error_reporting(E_ALL);
ini_set('display_errors', 1);

// File upload limits
ini_set('upload_max_filesize', '50M');
ini_set('post_max_size', '50M');
ini_set('max_execution_time', 600); // 10 minutes for long evaluations

// Timezone
date_default_timezone_set('UTC');

// Start session if not already started
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}