<?php
/**
 * file_handler.php
 *
 * File upload and management for the PHP referee app
 * Handles validation, storage, and cleanup of uploaded files
 */

require_once __DIR__ . '/../config.php';

class FileHandler {
    // Allowed file extensions
    private const ALLOWED_EXTENSIONS = ['pdf', 'txt', 'tex', 'docx'];

    // Maximum file size (50MB)
    private const MAX_FILE_SIZE = 50 * 1024 * 1024;

    /**
     * Handle multiple file uploads
     *
     * @param array $files $_FILES array
     * @return array ['uploaded' => [...], 'errors' => [...]]
     */
    public static function handleUpload($files) {
        $uploaded = [];
        $errors = [];

        // Handle single file or multiple files
        $file_count = is_array($files['name']) ? count($files['name']) : 1;

        for ($i = 0; $i < $file_count; $i++) {
            // Extract file info
            if (is_array($files['name'])) {
                $name = $files['name'][$i];
                $tmp_name = $files['tmp_name'][$i];
                $size = $files['size'][$i];
                $error = $files['error'][$i];
            } else {
                $name = $files['name'];
                $tmp_name = $files['tmp_name'];
                $size = $files['size'];
                $error = $files['error'];
            }

            // Validate upload error
            if ($error !== UPLOAD_ERR_OK) {
                $errors[] = "Upload error for $name: " . self::getUploadErrorMessage($error);
                continue;
            }

            // Validate file size
            if ($size > self::MAX_FILE_SIZE) {
                $errors[] = "$name exceeds 50MB limit";
                continue;
            }

            // Validate extension
            $ext = strtolower(pathinfo($name, PATHINFO_EXTENSION));
            if (!in_array($ext, self::ALLOWED_EXTENSIONS)) {
                $errors[] = "$name has invalid extension (allowed: " . implode(', ', self::ALLOWED_EXTENSIONS) . ")";
                continue;
            }

            // Validate MIME type (basic check)
            $finfo = finfo_open(FILEINFO_MIME_TYPE);
            $mime = finfo_file($finfo, $tmp_name);
            finfo_close($finfo);

            if (!self::isValidMimeType($mime, $ext)) {
                $errors[] = "$name has invalid MIME type";
                continue;
            }

            // Read file content
            $content = file_get_contents($tmp_name);
            if ($content === false) {
                $errors[] = "Failed to read $name";
                continue;
            }

            // Store in session
            $_SESSION['file_data'][$name] = $content;
            $uploaded[] = $name;

            // Save to temp directory for Python processing
            $temp_path = UPLOAD_DIR . '/' . session_id() . '_' . $name;
            file_put_contents($temp_path, $content);
        }

        return ['uploaded' => $uploaded, 'errors' => $errors];
    }

    /**
     * Handle pasted text
     *
     * @param string $text Pasted text content
     * @param string $format 'text' or 'latex'
     * @return string Filename
     */
    public static function handlePastedText($text, $format) {
        $ext = ($format === 'latex') ? '.tex' : '.txt';
        $name = 'Pasted_Text' . $ext;

        // Store in session
        $_SESSION['file_data'][$name] = $text;

        // Save to temp directory
        $temp_path = UPLOAD_DIR . '/' . session_id() . '_' . $name;
        file_put_contents($temp_path, $text);

        return $name;
    }

    /**
     * Get files from current session
     *
     * @return array {filename => content}
     */
    public static function getSessionFiles() {
        return $_SESSION['file_data'] ?? [];
    }

    /**
     * Get temp file path for a session file
     *
     * @param string $filename
     * @return string|null
     */
    public static function getTempPath($filename) {
        $temp_path = UPLOAD_DIR . '/' . session_id() . '_' . $filename;
        return file_exists($temp_path) ? $temp_path : null;
    }

    /**
     * Clear files from session and temp directory
     */
    public static function clearFiles() {
        // Clear session
        $_SESSION['file_data'] = [];

        // Delete temp files
        $pattern = UPLOAD_DIR . '/' . session_id() . '_*';
        foreach (glob($pattern) as $file) {
            if (is_file($file)) {
                unlink($file);
            }
        }
    }

    /**
     * Clean up old temp files (called periodically)
     *
     * @param int $max_age Maximum age in seconds (default 1 hour)
     */
    public static function cleanupOldFiles($max_age = 3600) {
        $pattern = UPLOAD_DIR . '/*';
        $now = time();

        foreach (glob($pattern) as $file) {
            if (is_file($file) && ($now - filemtime($file)) > $max_age) {
                unlink($file);
            }
        }
    }

    /**
     * Get upload error message
     *
     * @param int $error UPLOAD_ERR_* constant
     * @return string
     */
    private static function getUploadErrorMessage($error) {
        switch ($error) {
            case UPLOAD_ERR_INI_SIZE:
            case UPLOAD_ERR_FORM_SIZE:
                return 'File too large';
            case UPLOAD_ERR_PARTIAL:
                return 'File only partially uploaded';
            case UPLOAD_ERR_NO_FILE:
                return 'No file uploaded';
            case UPLOAD_ERR_NO_TMP_DIR:
                return 'Missing temporary folder';
            case UPLOAD_ERR_CANT_WRITE:
                return 'Failed to write file to disk';
            case UPLOAD_ERR_EXTENSION:
                return 'File upload stopped by extension';
            default:
                return 'Unknown upload error';
        }
    }

    /**
     * Validate MIME type matches extension
     *
     * @param string $mime MIME type
     * @param string $ext File extension
     * @return bool
     */
    private static function isValidMimeType($mime, $ext) {
        $valid_types = [
            'pdf' => ['application/pdf'],
            'txt' => ['text/plain', 'application/octet-stream'],
            'tex' => ['text/plain', 'text/x-tex', 'application/x-tex', 'application/x-latex', 'application/octet-stream'],
            'docx' => ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/octet-stream'],
        ];

        return isset($valid_types[$ext]) && in_array($mime, $valid_types[$ext]);
    }
}
