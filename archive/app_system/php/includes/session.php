<?php
/**
 * session.php
 *
 * Session management for the PHP referee app
 * Handles initialization, get/set operations, and cleanup
 */

class SessionManager {
    /**
     * Initialize session and set up default keys
     */
    public static function init() {
        if (session_status() === PHP_SESSION_NONE) {
            session_start();
        }
        self::initializeKeys();
    }

    /**
     * Initialize default session keys with default values
     */
    private static function initializeKeys() {
        $keys = [
            'file_data' => [],               // Uploaded files {filename => binary_content}
            'paper_type' => null,            // Selected paper type
            'active_tab' => 'referee',       // Current tab (referee | section_eval)
            'referee_results' => null,       // Referee evaluation results
            'section_eval_results' => null,  // Section evaluator results
            'custom_context' => '',          // User's custom evaluation context
            'selected_personas' => [],       // Manually selected personas
            'use_cache' => true,             // Caching preference
            'use_summarizer' => false,       // Summarization preference
        ];

        foreach ($keys as $key => $default) {
            if (!isset($_SESSION[$key])) {
                $_SESSION[$key] = $default;
            }
        }
    }

    /**
     * Set a session value
     *
     * @param string $key
     * @param mixed $value
     */
    public static function set($key, $value) {
        $_SESSION[$key] = $value;
    }

    /**
     * Get a session value with optional default
     *
     * @param string $key
     * @param mixed $default Default value if key doesn't exist
     * @return mixed
     */
    public static function get($key, $default = null) {
        return $_SESSION[$key] ?? $default;
    }

    /**
     * Check if a session key exists
     *
     * @param string $key
     * @return bool
     */
    public static function has($key) {
        return isset($_SESSION[$key]);
    }

    /**
     * Clear a specific session key
     *
     * @param string $key
     */
    public static function clear($key) {
        unset($_SESSION[$key]);
    }

    /**
     * Clear all session data and reinitialize
     */
    public static function clearAll() {
        session_destroy();
        session_start();
        self::initializeKeys();
    }

    /**
     * Clear file-related session data
     */
    public static function clearFiles() {
        self::set('file_data', []);
    }

    /**
     * Clear results from both workflows
     */
    public static function clearResults() {
        self::set('referee_results', null);
        self::set('section_eval_results', null);
    }

    /**
     * Get session ID
     *
     * @return string
     */
    public static function getId() {
        return session_id();
    }
}
