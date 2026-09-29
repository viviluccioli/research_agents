<?php
/**
 * python_bridge.php
 *
 * Bridge between PHP and Python backend
 * Executes Python scripts and returns JSON results
 */

require_once __DIR__ . '/../config.php';

class PythonBridge {
    /**
     * Execute a Python script and return JSON result
     *
     * @param string $script_name Script filename (e.g., 'run_referee.py')
     * @param array $args Associative array of command-line arguments
     * @param callable|null $progress_callback Optional callback for progress updates
     * @return array Parsed JSON result
     * @throws Exception on execution failure
     */
    public static function execute($script_name, $args = [], $progress_callback = null) {
        $python = PYTHON_VENV;
        $script = PYTHON_SCRIPTS_DIR . '/' . $script_name;

        // Validate Python executable exists
        if (!file_exists($python)) {
            throw new Exception("Python executable not found: $python");
        }

        // Validate script exists
        if (!file_exists($script)) {
            throw new Exception("Python script not found: $script");
        }

        // Build command
        $cmd = escapeshellarg($python) . ' ' . escapeshellarg($script);

        // Add arguments
        foreach ($args as $key => $value) {
            // Handle boolean values
            if (is_bool($value)) {
                $value = $value ? 'true' : 'false';
            }

            // Skip null values
            if ($value === null) {
                continue;
            }

            $cmd .= ' --' . escapeshellarg($key) . ' ' . escapeshellarg($value);
        }

        // Execute with proc_open for streaming output
        $descriptors = [
            0 => ['pipe', 'r'], // stdin
            1 => ['pipe', 'w'], // stdout
            2 => ['pipe', 'w'], // stderr
        ];

        $process = proc_open($cmd, $descriptors, $pipes);

        if (!is_resource($process)) {
            throw new Exception("Failed to start Python process");
        }

        // Close stdin (not needed)
        fclose($pipes[0]);

        // Set non-blocking mode for output streams
        stream_set_blocking($pipes[1], false);
        stream_set_blocking($pipes[2], false);

        $output = '';
        $errors = '';

        // Read output until process finishes
        while (true) {
            $status = proc_get_status($process);

            // Read stdout
            $chunk = fread($pipes[1], 8192);
            if ($chunk !== false && $chunk !== '') {
                $output .= $chunk;

                // Check for progress updates (JSON on separate line)
                if ($progress_callback && strpos($chunk, '{"progress":') !== false) {
                    if (preg_match('/\{\"progress\":[^}]+\}/', $chunk, $matches)) {
                        $progress_data = json_decode($matches[0], true);
                        if ($progress_data !== null) {
                            call_user_func($progress_callback, $progress_data);
                        }
                    }
                }
            }

            // Read stderr
            $err_chunk = fread($pipes[2], 8192);
            if ($err_chunk !== false && $err_chunk !== '') {
                $errors .= $err_chunk;
            }

            // Check if process finished
            if (!$status['running']) {
                break;
            }

            // Small delay to avoid CPU spinning
            usleep(100000); // 100ms
        }

        // Read any remaining output
        $output .= stream_get_contents($pipes[1]);
        $errors .= stream_get_contents($pipes[2]);

        fclose($pipes[1]);
        fclose($pipes[2]);

        $exit_code = proc_close($process);

        // Check exit code
        if ($exit_code !== 0) {
            throw new Exception("Python script failed (exit code $exit_code): $errors");
        }

        // Parse JSON result
        $result = json_decode($output, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new Exception(
                "Invalid JSON from Python: " . json_last_error_msg() .
                "\nOutput: " . substr($output, 0, 500)
            );
        }

        return $result;
    }

    /**
     * Execute Python script in background (non-blocking)
     *
     * @param string $script_name Script filename
     * @param array $args Command-line arguments
     * @param string $output_file File to store output
     * @return int Process ID
     * @throws Exception on execution failure
     */
    public static function executeAsync($script_name, $args = [], $output_file = null) {
        $python = PYTHON_VENV;
        $script = PYTHON_SCRIPTS_DIR . '/' . $script_name;

        if (!file_exists($python) || !file_exists($script)) {
            throw new Exception("Python or script not found");
        }

        // Build command
        $cmd = escapeshellarg($python) . ' ' . escapeshellarg($script);

        foreach ($args as $key => $value) {
            if (is_bool($value)) {
                $value = $value ? 'true' : 'false';
            }
            if ($value !== null) {
                $cmd .= ' --' . escapeshellarg($key) . ' ' . escapeshellarg($value);
            }
        }

        // Redirect output
        if ($output_file === null) {
            $output_file = CACHE_DIR . '/' . uniqid('python_output_') . '.json';
        }

        $cmd .= ' > ' . escapeshellarg($output_file) . ' 2>&1 & echo $!';

        // Execute and get PID
        $pid = trim(shell_exec($cmd));

        return ['pid' => (int)$pid, 'output_file' => $output_file];
    }

    /**
     * Check if a background process is still running
     *
     * @param int $pid Process ID
     * @return bool
     */
    public static function isProcessRunning($pid) {
        $result = shell_exec("ps -p $pid");
        return strpos($result, (string)$pid) !== false;
    }

    /**
     * Read progress from file (for background processes)
     *
     * @param string $session_id PHP session ID
     * @return array|null Progress data or null if not available
     */
    public static function readProgress($session_id) {
        $progress_file = '/tmp/referee_progress_' . $session_id . '.json';

        if (!file_exists($progress_file)) {
            return null;
        }

        $content = file_get_contents($progress_file);
        $data = json_decode($content, true);

        return ($data !== null) ? $data : null;
    }

    /**
     * Clean up progress file
     *
     * @param string $session_id PHP session ID
     */
    public static function cleanupProgress($session_id) {
        $progress_file = '/tmp/referee_progress_' . $session_id . '.json';
        if (file_exists($progress_file)) {
            unlink($progress_file);
        }
    }
}
