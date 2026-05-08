# Co-Economist Referee Report - PHP Version

This is a PHP-based web application that provides the same functionality as the Streamlit `app.py`, allowing traditional web hosting (Apache/nginx) without requiring a Python server.

## Architecture

**Hybrid PHP Frontend + Python Backend**:
- **PHP handles**: Web UI, file uploads, session management, form processing, results display
- **Python handles**: Evaluation logic, LLM calls, PDF extraction, scoring algorithms
- **Communication**: PHP calls Python scripts via CLI (`proc_open()`)

## Features

✅ **Two Evaluation Workflows**:
1. **Referee Report** - Multi-agent debate (MAD) system with 10 available personas
2. **Section Evaluator** - Paper-type-aware section evaluation

✅ **File Support**: PDF, LaTeX (.tex), plain text (.txt), DOCX
✅ **Caching**: SHA256-based granular caching for cost savings
✅ **Progress Tracking**: Real-time progress updates via AJAX polling
✅ **Session Persistence**: Files and results persist across page reloads

## Requirements

- **PHP 7.4+** with `proc_open()` enabled
- **Python 3.8+** with virtual environment at `../../venv/`
- **Python dependencies**: All packages from `requirements.txt`
- **Write permissions**: For `uploads/` and `cache/` directories
- **Web server**: Apache with mod_php or nginx with php-fpm

## Installation

### 1. Verify Python Environment

```bash
# From repository root
cd /path/to/research_agents
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

The PHP app reads from the same `.env` file as the Streamlit app:

```bash
cd app_system
cp .env.example .env
nano .env  # Add your API credentials
```

Required variables:
- `API_KEY` - Your API key
- `API_BASE` - API endpoint URL
- `MODEL_PRIMARY` - Claude 4.5 Sonnet model identifier

### 3. Set Directory Permissions

```bash
cd app_system/php
chmod 755 uploads cache
chmod +x python_scripts/*.py
```

### 4. Web Server Configuration

#### Apache (.htaccess)

Create `.htaccess` in `app_system/php/`:

```apache
# Enable rewrite engine
RewriteEngine On

# Route all requests through index.php
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule ^(.*)$ index.php [QSA,L]

# Set PHP limits
php_value upload_max_filesize 50M
php_value post_max_size 50M
php_value max_execution_time 600
php_value memory_limit 256M
```

#### nginx

Add to your server block:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/research_agents/app_system/php;
    index index.php;

    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }

    location ~ \.php$ {
        fastcgi_pass unix:/var/run/php/php7.4-fpm.sock;
        fastcgi_index index.php;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        include fastcgi_params;
    }

    location ~ /\. {
        deny all;
    }

    client_max_body_size 50M;
}
```

### 5. Test Installation

```bash
# Test Python bridge
cd app_system/php
php -r "require 'config.php'; echo 'Python: ' . PYTHON_VENV . PHP_EOL;"

# Test referee wrapper
../../venv/bin/python python_scripts/run_referee.py --help

# Test section eval wrapper
../../venv/bin/python python_scripts/run_section_eval.py --help
```

## Usage

### Development Server

```bash
cd app_system/php
php -S localhost:8080
```

Then visit: http://localhost:8080

### Production Deployment

1. Point web server document root to `app_system/php/`
2. Ensure `.env` file is readable by PHP process
3. Verify Python virtual environment path in `config.php`
4. Set up HTTPS (recommended for production)
5. Configure session storage (file-based or Redis)

## File Structure

```
php/
├── index.php                      # Main entry point
├── config.php                     # Configuration
├── includes/                      # PHP utilities
│   ├── session.php               # Session management
│   ├── file_handler.php          # File upload handling
│   └── python_bridge.php         # Python script executor
├── python_scripts/                # Python CLI wrappers
│   ├── run_referee.py            # Referee system wrapper
│   └── run_section_eval.py       # Section evaluator wrapper
├── assets/                        # Frontend resources
│   ├── css/main.css              # Styling
│   └── js/progress.js            # Progress tracking
├── uploads/                       # Temporary file storage (gitignored)
├── cache/                         # Session cache (gitignored)
└── README.md                      # This file
```

## How It Works

### Referee Report Workflow

1. User uploads paper (PDF/LaTeX/text)
2. PHP form captures configuration (paper type, personas, context)
3. PHP saves file to `uploads/{session_id}_{filename}`
4. PHP calls `python_scripts/run_referee.py` with arguments
5. Python script:
   - Extracts text from file
   - Runs `execute_debate_pipeline()` from `referee/engine.py`
   - Writes progress to `/tmp/referee_progress_{session_id}.json`
   - Returns JSON results to stdout
6. PHP polls progress file every second
7. PHP parses JSON results and displays formatted output

### Section Evaluator Workflow

1. User uploads paper and selects paper type
2. PHP calls `python_scripts/run_section_eval.py`
3. Python script:
   - Extracts text
   - Detects sections
   - Evaluates each section
   - Computes overall score
   - Returns JSON results
4. PHP displays section-by-section results

## Customization

### Adding New Paper Types

1. Add to `$paper_types` array in `index.php`
2. Ensure criteria defined in `section_eval/criteria/base.py`
3. Add prompt files in `prompts/section_evaluator/paper_type_contexts/`

### Styling

Edit `assets/css/main.css` to customize appearance. The default style mimics Streamlit's gradient theme.

### Progress Tracking

Edit `assets/js/progress.js` to customize progress polling interval (default 1 second).

## Troubleshooting

### "Python executable not found"

Check `PYTHON_VENV` path in `config.php`:

```php
define('PYTHON_VENV', realpath(__DIR__ . '/../../venv/bin/python'));
```

### "Permission denied" on uploads/cache

```bash
chmod 755 uploads cache
chown www-data:www-data uploads cache  # Apache/nginx user
```

### "Failed to start Python process"

Verify `proc_open()` is enabled:

```bash
php -r "echo function_exists('proc_open') ? 'Enabled' : 'Disabled';"
```

If disabled, edit `php.ini`:

```ini
; Remove proc_open from disable_functions
disable_functions = 
```

### Session not persisting

Check session configuration in `php.ini`:

```ini
session.save_path = "/tmp"
session.gc_maxlifetime = 3600
```

### Python script returns error

Test script directly:

```bash
cd app_system/php
../../venv/bin/python python_scripts/run_referee.py \
    --file /path/to/paper.pdf \
    --session-id test123 \
    --use-cache true
```

### File upload fails

Check PHP limits in `php.ini`:

```ini
upload_max_filesize = 50M
post_max_size = 50M
max_execution_time = 600
memory_limit = 256M
```

## Security Considerations

✅ **Input validation**: All user inputs are sanitized
✅ **File type validation**: Extension and MIME type checks
✅ **Shell escaping**: `escapeshellarg()` used for Python calls
✅ **Session security**: `httponly` and `secure` flags set
✅ **File size limits**: 50MB maximum per file
✅ **Temp file cleanup**: Old files auto-deleted after 1 hour

### Recommended Production Settings

1. **HTTPS only**: Force SSL/TLS
2. **CSRF tokens**: Add to forms (not yet implemented)
3. **Rate limiting**: Limit evaluation requests per IP
4. **Authentication**: Add user login system
5. **Logging**: Monitor failed attempts
6. **Firewall**: Restrict Python script execution to web user

## Performance

- **Caching**: 50-80% cost savings on repeated evaluations
- **Parallel personas**: Python uses `asyncio.gather()` for parallel LLM calls
- **Progress polling**: 1-second intervals (configurable)
- **File cleanup**: Automatic cleanup of files older than 1 hour

## Comparison to Streamlit Version

| Feature | Streamlit (app.py) | PHP (index.php) |
|---------|-------------------|-----------------|
| Deployment | Requires Python server | Apache/nginx |
| Session state | In-memory (process) | PHP sessions (file/Redis) |
| Progress tracking | Native callbacks | AJAX polling |
| File uploads | Native uploader | `<input type="file">` |
| Styling | Streamlit CSS | Custom CSS |
| Backend | Direct imports | CLI wrappers |

## Future Enhancements

- [ ] WebSocket-based progress (replace AJAX polling)
- [ ] Database storage for evaluation history
- [ ] User authentication system
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Horizontal scaling with Redis sessions
- [ ] Result export (PDF, Excel, JSON)
- [ ] Email notifications on completion

## License

Same as parent project.

## Support

For issues specific to the PHP version, check:
1. PHP error logs: `/var/log/apache2/error.log` or `/var/log/php-fpm/error.log`
2. Python script errors: Check stderr output in browser console
3. Session debugging: Enable `display_errors` in `config.php` (development only)

For evaluation logic issues, see main project documentation.
