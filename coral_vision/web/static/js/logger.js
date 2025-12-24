/**
 * Structured logging utility for frontend
 * Provides consistent logging with levels and optional error reporting
 */

const LogLevel = {
    DEBUG: 0,
    INFO: 1,
    WARN: 2,
    ERROR: 3,
    NONE: 4
};

class Logger {
    constructor(module = 'app', level = LogLevel.INFO) {
        this.module = module;
        this.level = level;
        this.isProduction = window.location.hostname !== 'localhost' &&
                           !window.location.hostname.startsWith('127.0.0.1');
    }

    debug(message, ...args) {
        if (this.level <= LogLevel.DEBUG && !this.isProduction) {
            console.debug(`[${this.module}] ${message}`, ...args);
        }
    }

    info(message, ...args) {
        if (this.level <= LogLevel.INFO) {
            console.info(`[${this.module}] ${message}`, ...args);
        }
    }

    warn(message, ...args) {
        if (this.level <= LogLevel.WARN) {
            console.warn(`[${this.module}] ${message}`, ...args);
        }
    }

    error(message, error = null, ...args) {
        if (this.level <= LogLevel.ERROR) {
            console.error(`[${this.module}] ${message}`, error, ...args);

            // In production, could send to error reporting service
            if (this.isProduction && error) {
                this.reportError(message, error);
            }
        }
    }

    reportError(message, error) {
        // Placeholder for error reporting service (e.g., Sentry)
        // In production, you could send errors to a monitoring service
        try {
            // Example: Send to error endpoint
            // fetch('/api/errors', {
            //     method: 'POST',
            //     headers: { 'Content-Type': 'application/json' },
            //     body: JSON.stringify({
            //         message,
            //         error: error?.toString(),
            //         stack: error?.stack,
            //         url: window.location.href,
            //         userAgent: navigator.userAgent
            //     })
            // }).catch(() => {});
        } catch (e) {
            // Silently fail error reporting
        }
    }
}

// Create default logger instance
const logger = new Logger('app', LogLevel.INFO);

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Logger, LogLevel, logger };
}

