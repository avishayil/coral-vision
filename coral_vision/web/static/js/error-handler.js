/**
 * Centralized error handling for the application
 * Provides consistent error handling and user feedback
 */

class ErrorHandler {
    constructor() {
        this.errorCallbacks = [];
    }

    /**
     * Register an error callback
     */
    onError(callback) {
        this.errorCallbacks.push(callback);
    }

    /**
     * Handle an error with user-friendly messages
     */
    handleError(error, context = {}) {
        let message = 'An unexpected error occurred';
        let userMessage = 'Something went wrong. Please try again.';

        if (error instanceof Error) {
            message = error.message;
        } else if (typeof error === 'string') {
            message = error;
        } else if (error?.error) {
            message = error.error;
        }

        // Map common errors to user-friendly messages
        if (message.includes('API key')) {
            userMessage = 'API key is required or invalid. Please check your API key configuration.';
        } else if (message.includes('NetworkError') || message.includes('Failed to fetch')) {
            userMessage = 'Network error. Please check your connection and try again.';
        } else if (message.includes('camera') || message.includes('Camera')) {
            userMessage = 'Camera access error. Please allow camera access and try again.';
        } else if (message.includes('WebSocket')) {
            userMessage = 'Connection error. Please refresh the page and try again.';
        } else if (message.includes('401') || message.includes('Unauthorized')) {
            userMessage = 'Authentication failed. Please check your API key.';
        } else if (message.includes('403') || message.includes('Forbidden')) {
            userMessage = 'Access denied. Please check your permissions.';
        } else if (message.includes('404')) {
            userMessage = 'Resource not found.';
        } else if (message.includes('429') || message.includes('rate limit')) {
            userMessage = 'Too many requests. Please wait a moment and try again.';
        } else if (message.includes('500') || message.includes('Internal Server Error')) {
            userMessage = 'Server error. Please try again later.';
        } else if (message.includes('503') || message.includes('Service Unavailable')) {
            userMessage = 'Service temporarily unavailable. Please try again later.';
        }

        // Notify registered callbacks
        this.errorCallbacks.forEach(callback => {
            try {
                callback(error, userMessage, context);
            } catch (e) {
                console.error('Error in error callback:', e);
            }
        });

        return { message, userMessage };
    }

    /**
     * Show error to user in UI
     */
    showError(userMessage, elementId = null) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `❌ ${userMessage}`;
        errorDiv.style.display = 'block';
        errorDiv.style.margin = '10px 0';
        errorDiv.style.padding = '10px';
        errorDiv.style.backgroundColor = '#fee';
        errorDiv.style.border = '1px solid #fcc';
        errorDiv.style.borderRadius = '4px';
        errorDiv.style.color = '#c33';

        if (elementId) {
            const container = document.getElementById(elementId);
            if (container) {
                container.innerHTML = '';
                container.appendChild(errorDiv);
                return;
            }
        }

        // Show as notification if no container specified
        document.body.appendChild(errorDiv);
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    /**
     * Wrap async function with error handling
     */
    wrapAsync(fn, context = {}) {
        return async (...args) => {
            try {
                return await fn(...args);
            } catch (error) {
                const { userMessage } = this.handleError(error, context);
                throw new Error(userMessage);
            }
        };
    }
}

// Create global error handler instance
const errorHandler = new ErrorHandler();

// Global error handler for unhandled errors
window.addEventListener('error', (event) => {
    errorHandler.handleError(event.error || event.message, {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
    });
});

// Global handler for unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    errorHandler.handleError(event.reason, {
        type: 'unhandledRejection'
    });
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ErrorHandler, errorHandler };
}

