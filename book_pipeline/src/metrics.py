"""
Metrics collection module for the book pipeline.
Uses Prometheus metrics for tracking pipeline performance and usage.
"""

import time
import logging
import contextlib
from typing import Optional, Dict, Any, Callable
from functools import wraps

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Define metrics
if PROMETHEUS_AVAILABLE:
    # Counters
    PIPELINE_RUNS_TOTAL = Counter(
        'pipeline_runs_total', 
        'Total number of pipeline runs', 
        ['status']
    )
    
    EXTRACTION_RUNS_TOTAL = Counter(
        'extraction_runs_total', 
        'Total number of extraction runs',
        ['method', 'status']
    )
    
    LLM_CALLS_TOTAL = Counter(
        'llm_calls_total',
        'Total number of LLM API calls',
        ['model', 'status']
    )
    
    DOCUMENTS_PROCESSED_TOTAL = Counter(
        'documents_processed_total',
        'Total number of documents processed'
    )
    
    ERRORS_TOTAL = Counter(
        'errors_total',
        'Total number of errors',
        ['error_type']
    )
    
    # Histograms for timings
    EXTRACTION_TIME = Histogram(
        'extraction_time_seconds',
        'Time spent on text extraction',
        ['method']
    )
    
    TRANSFORM_TIME = Histogram(
        'transform_time_seconds',
        'Time spent on text transformation',
        ['method']
    )
    
    LLM_PROCESSING_TIME = Histogram(
        'llm_processing_time_seconds',
        'Time spent on LLM processing',
        ['model']
    )
    
    HIGHLIGHTING_TIME = Histogram(
        'highlighting_time_seconds',
        'Time spent on PDF highlighting'
    )
    
    # Gauges for active processes
    ACTIVE_PIPELINES = Gauge(
        'active_pipelines',
        'Number of active pipeline runs'
    )

    # Cache metrics
    CACHE_HITS = Counter(
        'cache_hits_total',
        'Total number of cache hits'
    )
    
    CACHE_MISSES = Counter(
        'cache_misses_total',
        'Total number of cache misses'
    )
else:
    logger.warning("Prometheus client not available. Metrics will not be collected.")


def start_metrics_server(port: int = 8001) -> bool:
    """
    Start the Prometheus metrics server.
    
    Args:
        port: The port to run the server on
        
    Returns:
        True if server started successfully, False otherwise
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Cannot start metrics server: prometheus_client not installed")
        return False
    
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        return True
    except Exception as e:
        logger.error(f"Failed to start metrics server: {str(e)}")
        return False


def increment_counter(counter, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Safely increment a Prometheus counter.
    
    Args:
        counter: The counter to increment
        labels: Optional labels to apply to the counter
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    try:
        if labels:
            counter.labels(**labels).inc()
        else:
            counter.inc()
    except Exception as e:
        logger.error(f"Error incrementing counter: {str(e)}")


def record_error(error_type: str) -> None:
    """
    Record an error in the metrics.
    
    Args:
        error_type: The type of error to record
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(ERRORS_TOTAL, {"error_type": error_type})


def record_pipeline_run(status: str = "success") -> None:
    """
    Record a pipeline run in the metrics.
    
    Args:
        status: The status of the run ("success" or "failure")
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(PIPELINE_RUNS_TOTAL, {"status": status})
    increment_counter(DOCUMENTS_PROCESSED_TOTAL)


def record_extraction_run(method: str, status: str = "success") -> None:
    """
    Record an extraction run in the metrics.
    
    Args:
        method: The extraction method used
        status: The status of the run
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(EXTRACTION_RUNS_TOTAL, {"method": method, "status": status})


def record_llm_call(model: str, status: str = "success") -> None:
    """
    Record an LLM API call in the metrics.
    
    Args:
        model: The model used for the call
        status: The status of the call
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(LLM_CALLS_TOTAL, {"model": model, "status": status})


def record_cache_hit() -> None:
    """Record a cache hit in the metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(CACHE_HITS)


def record_cache_miss() -> None:
    """Record a cache miss in the metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    
    increment_counter(CACHE_MISSES)


def time_it(histogram, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to measure and record the execution time of a function.
    
    Args:
        histogram: The Prometheus histogram to record the time in
        labels: Optional labels to apply to the histogram
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            try:
                ACTIVE_PIPELINES.inc()
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                if labels:
                    histogram.labels(**labels).observe(execution_time)
                else:
                    histogram.observe(execution_time)
                
                return result
            except Exception as e:
                raise e
            finally:
                ACTIVE_PIPELINES.dec()
                
        return wrapper
    return decorator


@contextlib.contextmanager
def time_it_context(histogram, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to measure and record the execution time of a block of code.
    
    Args:
        histogram: The Prometheus histogram to record the time in
        labels: Optional labels to apply to the histogram
    """
    if not PROMETHEUS_AVAILABLE:
        yield
        return
        
    try:
        ACTIVE_PIPELINES.inc()
        start_time = time.time()
        yield
        execution_time = time.time() - start_time
        
        if labels:
            histogram.labels(**labels).observe(execution_time)
        else:
            histogram.observe(execution_time)
    except Exception as e:
        raise e
    finally:
        ACTIVE_PIPELINES.dec()


# Define convenience decorators for commonly timed functions
def time_extraction(method: str = "default"):
    """Time an extraction function."""
    return time_it(EXTRACTION_TIME, {"method": method})


def time_transform(method: str = "default"):
    """Time a transform function."""
    return time_it(TRANSFORM_TIME, {"method": method})


def time_llm_processing(model: str = "default"):
    """Time an LLM processing function."""
    return time_it(LLM_PROCESSING_TIME, {"model": model})


def time_highlighting():
    """Time a highlighting function."""
    return time_it(HIGHLIGHTING_TIME)


def time_highlighting_context():
    """Context manager to time a highlighting code block."""
    return time_it_context(HIGHLIGHTING_TIME)