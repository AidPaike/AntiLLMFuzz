"""Performance optimization components for the LLM fuzzer simulator."""

import time
import hashlib
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc

from src.fuzzer.data_models import TestCase, APISpec, ExecutionResult, CoverageMetrics
from src.utils.logger import get_logger


@dataclass
class CacheEntry:
    """Entry in the LLM response cache."""
    response: str
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CoverageState:
    """Incremental coverage calculation state."""
    covered_lines: Set[int] = field(default_factory=set)
    total_lines: int = 0
    covered_branches: Set[str] = field(default_factory=set)
    total_branches: int = 0
    covered_functions: Set[str] = field(default_factory=set)
    total_functions: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_coverage(self, new_coverage: CoverageMetrics) -> CoverageMetrics:
        """Update coverage incrementally and return new metrics."""
        # Update line coverage
        if new_coverage.covered_lines:
            self.covered_lines.update(new_coverage.covered_lines)
        
        if new_coverage.total_lines > 0:
            self.total_lines = max(self.total_lines, new_coverage.total_lines)
        
        # Calculate updated metrics
        line_coverage = len(self.covered_lines) / max(1, self.total_lines) * 100
        
        # For now, use the new coverage for other metrics (can be enhanced)
        branch_coverage = new_coverage.branch_coverage
        function_coverage = new_coverage.function_coverage
        
        self.last_update = datetime.now()
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            api_endpoint_coverage=new_coverage.api_endpoint_coverage,
            covered_lines=self.covered_lines.copy(),
            total_lines=self.total_lines
        )


class LLMResponseCache:
    """High-performance cache for LLM responses with intelligent eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        """Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.logger = get_logger("LLMResponseCache")
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def _generate_cache_key(self, prompt: str, model: str, temperature: float, 
                          max_tokens: int) -> str:
        """Generate a cache key for the given parameters."""
        key_data = {
            'prompt': prompt,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, prompt: str, model: str, temperature: float, 
            max_tokens: int) -> Optional[str]:
        """Get cached response if available.
        
        Args:
            prompt: LLM prompt
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            
        Returns:
            Cached response or None if not found
        """
        cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens)
        
        with self.lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check if entry is still valid (TTL)
                if datetime.now() - entry.timestamp < self.ttl:
                    entry.touch()
                    # Move to end (LRU)
                    self.cache.move_to_end(cache_key)
                    self.hits += 1
                    self.logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                    return entry.response
                else:
                    # Entry expired
                    del self.cache[cache_key]
                    self.logger.debug(f"Cache entry expired: {cache_key[:16]}...")
            
            self.misses += 1
            return None
    
    def put(self, prompt: str, model: str, temperature: float, 
            max_tokens: int, response: str) -> None:
        """Store response in cache.
        
        Args:
            prompt: LLM prompt
            model: Model name
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            response: LLM response to cache
        """
        cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens)
        
        with self.lock:
            # Remove oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key, _ = self.cache.popitem(last=False)
                self.evictions += 1
                self.logger.debug(f"Evicted cache entry: {oldest_key[:16]}...")
            
            # Add new entry
            entry = CacheEntry(
                response=response,
                timestamp=datetime.now()
            )
            self.cache[cache_key] = entry
            self.logger.debug(f"Cached response for key: {cache_key[:16]}...")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / max(1, total_requests)
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'memory_usage_estimate': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of cache in bytes."""
        total_size = 0
        for entry in self.cache.values():
            total_size += len(entry.response.encode('utf-8'))
            total_size += 200  # Estimate for metadata
        return total_size
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        removed_count = 0
        current_time = datetime.now()
        
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if current_time - entry.timestamp >= self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count


class IncrementalCoverageCalculator:
    """Efficient incremental coverage calculation."""
    
    def __init__(self):
        """Initialize the coverage calculator."""
        self.coverage_states: Dict[str, CoverageState] = {}
        self.lock = threading.RLock()
        self.logger = get_logger("IncrementalCoverageCalculator")
    
    def initialize_session(self, session_id: str) -> None:
        """Initialize coverage tracking for a session.
        
        Args:
            session_id: Session identifier
        """
        with self.lock:
            self.coverage_states[session_id] = CoverageState()
            self.logger.debug(f"Initialized coverage tracking for session: {session_id}")
    
    def update_coverage(self, session_id: str, new_coverage: CoverageMetrics) -> CoverageMetrics:
        """Update coverage incrementally for a session.
        
        Args:
            session_id: Session identifier
            new_coverage: New coverage metrics to incorporate
            
        Returns:
            Updated cumulative coverage metrics
        """
        with self.lock:
            if session_id not in self.coverage_states:
                self.initialize_session(session_id)
            
            state = self.coverage_states[session_id]
            updated_coverage = state.update_coverage(new_coverage)
            
            self.logger.debug(f"Updated coverage for session {session_id}: "
                            f"{updated_coverage.line_coverage:.1f}% line coverage")
            
            return updated_coverage
    
    def get_coverage(self, session_id: str) -> Optional[CoverageMetrics]:
        """Get current coverage for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Current coverage metrics or None if session not found
        """
        with self.lock:
            if session_id not in self.coverage_states:
                return None
            
            state = self.coverage_states[session_id]
            return CoverageMetrics(
                line_coverage=len(state.covered_lines) / max(1, state.total_lines) * 100,
                branch_coverage=0.0,  # Placeholder
                function_coverage=0.0,  # Placeholder
                api_endpoint_coverage=0.0,  # Placeholder
                covered_lines=state.covered_lines.copy(),
                total_lines=state.total_lines
            )
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up coverage tracking for a session.
        
        Args:
            session_id: Session identifier
        """
        with self.lock:
            if session_id in self.coverage_states:
                del self.coverage_states[session_id]
                self.logger.debug(f"Cleaned up coverage tracking for session: {session_id}")
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all tracked sessions.
        
        Returns:
            List of session IDs
        """
        with self.lock:
            return list(self.coverage_states.keys())


class ParallelProcessingOptimizer:
    """Optimizations for parallel processing of test cases."""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize the optimizer.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self.logger = get_logger("ParallelProcessingOptimizer")
        
        # Performance tracking
        self.execution_times: List[float] = []
        self.throughput_history: List[float] = []
        
    def optimize_batch_size(self, total_items: int, target_time_per_batch: float = 30.0) -> int:
        """Calculate optimal batch size based on performance history.
        
        Args:
            total_items: Total number of items to process
            target_time_per_batch: Target time per batch in seconds
            
        Returns:
            Optimal batch size
        """
        if not self.execution_times:
            # No history, use conservative default
            return min(10, total_items)
        
        # Calculate average time per item
        avg_time_per_item = sum(self.execution_times) / len(self.execution_times)
        
        if avg_time_per_item <= 0:
            return min(10, total_items)
        
        # Calculate optimal batch size
        optimal_batch_size = int(target_time_per_batch / avg_time_per_item)
        
        # Apply constraints
        optimal_batch_size = max(1, min(optimal_batch_size, total_items, 100))
        
        self.logger.debug(f"Calculated optimal batch size: {optimal_batch_size} "
                         f"(avg time per item: {avg_time_per_item:.3f}s)")
        
        return optimal_batch_size
    
    def execute_parallel_batches(self, items: List[Any], 
                               process_func: callable,
                               batch_size: Optional[int] = None) -> List[Any]:
        """Execute items in parallel batches with optimization.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            batch_size: Batch size (auto-calculated if None)
            
        Returns:
            List of results in original order
        """
        if not items:
            return []
        
        if batch_size is None:
            batch_size = self.optimize_batch_size(len(items))
        
        start_time = time.time()
        results = [None] * len(items)
        
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_indices = list(range(i, min(i + batch_size, len(items))))
            batches.append((batch, batch_indices))
        
        self.logger.debug(f"Processing {len(items)} items in {len(batches)} batches "
                         f"of size {batch_size}")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for batch_items, batch_indices in batches:
                future = executor.submit(self._process_batch, batch_items, process_func)
                future_to_batch[future] = batch_indices
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_indices = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for i, result in enumerate(batch_results):
                        results[batch_indices[i]] = result
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    # Fill with None for failed batch
                    for idx in batch_indices:
                        results[idx] = None
        
        # Update performance tracking
        total_time = time.time() - start_time
        avg_time_per_item = total_time / len(items)
        self.execution_times.append(avg_time_per_item)
        
        # Keep only recent history
        if len(self.execution_times) > 100:
            self.execution_times = self.execution_times[-50:]
        
        throughput = len(items) / total_time if total_time > 0 else 0
        self.throughput_history.append(throughput)
        
        if len(self.throughput_history) > 100:
            self.throughput_history = self.throughput_history[-50:]
        
        self.logger.debug(f"Parallel processing completed: {len(items)} items in "
                         f"{total_time:.2f}s (throughput: {throughput:.1f} items/s)")
        
        return results
    
    def _process_batch(self, batch_items: List[Any], process_func: callable) -> List[Any]:
        """Process a single batch of items.
        
        Args:
            batch_items: Items in the batch
            process_func: Function to process each item
            
        Returns:
            List of results for the batch
        """
        results = []
        for item in batch_items:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Item processing failed: {e}")
                results.append(None)
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.execution_times:
            return {
                'avg_time_per_item': 0.0,
                'avg_throughput': 0.0,
                'total_batches_processed': 0
            }
        
        return {
            'avg_time_per_item': sum(self.execution_times) / len(self.execution_times),
            'avg_throughput': sum(self.throughput_history) / len(self.throughput_history),
            'total_batches_processed': len(self.execution_times),
            'recent_throughput': self.throughput_history[-10:] if self.throughput_history else []
        }


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self):
        """Initialize the memory optimizer."""
        self.logger = get_logger("MemoryOptimizer")
        self.weak_refs: Set[weakref.ref] = set()
        
    def optimize_test_case_storage(self, test_cases: List[TestCase]) -> List[TestCase]:
        """Optimize memory usage of test case storage.
        
        Args:
            test_cases: List of test cases to optimize
            
        Returns:
            Optimized test cases
        """
        # Remove duplicate test cases based on parameters
        seen_params = set()
        optimized_cases = []
        
        for test_case in test_cases:
            # Create a hashable representation of parameters
            param_key = self._create_param_key(test_case.parameters)
            
            if param_key not in seen_params:
                seen_params.add(param_key)
                optimized_cases.append(test_case)
        
        removed_count = len(test_cases) - len(optimized_cases)
        if removed_count > 0:
            self.logger.debug(f"Removed {removed_count} duplicate test cases")
        
        return optimized_cases
    
    def _create_param_key(self, parameters: Dict[str, Any]) -> str:
        """Create a hashable key for parameters."""
        try:
            # Sort parameters for consistent hashing
            sorted_params = json.dumps(parameters, sort_keys=True, default=str)
            return hashlib.md5(sorted_params.encode()).hexdigest()
        except Exception:
            # Fallback for non-serializable parameters
            return str(hash(str(parameters)))
    
    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup and return statistics.
        
        Returns:
            Dictionary with cleanup statistics
        """
        # Clean up weak references
        dead_refs = []
        for ref in self.weak_refs:
            if ref() is None:
                dead_refs.append(ref)
        
        for ref in dead_refs:
            self.weak_refs.remove(ref)
        
        # Force garbage collection
        collected = gc.collect()
        
        stats = {
            'dead_references_cleaned': len(dead_refs),
            'objects_collected': collected,
            'memory_usage_mb': self._get_memory_usage()
        }
        
        self.logger.debug(f"Memory cleanup completed: {stats}")
        return stats
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def register_for_cleanup(self, obj: Any) -> None:
        """Register an object for automatic cleanup.
        
        Args:
            obj: Object to register for cleanup
        """
        weak_ref = weakref.ref(obj)
        self.weak_refs.add(weak_ref)


class PerformanceOptimizationManager:
    """Central manager for all performance optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the performance optimization manager.
        
        Args:
            config: Configuration dictionary for optimizations
        """
        self.config = config or {}
        self.logger = get_logger("PerformanceOptimizationManager")
        
        # Initialize optimization components
        cache_size = self.config.get('cache_size', 1000)
        cache_ttl = self.config.get('cache_ttl_hours', 24)
        self.llm_cache = LLMResponseCache(max_size=cache_size, ttl_hours=cache_ttl)
        
        self.coverage_calculator = IncrementalCoverageCalculator()
        
        max_workers = self.config.get('max_workers')
        self.parallel_optimizer = ParallelProcessingOptimizer(max_workers=max_workers)
        
        self.memory_optimizer = MemoryOptimizer()
        
        # Performance monitoring
        self.start_time = time.time()
        self.optimization_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_cleanups': 0,
            'parallel_batches': 0
        }
    
    def get_llm_cache(self) -> LLMResponseCache:
        """Get the LLM response cache."""
        return self.llm_cache
    
    def get_coverage_calculator(self) -> IncrementalCoverageCalculator:
        """Get the incremental coverage calculator."""
        return self.coverage_calculator
    
    def get_parallel_optimizer(self) -> ParallelProcessingOptimizer:
        """Get the parallel processing optimizer."""
        return self.parallel_optimizer
    
    def get_memory_optimizer(self) -> MemoryOptimizer:
        """Get the memory optimizer."""
        return self.memory_optimizer
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform routine maintenance and optimization.
        
        Returns:
            Dictionary with maintenance statistics
        """
        self.logger.debug("Performing routine maintenance")
        
        # Clean up expired cache entries
        expired_cleaned = self.llm_cache.cleanup_expired()
        
        # Memory cleanup
        memory_stats = self.memory_optimizer.cleanup_memory()
        
        # Update statistics
        self.optimization_stats['memory_cleanups'] += 1
        
        maintenance_stats = {
            'expired_cache_entries_cleaned': expired_cleaned,
            'memory_cleanup_stats': memory_stats,
            'cache_stats': self.llm_cache.get_statistics(),
            'parallel_stats': self.parallel_optimizer.get_performance_stats(),
            'uptime_seconds': time.time() - self.start_time
        }
        
        self.logger.debug(f"Maintenance completed: {maintenance_stats}")
        return maintenance_stats
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary with all performance statistics
        """
        return {
            'optimization_stats': self.optimization_stats.copy(),
            'cache_stats': self.llm_cache.get_statistics(),
            'parallel_stats': self.parallel_optimizer.get_performance_stats(),
            'uptime_seconds': time.time() - self.start_time,
            'active_coverage_sessions': len(self.coverage_calculator.get_all_sessions())
        }
    
    def shutdown(self) -> None:
        """Shutdown and cleanup all optimization components."""
        self.logger.info("Shutting down performance optimization manager")
        
        # Clear caches
        self.llm_cache.clear()
        
        # Final memory cleanup
        self.memory_optimizer.cleanup_memory()
        
        self.logger.info("Performance optimization manager shutdown completed")