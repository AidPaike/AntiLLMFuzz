"""Random seed management for deterministic execution."""

import random
import hashlib
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SeedState:
    """State information for a random seed."""
    seed: int
    component: str
    operation: str
    timestamp: datetime = field(default_factory=datetime.now)
    call_count: int = 0


class RandomSeedManager:
    """Manages random seeds for deterministic execution across all components."""
    
    def __init__(self, master_seed: Optional[int] = None):
        """Initialize random seed manager.
        
        Args:
            master_seed: Master seed for all random operations. If None, uses current time.
        """
        self._lock = threading.Lock()
        self._master_seed = master_seed if master_seed is not None else self._generate_master_seed()
        self._component_seeds: Dict[str, int] = {}
        self._seed_history: List[SeedState] = []
        self._random_instances: Dict[str, random.Random] = {}
        
        # Initialize global random state
        random.seed(self._master_seed)
        
        logger.info(f"RandomSeedManager initialized with master seed: {self._master_seed}")
    
    def _generate_master_seed(self) -> int:
        """Generate a master seed based on current timestamp."""
        timestamp = datetime.now().timestamp()
        # Use hash to ensure we get a good distribution
        seed_bytes = hashlib.md5(str(timestamp).encode()).digest()
        return int.from_bytes(seed_bytes[:4], byteorder='big')
    
    def get_master_seed(self) -> int:
        """Get the master seed."""
        return self._master_seed
    
    def set_master_seed(self, seed: int) -> None:
        """Set a new master seed and reset all component seeds.
        
        Args:
            seed: New master seed
        """
        with self._lock:
            self._master_seed = seed
            self._component_seeds.clear()
            self._random_instances.clear()
            self._seed_history.clear()
            
            # Reset global random state
            random.seed(self._master_seed)
            
            logger.info(f"Master seed updated to: {self._master_seed}")
    
    def get_component_seed(self, component: str, operation: str = "default") -> int:
        """Get a deterministic seed for a specific component and operation.
        
        Args:
            component: Component name (e.g., 'llm_generator', 'target_system')
            operation: Operation name (e.g., 'test_generation', 'response_simulation')
            
        Returns:
            Deterministic seed for the component/operation
        """
        with self._lock:
            key = f"{component}:{operation}"
            
            if key not in self._component_seeds:
                # Generate deterministic seed based on master seed and component/operation
                seed_input = f"{self._master_seed}:{component}:{operation}"
                seed_bytes = hashlib.md5(seed_input.encode()).digest()
                component_seed = int.from_bytes(seed_bytes[:4], byteorder='big')
                
                self._component_seeds[key] = component_seed
                
                # Record seed state
                seed_state = SeedState(
                    seed=component_seed,
                    component=component,
                    operation=operation
                )
                self._seed_history.append(seed_state)
                
                logger.debug(f"Generated seed {component_seed} for {key}")
            
            # Update call count
            for state in self._seed_history:
                if state.component == component and state.operation == operation:
                    state.call_count += 1
                    break
            
            return self._component_seeds[key]
    
    def get_random_instance(self, component: str, operation: str = "default") -> random.Random:
        """Get a Random instance seeded for a specific component and operation.
        
        Args:
            component: Component name
            operation: Operation name
            
        Returns:
            Random instance with deterministic seed
        """
        with self._lock:
            key = f"{component}:{operation}"
            
            if key not in self._random_instances:
                seed = self.get_component_seed(component, operation)
                self._random_instances[key] = random.Random(seed)
                logger.debug(f"Created Random instance for {key} with seed {seed}")
            
            return self._random_instances[key]
    
    def seed_component(self, component: str, operation: str = "default") -> None:
        """Seed the global random state for a specific component operation.
        
        Args:
            component: Component name
            operation: Operation name
        """
        seed = self.get_component_seed(component, operation)
        random.seed(seed)
        logger.debug(f"Seeded global random state with {seed} for {component}:{operation}")
    
    def reset_component(self, component: str, operation: str = "default") -> None:
        """Reset the random state for a specific component operation.
        
        Args:
            component: Component name
            operation: Operation name
        """
        with self._lock:
            key = f"{component}:{operation}"
            
            if key in self._component_seeds:
                seed = self._component_seeds[key]
                
                # Reset random instance if it exists
                if key in self._random_instances:
                    self._random_instances[key] = random.Random(seed)
                
                logger.debug(f"Reset random state for {key}")
    
    def get_seed_history(self) -> List[SeedState]:
        """Get the history of all generated seeds.
        
        Returns:
            List of SeedState objects
        """
        with self._lock:
            return self._seed_history.copy()
    
    def get_component_seeds(self) -> Dict[str, int]:
        """Get all component seeds.
        
        Returns:
            Dictionary mapping component:operation to seed
        """
        with self._lock:
            return self._component_seeds.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export the current state for reproduction.
        
        Returns:
            Dictionary containing all state information
        """
        with self._lock:
            return {
                'master_seed': self._master_seed,
                'component_seeds': self._component_seeds.copy(),
                'seed_history': [
                    {
                        'seed': state.seed,
                        'component': state.component,
                        'operation': state.operation,
                        'timestamp': state.timestamp.isoformat(),
                        'call_count': state.call_count
                    }
                    for state in self._seed_history
                ]
            }
    
    def import_state(self, state_dict: Dict[str, Any]) -> None:
        """Import state from a previous export.
        
        Args:
            state_dict: State dictionary from export_state()
        """
        with self._lock:
            self._master_seed = state_dict['master_seed']
            self._component_seeds = state_dict['component_seeds'].copy()
            
            # Reconstruct seed history
            self._seed_history = []
            for hist_item in state_dict['seed_history']:
                seed_state = SeedState(
                    seed=hist_item['seed'],
                    component=hist_item['component'],
                    operation=hist_item['operation'],
                    timestamp=datetime.fromisoformat(hist_item['timestamp']),
                    call_count=hist_item['call_count']
                )
                self._seed_history.append(seed_state)
            
            # Clear random instances to force regeneration
            self._random_instances.clear()
            
            # Reset global random state
            random.seed(self._master_seed)
            
            logger.info(f"Imported seed state with master seed: {self._master_seed}")
    
    def generate_deterministic_value(self, component: str, operation: str, 
                                   min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Generate a deterministic random value for a component operation.
        
        Args:
            component: Component name
            operation: Operation name
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Deterministic random value in range [min_val, max_val)
        """
        rng = self.get_random_instance(component, operation)
        return rng.uniform(min_val, max_val)
    
    def generate_deterministic_int(self, component: str, operation: str, 
                                 min_val: int = 0, max_val: int = 100) -> int:
        """Generate a deterministic random integer for a component operation.
        
        Args:
            component: Component name
            operation: Operation name
            min_val: Minimum value (inclusive)
            max_val: Maximum value (exclusive)
            
        Returns:
            Deterministic random integer in range [min_val, max_val)
        """
        rng = self.get_random_instance(component, operation)
        return rng.randint(min_val, max_val - 1)
    
    def generate_deterministic_choice(self, component: str, operation: str, 
                                    choices: List[Any]) -> Any:
        """Make a deterministic random choice for a component operation.
        
        Args:
            component: Component name
            operation: Operation name
            choices: List of choices to select from
            
        Returns:
            Deterministic random choice from the list
        """
        if not choices:
            raise ValueError("Cannot choose from empty list")
        
        rng = self.get_random_instance(component, operation)
        return rng.choice(choices)
    
    def shuffle_deterministic(self, component: str, operation: str, 
                            items: List[Any]) -> List[Any]:
        """Shuffle a list deterministically for a component operation.
        
        Args:
            component: Component name
            operation: Operation name
            items: List to shuffle
            
        Returns:
            New shuffled list
        """
        rng = self.get_random_instance(component, operation)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        return shuffled


# Global seed manager instance
_seed_manager: Optional[RandomSeedManager] = None


def get_seed_manager(master_seed: Optional[int] = None) -> RandomSeedManager:
    """Get global random seed manager instance.
    
    Args:
        master_seed: Master seed for initialization. Only used on first call.
        
    Returns:
        RandomSeedManager instance
    """
    global _seed_manager
    if _seed_manager is None:
        _seed_manager = RandomSeedManager(master_seed)
    elif master_seed is not None:
        # Update master seed if provided
        _seed_manager.set_master_seed(master_seed)
    return _seed_manager


def set_global_seed(seed: int) -> None:
    """Set the global master seed.
    
    Args:
        seed: Master seed to set
    """
    manager = get_seed_manager()
    manager.set_master_seed(seed)


def get_component_random(component: str, operation: str = "default") -> random.Random:
    """Get a Random instance for a component operation.
    
    Args:
        component: Component name
        operation: Operation name
        
    Returns:
        Random instance with deterministic seed
    """
    manager = get_seed_manager()
    return manager.get_random_instance(component, operation)


def seed_component_random(component: str, operation: str = "default") -> None:
    """Seed the global random state for a component operation.
    
    Args:
        component: Component name
        operation: Operation name
    """
    manager = get_seed_manager()
    manager.seed_component(component, operation)