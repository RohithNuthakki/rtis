import numpy as np


# Helper functions for safe operations
def add(x, y):
    return x + y

def sub(x, y):
    return x - y
    
def mul(x, y):
    return x * y
    
def div(x, y):
    if abs(y) < 0.001:
        return x  # Return numerator when denominator is close to zero
    return x / y


def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, global_avg_speed, local_avg_speed=None, num_vehicles=None, incoming_length=200, outgoing_length=200):
    """
    Predict speed improvement using only pure speed features.
    
    Formula: div(sub(calculated_global_avg_speed, avg_incoming_speed), 2.255)
    
    Parameters:
    -----------
    chosen_phase : int
        The phase to evaluate (2-5)
    transition_needed : int
        Whether a transition is needed (0 or 1)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    global_avg_speed : float
        Global average speed across the network (average of all incoming and outgoing speeds)
        Note: This can be either 'global_avg_speed' or 'calculated_global_avg_speed' from the data
    local_avg_speed : float, optional
        Local average speed (not used in this formula)
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (not used in this formula)
    outgoing_length : float, optional
        Length of outgoing roads (not used in this formula)
        
    Returns:
    --------
    float
        Predicted speed improvement
    """
    # Define features used in the formula
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = global_avg_speed  # This works with either column name since we use the parameter name
    
    # Return the predicted improvement using the discovered formula
    return div(sub(X4, avg_incoming_speed), 2.255)


def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
    """
    Select the best phase using the pure speed formula.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float
        Local average speed around the intersection
    global_avg_speed : float
        Global average speed (average of all intersection incoming and outgoing speeds)
        Note: This can be either 'global_avg_speed' or 'calculated_global_avg_speed' from the data
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (not used in this formula)
    outgoing_length : float, optional
        Length of outgoing roads (not used in this formula)
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    """
    best_improvement = float('-inf')
    best_phase = current_phase  # Default to keeping current phase
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Check if this would require a transition
        transition_needed = 1 if phase != current_phase else 0
        
        # Predict speed improvement
        improvement = predict_speed_improvement(
            phase, transition_needed, avg_incoming_speed, avg_outgoing_speed,
            global_avg_speed, local_avg_speed, num_vehicles,
            incoming_length, outgoing_length
        )
        
        # Apply a small penalty for transitions
        if transition_needed:
            improvement -= 0.2
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase


def calculate_max_pressure_score(chosen_phase, avg_incoming_speed, avg_outgoing_speed, 
                                incoming_length=200, outgoing_length=200):
    """
    Calculate the Max Pressure score: (incoming speed / incoming length) - (outgoing speed / outgoing length)
    
    This measures the differential pressure across the intersection.
    """
    # Avoid division by zero
    if incoming_length <= 0:
        incoming_length = 0.001
    if outgoing_length <= 0:
        outgoing_length = 0.001
        
    # Calculate pressure differential
    incoming_pressure = avg_incoming_speed / incoming_length
    outgoing_pressure = avg_outgoing_speed / outgoing_length
    
    # Return the pressure differential
    return incoming_pressure - outgoing_pressure

def select_max_pressure_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed=None, 
                             global_avg_speed=None, num_vehicles=None, incoming_length=200, outgoing_length=200):
    """
    Select the best phase using the Max Pressure formula.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float, optional
        Local average speed (not used in this formula)
    global_avg_speed : float, optional
        Global average speed (not used in this formula)
    num_vehicles : int, optional
        Number of vehicles (not used in this formula)
    incoming_length : float, optional
        Length of incoming roads (default 200)
    outgoing_length : float, optional
        Length of outgoing roads (default 200)
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    """
    best_score = float('-inf')
    best_phase = current_phase  # Default to keeping current phase
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Calculate Max Pressure score
        score = calculate_max_pressure_score(
            phase, avg_incoming_speed, avg_outgoing_speed, 
            incoming_length, outgoing_length
        )
        
        # Apply a small penalty for transitions
        if phase != current_phase:
            score -= 0.2
        
        if score > best_score:
            best_score = score
            best_phase = phase
    
    return best_phase
