
def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles):
    # Formula: sub(div(sub(sub(global_avg_speed, local_avg_speed), sub(local_avg_speed, global_avg_speed)), avg_incoming_speed), div(local_avg_speed, global_avg_speed))
    
    # Implementation of the formula
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = local_avg_speed
    X5 = global_avg_speed
    X6 = num_vehicles
    
    # Calculate predicted improvement (this is the formula discovered by symbolic regression)
    return sub(div(sub(sub(global_avg_speed, local_avg_speed), sub(local_avg_speed, global_avg_speed)), avg_incoming_speed), div(local_avg_speed, global_avg_speed))

def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles):
    """
    Select the best phase based on predicted speed improvement.
    
    Parameters:
    -----------
    current_phase : int
        The current phase (2-5)
    avg_incoming_speed : float
        Average speed on incoming edges
    avg_outgoing_speed : float
        Average speed on outgoing edges
    local_avg_speed : float
        Average speed around the intersection
    global_avg_speed : float
        Global average speed
    num_vehicles : int
        Number of vehicles in the simulation
        
    Returns:
    --------
    int
        The best phase (2-5) to select next
    """
    best_improvement = float('-inf')
    best_phase = current_phase  # Default to keeping current phase to avoid transitions
    
    # Evaluate each possible phase
    for phase in range(2, 6):  # Phases 2-5
        # Check if this would require a transition
        transition_needed = 1 if phase != current_phase else 0
        
        # Predict speed improvement
        improvement = predict_speed_improvement(
            phase, transition_needed, avg_incoming_speed, 
            avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles
        )
        
        # Apply a penalty for transitions to account for the transition phase cost
        if transition_needed:
            improvement -= 0.5  # This penalty might need tuning
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase
