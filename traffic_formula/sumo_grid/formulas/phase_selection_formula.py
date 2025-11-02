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
    
def sqrt(x):
    if x < 0:
        return 0
    return np.sqrt(x)


def predict_speed_improvement(chosen_phase, transition_needed, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
    """
    Predict the speed improvement based on the discovered formula.
    
    Formula: sub(div(global_speed_per_meter, speed_time_efficiency), local_to_global_ratio)
    """
    # Original features
    X0 = chosen_phase
    X1 = transition_needed
    X2 = avg_incoming_speed
    X3 = avg_outgoing_speed
    X4 = local_avg_speed
    X5 = global_avg_speed
    
    # Segment lengths
    X6 = incoming_length
    X7 = outgoing_length
    X8 = incoming_length + outgoing_length  # total_segment_length
    
    # Speed ratios
    X9 = div(local_avg_speed, global_avg_speed)  # local_to_global_ratio
    X10 = div(avg_incoming_speed, local_avg_speed)  # incoming_to_local_ratio
    X11 = div(avg_outgoing_speed, local_avg_speed)  # outgoing_to_local_ratio
    X12 = div(avg_incoming_speed, avg_outgoing_speed)  # incoming_to_outgoing_ratio
    
    # Local-global speed relationships
    X13 = local_avg_speed - global_avg_speed  # local_global_diff
    X14 = local_avg_speed + global_avg_speed  # local_global_sum
    X15 = local_avg_speed * global_avg_speed  # local_global_product
    
    # Basic speed differences and relationships
    X16 = avg_incoming_speed - avg_outgoing_speed  # incoming_outgoing_diff
    X17 = avg_incoming_speed - local_avg_speed  # incoming_local_diff
    X18 = avg_outgoing_speed - local_avg_speed  # outgoing_local_diff
    X19 = avg_incoming_speed + avg_outgoing_speed  # incoming_outgoing_sum
    X20 = avg_incoming_speed + local_avg_speed  # incoming_local_sum
    X21 = avg_outgoing_speed + local_avg_speed  # outgoing_local_sum
    X22 = avg_incoming_speed * avg_outgoing_speed  # incoming_outgoing_product
    X23 = avg_incoming_speed * local_avg_speed  # incoming_local_product
    X24 = avg_outgoing_speed * local_avg_speed  # outgoing_local_product
    
    # Length-normalized speed features
    X25 = div(avg_incoming_speed, incoming_length)  # incoming_speed_per_meter
    X26 = div(avg_outgoing_speed, outgoing_length)  # outgoing_speed_per_meter
    X27 = div(local_avg_speed, (incoming_length + outgoing_length))  # local_speed_per_meter
    X28 = div(global_avg_speed, (incoming_length + outgoing_length))  # global_speed_per_meter
    
    # Differences for length-normalized speeds
    X29 = X25 - X26  # incoming_outgoing_speed_per_meter_diff
    X30 = X27 - X28  # local_global_speed_per_meter_diff
    X31 = X25 - X27  # incoming_local_speed_per_meter_diff
    X32 = X26 - X27  # outgoing_local_speed_per_meter_diff
    
    # Sum of length-normalized speeds
    X33 = X25 + X26  # incoming_outgoing_speed_per_meter_sum
    X34 = X27 + X28  # local_global_speed_per_meter_sum
    X35 = X25 + X26 + X27 + X28  # all_speed_per_meter_sum
    
    # Products of length-normalized speeds
    X36 = X25 * X26  # incoming_outgoing_speed_per_meter_product
    X37 = X27 * X28  # local_global_speed_per_meter_product
    
    # Travel time estimates (in seconds)
    X38 = div(incoming_length, avg_incoming_speed)  # incoming_travel_time
    X39 = div(outgoing_length, avg_outgoing_speed)  # outgoing_travel_time
    X40 = X38 + X39  # total_travel_time
    
    # Travel time differences and relationships
    X41 = X38 - X39  # incoming_outgoing_travel_time_diff
    X42 = div(X38, X40)  # incoming_total_travel_time_ratio
    X43 = div(X39, X40)  # outgoing_total_travel_time_ratio
    X44 = X38 * X39  # travel_time_product
    
    # Combined speed-time metrics
    X45 = local_avg_speed * X40  # speed_time_balance
    X46 = div(local_avg_speed, X40)  # speed_time_efficiency
    
    # Phase-specific features
    X47 = 1 if (chosen_phase == 2 or chosen_phase == 3) else 0  # phase_is_straight
    X48 = 1 if (chosen_phase == 4 or chosen_phase == 5) else 0  # phase_is_left
    X49 = 1 if (chosen_phase == 2 or chosen_phase == 4) else 0  # phase_is_east_west
    X50 = 1 if (chosen_phase == 3 or chosen_phase == 5) else 0  # phase_is_north_south
    
    # Define variables for readability
    # Segment lengths
    incoming_length = X6
    outgoing_length = X7
    total_segment_length = X8
    
    # Speed ratios
    local_to_global_ratio = X9
    incoming_to_local_ratio = X10
    outgoing_to_local_ratio = X11
    incoming_to_outgoing_ratio = X12
    
    # Local-global speed relationships
    local_global_diff = X13
    local_global_sum = X14
    local_global_product = X15
    
    # Basic speed differences and relationships
    incoming_outgoing_diff = X16
    incoming_local_diff = X17
    outgoing_local_diff = X18
    incoming_outgoing_sum = X19
    incoming_local_sum = X20
    outgoing_local_sum = X21
    incoming_outgoing_product = X22
    incoming_local_product = X23
    outgoing_local_product = X24
    
    # Length-normalized speed features
    incoming_speed_per_meter = X25
    outgoing_speed_per_meter = X26
    local_speed_per_meter = X27
    global_speed_per_meter = X28
    
    # Differences for length-normalized speeds
    incoming_outgoing_speed_per_meter_diff = X29
    local_global_speed_per_meter_diff = X30
    incoming_local_speed_per_meter_diff = X31
    outgoing_local_speed_per_meter_diff = X32
    
    # Sum of length-normalized speeds
    incoming_outgoing_speed_per_meter_sum = X33
    local_global_speed_per_meter_sum = X34
    all_speed_per_meter_sum = X35
    
    # Products of length-normalized speeds
    incoming_outgoing_speed_per_meter_product = X36
    local_global_speed_per_meter_product = X37
    
    # Travel time estimates
    incoming_travel_time = X38
    outgoing_travel_time = X39
    total_travel_time = X40
    
    # Travel time differences and relationships
    incoming_outgoing_travel_time_diff = X41
    incoming_total_travel_time_ratio = X42
    outgoing_total_travel_time_ratio = X43
    travel_time_product = X44
    
    # Combined speed-time metrics
    speed_time_balance = X45
    speed_time_efficiency = X46
    
    # Phase-specific features
    phase_is_straight = X47
    phase_is_left = X48
    phase_is_east_west = X49
    phase_is_north_south = X50
    
    # Calculate predicted improvement using the formula
    return sub(div(global_speed_per_meter, speed_time_efficiency), local_to_global_ratio)


def select_best_phase(current_phase, avg_incoming_speed, avg_outgoing_speed, local_avg_speed, global_avg_speed, num_vehicles=None, incoming_length=200, outgoing_length=200):
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
    num_vehicles : int, optional
        Number of vehicles in the simulation (not used in HERE API-like scenario)
    incoming_length : float, optional
        Length of incoming road segment in meters (default 200)
    outgoing_length : float, optional
        Length of outgoing road segment in meters (default 200)
        
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
            avg_outgoing_speed, local_avg_speed, global_avg_speed, 
            num_vehicles, incoming_length, outgoing_length
        )
        
        # Apply a penalty for transitions to account for the transition phase cost
        if transition_needed:
            improvement -= 0.3  # Reduced penalty to encourage more exploration
        
        if improvement > best_improvement:
            best_improvement = improvement
            best_phase = phase
    
    return best_phase
