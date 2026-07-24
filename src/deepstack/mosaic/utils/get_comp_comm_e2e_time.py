import logging
log = logging.getLogger(__name__) 
import math

def get_comp_comm_e2e_time(compute_time:float, network_hop_latency:float, network_link_time:float, waves, overlap:bool):
    
    log.info("waves before ceiling: %s", waves)
    waves = math.ceil(waves)
    log.info("waves after ceiling: %s", waves)

    log.info("compute_time: %s, network_hop_latency: %s, network_link_time: %s", compute_time, network_hop_latency, network_link_time)

    if overlap and waves > 1:
        
        # prologue: only compute
        compute_wave1_time = compute_time / waves
        # loop: loop
        compute_loop_time = compute_time * (waves - 1) / waves 
        communication_loop_time = network_hop_latency + network_link_time * (waves - 1) / waves 
        loop_time = max(compute_loop_time, communication_loop_time)
        # epilogue
        communication_last_wave_time = network_hop_latency + network_link_time / waves
        log.info("Computation communication overlap: True")
        
        log.info(
            "waves: %s, get_comp_comm_e2e_time, compute_wave1_time: %s, compute_loop_time: %s, communication_loop_time: %s, communication_last_wave_time: %s",
            waves,
            compute_wave1_time,
            compute_loop_time,
            communication_loop_time,
            communication_last_wave_time,
        )
        log.info("e2e overall time: %s", compute_wave1_time + loop_time + communication_last_wave_time)
        return compute_wave1_time + loop_time + communication_last_wave_time
        # return compute_time + max(network_hop_latency, network_link_time)
    elif waves<=1:
        log.info("Computation communication overlap: False")
        log.info("No computation communication overlap due to waves: %s", waves)
        return compute_time + network_hop_latency + network_link_time
    elif overlap == False:
        log.info("Computation communication overlap: False")
        log.info("No computation communication overlap due to modeling configuration: overlap = %s", overlap)
        return compute_time + network_hop_latency + network_link_time
    else:
        raise ValueError("waves: %s or config not valid", waves)    
