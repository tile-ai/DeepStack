# 文件名: matmul_op.py
from ..util import *
# M, N, K, tb_m, tb_n, tb_k, L2_Cap, SM_Count, BYTE_per_num, stage_num, block_per_sm
import numpy as np
import math

# def calculate_conv_implicit_gemm_resource_utilization(m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,arch.l2_capacity,arch.sm_count, bytes_per_num, stage_num,arch):
def calculate_conv_nchw_resource_utilization(op_shape,tb_shape,dim_threads,r_step, bytes_per_num, arch,mem_levels):

    # N,F,H,W,C,conv_kh,conv_kw,S,D,P,tb_n,tb_f,tb_h,tb_w,c_rstep,100, ddr_bw_req*100/DDR_BW,l2_bw_req*100/L2_BW,hit_rate*100
    conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw,conv_s,conv_d,conv_p=op_shape
    tb_n,tb_f,tb_h,tb_w=tb_shape
    # 'block': [1, 32, 7, 28], 'thread': [1, 32, 1, 4], 'rstep': [8, 3, 3]
    n_thread,f_thread,h_thread,w_thread=dim_threads
    threads_per_tb=n_thread*f_thread*h_thread*w_thread
    c_inner_step,kh_step,kw_step=r_step

    c_iter=math.ceil(conv_c/c_inner_step)

    thread_n=math.ceil(tb_n/n_thread)
    thread_f=math.ceil(tb_f/f_thread)
    thread_h=math.ceil(tb_h/h_thread)
    thread_w=math.ceil(tb_w/w_thread)

    in1_level = mem_levels['in1']
    in2_level = mem_levels['in2']
    out1_level = mem_levels['out1']

    # Calculating the input height and width
    inh = (conv_h - 1) * conv_s + (conv_kh - 1) * conv_d + 1 - 2 * conv_p
    inw = (conv_w - 1) * conv_s + (conv_kw - 1) * conv_d + 1 - 2 * conv_p
    # Calculating the padded height and width
    padh = inh + 2 * conv_p
    padw = inw + 2 * conv_p
    # Calculating the input height and width
    tb_inh = (tb_h - 1) * conv_s + (conv_kh - 1) * conv_d + 1 - 2 * conv_p
    tb_inw = (tb_w - 1) * conv_s + (conv_kw - 1) * conv_d + 1 - 2 * conv_p
    # Calculating the padded height and width
    tb_padh = tb_inh + 2 * conv_p
    tb_padw = tb_inw + 2 * conv_p

    # m,n,k,tb_m,tb_n,tb_k,wp_m,wp_n,wp_k,arch.l2_capacity,arch.sm_count, bytes_per_num, stage_num,arch

    DDR_non_ideal_para=1.1 #Unmerged fragmented accesses will be scaled to sector (32bytes) as the smallest unit
    REG_spill_para=1.1 # just like the last one, something we don't know clearly about ncu

    compute_flops=2*conv_n*conv_f*conv_h*conv_w*conv_c*conv_kh*conv_kw
    # nchw_conv_l2_hitrate(N, F, H, W, C, conv_kh, conv_kw, S, D, P, tb_n, tb_f, tb_h, tb_w, c_rstep, L2_Cap, SM_Count, BYTE_per_num)
    l2_hit_rate=nchw_conv_l2_hitrate(conv_n,conv_f,conv_h,conv_w,conv_c,conv_kh,conv_kw,conv_s,conv_d,conv_p,tb_n,tb_f,tb_h,tb_w,c_inner_step,arch.l2_capacity, arch.sm_count,bytes_per_num)
    smem_footprint=bytes_per_num*(tb_n*c_inner_step*tb_padh*tb_padw+tb_f*c_inner_step*kh_step*kw_step)
    
    
    gridN=math.ceil(conv_n/tb_n)
    gridF=math.ceil(conv_f/tb_f)
    gridH=math.ceil(conv_h/tb_h)
    gridW=math.ceil(conv_w/tb_w)
    # print("gridN*gridF*gridH*gridW",gridN*gridF*gridH*gridW)
    # print(inh,inw)
    # l2_read_io=gridN*gridF*gridH*gridW*(tb_n*conv_c*tb_padh*tb_padw**in1_level[0]+tb_f*conv_c*conv_kh*conv_kw*in2_level[0])*bytes_per_num
    l2_read_io=gridN*gridF*gridH*gridW*(tb_n*conv_c*tb_padh*tb_padw*(inh*inw/(padh*padw))*in1_level[0]+tb_f*conv_c*conv_kh*conv_kw*in2_level[0])*bytes_per_num
    l2_store_io=gridN*gridF*gridH*gridW*(tb_n*tb_f*tb_h*tb_w)*bytes_per_num*out1_level[0]

    # print(l2_read_io,l2_store_io)

    ddr_io=l2_read_io*(1-l2_hit_rate)+l2_store_io
    ddr_io=ddr_io*DDR_non_ideal_para

    # reg_footprint=math.ceil((thread_n*thread_f*thread_h*thread_w+4)*REG_spill_para) #no local keep
    # reg_footprint=math.ceil((thread_n*thread_f*thread_h*thread_w+4+thread_n*thread_h*thread_w+thread_f)*REG_spill_para) # only the innersest
    reg_footprint=math.ceil((thread_n*thread_f*thread_h*thread_w+4+thread_n*thread_h*(thread_w+conv_kw-1)+thread_f*conv_kw)*REG_spill_para*bytes_per_num/4) #keep the kw step
    # reg_footprint=math.ceil((thread_n*thread_f*thread_h*thread_w+4+thread_n*(thread_h+conv_kh-1)*(thread_w+conv_kw-1)+thread_f*conv_kh*conv_kw)*REG_spill_para)# keep the kh*kw step

    # active_warp_per_tb=(tb_m/wp_m)*(tb_n/wp_n)

    if (arch.core=="A100" or arch.core=="H100" or arch.core=="B200"):
        #a100 with special two-part l2 cache structure
        l2_io=l2_read_io*(l2_hit_rate)+l2_read_io*(1-l2_hit_rate)*2+l2_store_io*2
        # l2_io=l2_read_io+l2_store_io
    else:
        l2_io=l2_read_io+l2_store_io

    #load from global, L2->L1->reg,L1 bypassing
    l1_io_avg=gridN*gridF*gridH*gridW*(tb_n*conv_c*tb_padh*tb_padw*(inh*inw/(padh*padw))*in1_level[0]+tb_f*conv_c*conv_kh*conv_kw*in2_level[0])*bytes_per_num
    #store shared, reg->smem
    smem_io_avg=gridN*gridF*gridH*gridW*(tb_n*conv_c*tb_padh*tb_padw*(inh*inw/(padh*padw))*in1_level[0]+tb_f*conv_c*conv_kh*conv_kw*in2_level[0])*bytes_per_num
    #load shared to all reduce threads
    # smem_io_avg+=gridN*gridF*gridH*gridW*tb_n*tb_f*tb_h*tb_w*conv_c*conv_kh*conv_kw*(in1_level[1]+in2_level[1])*bytes_per_num # no local keep
    # smem_io_avg+=gridN*gridF*gridH*gridW*threads_per_tb*c_iter*c_inner_step*kh_step*kw_step*(thread_n*thread_h*thread_w*in1_level[1]+thread_f*in2_level[1])*bytes_per_num # only keep the innerest
    smem_io_avg+=gridN*gridF*gridH*gridW*threads_per_tb*c_iter*c_inner_step*kh_step*(thread_n*thread_h*(thread_w+kw_step-1)*in1_level[1]+thread_f*kw_step*in2_level[1])*bytes_per_num # keep the kh*kw step
    # smem_io_avg+=gridN*gridF*gridH*gridW*threads_per_tb*c_iter*c_inner_step*(thread_n*(thread_h+kh_step-1)*(thread_w+kw_step-1)*in1_level[1]+thread_f*kh_step*kw_step*in2_level[1])*bytes_per_num # keep the kh*kw step

    # smem_io_avg=smem_io_avg+thread_per_tb*conv_c*conv_kh*conv_kw*(thread_n*thread_h*thread_w+thread_f)*bytes_per_num
    #store global, reg->L2 ? or reg->L1, L1->L2
    l1_io_avg+=gridN*gridF*gridH*gridW*(tb_n*tb_f*tb_h*tb_w)*bytes_per_num*out1_level[1]
    smem_l1_io=smem_io_avg+l1_io_avg
    
    ddr_read_io=l2_read_io*(1-l2_hit_rate)

    return ddr_io, l2_hit_rate, l2_io, smem_footprint, smem_l1_io, reg_footprint, compute_flops, ddr_read_io, l2_read_io
        
