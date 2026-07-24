# ------------    element-wise
# def calculate_N_0_elementwise_resource_utilization
# (op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1):

# def calculate_N_1_elementwise_resource_utilization
# (in1_shape,in2_shape,in1_tb_shape,dim_threads, arch,mem_levels,num_ops=1):

# def calculate_N_N_elementwise_resource_utilization
# (op_shape,tb_shape,dim_threads, arch,mem_levels,num_ops=1):

# ------------    matmul
# def calculate_matmul_resource_utilization
# (op_shape,tb_shape,wp_shape, stage_num,arch,mem_levels,row_panel=108,batch=1):

# ------------    reduce inter-thread
# def calculate_N_0_general_ruduce_inter_thread_resource_utilization
# (out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=1):

# def calculate_general_ruduce_inter_thread_resource_utilization
# (out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads,reduce_threads_info, arch,mem_levels,compute_at=1)

# ------------    reduce no inter-thread
# def calculate_N_0_general_ruduce_resource_utilization
# (out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch,mem_levels,compute_at=1):

# def calculate_general_ruduce_resource_utilization
# (out_shape,reduction_shape,out_axis_mapping,reduction_axis_mapping,out_tb_shape,dim_threads, arch,mem_levels,compute_at=1):


# 8 function in total. The template actually divides the op into four broad categories
# 1\element wise
# 2\matmul
# 3\inter thread reduce
# 4\no inter thread reduce



class Operation:
    def __init__(self, op_name):
        self.op_name = op_name
        self.input1_shape = [] #  [1,4,16,16]
        self.input2_shape = [] #  [1,1,8,8]
        self.output_shape = [] # [1,4,16,16]

        self.input1_bytes= None
        self.input2_bytes= None
        self.output_bytes= None

        self.input1_level=""
        self.input2_level=""
        self.output_level=""

        # tiling now
        self.block = []
        self.thread = []
        self.rstep = []
        self.step = []
        self.reduce_thread = []
        # {'node_name': 'welder_C2DImplicitGemm_5', 'block': [32, 256], 'rstep': [16], 'warp': [16, 128], 'wmma': [8, 32, 16], 'use_cutlass': False, 'use_tc': 80}
        self.use_tc= None
        self.use_cutlass = None
        self.warp=[]
        self.wmma=[]
        # fuse levels & bits
        # self.mem_levels = {'in1': [], 'in2': [], 'out1': []} # 初始化为空字典,这个最后要送到modeling里去
        self.mem_levels={}
        # unique attributes
        self.num_ops = None
        self.compute_at = None
        self.row_panel= None
        self.op_special_metric_dict={}
    
    def __str__(self):
        return (f"Operation Name: {self.op_name}\n"
                f"Input 1 Shape: {self.input1_shape}, Input 2 Shape: {self.input2_shape}, Output Shape: {self.output_shape}\n"
                f"Input 1 Bytes: {self.input1_bytes}, Input 2 Bytes: {self.input2_bytes}, Output Bytes: {self.output_bytes}\n"
                f"Input 1 Level: {self.input1_level}, Input 2 Level: {self.input2_level}, Output Level: {self.output_level}\n"
                f"Op Special Metrics: {self.op_special_metric_dict}\n"
                )
                # f"block: {self.block}, thread: {self.thread}, rstep: {self.rstep}, step: {self.step}, reduce thread: {self.reduce_thread}\n"
                # f"use tc: {self.use_tc}, use cutlass: {self.use_cutlass}, warp: {self.warp}, wmma: {self.wmma}\n"
                # f"memory levels: {self.mem_levels}, number of operations: {self.num_ops}, compute at: {self.compute_at}, row panel: {self.row_panel}")

    def __eq__(self, other):
        if isinstance(other, Operation):
            return (self.op_name == other.op_name and
                    self.input1_shape == other.input1_shape and
                    self.input2_shape == other.input2_shape and
                    self.output_shape == other.output_shape and
                    self.input1_bytes == other.input1_bytes and
                    self.input2_bytes == other.input2_bytes and
                    self.output_bytes == other.output_bytes and
                    self.op_special_metric_dict == other.op_special_metric_dict)
        return False

    def __hash__(self):
        # Convert the op_special_metric_dict to a hashable type
        hashable_dict = {k: frozenset(v) if isinstance(v, list) else v for k, v in self.op_special_metric_dict.items()}
        return hash((self.op_name, tuple(self.input1_shape), tuple(self.input2_shape), tuple(self.output_shape), 
                     self.input1_bytes, self.input2_bytes, self.output_bytes, frozenset(hashable_dict.items())))

    def to_dict(self):
        return {
            "Operation Name": self.op_name,
            "Input 1 Shape": self.input1_shape,
            "Input 2 Shape": self.input2_shape,
            "Output Shape": self.output_shape,
            "Input 1 Bytes": self.input1_bytes,
            "Input 2 Bytes": self.input2_bytes,
            "Output Bytes": self.output_bytes,
            "Op Special Metrics": self.op_special_metric_dict
        }

    @classmethod
    def from_dict(cls, dict_data):
        op = cls(dict_data["Operation Name"])
        op.input1_shape = dict_data["Input 1 Shape"]
        op.input2_shape = dict_data["Input 2 Shape"]
        op.output_shape = dict_data["Output Shape"]
        op.input1_bytes = dict_data["Input 1 Bytes"]
        op.input2_bytes = dict_data["Input 2 Bytes"]
        op.output_bytes = dict_data["Output Bytes"]
        op.op_special_metric_dict = dict_data["Op Special Metrics"]
        return op

    def set_input_shapes(self, input_shapes):
        self.input_shapes = input_shapes
    
    def exchange_input1_input2(self):
        self.input1_shape, self.input2_shape = self.input2_shape, self.input1_shape
        self.input1_bytes, self.input2_bytes = self.input2_bytes, self.input1_bytes
        self.input1_level, self.input2_level = self.input2_level, self.input1_level

    def set_output_shapes(self, output_shapes):
        self.output_shapes = output_shapes

    def set_tiling(self, tiling):
        self.tiling = tiling

    def set_mem_levels(self, mem_levels):
        self.mem_levels = mem_levels

    def get_mem_levels(self):
        if self.mem_levels != {}:
            return self.mem_levels
        else:
            if self.input1_shape != []:
                # in1_level=[]
                if self.input1_level == "ddr":
                    in1_level=[1,1,1]
                elif self.input1_level == "smem":
                    in1_level=[0,1,1]
                elif self.input1_level == "reg":
                    in1_level=[0,0,1]
                
                in1_level.append(self.input1_bytes)
                self.mem_levels.update({'in1': in1_level})
            
            if self.input2_shape != []:
                # in1_level=[]
                if self.input2_level == "ddr":
                    in2_level=[1,1,1]
                elif self.input2_level == "smem":
                    in2_level=[0,1,1]
                elif self.input2_level == "reg":
                    in2_level=[0,0,1]
                
                in2_level.append(self.input2_bytes)
                self.mem_levels.update({'in2': in2_level})

            if self.output_shape != []:
                # in1_level=[]
                if self.output_level == "ddr":
                    out1_level=[1,1,1]
                elif self.output_level == "smem":
                    out1_level=[0,1,1]
                elif self.output_level == "reg":
                    out1_level=[0,0,1]
                
                out1_level.append(self.output_bytes)
                self.mem_levels.update({'out1': out1_level})
            # print(self.mem_levels)
            
            return self.mem_levels
