import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from clip.model import VisionTransformer as CLIP_ViT
from model.utils import low_rank_approx

from .peft_modules import *


class ViT_Tuner(nn.Module):
    """ All instance variables in this class will be optimized.
    """
    def __init__(self, cfg, vit_model: CLIP_ViT):
        super().__init__()

        self.cfg = cfg

        n_layers = len(vit_model.transformer.resblocks)
        emb_dim = vit_model.positional_embedding.shape[1]
        seq_len = vit_model.positional_embedding.shape[0]
        patch_size = vit_model.conv1.kernel_size
        dtype = vit_model.conv1.weight.dtype

        blocks = vit_model.transformer.resblocks
        #Định nghĩa các hàm lambda để lấy trọng số và bias của các thành phần attention và MLP trong mỗi block
        get_attn_in_weight = lambda i: blocks[i].attn.in_proj_weight
        get_attn_in_bias = lambda i: blocks[i].attn.in_proj_bias
        get_attn_out_weight = lambda i: blocks[i].attn.out_proj.weight
        get_attn_out_bias = lambda i: blocks[i].attn.out_proj.bias
        get_mlp_in_weight = lambda i: blocks[i].mlp[0].weight
        get_mlp_in_bias = lambda i: blocks[i].mlp[0].bias
        get_mlp_out_weight = lambda i: blocks[i].mlp[2].weight
        get_mlp_out_bias = lambda i: blocks[i].mlp[2].bias

        attn_in_dim = get_attn_in_bias(0).shape[0]
        attn_out_dim = get_attn_out_bias(0).shape[0]
        mlp_in_dim = get_mlp_in_bias(0).shape[0]
        mlp_out_dim = get_mlp_out_bias(0).shape[0]

        # Define the tuning strategy based on the configuration
        use_full_tuning = cfg.v_full_tuning
        partial = cfg.v_partial

        use_keeplora = cfg.v_keeplora #use_keeplora is a list of strings indicating which components to apply KeepLoRA to, e.g., ['q', 'k', 'v', 'o']
        lora_rank = cfg.v_adapter_dim
        #Partial: chọn phạm vi tầng sẽ áp dụng PEFT
        if partial is None:
            _start, _end = 0, n_layers
        elif isinstance(partial, int):
            _start, _end = n_layers - partial, n_layers
        elif isinstance(partial, list):
            _start, _end = partial[0], partial[1]

        #Nếu lựa chọn full tuning (Thay vì sử dụng PEFT) được bật, thì block_tuned sẽ chứa các block từ _start đến _end, ngược lại block_tuned sẽ là None
        if use_full_tuning:
            block_tuned = blocks[_start: _end]
        else:
            block_tuned = None

        # Initialize KeepLoRA modules
        if use_keeplora:
            valid_keeplora_keys = {'q', 'k', 'v', 'o'}
            if not set(use_keeplora).issubset(valid_keeplora_keys):
                raise ValueError(f"use_keeplora can only contain a subset of {valid_keeplora_keys}, got {use_keeplora}")
            keeplora_list = nn.ModuleList([
                *[None] * (_start),
                *[nn.ModuleDict({
                    k: KeepLoRA(in_dim=emb_dim, out_dim=emb_dim, r=lora_rank, lora_alpha=1, use_rslora=False, dtype=dtype) for k in use_keeplora
                }) for _ in range(_start, _end)],
                *[None] * (n_layers - _end)
            ])
            #Đoạn code đang xây dựng một danh sách có độ dài n_layers:

            ###Các layer trước _start: None.
            # Các layer từ _start đến _end - 1: mỗi layer có một ModuleDict chứa các module KeepLoRA được chỉ định bởi use_keeplora (Mỗi một layer sẽ có các module KeepLoRA tương ứng với từng q, k, v, o).
            # Các layer từ _end trở đi: None.
            # Nói ngắn gọn: đây là cách chọn một khoảng layer để gắn các module KeepLoRA, đồng thời giữ nguyên vị trí tương ứng với các layer còn lại trong mô hình.
            ##Cấu trúc keeplora_list: là một nn.ModuleList đúng bằng số tầng n_layers (12), ghép từ ba đoạn:
            # [None] * _start: các tầng trước phạm vi được chọn → không có LoRA (giá trị None).[nn.ModuleDict({...}) for _ in range(_start, _end)]: các tầng trong phạm vi → mỗi tầng có một nn.ModuleDict, trong đó mỗi key ('q', 'k', 'v', hoặc 'o' — tùy use_keeplora chứa gì) trỏ tới một instance KeepLoRA riêng biệt (đã phân tích kỹ ở các lượt trước — nhớ lại: lora_A cố định, chỉ lora_B học).
            # [None] * (n_layers - _end): các tầng sau phạm vi → cũng không có LoRA.
        else:
            keeplora_list = nn.ModuleList([None] * n_layers)

        #Khởi tạo list các module KeepLoRA cho từng block trong mô hình ViT, dựa trên cấu hình được cung cấp. Nếu use_keeplora là True, thì sẽ tạo ra một danh sách các module KeepLoRA cho các thành phần attention (q, k, v, o) trong mỗi block từ _start đến _end. Nếu không, danh sách này sẽ chứa None cho tất cả các block.

        # To be optimized
        self.block_tuned = block_tuned
        self.keeplora_list = keeplora_list
        #Gán làm thuộc tính của module — vì keeplora_list là nn.ModuleList, gán nó làm self.keeplora_list khiến toàn bộ tham số bên trong (mọi lora_B của mọi KeepLoRA instance) tự động xuất hiện trong self.parameters() của ViT_Tuner — đây chính là câu comment ở đầu class: "All instance variables in this class will be optimized."
        param_names = self.cfg.v_svd_param_names
        alphas = self.cfg.v_svd_alphas
        if param_names is not None and alphas is not None:
            self.apply_low_rank_approx_to_params(vit_model, param_names, alphas)

    def apply_low_rank_approx_to_params(self, vit_model, param_names, alphas):
        """
        Apply SVD decomposition and low-rank approximation to specified parameters.
        """
        #Đây KHÔNG phải một phần của cơ chế KeepLoRA khi huấn luyện — nó sửa trực tiếp trọng số gốc của vit_model (chính CLIP backbone, .data = W_r), biến backbone thành một phiên bản đã bị nén hạng thấp cố định, dùng để đo xem zero-shot accuracy suy giảm ra sao khi loại bỏ dần các thành phần "năng lượng thấp" của trọng số pretrained — đúng như bạn đã đọc ở Fig. 1 của paper ("we measure zero-shot performance after reconstructing attention weights using only the top principal singular components"). Nó không liên quan gì tới quá trình huấn luyện LoRA hay chiếu gradient trực giao đã phân tích ở các lượt trước.
        #check that param_names and alphas have the same length
        assert len(param_names) == len(alphas), "param_names and alphas must have the same length"

        device = [int(s) for s in self.cfg.gpu_id.split(',')][0]
        emb_dim = vit_model.positional_embedding.shape[1]
        for param_name, alpha in zip(param_names, alphas):
            for i in range(len(vit_model.transformer.resblocks)):
                # print(f"Layer {i}: ", end="")
                block = vit_model.transformer.resblocks[i]
                if param_name == 'attn.in_proj_weight':
                    W = block.attn.in_proj_weight
                    W_q = W[:emb_dim, :]
                    W_k = W[emb_dim:2*emb_dim, :]
                    W_v = W[2*emb_dim:, :]

                    # print(f"Q: ", end="")
                    W_q_r = low_rank_approx(W_q, alpha, device)
                    # print(f"/{min(W_q.shape)}, K: ", end="")
                    W_k_r = low_rank_approx(W_k, alpha, device)
                    # print(f"/{min(W_k.shape)}, V: ", end="")
                    W_v_r = low_rank_approx(W_v, alpha, device)
                    # print(f"/{min(W_v.shape)}", end="")

                    W_r = torch.cat([W_q_r, W_k_r, W_v_r], dim=0)
                    block.attn.in_proj_weight.data = W_r

                elif param_name == 'attn.out_proj.weight':
                    # print(f"O: ", end="")
                    W = block.attn.out_proj.weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.attn.out_proj.weight.data = W_r
                    # print(f"/{min(W.shape)}", end="")

                elif param_name == 'mlp.c_fc.weight':
                    # print(f"MLP_1: ", end="")
                    W = block.mlp[0].weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.mlp[0].weight.data = W_r
                    # print(f"/{min(W.shape)}", end="")
                
                elif param_name == 'mlp.c_proj.weight':
                    # print(f"MLP_2: ", end="")
                    W = block.mlp[2].weight
                    W_r = low_rank_approx(W, alpha, device)
                    block.mlp[2].weight.data = W_r
                    # print(f"/{min(W.shape)}", end="")
                else:
                    raise ValueError(f"Unsupported parameter name: {param_name}")
                # print("")


class Peft_ViT(nn.Module):
    def __init__(self, vit_model: CLIP_ViT):
        super().__init__()

        """
        Class này không tự tạo bất kỳ tham số nào mới — nó chỉ gán tham chiếu trực tiếp tới các thành phần đã tồn tại sẵn trong vit_model (chính là instance VisionTransformer của CLIP gốc,
        đã phân tích kỹ ở lượt trước — conv1, class_embedding, positional_embedding, ln_pre, transformer.resblocks, ln_post, proj). Vì đây là gán tham chiếu (không phải copy), 
        mọi thay đổi lên các tham số này (ví dụ qua merge_keeplora_weights/subtract_keeplora_weights đã xem ở các lượt trước, thao tác trực tiếp trên attn.in_proj_weight.data) sẽ tác động trực tiếp lên đúng những tensor mà Peft_ViT đang dùng — không cần đồng bộ hóa gì thêm."""
        self.patch_embedding = vit_model.conv1
        self.class_embedding = vit_model.class_embedding
        self.positional_embedding = vit_model.positional_embedding
        self.ln_pre = vit_model.ln_pre
        self.blocks = vit_model.transformer.resblocks
        self.ln_post = vit_model.ln_post
        self.proj = vit_model.proj
        self.out_dim = self.proj.shape[1] #Lấy chiều đầu ra của Peft_ViT từ chiều thứ 1 của ma trận proj (shape = [width, out_dim]) — đây là chiều embedding cuối cùng của CLIP ViT, dùng để tính cosine similarity với embedding text.

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    def forward(self, x: torch.Tensor, tuner: ViT_Tuner=None, accumulate_mode: bool = False):
        
        x = x.to(self.dtype)
        #Biến đổi đầu vào x từ [batch_size, channels, height, width] sang embedding sequence của ViT: (từ giạng ảnh sang dạng chuỗi các vector embedding: số patch x số chiều vector embedding ).
        # -> Phù hợp với cách hoạt động của Vision Transformer (ViT), nơi mỗi patch ảnh được ánh xạ thành một vector embedding, và các vector này được xử lý tuần tự qua các block transformer.
        x = self.patch_embedding(x)  # shape = [*, width, grid, grid] [* là batch_size, width là embedding dimension, grid là số patch theo chiều cao và chiều rộng]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] [* là batch_size, width là embedding dimension, grid ** 2 là số patch]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] [* là batch_size, grid ** 2 là số patch, width là embedding dimension]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x) #layer normalization trước khi đưa vào các block transformer. Nó giúp chuẩn hóa các vector embedding đầu vào, cải thiện sự hội tụ và hiệu suất của mô hình.

        _bsz = x.shape[0] #Batch size
        _seq_len = x.shape[1] #Sequence length (số lượng patch + 1 cho class token)
        _emb_dim = x.shape[2] #Embedding dimension (chiều của vector embedding cho mỗi patch/class token)

        n_layers = len(self.blocks) # số lượng block transformer trong mô hình ViT (ví dụ: 12 cho ViT-B/16)
        #blocks = resblocks của Vit , mỗi block chứa một multi-head self-attention layer và một feed-forward network (MLP), cùng với các layer normalization và residual connections.
        for i in range(n_layers):
            block = self.blocks[i]

            if tuner is not None:
                keeplora = tuner.keeplora_list[i]
            else:
                keeplora = None

            _seq_len_after_vpt = x.shape[1] # lấy chiều dài sequence hiện tại của x trước khi đi qua block transformer, để sử dụng sau này khi reshape lại x sau attention và MLP.

            x = x.permute(1, 0, 2)  # NLD -> LND Vì nn.MultiheadAttention mặc định của PyTorch sử dụng: (sequence_length, batch_size, embedding_dim)

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2

            #.weights , .bias -> trỏ trực tiếp tới các tensor trọng số (chỉ lấy trọng số ra để truyền vào linear hoặc sequence) và bias của các thành phần attention và MLP trong block hiện tại, để sử dụng trong forward pass.
            _attn_in_proj_weight = _attn.in_proj_weight 
            _attn_in_proj_bias = _attn.in_proj_bias
            _attn_out_proj_weight = _attn.out_proj.weight
            _attn_out_proj_bias = _attn.out_proj.bias
            _mlp_in_proj_weight = _mlp[0].weight
            _mlp_in_proj_bias = _mlp[0].bias
            _mlp_act = _mlp[1]
            _mlp_out_proj_weight = _mlp[2].weight
            _mlp_out_proj_bias = _mlp[2].bias

            #-> các bước này chỉ là lấy trọng số ra khỏi các thành phần attention và MLP của block hiện tại, để sử dụng trong quá trình tính toán forward pass. Việc này giúp tách biệt các tham số của mô hình, thuận tiện cho việc áp dụng các kỹ thuật như KeepLoRA hoặc low-rank approximation mà không làm thay đổi trực tiếp cấu trúc của block transformer.
            #chứ không trỏ đến các phép toán hay forward gì cả
            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads
            
            ###############################
            ## Multi-Head Self-Attention ##
            ###############################
            identity = x

            x = _ln_1(x)

            #truyền trọng số và bias của attention vào hàm F.linear để tính toán Q, K, V từ input x. Sau đó, nếu keeplora không phải None, sẽ kiểm tra xem có đang ở chế độ accumulate hay không. Nếu có, sẽ gọi phương thức accumulate_features của các module KeepLoRA tương ứng (q, k, v) để tích lũy thông tin đặc trưng từ x. Nếu không ở chế độ accumulate, sẽ cộng thêm output của các module KeepLoRA vào Q, K, V tương ứng.
            qkv = F.linear(x, _attn_in_proj_weight, _attn_in_proj_bias) #Nhân x với một Projection lớn chứa cả Wq,Wv,Wk của multi-head attention, để tạo ra Q, K, V trong một bước duy nhất. Đây là cách tối ưu hóa tính toán trong PyTorch, thay vì tính từng Q, K, V riêng lẻ.
            q, k, v = qkv.chunk(3, dim=-1) #lấy Q, K, V bằng cách chia tensor qkv thành 3 phần dọc theo chiều cuối cùng -chiều thứ 3 (embedding dimension).

            if keeplora is not None:
                if accumulate_mode: #chế độ accumulate: chỉ tích lũy thông tin đặc trưng từ x vào các module KeepLoRA mà không thay đổi Q, K, V hiện tại.
                    keeplora['q'].accumulate_features(x) if 'q' in keeplora else None
                    keeplora['k'].accumulate_features(x) if 'k' in keeplora else None
                    keeplora['v'].accumulate_features(x) if 'v' in keeplora else None
                else:
                    q = q + keeplora["q"](x) if 'q' in keeplora else q
                    k = k + keeplora["k"](x) if 'k' in keeplora else k
                    v = v + keeplora["v"](x) if 'v' in keeplora else v

            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            attn_mask = block.attn_mask.to(dtype=x.dtype, device=x.device) if block.attn_mask is not None else None
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)

            if keeplora is not None:
                x_hat = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)
                if accumulate_mode:
                    keeplora['o'].accumulate_features(x) if 'o' in keeplora else None
                    x = x_hat
                else:
                    x = x_hat + keeplora["o"](x) if 'o' in keeplora else x_hat
            else:
                x = F.linear(x, _attn_out_proj_weight, _attn_out_proj_bias)

            x = x.view(_seq_len_after_vpt, _bsz, _emb_dim)
            x = x + identity #KeepLora + frozen weight: x = x_hat + Lora(x) + identity

            ##########################
            ## Feed-Forward Network ##
            ##########################
            identity = x

            x = _ln_2(x)
            
            x = F.linear(x, _mlp_in_proj_weight, _mlp_in_proj_bias)
            
            x = _mlp_act(x)
            
            x = F.linear(x, _mlp_out_proj_weight, _mlp_out_proj_bias)
            
            x = x + identity
            x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:, 0, :]
        x = self.ln_post(x)
        x = x @ self.proj

        return x

