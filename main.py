import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from torch import nn


#utils
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Pos embedding
def pos_emb_sincos_2d(
    h,
    w,
    dim,
    temperature: int = 10000,
    dtype = torch.float32
):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature ** omega)
    
    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


#classes
class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        hidden_dim, 
        dropout = 0.
    ):
        """
        Feedforward layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. GELU
        4. Dropout
        5. Linear
        6. Dropout

        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Apply dropout to the input
        3. Apply non-linearity to the input
        4. Apply dropout to the input

        Args:
        -----
        dim: int
            Dimension of input
        hidden_dim: int
            Dimension of hidden layer
        dropout: float
            Dropout rate
        
        Returns:
        --------
        torch.Tensor
            Output of feedforward layer

        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        #apply feedforward layer to input tensor
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        """
        Attention Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Linear
        3. Rearrange
        4. LayerNorm
        5. Linear
        6. Rearrange
        7. Softmax
        8. Dropout
        9. Rearrange
        10. Linear
        11. Dropout
        
        Purpose:
        --------
        1. Apply non-linearity to the input
        2. Rearrange input tensor
        3. Apply non-linearity to the input
        4. Rearrange input tensor
        5. Apply softmax to the input
        6. Apply dropout to the input
        7. Rearrange input tensor
        8. Apply non-linearity to the input
        
        """
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        #layer norm
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)

        #sftmx
        self.attend = nn.Softmax(dim = -1)

        #dropout
        self.dropout = nn.Dropout(dropout)

        #projections, split from x -> q, k, v
        self.to_qkv = nn.Linear(
            dim, 
            inner_dim * 3, 
            bias = False
        )
        
        #project out
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        #apply layernorm to x
        x = self.norm(x)

        #apply linear layer to x
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        #rearrange x to original shape
        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)

        # #normalize key and values, known QK Normalization
        k = self.norm_k(k)
        v = self.norm_v(v)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_map = self.attend(attn_scores)  # This is the Attention Map
        
        # attn
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            #Flash Attention
            out = F.scaled_dot_product_attention(q, k, v)
            
            #dropout
            out = self.dropout(out)

        #rearrange to originalf shape
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        #project out
        return out, attn_map
        
        
class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        mlp_dim, 
        dropout = 0.
    ):
        """
        Transformer Layer

        Architecture:
        -------------
        1. LayerNorm
        2. Attention
        3. FeedForward

        Args:
        -----
        dim: int
            Dimension of input
        depth: int
            layers of transformers
        heads: int
            Number of heads
        dim_head: int
            Dimension of head
        mlp_dim: int
            Dimension of MLP
        dropout: float
            Dropout rate

        
        """
        super().__init__()
        
        #layer norm
        self.norm = nn.LayerNorm(dim)

        #transformer layers data array
        self.layers = nn.ModuleList([])
        
        #add transformer layers as depth = transformer blocks
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #attention
                Attention(
                    dim, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = dropout
                ),
                #feedforward
                FeedForward(dim, mlp_dim, dropout)
            ]))

    def forward(self, x):
        attn_maps = []  # List to store attention maps
        for attn, ff in self.layers:
            # Attention 출력과 Attention 맵 분리
            attn_output, attn_map = attn(x)  # `attn`이 tuple을 반환
            x = x + attn_output  # Skip connection with attention output
            x = x + ff(x)  # Skip connection with feedforward output
            attn_maps.append(attn_map)  # Collect attention maps

        x = self.norm(x)  # 마지막에 LayerNorm 적용
        return x, attn_maps  # Return the output and the attention maps
    
class VitRGTS(nn.Module):
    """
    VitRGTS model from https://arxiv.org/abs/2106.14759

    Args:
    -------
    image_size: int
        Size of image
    patch_size: int
        Size of patch
    num_classes: int
        Number of classes
    dim: int
        Dimension of embedding
    depth: int
        Depth of transformer
    heads: int
        Number of heads
    mlp_dim: int
        Dimension of MLP
    pool: str
        Type of pooling
    channels: int
        Number of channels
    dim_head: int
        Dimension of head
    dropout: float
        Dropout rate
    emb_dropout: float
        Dropout rate for embedding
    
    Returns:
    --------
    torch.Tensor
        Predictions
    
    Methods:
    --------
    forward(img: torch.Tensor) -> torch.Tensor:
        Forward pass
    
    Architecture:
    -------------
    1. Input image is passed through a patch embedding layer
    2. Positional embedding is added
    3. Dropout is applied
    4. Transformer is applied
    5. Pooling is applied
    6. MLP head is applied
    7. Output is returned
    """
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim,
        num_register_tokens=4,
        use_register_tokens=True, 
        pool='cls', 
        channels=3, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.
    ):
        super().__init__()
        
        self.use_register_tokens = use_register_tokens

        # Patch embedding setup
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        
        # Register tokens
        if use_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        # Positional embedding
        self.pos_embedding = pos_emb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        # Patch embedding
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device)

        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # CLS 토큰을 패치 임베딩 앞에 추가

        ps = None  # Initialize `ps` as None

        if self.use_register_tokens:
            # Include register tokens
            r = repeat(self.register_tokens, 'n d -> b n d', b=batch)

            # 디버깅 출력문 추가
            # print(f"Shape of x before packing: {x.shape}")  # 디버깅: 패치 임베딩의 크기
            # print(f"Shape of register tokens: {r.shape}")  # 디버깅: 레지스터 토큰의 크기

            x, ps = pack([x, r], 'b * d')  # `x` shape becomes [batch, 200, dim]

            # 디버깅 출력문 추가
            # print(f"Shape of x after packing: {x.shape}")  # 디버깅: 결합 후 크기

        # Transformer and Attention maps
        x, attn_maps = self.transformer(x)  # `x` shape should still be [batch, 200, dim]

        # Debugging output to check `ps` and `x` shapes
        # print(f"x shape before unpack: {x.shape}")
        # if ps is not None:
            # print(f"ps: {ps}")

        # Ensure `x` is a Tensor and not a tuple
        if not isinstance(x, torch.Tensor):
            print("Error: `x` is not a Tensor!")
            return None, None

        # Unpack only if `ps` is not None and shapes match
        if ps is not None:
            expected_length = sum(p.numel() for p in ps)
            if x.shape[1] == expected_length:  # Check if shapes match
                try:
                    x, _ = unpack(x, ps, 'b * d')  # `x` shape becomes [batch, 196, dim]
                except RuntimeError as e:
                    print(f"Error during unpack: {e}")
                    return None, None
            else:
                # print(f"Shape mismatch: x.shape[1] = {x.shape[1]}, expected = {expected_length}")
                return None, None

        # Pooling and output
        x = x.mean(dim=1)
        x = self.to_latent(x)
        out = self.linear_head(x)
        return out, attn_maps
    
    def get_cls_and_patch_embeddings(self, img):
        """
        이미지에서 CLS 토큰 및 패치 임베딩을 추출.

        Args:
            img (torch.Tensor): [Batch, Channels, Height, Width] 크기의 입력 이미지

        Returns:
            cls_embedding (torch.Tensor): [Batch, Dim] 크기의 CLS 토큰 임베딩
            patch_embeddings (torch.Tensor): [Batch, Patches, Dim] 크기의 패치 임베딩
        """
        batch, device = img.shape[0], img.device

        # Patch embedding
        x = self.to_patch_embedding(img)  # 패치 임베딩 계산
        x += self.pos_embedding.to(device)  # 위치 임베딩 추가

        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # CLS 토큰을 패치 임베딩 앞에 추가

        # Transformer 통과
        x, _ = self.transformer(x)  # Transformer 결과 반환 (CLS + 패치 토큰)

        # CLS 토큰과 패치 토큰 분리
        cls_embedding = x[:, 0]  # 첫 번째 토큰이 CLS 토큰
        patch_embeddings = x[:, 1:]  # 나머지가 패치 토큰
        return cls_embedding, patch_embeddings

