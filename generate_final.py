import os
import math
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DynamicResBlock(nn.Module):
    def __init__(self, state_dict, base_str):
        super().__init__()
        w_keys = [k for k in state_dict.keys() if k.startswith(base_str)]
        
        def find_conv_kv(sub):
            w = [k for k in w_keys if f".{sub}." in k and k.endswith(".weight") and len(state_dict[k].shape) == 4]
            if not w:
                return None, None
            w_tensor = state_dict[w[0]]
            target_out_channels = w_tensor.shape[0]
            b = [k for k in w_keys if f".{sub}." in k and k.endswith(".bias") and state_dict[k].shape[0] == target_out_channels]
            return nn.Parameter(w_tensor), (nn.Parameter(state_dict[b[0]]) if b else None)

        self.in_w, self.in_b = find_conv_kv("in_layers")
        
        def find_linear_kv(sub):
            w = [k for k in w_keys if f".{sub}." in k and k.endswith(".weight") and len(state_dict[k].shape) == 2]
            b = [k for k in w_keys if f".{sub}." in k and k.endswith(".bias")]
            return nn.Parameter(state_dict[w[0]]) if w else None, nn.Parameter(state_dict[b[0]]) if b else None

        self.emb_w, self.emb_b = find_linear_kv("emb_layers")
        self.out_w, self.out_b = find_conv_kv("out_layers")
        
        skip_w = [k for k in w_keys if ".skip_connection." in k and k.endswith(".weight") and len(state_dict[k].shape) == 4]
        self.skip_w = nn.Parameter(state_dict[skip_w[0]]) if skip_w else None
        if skip_w:
            target_skip_out = self.skip_w.shape[0]
            skip_b = [k for k in w_keys if ".skip_connection." in k and k.endswith(".bias") and state_dict[k].shape[0] == target_skip_out]
            self.skip_b = nn.Parameter(state_dict[skip_b[0]]) if skip_b else None
        else:
            self.register_parameter("skip_b", None)

    def forward(self, x, emb):
        h = nn.functional.conv2d(x, self.in_w, self.in_b, padding=1)
        res_emb = nn.functional.linear(emb, self.emb_w, self.emb_b)
        if res_emb.shape[1] == h.shape[1] * 2:
            scale, shift = res_emb.chunk(2, dim=1)
            h = h * (1 + scale[..., None, None]) + shift[..., None, None]
        else:
            h = h + res_emb[..., None, None]
        h = SiLU()(h)
        h = nn.functional.conv2d(h, self.out_w, self.out_b, padding=1)
        
        if self.skip_w is not None:
            return nn.functional.conv2d(x, self.skip_w, self.skip_b) + h
            
        # Channel Alignment Patch for UNet upsampling concatenation
        if x.shape[1] != h.shape[1]:
            return x[:, :h.shape[1]] + h
            
        return x + h

class RobustConditionalUNet(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.t_1_w = nn.Parameter(state_dict["time_embed.0.weight"])
        self.t_1_b = nn.Parameter(state_dict["time_embed.0.bias"])
        self.t_2_w = nn.Parameter(state_dict["time_embed.2.weight"])
        self.t_2_b = nn.Parameter(state_dict["time_embed.2.bias"])
        self.lbl_w = nn.Parameter(state_dict["label_emb.weight"])
        
        self.in_0_0_w = nn.Parameter(state_dict["input_blocks.0.0.weight"])
        self.in_0_0_b = nn.Parameter(state_dict["input_blocks.0.0.bias"])
        
        self.in_1 = DynamicResBlock(state_dict, "input_blocks.1.0")
        self.in_2 = DynamicResBlock(state_dict, "input_blocks.2.0")
        self.in_3 = DynamicResBlock(state_dict, "input_blocks.3.0")
        self.in_4 = DynamicResBlock(state_dict, "input_blocks.4.0")
        
        self.mid_1 = DynamicResBlock(state_dict, "middle_block.0")
        self.mid_attn_qkv_w = nn.Parameter(state_dict["middle_block.1.qkv.weight"])
        self.mid_attn_qkv_b = nn.Parameter(state_dict["middle_block.1.qkv.bias"])
        self.mid_attn_out_w = nn.Parameter(state_dict["middle_block.1.proj_out.weight"])
        self.mid_attn_out_b = nn.Parameter(state_dict["middle_block.1.proj_out.bias"])
        self.mid_2 = DynamicResBlock(state_dict, "middle_block.2")
        
        self.out_0 = DynamicResBlock(state_dict, "output_blocks.0.0")
        self.out_1 = DynamicResBlock(state_dict, "output_blocks.1.0")
        self.out_2 = DynamicResBlock(state_dict, "output_blocks.2.0")
        self.out_3 = DynamicResBlock(state_dict, "output_blocks.3.0")
        
        self.final_norm_w = nn.Parameter(state_dict["out.0.weight"])
        self.final_norm_b = nn.Parameter(state_dict["out.0.bias"])
        self.final_conv_w = nn.Parameter(state_dict["out.2.weight"])
        self.final_conv_b = nn.Parameter(state_dict["out.2.bias"])

    def forward(self, x, timesteps, y):
        half_dim = self.t_1_w.shape[1] // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim).to(x.device)
        args = timesteps[:, None] * freqs[None, :]
        time_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        emb = nn.functional.linear(time_emb, self.t_1_w, self.t_1_b)
        emb = SiLU()(emb)
        emb = nn.functional.linear(emb, self.t_2_w, self.t_2_b)
        emb = emb + nn.functional.embedding(y, self.lbl_w)

        h0 = nn.functional.conv2d(x, self.in_0_0_w, self.in_0_0_b, padding=1)
        h1 = self.in_1(h0, emb)
        h2 = self.in_2(h1, emb)
        h3 = self.in_3(h2, emb)
        h4 = self.in_4(h3, emb)

        hm = self.mid_1(h4, emb)
        b, c, height, w = hm.shape
        hm_flat = hm.view(b, c, -1).permute(0, 2, 1)
        
        qkv = nn.functional.linear(hm_flat, self.mid_attn_qkv_w.squeeze(-1).squeeze(-1), self.mid_attn_qkv_b)
        q, k, v = qkv.chunk(3, dim=-1)
        
        scale = 1.0 / math.sqrt(c // 4)
        attn_scores = torch.softmax(torch.matmul(q, k.permute(0, 2, 1)) * scale, dim=-1)
        attn_context = torch.matmul(attn_scores, v)
        
        attn_out = nn.functional.linear(attn_context, self.mid_attn_out_w.squeeze(-1).squeeze(-1), self.mid_attn_out_b)
        hm = hm + attn_out.permute(0, 2, 1).view(b, c, height, w)
        hm = self.mid_2(hm, emb)

        ho0 = self.out_0(torch.cat([hm, h4], dim=1), emb)
        ho1 = self.out_1(torch.cat([ho0, h3], dim=1), emb)
        ho2 = self.out_2(torch.cat([ho1, h2], dim=1), emb)
        ho3 = self.out_3(torch.cat([ho2, h1], dim=1), emb)

        out = ho3 * self.final_norm_w[None, :, None, None] + self.final_norm_b[None, :, None, None]
        out = SiLU()(out)
        return nn.functional.conv2d(out, self.final_conv_w, self.final_conv_b, padding=1)

@torch.no_grad()
def generate_images(model, num_steps=50, batch_size=16, device="cuda", class_label=1, cfg_scale=4.0):
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    dt = 1.0 / num_steps
    
    print(f"Executing mathematical trajectory loop for Class {class_label} on GPU...")
    for step in range(num_steps):
        t_scalar = step * dt
        t = torch.ones(batch_size, device=device) * t_scalar
        
        v_uncond = model(x, t, torch.ones(batch_size, dtype=torch.long, device=device) * 10)
        v_cond = model(x, t, torch.ones(batch_size, dtype=torch.long, device=device) * class_label)
        
        velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
        x = x + velocity * dt
        
    print("Sampling complete!")
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)

if __name__ == "__main__":
    checkpoint_path = "./conditional_checkpoint.pth"

    if not os.path.exists(checkpoint_path):
        print("Error: Could not locate conditional_checkpoint.pth.")
        exit()

    print("Loading weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    corrected_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    model = RobustConditionalUNet(corrected_state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    images = generate_images(model=model, num_steps=50, batch_size=16, device=device, class_label=1, cfg_scale=4.0)

    grid = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.cpu().permute(1, 2, 0).numpy())
    plt.axis("off")
    
    output_image_path = "./generated_grid.png"
    plt.savefig(output_image_path, bbox_inches="tight")
    print(f"Success! Image grid saved to cluster directory: {output_image_path}")
