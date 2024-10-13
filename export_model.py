from model.modules import *


class EncodeKey(nn.Module):
    def __init__(self, encoder: KeyEncoder, projection: KeyProjection):
        super().__init__()
        self.enc = encoder
        self.proj = projection

    def forward(self, f, need_sk: torch.Tensor, need_ek: torch.Tensor):
        # keyencoder
        f16, f8, f4 = self.enc(f)

        # keyprojection
        key = self.proj.key_proj(f16)

        shrinkage = torch.where(
            need_sk == torch.ones(1),
            self.proj.d_proj(f16)**2 + 1,
            torch.zeros_like(self.proj.d_proj(f16)**2 + 1)
        )
        selection = torch.where(
            need_ek == torch.ones(1),
            torch.sigmoid(self.proj.e_proj(f16)),
            torch.sigmoid(self.proj.e_proj(f16)).zero_()
        )

        return key, shrinkage, selection, f16, f8, f4



class EncodeValue(nn.Module):
    def __init__(self, encoder: ValueEncoder) -> None:
        super().__init__()
        self.enc = encoder
        self.is_hidden_dim = torch.tensor([encoder.hidden_reinforce is not None]).type(torch.bool)

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update:torch.Tensor):
        # image_feat_f16 is the feature from the key encoder

        g = torch.zeros_like(masks)
        if not self.enc.single_object:
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        g = self.enc.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.enc.conv1(g)
        g = self.enc.bn1(g) # 1/2, 64
        g = self.enc.maxpool(g)  # 1/4, 64
        g = self.enc.relu(g) 

        g = self.enc.layer1(g) # 1/4
        g = self.enc.layer2(g) # 1/8
        g = self.enc.layer3(g) # 1/16

        g = g.view(batch_size, num_objects, *g.shape[1:])
        g = self.enc.fuser(image_feat_f16, g)

        # if self.is_hidden_reinforce[0] and is_deep_update[0]==1:
        #     h = self.enc.hidden_reinforce(g, h)
        h = torch.where(
            torch.logical_and(self.is_hidden_dim, is_deep_update==torch.ones(1)),
            self.enc.hidden_reinforce(g, h),
            h
        )

        return g, h


class Segment(nn.Module):
    def __init__(self, decoder: Decoder, val_dim):
        super().__init__()
        self.dec = decoder
        self.fuser = FeatureFusionBlock(1024, val_dim, 512, 512)
        self.is_hidden_dim = torch.tensor([decoder.hidden_update is not None]).type(torch.bool)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out: torch.Tensor):
        batch_size, num_objects = memory_readout.shape[:2]
        g16 = torch.zeros_like(f16)
        
        # if is_hidden[0]==1:
        #     g16 = self.dec.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        # else:
        #     g16 = self.fuser(f16, memory_readout)
        g16 = torch.where(
            self.is_hidden_dim,
            self.dec.fuser(f16, torch.cat([memory_readout, hidden_state], 2)),
            self.fuser(f16, memory_readout)
            )

        g8 = self.dec.up_16_8(f8, g16)
        g4 = self.dec.up_8_4(f4, g8)
        logits = self.dec.pred(F.relu(g4.flatten(start_dim=0, end_dim=1)))

        # if h_out[0]==1 and self.is_hidden_updata:
        #     g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
        #     hidden_state = self.dec.hidden_update([g16, g8, g4], hidden_state)
        hidden_state = torch.where(
            torch.logical_and( self.is_hidden_dim, h_out==torch.ones(1) ),
            self.update(hidden_state, g16, g8, g4, logits, batch_size, num_objects),
            hidden_state
        )

        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])

        prob = torch.sigmoid(logits)
        # logits, prob = aggregate(prob, dim=1, return_logits=True)

        return hidden_state, logits, prob
    
    def update(self, h16, g16, g8, g4, logits, batch_size, num_objects) -> torch.Tensor:
        g4 =  torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
        return self.dec.hidden_update([g16, g8, g4], h16)