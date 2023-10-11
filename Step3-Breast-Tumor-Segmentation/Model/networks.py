import torch
import torch.nn as nn

basic_dims = 8
num_modals = 3
patch_size = [2, 8, 8]


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation, )

class general_conv3d_prenorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', is_training=True,
                 act_type='lrelu', relufactor=0.2):
        super(general_conv3d_prenorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

        self.norm = normalization(out_ch, norm=norm)
        if act_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        x = self.conv(x)
        return x

class fusion_prenorm(nn.Module):
    def __init__(self, in_channel=64, num_cls=1):
        super(fusion_prenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)


class fusion_layer(nn.Module):
    def __init__(self, in_channel=64, num_cls=1):
        super(fusion_layer, self).__init__()
        self.fusion_layer = nn.Sequential(
            general_conv3d_prenorm(in_channel * 2, in_channel, k_size=1, padding=0, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            general_conv3d_prenorm(in_channel, in_channel, k_size=1, padding=0, stride=1))

    def forward(self, x):
        return self.fusion_layer(x)

class MSA(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(MSA, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv3d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out

class Encoder(nn.Module):
    def __init__(self, flag=True):
        super(Encoder, self).__init__()
        if flag:
            self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                                   padding_mode='zeros', bias=True)
        else:
            self.e1_c1 = nn.Conv3d(in_channels=2, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                                   padding_mode='zeros', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='zeros')

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, pad_type='zeros')

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, pad_type='zeros')

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, pad_type='zeros')

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16, pad_type='zeros')

        self.attehtion_block1 = MSA(basic_dims * 1, basic_dims * 1, basic_dims * 1)
        self.attehtion_block2 = MSA(basic_dims * 2, basic_dims * 2, basic_dims * 2)
        self.attehtion_block3 = MSA(basic_dims * 4, basic_dims * 4, basic_dims * 4)
        self.attehtion_block4 = MSA(basic_dims * 8, basic_dims * 8, basic_dims * 8)
        self.attehtion_block5 = MSA(basic_dims * 16, basic_dims * 16, basic_dims * 16)

        self.Downsample1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.Downsample2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.Downsample3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.Downsample4 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')

        self.inforconv = general_conv3d_prenorm(1, basic_dims, pad_type='zeros')

    def forward(self, x, infor):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))
        infor1 = self.inforconv(infor)
        x1 = self.attehtion_block1(infor1, x1)

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))
        infor2 = self.Downsample1(infor1)
        x2 = self.attehtion_block2(infor2, x2)

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))
        infor3 = self.Downsample2(infor2)
        x3 = self.attehtion_block3(infor3, x3)

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        infor4 = self.Downsample3(infor3)
        x4 = self.attehtion_block4(infor4, x4)

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        infor5 = self.Downsample4(infor4)
        x5 = self.attehtion_block5(infor5, x5)

        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=1):
        super(Decoder_sep, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8, pad_type='zeros')
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0, pad_type='zeros')

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4, pad_type='zeros')
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0, pad_type='zeros')

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2, pad_type='zeros')
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0, pad_type='zeros')

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims, pad_type='zeros')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='zeros')

        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        logits = self.seg_layer(de_x1)
        return de_x5, de_x4, de_x3, de_x2, torch.sigmoid(logits)


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()

        self.inforconv = general_conv3d_prenorm(1, basic_dims, pad_type='zeros')
        self.Downsample1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2, pad_type='zeros')
        self.Downsample2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2, pad_type='zeros')
        self.Downsample3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2, pad_type='zeros')
        self.Downsample4 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2, pad_type='zeros')

    def forward(self, x):
        x1 = self.inforconv(x)
        x2 = self.Downsample1(x1)
        x3 = self.Downsample2(x2)
        x4 = self.Downsample3(x3)
        x5 = self.Downsample4(x4)
        return x1, x2, x3, x4, x5

class att(nn.Module):
    def __init__(self, in_channel=8):
        super(att, self).__init__()
        self.weight_layer = nn.Sequential(
                            nn.Conv3d(4*in_channel, 128, 1, padding=0, bias=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv3d(128, 1, 1, padding=0, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, infor, all):
        B, C, H, W, Z = infor.size()

        infor_avg = torch.mean(infor, dim=(2, 3, 4), keepdim=False)
        all_avg = torch.mean(all, dim=(2, 3, 4), keepdim=False)

        infor_avg = infor_avg.view(B, C, 1, 1, 1)
        all_avg = all_avg.view(B, 3 * C, 1, 1, 1)
        infor_avg = torch.cat((infor_avg, all_avg), dim=1)
        weight = torch.reshape(self.weight_layer(infor_avg), (B, 1))
        weight = self.sigmoid(weight).view(B, 1, 1, 1, 1)

        region_feat = infor * weight
        return region_feat

class MTG(nn.Module):
    def __init__(self, in_channel=8):
        super(MTG, self).__init__()

        self.att_adc = att(in_channel)
        self.att_t2 = att(in_channel)
        self.conv_layer = general_conv3d_prenorm(in_channel, in_channel, pad_type='zeros')
        self.fusion_prenorm = fusion_prenorm(in_channel)

    def forward(self, DCE, ADC, T2, InforDCE, InforADC, InforT2):

        newDCE = DCE * InforDCE
        newADC = ADC * InforADC
        newT2 = T2 * InforT2
        all = torch.cat((newDCE, newADC, newT2), dim=1)

        trust_ADC = self.att_adc(newADC, all)
        trust_T2 = self.att_t2(newT2, all)

        weight = self.fusion_prenorm(all) + trust_ADC + trust_T2

        x = self.conv_layer(weight)
        return x

class MoSID(nn.Module):
    def __init__(self, num_cls=1):
        super(MoSID, self).__init__()
        self.DCE_encoder = Encoder(flag=False)
        self.ADC_encoder = Encoder(flag=True)
        self.T2_encoder = Encoder(flag=True)

        self.decoder_sep = Decoder_sep(num_cls=num_cls)

        self.downsample1 = Downsample()
        self.downsample2 = Downsample()
        self.downsample3 = Downsample()
        self.MTG1 = MTG(in_channel=basic_dims * 1)
        self.MTG2 = MTG(in_channel=basic_dims * 2)
        self.MTG3 = MTG(in_channel=basic_dims * 4)
        self.MTG4 = MTG(in_channel=basic_dims * 8)
        self.MTG5 = MTG(in_channel=basic_dims * 16)

        act = nn.Sigmoid()
        self.out5 = conv_block_3d(basic_dims * 8, 1, act)
        self.out4 = conv_block_3d(basic_dims * 4, 1, act)
        self.out3 = conv_block_3d(basic_dims * 2, 1, act)
        self.out2 = conv_block_3d(basic_dims * 1, 1, act)

        self.is_training = True

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, DCE, Sub, ADC, T2, InforDCE, InforADC, InforT2):

        input1 = torch.cat((DCE, Sub), dim=1)

        DCE_x1, DCE_x2, DCE_x3, DCE_x4, DCE_x5 = self.DCE_encoder(input1, InforDCE)
        ADC_x1, ADC_x2, ADC_x3, ADC_x4, ADC_x5 = self.ADC_encoder(ADC, InforADC)
        T2_x1, T2_x2, T2_x3, T2_x4, T2_x5 = self.T2_encoder(T2, InforT2)

        InforDCE_x1, InforDCE_x2, InforDCE_x3, InforDCE_x4, InforDCE_x5 = self.downsample1(InforDCE)
        InforADC_x1, InforADC_x2, InforADC_x3, InforADC_x4, InforADC_x5 = self.downsample2(InforADC)
        InforT2_x1, InforT2_x2, InforT2_x3, InforT2_x4, InforT2_x5 = self.downsample3(InforT2)

        x1 = self.MTG1(DCE_x1, ADC_x1, T2_x1, InforDCE_x1, InforADC_x1, InforT2_x1)
        x2 = self.MTG2(DCE_x2, ADC_x2, T2_x2, InforDCE_x2, InforADC_x2, InforT2_x2)
        x3 = self.MTG3(DCE_x3, ADC_x3, T2_x3, InforDCE_x3, InforADC_x3, InforT2_x3)
        x4 = self.MTG4(DCE_x4, ADC_x4, T2_x4, InforDCE_x4, InforADC_x4, InforT2_x4)
        x5 = self.MTG5(DCE_x5, ADC_x5, T2_x5, InforDCE_x5, InforADC_x5, InforT2_x5)

        de_x5, de_x4, de_x3, de_x2, fuse_pred = self.decoder_sep(x1, x2, x3, x4, x5)
        out5, out4, out3, out2 = self.out5(de_x5), self.out4(de_x4), self.out3(de_x3), self.out2(de_x2)

        return out5, out4, out3, out2, fuse_pred




