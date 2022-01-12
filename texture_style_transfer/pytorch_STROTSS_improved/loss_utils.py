import math
import numpy as np
import torch

from pytorch_STROTSS_improved import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pairwise_distances_sq_l2(x, y):
    # NOTE: understand
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())

    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def pairwise_distances_cos(x, y):
    x_normalized = x / x.norm(dim=1, keepdim=True)
    y_normalized = y / y.norm(dim=1, keepdim=True)

    return 1 - torch.mm(x_normalized, y_normalized.t())


def compute_distance_matrix(X, Y, h=1.0, start_index=0, splits=[128*3+256*3+512*4], use_cosine_distance=True):
    M = torch.zeros(X.size(0), Y.size(0)).to(device)
    start_index = 0
    end_index = 0

    for i in range(len(splits)):
        if use_cosine_distance:
            end_index = start_index + splits[i]
            M = M + pairwise_distances_cos(X[:,start_index:end_index],Y[:,start_index:end_index])

            start_index = end_index
        else:
            end_index = start_index + splits[i]
            M = M + torch.sqrt(pairwise_distances_sq_l2(X[:,start_index:end_index],Y[:,start_index:end_index]))

            start_index = end_index

    return M


def compute_remd_loss(X, Y, h=None, use_cosine_distance=True,
              splits=[3+64+64+128+128+256+256+256+512+512],
              use_sinkhorn=False, sinkhorn_reg=0.1,
              sinkhorn_maxiter=30):

    d = X.size(1)


    if d == 3:
        X = utils.rgb_to_yuv(X.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        Y = utils.rgb_to_yuv(Y.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)

    else:
        X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    cost_matrix = compute_distance_matrix(X, Y, 1., use_cosine_distance=use_cosine_distance, splits=splits)

    if d==3:
        cost_matrix = cost_matrix+compute_distance_matrix(X, Y, 1.,use_cosine_distance=False, splits=splits)

    if use_sinkhorn:
        remd = sinkhorn_logsumexp(cost_matrix, reg=sinkhorn_reg, maxiter=sinkhorn_maxiter)
    else:
        m1, _ = cost_matrix.min(1)
        m2, _ = cost_matrix.min(0)
        remd = torch.max(m1.mean(),m2.mean())

    # # compare with exact OT distance
    # m, n = cost_matrix.size()
    # M = cost_matrix.detach().cpu().numpy()
    # a, b = (np.ones(m)/m).astype(float), (np.ones(n)/n).astype(float)
    # emd2 = ot.emd2(a, b, M)
    # print('REMD ', remd.item(), ' POT exact ', emd2, 'ratio', remd.item()/emd2)

    return remd


def compute_moment_loss(X, Y, moments=[1, 2]):

    d = X.size(1)
    ell = 0.

    Xo = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Yo = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    splits = [Xo.size(1)]

    cb = 0
    ce = 0
    for i in range(len(splits)):
        ce = cb + splits[i]
        X = Xo[:,cb:ce]
        Y = Yo[:,cb:ce]
        cb = ce

        mu_x = torch.mean(X,0,keepdim=True)
        mu_y = torch.mean(Y,0,keepdim=True)
        mu_d = torch.abs(mu_x-mu_y).mean()



        if 1 in moments:
            ell = ell + mu_d


        if 2 in moments:
            sig_x = torch.mm((X-mu_x).transpose(0,1), (X-mu_x))/X.size(0)
            sig_y = torch.mm((Y-mu_y).transpose(0,1), (Y-mu_y))/Y.size(0)


            sig_d = torch.abs(sig_x-sig_y).mean()
            ell = ell + sig_d


    return ell


def compute_dp_loss(X,Y):

    d = X.size(1)

    X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Xc = X[:,-2:]
    Y = Y[:,:-2]
    X = X[:,:-2]

    if 0:
        dM = torch.exp(-2.*compute_distance_matrix(Xc,Xc,1., use_cosine_distance=False))
        dM = dM/dM.sum(0,keepdim=True).detach()*dM.size(0)
    else:
        dM = 1.

    Mx = compute_distance_matrix(X,X,1.,use_cosine_distance=True,splits=[X.size(1)])
    Mx = Mx/Mx.sum(0,keepdim=True)

    My = compute_distance_matrix(Y,Y,1.,use_cosine_distance=True,splits=[X.size(1)])
    My = My/My.sum(0,keepdim=True)

    d = torch.abs(dM*(Mx-My)).mean()*X.size(0)

    return d


def compute_modified_cost_matrix(cost_matrix, u, v, reg):
    return (-cost_matrix + u.unsqueeze(1) + v.unsqueeze(0)) / reg


def sinkhorn_logsumexp(cost_matrix, reg=1e-1, maxiter=30, momentum=0.):
    m, n = cost_matrix.size()

    mu = torch.FloatTensor(m).fill_(1./m)
    nu = torch.FloatTensor(n).fill_(1./n)

    if torch.cuda.is_available():
        mu, nu = mu.cuda(), nu.cuda()

    u, v = 0. * mu, 0. * nu

    for i in range(maxiter):
        u1, v1 = u, v
        modified_cost_matrix = compute_modified_cost_matrix(cost_matrix, u, v, reg)
        u = reg * (torch.log(mu) - torch.logsumexp(modified_cost_matrix, dim=1)) + u
        v = reg * (torch.log(nu) - torch.logsumexp(modified_cost_matrix.t(), dim=1)) + v
        if momentum > 0.:
            u += momentum * (u - u1)
            v += momentum * (v - v1)

    modified_cost_matrix = compute_modified_cost_matrix(cost_matrix, u, v, reg)
    pi = torch.exp(modified_cost_matrix)
    sinkhorn_distance = torch.sum(pi * cost_matrix)

    return sinkhorn_distance

class RelaxedOptimalTransportSelfSimilarityLoss():

    def __init__(self, use_sinkhorn=False, sinkhorn_reg=1e-1, sinkhorn_maxiter=30, use_random=True):

        self.z_dist = torch.zeros(1).cuda()
        self.use_random = use_random

        self.rand_ixx = {}
        self.rand_ixy = {}
        self.rand_iy = {}
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_maxiter = sinkhorn_maxiter


    def eval(self, stylized_im_cnn_features, content_im_cnn_features, style_features,
             content_weight=4.0, moment_weight=1.0):

        final_loss = 0.
        for ri in range(len(self.rand_ixx.keys())):
            xx, xy, yx = self.get_feature_inds(ri=ri)
            x_st, c_st = self.spatial_feature_extract(stylized_im_cnn_features,
                                                      content_im_cnn_features,
                                                      xx, xy)

            ## Reshape Features from Style Distribution ##
            d = style_features[ri][0].size(1)
            reshaped_style_features = style_features[ri][0].view(1, d, -1, 1)

            content_loss = compute_dp_loss(x_st[:,:,:,:], c_st[:,:,:,:])

            fm = 3+2*64+128*2+256*3+512*2
            style_loss = compute_remd_loss(
                    x_st[:,:fm,:,:], reshaped_style_features[:,:fm,:,:], self.z_dist, splits=[fm],
                    use_cosine_distance=True, use_sinkhorn=self.use_sinkhorn, sinkhorn_reg=self.sinkhorn_reg,
                    sinkhorn_maxiter=self.sinkhorn_maxiter)

            moment_loss = compute_moment_loss(x_st[:,:-2,:,:], reshaped_style_features, moments=[1,2])

            palette_matching_loss = 1. / max(content_weight, 1.) * compute_remd_loss(
                    x_st[:,:3,:,:], reshaped_style_features[:,:3,:,:], self.z_dist, splits=[3],
                    use_cosine_distance=False, use_sinkhorn=self.use_sinkhorn)

            final_loss += (
                    content_weight * content_loss + style_loss + moment_weight * (moment_loss + palette_matching_loss))


        return final_loss / ((content_weight + 1. + moment_weight) *  len(self.rand_ixx.keys()))


    def init_inds(self, stylized_image_features, region_style_features, region_mask, i_region):
        const = 128**2

        # TODO: understand ixx, ixy, iy and rename
        try:
            tmp = self.rand_ixx[i_region]
            del tmp
        except:
            self.rand_ixx[i_region]= []
            self.rand_ixy[i_region]= []
            self.rand_iy[i_region]= []

        for i in range(len(region_style_features)):

            d = region_style_features[i].size(1)
            reshaped_region_style_features = region_style_features[i].view(1,d,-1,1)
            x_st = stylized_image_features[i]

            big_size = x_st.size(3)*x_st.size(2)

            if self.use_random:
                stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
                offset_x = np.random.randint(stride_x)

                stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
                offset_y = np.random.randint(stride_y)
            else:
                stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
                offset_x = stride_x//2

                stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
                offset_y = stride_y//2

            xx,xy = np.meshgrid(np.array(range(x_st.size(2)))[offset_x::stride_x], np.array(range(x_st.size(3)))[offset_y::stride_y] )

            xx = np.expand_dims(xx.flatten(),1)
            xy = np.expand_dims(xy.flatten(),1)
            xc = np.concatenate([xx,xy],1)

            try:
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]
            except:
                region_mask = region_mask[:,:]
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]

            self.rand_ixx[i_region].append(xc[:,0])
            self.rand_ixy[i_region].append(xc[:,1])

            zx = np.array(range(reshaped_region_style_features.size(2))).astype(np.int32)

            self.rand_iy[i_region].append(zx)


    def spatial_feature_extract(self, stylized_image_features, content_image_features, xx, xy):

        l2 = []
        l3 = []

        for i in range(len(stylized_image_features)):

            style_feats_i = stylized_image_features[i]
            content_feats_i = content_image_features[i]

            # TODO: understand xx, xxr, w00 ...
            if i > 0 and stylized_image_features[i-1].size(2) > stylized_image_features[i].size(2):
                xx = xx / 2.0
                xy = xy / 2.0

            xxm = np.floor(xx).astype(np.float32)
            xxr = (xx - xxm).astype(np.float32)

            xym = np.floor(xy).astype(np.float32)
            xyr = (xy - xym).astype(np.float32)

            w00 = torch.from_numpy((1.-xxr)*(1.-xyr)).view(1, 1, -1, 1).to(device)
            w01 = torch.from_numpy((1.-xxr)*xyr).view(1, 1, -1, 1).to(device)
            w10 = torch.from_numpy(xxr*(1.-xyr)).view(1, 1, -1, 1).to(device)
            w11 = torch.from_numpy(xxr*xyr).view(1, 1, -1, 1).to(device)


            xxm = np.clip(xxm.astype(np.int32),0,style_feats_i.size(2)-1)
            xym = np.clip(xym.astype(np.int32),0,style_feats_i.size(3)-1)

            s00 = xxm*style_feats_i.size(3)+xym
            s01 = xxm*style_feats_i.size(3)+np.clip(xym+1,0,style_feats_i.size(3)-1)
            s10 = np.clip(xxm+1,0,style_feats_i.size(2)-1)*style_feats_i.size(3)+(xym)
            s11 = np.clip(xxm+1,0,style_feats_i.size(2)-1)*style_feats_i.size(3)+np.clip(xym+1,0,style_feats_i.size(3)-1)


            style_feats_i = style_feats_i.view(1,style_feats_i.size(1),style_feats_i.size(2)*style_feats_i.size(3),1)
            style_feats_i = style_feats_i[:,:,s00,:].mul_(w00).add_(style_feats_i[:,:,s01,:].mul_(w01)).add_(style_feats_i[:,:,s10,:].mul_(w10)).add_(style_feats_i[:,:,s11,:].mul_(w11))


            content_feats_i = content_feats_i.view(1,content_feats_i.size(1),content_feats_i.size(2)*content_feats_i.size(3),1)
            content_feats_i = content_feats_i[:,:,s00,:].mul_(w00).add_(content_feats_i[:,:,s01,:].mul_(w01)).add_(content_feats_i[:,:,s10,:].mul_(w10)).add_(content_feats_i[:,:,s11,:].mul_(w11))

            l2.append(style_feats_i)
            l3.append(content_feats_i)

        x_st = torch.cat([li.contiguous() for li in l2],1)
        c_st = torch.cat([li.contiguous() for li in l3],1)


        xx = torch.from_numpy(xx).cuda().view(1,1,x_st.size(2),1).float()
        yy = torch.from_numpy(xy).cuda().view(1,1,x_st.size(2),1).float()


        x_st = torch.cat([x_st,xx,yy],1)
        c_st = torch.cat([c_st,xx,yy],1)

        return x_st, c_st


    def shuffle_feature_inds(self, i=0):
        if self.use_random:
            for ri in self.rand_ixx.keys():
                np.random.shuffle(self.rand_ixx[ri][i])
                np.random.shuffle(self.rand_ixy[ri][i])
                np.random.shuffle(self.rand_iy[ri][i])


    def get_feature_inds(self, ri=0, i=0, cnt=32**2):
        if self.use_random:
            xx = self.rand_ixx[ri][i][:cnt]
            xy = self.rand_ixy[ri][i][:cnt]
            yx = self.rand_iy[ri][i][:cnt]
        else:
            xx = self.rand_ixx[ri][i][::(self.rand_ixx[ri][i].shape[0]//cnt)]
            xy = self.rand_ixy[ri][i][::(self.rand_ixy[ri][i].shape[0]//cnt)]
            yx =  self.rand_iy[ri][i][::(self.rand_iy[ri][i].shape[0]//cnt)]

        return xx, xy, yx
