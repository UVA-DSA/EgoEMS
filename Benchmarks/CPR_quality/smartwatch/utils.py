class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    

def get_data_stats(data_loader):
    import torch
    depth_norm_vals=torch.empty(0)
    acc_vals=torch.empty(0)

    for i, batch in enumerate(data_loader):
        data=batch['smartwatch'].float()
        depth_gt=batch['depth_sensor'].squeeze()
        depth_gt_mask=depth_gt>0
        depth_gt_min = torch.where(depth_gt_mask, depth_gt, torch.tensor(float('inf')))
        mins=depth_gt_min.min(dim=1).values
        depth_norm=depth_gt-mins.unsqueeze(1)
        depth_norm_vals=torch.concat([depth_norm_vals,depth_norm[depth_gt_mask]])
        acc_vals=torch.concat([acc_vals,data])

    print(f'max depth: {depth_norm_vals.max()}')
    print(f'min depth: {depth_norm_vals.min()}')
    print(f'max acc: {torch.max(torch.max(acc_vals,dim=0).values,dim=0).values}')
    print(f'min acc: {torch.min(torch.min(acc_vals,dim=0).values,dim=0).values}')
