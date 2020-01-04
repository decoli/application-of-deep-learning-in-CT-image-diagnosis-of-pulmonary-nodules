import numpy as np
from visdom import Visdom

def visdom_loss(vis, epoch, loss, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([float(loss)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='loss',
        )
    )

def visdom_acc(vis, epoch, acc, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([float(acc)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='acc %',
        )
    )

def visdom_se(vis, epoch, se, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([float(se)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='se %',
        )
    )

def visdom_sp(vis, epoch, sp, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([float(sp)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='sp %',
        )
    )

def visdom_roc_auc(vis, epoch, roc_auc, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([float(roc_auc)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='roc_auc',
        )
    )

def visdom_scatter(vis, x, y, win, name):
    vis.scatter(
        X=x,
        Y=y,
        win=win,
        name=name,
        opts=dict(
            legend=['mu', 'logvar'],
            markersize=5,
        )
    )
