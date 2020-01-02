import numpy as np
from visdom import Visdom

def visdom_loss(vis, epoch, loss, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([int(loss)]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='loss %',
        )
    )

def visdom_acc(vis, epoch, acc, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([int(acc)]),
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
        Y=np.array([se]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='validation se and sp',
        )
    )

def visdom_sp(vis, epoch, sp, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([sp]),
        win=win,
        name=name,
        update='append',
        opts=dict(
            markers=True,
            showlegend=True,
            xlabel='epoch',
            ylabel='validation se and sp',
        )
    )

def visdom_roc_auc(vis, epoch, roc_auc, win, name):
    vis.line(
        X=np.array([int(epoch)]),
        Y=np.array([roc_auc]),
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
