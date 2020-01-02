import numpy as np
from visdom import Visdom

def visdom_loss(vis, win, name, epoch, loss):
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

def visdom_acc(vis, win, name, epoch, acc):
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

def visdom_se(vis, win, name, epoch, se):
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

def visdom_sp(vis, win, name, epoch, sp):
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

def visdom_roc_auc(vis, win, name, epoch):
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
